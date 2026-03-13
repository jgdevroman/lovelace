"""Community detection with hybrid weighting for microservice decomposition."""

import itertools
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx.algorithms import community

from lovelace.core.config import ClusteringConfig, ConstraintConfig
from lovelace.core.graph import DependencyGraph
from lovelace.core.vector import VectorEngine

logger = logging.getLogger(__name__)


@dataclass
class ClusterInfo:
    """Information about a detected cluster."""

    id: int
    suggested_name: str
    classes: List[str]
    class_count: int
    internal_cohesion: float
    external_coupling: float
    complexity_score: int
    dominant_type: str
    entities: List[str]
    entry_points: List[str]
    description: str = ""  # LLM-generated description
    rationale: str = ""  # LLM-generated rationale for grouping


@dataclass
class BoundaryEdge:
    """An edge that crosses cluster boundaries."""

    from_service: str
    to_service: str
    from_class: str
    to_class: str
    method: Optional[str]
    edge_type: str
    weight: float


class ClusterEngine:
    """Community detection with hybrid weighting."""

    def __init__(
        self,
        graph: DependencyGraph,
        vector_engine: VectorEngine,
        config: ClusteringConfig,
        constraints: List[ConstraintConfig],
    ):
        """
        Initialize the cluster engine.

        Args:
            graph: Dependency graph from Phase 1.
            vector_engine: Vector store with embeddings.
            config: Clustering configuration.
            constraints: List of constraint configurations from YAML.
        """
        self.graph = graph
        self.vector_engine = vector_engine
        self.config = config
        self.constraints = constraints

        # Extract weight coefficients
        weights = config.weights
        self.alpha = weights.get("structural", 0.5)
        self.beta = weights.get("semantic", 0.2)
        self.gamma = weights.get("data_gravity", 0.3)

        # Normalize weights to sum to 1.0
        total = self.alpha + self.beta + self.gamma
        if total > 0:
            self.alpha /= total
            self.beta /= total
            self.gamma /= total

        # Cluster assignments: class_id -> cluster_id
        self.clusters: Dict[str, int] = {}
        self.weighted_graph: Optional[nx.Graph] = None

    def _apply_constraints(self) -> Dict[str, str]:
        """
        Apply manual constraints by creating super-nodes.

        Returns:
            Dictionary mapping original class_id -> super_node_id.
        """
        node_mapping: Dict[str, str] = {}
        constraint_groups: Dict[str, Set[str]] = {}

        # Build constraint groups
        for constraint in self.constraints:
            group_name = f"__CONSTRAINT_{constraint.group}__"
            constraint_groups[group_name] = set()

            # Resolve class patterns
            for pattern in constraint.classes:
                # Simple pattern matching (supports * wildcard)
                if "*" in pattern:
                    prefix = pattern.replace("*", "")
                    for node_id in self.graph.graph.nodes():
                        if node_id.startswith(prefix):
                            constraint_groups[group_name].add(node_id)
                else:
                    constraint_groups[group_name].add(pattern)

        # Create mapping
        for group_name, class_set in constraint_groups.items():
            if len(class_set) > 1:
                # Map all classes in the group to the super-node
                for class_id in class_set:
                    if class_id in self.graph.graph.nodes():
                        node_mapping[class_id] = group_name

        return node_mapping

    def _get_data_gravity_weight(self, class_a: str, class_b: str) -> float:
        """
        Calculate data gravity weight between two classes.

        Args:
            class_a: First class fully qualified name.
            class_b: Second class fully qualified name.

        Returns:
            Data gravity weight (0.0 to 1.0).
        """
        # Check if both are entities
        node_a = self.graph.graph.nodes.get(class_a, {})
        node_b = self.graph.graph.nodes.get(class_b, {})

        if node_a.get("type") != "Entity" or node_b.get("type") != "Entity":
            return 0.0

        # Check for JPA relationships
        # We need to check if there's a JPA relationship between them
        # This requires access to the original ClassMetadata, which we'll need to store
        # For now, check if there's a direct edge with high weight (indicating strong coupling)
        if self.graph.graph.has_edge(class_a, class_b):
            edge_data = self.graph.graph[class_a][class_b]
            # If there's a strong structural connection, assume data gravity
            if edge_data.get("weight", 0) > 5.0:
                return 1.0

        return 0.0

    def _get_structural_weight(self, class_a: str, class_b: str) -> float:
        """
        Calculate structural weight (method calls, imports).

        Args:
            class_a: First class fully qualified name.
            class_b: Second class fully qualified name.

        Returns:
            Normalized structural weight (0.0 to 1.0).
        """
        weight = 0.0

        # Check for edges in both directions
        if self.graph.graph.has_edge(class_a, class_b):
            edge_data = self.graph.graph[class_a][class_b]
            edge_weight = edge_data.get("weight", 0.0)
            edge_types = edge_data.get("types", [])

            # Weight by edge type
            for edge_type in edge_types:
                if edge_type == "CALLS":
                    weight += edge_weight * 2.0  # Method calls are strong
                elif edge_type == "INHERITS":
                    weight += edge_weight * 3.0  # Inheritance is very strong
                elif edge_type == "IMPORTS":
                    weight += edge_weight * 0.5  # Imports are weak

        if self.graph.graph.has_edge(class_b, class_a):
            edge_data = self.graph.graph[class_b][class_a]
            edge_weight = edge_data.get("weight", 0.0)
            edge_types = edge_data.get("types", [])

            for edge_type in edge_types:
                if edge_type == "CALLS":
                    weight += edge_weight * 2.0
                elif edge_type == "INHERITS":
                    weight += edge_weight * 3.0
                elif edge_type == "IMPORTS":
                    weight += edge_weight * 0.5

        # Normalize (max weight is roughly 10.0 for strong relationships)
        return min(weight / 10.0, 1.0)

    def _get_semantic_weight(self, class_a: str, class_b: str) -> float:
        """
        Calculate semantic similarity weight.

        Args:
            class_a: First class fully qualified name.
            class_b: Second class fully qualified name.

        Returns:
            Semantic similarity (0.0 to 1.0).
        """
        try:
            return self.vector_engine.get_similarity(class_a, class_b)
        except Exception as e:
            logger.warning(f"Error getting semantic similarity: {e}")
            return 0.0

    def compute_hybrid_weights(self) -> nx.Graph:
        """
        Create weighted undirected graph for clustering.

        Returns:
            NetworkX Graph with hybrid edge weights.
        """
        logger.info("Computing hybrid weights...")

        # Create undirected graph
        weighted_graph = nx.Graph()

        # Add all nodes
        for node_id in self.graph.graph.nodes():
            weighted_graph.add_node(node_id)

        # Compute weights for all pairs of connected nodes
        all_nodes = list(self.graph.graph.nodes())
        edge_count = 0

        for i, node_a in enumerate(all_nodes):
            for node_b in all_nodes[i + 1 :]:
                # Only consider pairs that have some connection
                has_edge_ab = self.graph.graph.has_edge(node_a, node_b)
                has_edge_ba = self.graph.graph.has_edge(node_b, node_a)

                if has_edge_ab or has_edge_ba:
                    # Calculate components
                    structural = self._get_structural_weight(node_a, node_b)
                    semantic = self._get_semantic_weight(node_a, node_b)
                    data_gravity = self._get_data_gravity_weight(node_a, node_b)

                    # Apply hybrid formula
                    hybrid_weight = (
                        self.alpha * structural + self.beta * semantic + self.gamma * data_gravity
                    )

                    if hybrid_weight > 0:
                        weighted_graph.add_edge(node_a, node_b, weight=hybrid_weight)
                        edge_count += 1

        logger.info(f"Created weighted graph with {edge_count} edges")
        self.weighted_graph = weighted_graph
        return weighted_graph

    def detect_communities(
        self, algorithm: Optional[str] = None, resolution: Optional[float] = None
    ) -> Dict[str, int]:
        """
        Detect communities/clusters using specified algorithm.

        Args:
            algorithm: 'louvain' (fast) or 'girvan_newman' (accurate). If None, uses config.
            resolution: Louvain resolution parameter. If None, uses config.

        Returns:
            Dict mapping class_id -> cluster_id.
        """
        if self.weighted_graph is None:
            self.compute_hybrid_weights()

        algorithm = algorithm or self.config.algorithm
        resolution = resolution or self.config.resolution

        logger.info(f"Detecting communities using {algorithm} algorithm...")

        # Apply constraints first
        constraint_mapping = self._apply_constraints()

        # Create a working graph with super-nodes
        working_graph = self.weighted_graph.copy()

        # Merge constrained nodes into super-nodes
        super_nodes: Dict[str, Set[str]] = {}
        for original_node, super_node in constraint_mapping.items():
            if super_node not in super_nodes:
                super_nodes[super_node] = set()
            super_nodes[super_node].add(original_node)

        # Create super-node graph
        if super_nodes:
            # Remove original nodes and add super-nodes
            for super_node, members in super_nodes.items():
                working_graph.add_node(super_node)
                # Aggregate edges
                for member in members:
                    if member in working_graph:
                        # Transfer edges
                        neighbors = list(working_graph.neighbors(member))
                        for neighbor in neighbors:
                            if neighbor not in members:
                                edge_weight = working_graph[member][neighbor].get("weight", 0.0)
                                if working_graph.has_edge(super_node, neighbor):
                                    # Sum weights
                                    working_graph[super_node][neighbor]["weight"] += edge_weight
                                else:
                                    working_graph.add_edge(super_node, neighbor, weight=edge_weight)
                        working_graph.remove_node(member)

        # Run clustering algorithm
        if algorithm == "louvain":
            communities = community.louvain_communities(
                working_graph, weight="weight", resolution=resolution, seed=42
            )
        elif algorithm == "girvan_newman":
            # Girvan-Newman requires number of communities
            # Use modularity to determine optimal number
            comp = community.girvan_newman(working_graph)
            # Use first partition (2 communities) as starting point
            # This is a simplification - in practice, you'd want to optimize
            communities = next(comp)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Map communities back to original nodes
        cluster_assignments: Dict[str, int] = {}
        for cluster_id, community_set in enumerate(communities):
            for node in community_set:
                if node in super_nodes:
                    # Expand super-node
                    for original_node in super_nodes[node]:
                        cluster_assignments[original_node] = cluster_id
                else:
                    cluster_assignments[node] = cluster_id

        # Assign unassigned nodes to their own clusters
        all_nodes = set(self.graph.graph.nodes())
        assigned_nodes = set(cluster_assignments.keys())
        unassigned = all_nodes - assigned_nodes

        next_cluster_id = len(communities)
        for node in unassigned:
            cluster_assignments[node] = next_cluster_id
            next_cluster_id += 1

        self.clusters = cluster_assignments
        logger.info(f"Detected {len(set(cluster_assignments.values()))} clusters")

        return cluster_assignments

    def get_cluster_report(self) -> List[ClusterInfo]:
        """
        Generate detailed cluster information.

        Returns:
            List of ClusterInfo objects with metrics.
        """
        if not self.clusters:
            self.detect_communities()

        cluster_groups: Dict[int, List[str]] = {}
        for class_id, cluster_id in self.clusters.items():
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(class_id)

        cluster_infos = []
        used_names: Dict[str, int] = {}  # Track name usage for deduplication

        for cluster_id, classes in cluster_groups.items():
            # Calculate metrics
            internal_edges = 0
            external_edges = 0
            total_complexity = 0
            type_counts: Dict[str, int] = {}
            entities = []
            entry_points = []

            class_set = set(classes)

            for class_id in classes:
                node_data = self.graph.graph.nodes[class_id]
                total_complexity += node_data.get("complexity_score", 0)

                node_type = node_data.get("type", "Unknown")
                type_counts[node_type] = type_counts.get(node_type, 0) + 1

                if node_type == "Entity":
                    entities.append(class_id)
                elif node_type == "Controller":
                    entry_points.append(class_id)

                # Count edges
                for neighbor in self.graph.graph.successors(class_id):
                    if neighbor in class_set:
                        internal_edges += 1
                    else:
                        external_edges += 1

                for neighbor in self.graph.graph.predecessors(class_id):
                    if neighbor not in class_set:
                        external_edges += 1

            # Calculate cohesion and coupling
            total_possible_internal = len(classes) * (len(classes) - 1) / 2
            internal_cohesion = internal_edges / total_possible_internal if total_possible_internal > 0 else 0.0
            external_coupling = external_edges / len(classes) if classes else 0.0

            # Find dominant type
            dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "Unknown"

            # Suggest service name with deduplication
            base_name = self._suggest_service_name(classes, entities, dominant_type)
            suggested_name = self._deduplicate_name(base_name, used_names)

            cluster_info = ClusterInfo(
                id=cluster_id,
                suggested_name=suggested_name,
                classes=classes,
                class_count=len(classes),
                internal_cohesion=internal_cohesion,
                external_coupling=external_coupling,
                complexity_score=total_complexity,
                dominant_type=dominant_type,
                entities=entities,
                entry_points=entry_points,
            )

            cluster_infos.append(cluster_info)

        return cluster_infos

    def _suggest_service_name(self, classes: List[str], entities: List[str], dominant_type: str) -> str:
        """
        Suggest a service name based on cluster contents.

        Args:
            classes: List of class IDs in the cluster.
            entities: List of entity class IDs.
            dominant_type: Dominant node type.

        Returns:
            Suggested service name.
        """
        # Rule 1: If cluster has a dominant entity, use "{entity}-service"
        if entities:
            # Use the first entity's simple name
            entity_id = entities[0]
            entity_node = self.graph.graph.nodes[entity_id]
            simple_name = entity_node.get("simple_name", entity_id.split(".")[-1])
            # Remove "Entity" suffix if present
            if simple_name.endswith("Entity"):
                simple_name = simple_name[:-6]
            return f"{simple_name.lower()}-service"

        # Rule 2: If cluster has a dominant package, use "{package}-service"
        package_counts: Dict[str, int] = {}
        for class_id in classes:
            node_data = self.graph.graph.nodes[class_id]
            package = node_data.get("package", "")
            if package:
                package_counts[package] = package_counts.get(package, 0) + 1

        if package_counts:
            dominant_package = max(package_counts.items(), key=lambda x: x[1])[0]
            # Extract last part of package
            package_parts = dominant_package.split(".")
            if package_parts:
                return f"{package_parts[-1].lower()}-service"

        # Rule 3: Fall back to "service-{cluster_id}"
        return f"service-{self.clusters.get(classes[0], 0)}"

    def _deduplicate_name(self, base_name: str, used_names: Dict[str, int]) -> str:
        """
        Ensure service name is unique by appending a suffix if needed.

        Args:
            base_name: The proposed service name.
            used_names: Dictionary tracking how many times each name has been used.

        Returns:
            A unique service name.
        """
        if base_name not in used_names:
            used_names[base_name] = 1
            return base_name

        # Name already used, append counter
        count = used_names[base_name]
        used_names[base_name] = count + 1
        unique_name = f"{base_name}-{count + 1}"

        # Track the new unique name too
        used_names[unique_name] = 1
        return unique_name

    def get_boundary_edges(self) -> List[BoundaryEdge]:
        """
        Find edges that cross cluster boundaries.

        These represent API calls that will need to be
        converted to HTTP/gRPC calls after extraction.

        Returns:
            List of BoundaryEdge objects.
        """
        if not self.clusters:
            self.detect_communities()

        boundary_edges = []

        # Get cluster names for services
        cluster_names: Dict[int, str] = {}
        cluster_report = self.get_cluster_report()
        for info in cluster_report:
            cluster_names[info.id] = info.suggested_name

        # Find edges that cross boundaries
        for source, target, data in self.graph.graph.edges(data=True):
            source_cluster = self.clusters.get(source)
            target_cluster = self.clusters.get(target)

            if source_cluster is not None and target_cluster is not None and source_cluster != target_cluster:
                boundary_edge = BoundaryEdge(
                    from_service=cluster_names.get(source_cluster, f"service-{source_cluster}"),
                    to_service=cluster_names.get(target_cluster, f"service-{target_cluster}"),
                    from_class=source,
                    to_class=target,
                    method=data.get("method_name"),
                    edge_type=data.get("type", "UNKNOWN"),
                    weight=data.get("weight", 0.0),
                )
                boundary_edges.append(boundary_edge)

        return boundary_edges

