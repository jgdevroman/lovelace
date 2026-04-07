"""LLM-first clustering for microservice boundary detection."""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from lovelace.core.clustering import BoundaryEdge, ClusterInfo
from lovelace.core.graph import DependencyGraph
from lovelace.core.llm import LLMClient
from lovelace.core.parser import ClassMetadata
from lovelace.core.token_budget import TokenBudget

logger = logging.getLogger(__name__)


@dataclass
class DomainClass:
    """A class identified as belonging to the application domain."""
    
    fqn: str  # Fully qualified name
    simple_name: str
    package: str
    summary: str  # Brief description for LLM context
    class_type: str  # Entity, Controller, Service, Repository, etc.


class LLMClusterEngine:
    """LLM-driven clustering engine for microservice decomposition."""

    def __init__(
        self,
        graph: DependencyGraph,
        parsed_classes: List[ClassMetadata],
        llm_client: LLMClient,
    ):
        """
        Initialize the LLM cluster engine.

        Args:
            graph: Dependency graph from analysis phase.
            parsed_classes: List of parsed Java classes.
            llm_client: LLM client for API calls.
        """
        self.graph = graph
        self.parsed_classes = parsed_classes
        self.llm_client = llm_client
        self.token_budget = TokenBudget(model=llm_client.model)
        
        # Results (cached)
        self.domain_classes: List[DomainClass] = []
        self.excluded_classes: List[str] = []  # Classes filtered out (not domain)
        self.service_proposals: List[dict] = []  # Cache service proposals
        self.clusters: Dict[str, int] = {}  # class_fqn -> cluster_id
        self.cluster_info: List[ClusterInfo] = []

    def _create_class_summary(self, class_meta: ClassMetadata) -> str:
        """Create a brief summary of a class for LLM context."""
        parts = [class_meta.fully_qualified_name]
        
        if class_meta.annotations:
            parts.append(f"[@{', @'.join(class_meta.annotations[:3])}]")
        
        if class_meta.methods:
            method_names = [m.name for m in class_meta.methods[:5]]
            parts.append(f"methods: {', '.join(method_names)}")
        
        return " | ".join(parts)

    def _get_class_type(self, class_meta: ClassMetadata) -> str:
        """Determine the type of class based on annotations."""
        annotations = set(class_meta.annotations)
        
        if "Entity" in annotations or "Table" in annotations:
            return "Entity"
        elif "Controller" in annotations or "RestController" in annotations:
            return "Controller"
        elif "Service" in annotations:
            return "Service"
        elif "Repository" in annotations:
            return "Repository"
        elif "Component" in annotations:
            return "Component"
        elif "Configuration" in annotations:
            return "Configuration"
        elif class_meta.is_interface:
            return "Interface"
        else:
            return "Class"

    def filter_domain_classes(self) -> List[DomainClass]:
        """
        Phase 1: Use LLM to identify domain classes vs external dependencies.
        
        Returns:
            List of DomainClass objects representing application domain.
        """
        logger.info("Phase 1: Filtering domain classes with LLM...")
        
        # Build class list for LLM
        class_list = []
        for cls in self.parsed_classes:
            class_list.append({
                "fqn": cls.fully_qualified_name,
                "simple_name": cls.simple_name,
                "package": cls.package_name,
                "type": self._get_class_type(cls),
                "annotations": cls.annotations[:5] if cls.annotations else [],
            })
        
        # Check if we need to chunk
        class_json = json.dumps(class_list, indent=2)
        if not self.token_budget.check_fits(class_json):
            logger.info("Class list too large, using chunked processing")
            return self._filter_domain_classes_chunked(class_list)
        
        prompt = f"""Analyze these Java classes and identify which ones belong to the APPLICATION DOMAIN vs which are EXTERNAL DEPENDENCIES or INFRASTRUCTURE.

Classes:
```json
{class_json}
```

For each class, determine if it's a DOMAIN class (application business logic) or EXTERNAL (framework, library, infrastructure).

Criteria for DOMAIN classes:
- Part of the application's core business logic
- Custom entities, services, controllers, repositories
- Application-specific DTOs, value objects

Criteria for EXTERNAL/INFRASTRUCTURE:
- Framework configuration classes (Spring, JPA config)
- Generic utility/helper classes from libraries
- Auto-generated classes
- Test classes

Return ONLY the fully qualified names of DOMAIN classes as a JSON array:
```json
["com.example.app.DomainClass1", "com.example.app.DomainClass2"]
```"""

        messages = [
            {"role": "system", "content": "You are an expert Java architect specializing in domain-driven design."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.chat(messages, temperature=0.1)
        
        # Parse response
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            domain_fqns = set(json.loads(content))
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to parse LLM response: {e}, using all classes")
            domain_fqns = {cls.fully_qualified_name for cls in self.parsed_classes}
        
        # Build DomainClass list and track excluded classes
        self.domain_classes = []
        self.excluded_classes = []
        for cls in self.parsed_classes:
            if cls.fully_qualified_name in domain_fqns:
                self.domain_classes.append(DomainClass(
                    fqn=cls.fully_qualified_name,
                    simple_name=cls.simple_name,
                    package=cls.package_name,
                    summary=self._create_class_summary(cls),
                    class_type=self._get_class_type(cls),
                ))
            else:
                self.excluded_classes.append(cls.fully_qualified_name)
        
        logger.info(f"Identified {len(self.domain_classes)} domain classes, excluded {len(self.excluded_classes)}")
        return self.domain_classes

    def _filter_domain_classes_chunked(self, class_list: List[dict]) -> List[DomainClass]:
        """Process class filtering in chunks for large codebases."""
        chunk_size = 100
        domain_fqns: Set[str] = set()
        
        for i in range(0, len(class_list), chunk_size):
            chunk = class_list[i:i + chunk_size]
            chunk_json = json.dumps(chunk, indent=2)
            
            prompt = f"""Identify DOMAIN classes (application business logic) from this list.
Return only FQNs of domain classes as JSON array.

Classes:
```json
{chunk_json}
```"""
            
            messages = [
                {"role": "system", "content": "You are a Java architect. Return only a JSON array of domain class FQNs."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.chat(messages, temperature=0.1)
            
            try:
                content = response.content.strip()
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.split("```")[0].strip()
                domain_fqns.update(json.loads(content))
            except (json.JSONDecodeError, IndexError):
                # On failure, include all classes from this chunk
                domain_fqns.update(c["fqn"] for c in chunk)
        
        # Build DomainClass list
        self.domain_classes = []
        for cls in self.parsed_classes:
            if cls.fully_qualified_name in domain_fqns:
                self.domain_classes.append(DomainClass(
                    fqn=cls.fully_qualified_name,
                    simple_name=cls.simple_name,
                    package=cls.package_name,
                    summary=self._create_class_summary(cls),
                    class_type=self._get_class_type(cls),
                ))
        
        return self.domain_classes

    def propose_services(self) -> List[dict]:
        """
        Phase 2-3: LLM proposes service boundaries and assigns classes.
        
        Returns:
            List of service proposals with class assignments.
        """
        # Return cached results if available
        if self.service_proposals:
            return self.service_proposals
            
        if not self.domain_classes:
            self.filter_domain_classes()
        
        logger.info("Phase 2-3: Proposing service boundaries...")
        
        # Build domain summary
        domain_summary = []
        for dc in self.domain_classes:
            domain_summary.append(f"- {dc.fqn} [{dc.class_type}]")
        
        domain_text = "\n".join(domain_summary)
        
        # Check context window
        if not self.token_budget.check_fits(domain_text):
            return self._propose_services_hierarchical()
        
        prompt = f"""You are decomposing a Java monolith into microservices.

DOMAIN CLASSES:
{domain_text}

Analyze these classes and propose microservice boundaries. Follow these rules:

1. **BOUNDED CONTEXTS**: Group by aggregate root with its closely related entities.
   - An aggregate root and its child entities belong TOGETHER (e.g., Order with OrderLine)
   - Classes in the same package that share a domain concept belong together
   - Controller + Service + Repository + Entity for the same domain = one service

2. **SHARED/BASE CLASSES**: Abstract classes, utilities, and base entities should be:
   - Put in a cluster named "shared-service"
   - These will become a shared dependency, not a microservice

3. **NAMING**: Name services after their primary domain concept
   - Use "[domain]-service" pattern based on the aggregate root or main entity

4. **PRACTICAL GRANULARITY**: Let the number of services emerge from the domain structure.
   - Each service should own a cohesive set of aggregate roots — do not merge unrelated domains just to reduce count.
   - Equally, do not split a single aggregate across multiple services — that creates a distributed monolith.
   - If classes are in the same package and work together, keep them together.

5. **AGGREGATE INTEGRITY**: Never split a single aggregate across multiple services.
   - Child entities, value objects, and lookup types belong with their aggregate root.
   - Only separate entities into different services if they are truly independent aggregate roots with no parent-child relationship.
   - A service with many entities is acceptable if they form a single cohesive aggregate (e.g., an Order aggregate with OrderLine, OrderItem, ShippingAddress).

Return ONLY valid JSON:
```json
{{
  "services": [
    {{
      "name": "domain-service",
      "description": "What this service does",
      "rationale": "Why these classes belong together",
      "classes": ["fully.qualified.ClassName1", "fully.qualified.ClassName2"]
    }}
  ]
}}
```"""

        messages = [
            {"role": "system", "content": "You are a microservices architect. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.chat(messages, temperature=0.3)
        
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            self.service_proposals = result.get("services", [])
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to parse service proposals: {e}")
            # Fallback: single service with all classes
            self.service_proposals = [{
                "name": "monolith-service",
                "description": "All domain classes",
                "rationale": "LLM parsing failed",
                "classes": [dc.fqn for dc in self.domain_classes]
            }]
        
        return self.service_proposals

    def _propose_services_hierarchical(self) -> List[dict]:
        """Propose services using hierarchical summarization for large codebases."""
        logger.info("Using hierarchical approach for large codebase")
        
        # Group by package
        packages: Dict[str, List[DomainClass]] = {}
        for dc in self.domain_classes:
            if dc.package not in packages:
                packages[dc.package] = []
            packages[dc.package].append(dc)
        
        # Summarize packages
        package_summaries = []
        for pkg, classes in packages.items():
            types = [dc.class_type for dc in classes]
            type_counts = {t: types.count(t) for t in set(types)}
            package_summaries.append({
                "package": pkg,
                "class_count": len(classes),
                "types": type_counts,
                "classes": [dc.fqn for dc in classes]
            })
        
        pkg_json = json.dumps(package_summaries, indent=2)
        
        prompt = f"""Decompose these packages into FINE-GRAINED microservices.

PACKAGES:
```json
{pkg_json}
```

Rules:
1. ONE ENTITY PER SERVICE - create separate services for each distinct business entity
2. SHARED CLASSES - put base/abstract classes in "common-library" 
3. Use descriptive service names based on the domain entity they handle

Return JSON:
```json
{{
  "services": [
    {{"name": "descriptive-name", "description": "...", "rationale": "...", "classes": ["fqn1", "fqn2"]}}
  ]
}}
```"""

        messages = [
            {"role": "system", "content": "You are a microservices architect."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.chat(messages, temperature=0.3)
        
        try:
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.split("```")[0].strip()
            
            result = json.loads(content)
            return result.get("services", [])
        except (json.JSONDecodeError, IndexError):
            # Fallback: one service per package
            return [
                {
                    "name": f"{pkg.split('.')[-1]}-service",
                    "description": f"Classes from {pkg}",
                    "rationale": "Package-based grouping",
                    "classes": [dc.fqn for dc in classes]
                }
                for pkg, classes in packages.items()
            ]

    def detect_communities(self) -> Dict[str, int]:
        """
        Run full LLM clustering pipeline.
        
        Returns:
            Dict mapping class FQN to cluster ID.
        """
        logger.info("Starting LLM-based clustering...")
        
        # Phase 1: Filter domain classes
        self.filter_domain_classes()
        
        # Phase 2-3: Propose and assign
        services = self.propose_services()
        
        # Build cluster mapping
        self.clusters = {}
        for cluster_id, service in enumerate(services):
            for class_fqn in service.get("classes", []):
                self.clusters[class_fqn] = cluster_id
        
        # Assign unassigned domain classes to their own cluster
        next_id = len(services)
        for dc in self.domain_classes:
            if dc.fqn not in self.clusters:
                self.clusters[dc.fqn] = next_id
                next_id += 1
        
        logger.info(f"Created {len(set(self.clusters.values()))} clusters")
        return self.clusters

    def get_cluster_report(self) -> List[ClusterInfo]:
        """Generate ClusterInfo objects compatible with existing system."""
        if not self.clusters:
            self.detect_communities()
        
        # Get service proposals for names
        services = self.propose_services()
        service_map = {i: s for i, s in enumerate(services)}
        
        # Group classes by cluster
        cluster_groups: Dict[int, List[str]] = {}
        for class_fqn, cluster_id in self.clusters.items():
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(class_fqn)
        
        self.cluster_info = []
        for cluster_id, classes in cluster_groups.items():
            service = service_map.get(cluster_id, {})
            
            # Determine entities and entry points
            entities = []
            entry_points = []
            type_counts: Dict[str, int] = {}
            
            for class_fqn in classes:
                node_data = self.graph.graph.nodes.get(class_fqn, {})
                node_type = node_data.get("type", "Unknown")
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
                
                if node_type == "Entity":
                    entities.append(class_fqn)
                elif node_type == "Controller":
                    entry_points.append(class_fqn)
            
            dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "Unknown"
            
            # Calculate metrics - count edges in BOTH directions
            internal_edges = 0
            external_edges = 0
            class_set = set(classes)
            
            for class_fqn in classes:
                if class_fqn not in self.graph.graph:
                    continue
                # Count successors (outgoing edges)
                for neighbor in self.graph.graph.successors(class_fqn):
                    if neighbor in class_set:
                        internal_edges += 1
                    else:
                        external_edges += 1
                # Count predecessors (incoming edges) 
                for neighbor in self.graph.graph.predecessors(class_fqn):
                    if neighbor in class_set:
                        internal_edges += 1
                    else:
                        external_edges += 1
            
            # Avoid double-counting: divide by 2 for internal edges
            internal_edges = internal_edges // 2
            
            # Cohesion: actual internal edges / max possible edges
            total_possible = len(classes) * (len(classes) - 1) / 2
            edge_cohesion = internal_edges / total_possible if total_possible > 0 else 0.0
            
            # Package-based cohesion (since DI relationships aren't captured as imports)
            packages = set()
            for class_fqn in classes:
                pkg = ".".join(class_fqn.split(".")[:-1])
                packages.add(pkg)
            
            # Package cohesion: 1.0 if all same package, decreases with more packages
            package_cohesion = 1.0 / len(packages) if packages else 0.0

            # Architectural cohesion: score stack completeness per package, then average.
            # A complete package stack has Entity + Controller + Repository.
            package_stack = {
                pkg: {
                    "has_entity": False,
                    "has_controller": False,
                    "has_repository": False,
                }
                for pkg in packages
            }

            for class_fqn in classes:
                pkg = ".".join(class_fqn.split(".")[:-1])
                node_type = self.graph.graph.nodes.get(class_fqn, {}).get("type", "Unknown")

                if pkg not in package_stack:
                    continue

                if node_type == "Entity":
                    package_stack[pkg]["has_entity"] = True
                if node_type == "Controller":
                    package_stack[pkg]["has_controller"] = True
                if node_type == "Repository" or "Repository" in class_fqn:
                    package_stack[pkg]["has_repository"] = True

            package_completeness_scores = [
                (
                    int(flags["has_entity"])
                    + int(flags["has_controller"])
                    + int(flags["has_repository"])
                ) / 3.0
                for flags in package_stack.values()
            ]
            avg_package_completeness = (
                sum(package_completeness_scores) / len(package_completeness_scores)
                if package_completeness_scores
                else 0.0
            )

            # Combine: reward clusters with fewer packages and more complete per-package stacks.
            cohesion = package_cohesion * (0.5 + 0.5 * avg_package_completeness)
            
            # Use edge cohesion if it's higher (rare but possible)
            cohesion = max(edge_cohesion, cohesion)
            coupling = external_edges / len(classes) if classes else 0.0
            
            info = ClusterInfo(
                id=cluster_id,
                suggested_name=service.get("name", f"service-{cluster_id}"),
                classes=classes,
                class_count=len(classes),
                internal_cohesion=cohesion,
                external_coupling=coupling,
                complexity_score=sum(
                    self.graph.graph.nodes.get(c, {}).get("complexity_score", 0)
                    for c in classes
                ),
                dominant_type=dominant_type,
                entities=entities,
                entry_points=entry_points,
                description=service.get("description", ""),
                rationale=service.get("rationale", ""),
            )
            self.cluster_info.append(info)
        
        return self.cluster_info

    def get_boundary_edges(self) -> List[BoundaryEdge]:
        """Find edges that cross cluster boundaries."""
        if not self.clusters:
            self.detect_communities()
        
        # Get cluster names
        cluster_names: Dict[int, str] = {}
        for info in self.get_cluster_report():
            cluster_names[info.id] = info.suggested_name
        
        boundary_edges = []
        for source, target, data in self.graph.graph.edges(data=True):
            source_cluster = self.clusters.get(source)
            target_cluster = self.clusters.get(target)
            
            if source_cluster is not None and target_cluster is not None and source_cluster != target_cluster:
                boundary_edges.append(BoundaryEdge(
                    from_service=cluster_names.get(source_cluster, f"service-{source_cluster}"),
                    to_service=cluster_names.get(target_cluster, f"service-{target_cluster}"),
                    from_class=source,
                    to_class=target,
                    method=data.get("method_name"),
                    edge_type=data.get("type", "UNKNOWN"),
                    weight=data.get("weight", 0.0),
                ))
        
        return boundary_edges
