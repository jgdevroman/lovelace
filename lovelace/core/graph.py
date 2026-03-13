"""Dependency graph builder using NetworkX."""

from typing import Dict, List, Optional, Set

import networkx as nx

from lovelace.core.parser import ClassMetadata


class DependencyGraph:
    """Wrapper around NetworkX DiGraph for dependency analysis."""

    def __init__(self):
        """Initialize an empty dependency graph."""
        self.graph = nx.DiGraph()

    def add_class_node(self, class_metadata: ClassMetadata) -> None:
        """
        Add a class node to the graph.

        Args:
            class_metadata: ClassMetadata object with class information.
        """
        node_id = class_metadata.fully_qualified_name

        node_type = self._classify_node_type(class_metadata)
        complexity_score = self._calculate_complexity(class_metadata)

        self.graph.add_node(
            node_id,
            type=node_type,
            complexity_score=complexity_score,
            file_path=str(class_metadata.file_path),
            package=class_metadata.package_name,
            simple_name=class_metadata.simple_name,
            is_interface=class_metadata.is_interface,
            is_abstract=class_metadata.is_abstract,
            annotations=class_metadata.annotations,
            method_count=len(class_metadata.methods),
            field_count=len(class_metadata.fields),
            superclass=class_metadata.superclass,  # For inheritance flattening
            fields=class_metadata.fields,  # For inheritance flattening (type, name) tuples
        )

    def _classify_node_type(self, class_metadata: ClassMetadata) -> str:
        """
        Classify node type based on annotations and naming conventions.

        Args:
            class_metadata: ClassMetadata object.

        Returns:
            Node type: Service, Entity, Controller, Utility, or Unknown.
        """
        annotations_lower = [ann.lower() for ann in class_metadata.annotations]

        # Check Spring annotations
        if any("controller" in ann or "restcontroller" in ann for ann in annotations_lower):
            return "Controller"
        if any("service" in ann for ann in annotations_lower):
            return "Service"
        if any("entity" in ann or "table" in ann for ann in annotations_lower):
            return "Entity"
        if any("repository" in ann or "dao" in ann for ann in annotations_lower):
            return "Repository"

        # Fallback to naming conventions
        simple_name_lower = class_metadata.simple_name.lower()
        if simple_name_lower.endswith("service"):
            return "Service"
        if simple_name_lower.endswith("controller"):
            return "Controller"
        if simple_name_lower.endswith("entity") or simple_name_lower.endswith("model"):
            return "Entity"
        if simple_name_lower.endswith("util") or simple_name_lower.endswith("utils"):
            return "Utility"

        return "Unknown"

    def _calculate_complexity(self, class_metadata: ClassMetadata) -> int:
        """
        Calculate a simple complexity score for a class.

        Args:
            class_metadata: ClassMetadata object.

        Returns:
            Complexity score (method count + field count).
        """
        return len(class_metadata.methods) + len(class_metadata.fields)

    def add_dependency_edge(
        self,
        source_class: str,
        target_class: str,
        edge_type: str,
        weight: float = 1.0,
        method_name: Optional[str] = None,
    ) -> None:
        """
        Add a dependency edge between two classes.

        Args:
            source_class: Fully qualified name of source class.
            target_class: Fully qualified name of target class.
            edge_type: Type of dependency (CALLS, INHERITS, IMPORTS).
            weight: Edge weight (default 1.0).
            method_name: Optional method name for CALLS edges.
        """
        # Add placeholder nodes for classes not yet in graph
        # This preserves connectivity through framework classes
        if not self.graph.has_node(source_class):
            self.graph.add_node(source_class, type="Unknown")

        if not self.graph.has_node(target_class):
            self.graph.add_node(target_class, type="Unknown")

        # Check if edge already exists
        if self.graph.has_edge(source_class, target_class):
            # Update weight (sum for multiple dependencies)
            current_weight = self.graph[source_class][target_class].get("weight", 0.0)
            self.graph[source_class][target_class]["weight"] = current_weight + weight
            # Merge edge types
            current_types = self.graph[source_class][target_class].get("types", [])
            if edge_type not in current_types:
                current_types.append(edge_type)
                self.graph[source_class][target_class]["types"] = current_types
        else:
            self.graph.add_edge(
                source_class,
                target_class,
                type=edge_type,
                weight=weight,
                types=[edge_type],
                method_name=method_name,
            )

    def add_class_dependencies(self, class_metadata: ClassMetadata) -> None:
        """
        Add all dependencies for a class to the graph.

        Args:
            class_metadata: ClassMetadata object with dependencies.
        """
        source_class = class_metadata.fully_qualified_name

        for dep in class_metadata.dependencies:
            # Determine edge weight based on type
            if dep.dependency_type == "INHERITS":
                weight = 10.0  # Strong coupling
            elif dep.dependency_type == "IMPORTS":
                weight = 1.0  # Weak coupling
            elif dep.dependency_type == "CALLS":
                weight = 5.0  # Medium coupling
            else:
                weight = 1.0

            self.add_dependency_edge(
                source_class,
                dep.target_class,
                dep.dependency_type,
                weight=weight,
                method_name=dep.method_name,
            )

    def get_node_dependencies(self, node_id: str) -> List[str]:
        """
        Get all nodes that the given node depends on.

        Args:
            node_id: Fully qualified class name.

        Returns:
            List of dependent class names.
        """
        if not self.graph.has_node(node_id):
            return []
        return list(self.graph.successors(node_id))

    def get_node_dependents(self, node_id: str) -> List[str]:
        """
        Get all nodes that depend on the given node.

        Args:
            node_id: Fully qualified class name.

        Returns:
            List of dependent class names.
        """
        if not self.graph.has_node(node_id):
            return []
        return list(self.graph.predecessors(node_id))

    def get_cluster_subgraph(self, node_ids: Set[str]) -> "DependencyGraph":
        """
        Extract a subgraph containing only the specified nodes and edges between them.

        Args:
            node_ids: Set of node IDs to include.

        Returns:
            New DependencyGraph containing the subgraph.
        """
        subgraph = DependencyGraph()
        subgraph.graph = self.graph.subgraph(node_ids).copy()
        return subgraph

    def to_json(self) -> Dict:
        """
        Export graph to JSON-serializable format.

        Returns:
            Dictionary with nodes and edges.
        """
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append({"id": node_id, **data})

        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append({"source": source, "target": target, **data})

        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_json(cls, data: Dict) -> "DependencyGraph":
        """
        Reconstruct graph from JSON export.

        Args:
            data: Dictionary with 'nodes' and 'edges' keys, as produced by to_json().

        Returns:
            New DependencyGraph instance.
        """
        import copy

        graph = cls()

        # Add all nodes (copy to avoid modifying input)
        for node_data in data.get("nodes", []):
            node_data_copy = copy.deepcopy(node_data)
            node_id = node_data_copy.pop("id")
            graph.graph.add_node(node_id, **node_data_copy)

        # Add all edges (copy to avoid modifying input)
        for edge_data in data.get("edges", []):
            edge_data_copy = copy.deepcopy(edge_data)
            source = edge_data_copy.pop("source")
            target = edge_data_copy.pop("target")
            graph.graph.add_edge(source, target, **edge_data_copy)

        return graph

    def to_graphml(self, file_path: str) -> None:
        """
        Export graph to GraphML format.

        Args:
            file_path: Path to save GraphML file.
        """
        nx.write_graphml(self.graph, file_path)

    def get_statistics(self) -> Dict:
        """
        Get graph statistics.

        Returns:
            Dictionary with graph metrics.
        """
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
            "strongly_connected_components": len(list(nx.strongly_connected_components(self.graph))),
            "weakly_connected_components": len(list(nx.weakly_connected_components(self.graph))),
            "node_types": self._count_node_types(),
        }

    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type."""
        type_counts = {}
        for _, data in self.graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts

    def visualize(
        self,
        output_path: Optional[str] = None,
        layout: str = "spring",
        show_labels: bool = True,
        node_size: int = 300,
        figsize: tuple = (16, 12),
    ) -> None:
        """
        Visualize the dependency graph using matplotlib.

        Args:
            output_path: Path to save the image. If None, displays interactively.
            layout: Layout algorithm ('spring', 'circular', 'hierarchical', 'kamada_kawai').
            show_labels: Whether to show node labels.
            node_size: Size of nodes.
            figsize: Figure size (width, height).
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualization. Install it with: pip install matplotlib"
            )

        # Color mapping for node types
        type_colors = {
            "Entity": "#FF6B6B",  # Red
            "Service": "#4ECDC4",  # Teal
            "Controller": "#95E1D3",  # Light teal
            "Repository": "#F38181",  # Pink
            "Utility": "#AA96DA",  # Purple
            "Unknown": "#C7C7C7",  # Gray
        }

        fig, ax = plt.subplots(figsize=figsize)

        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout == "hierarchical":
            try:
                pos = nx.nx_agraph.graphviz_layout(self.graph, prog="dot")
            except:
                pos = nx.spring_layout(self.graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)

        # Draw edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            ax=ax,
            alpha=0.3,
            edge_color="gray",
            arrows=True,
            arrowsize=10,
            arrowstyle="->",
        )

        # Draw nodes grouped by type
        for node_type, color in type_colors.items():
            nodes = [
                node
                for node, data in self.graph.nodes(data=True)
                if data.get("type", "Unknown") == node_type
            ]
            if nodes:
                nx.draw_networkx_nodes(
                    self.graph,
                    pos,
                    nodelist=nodes,
                    node_color=color,
                    node_size=node_size,
                    ax=ax,
                    alpha=0.8,
                )

        # Draw labels
        if show_labels:
            # Only show simple names for readability
            labels = {}
            for node_id, data in self.graph.nodes(data=True):
                simple_name = data.get("simple_name", node_id.split(".")[-1])
                # Truncate long names
                if len(simple_name) > 20:
                    simple_name = simple_name[:17] + "..."
                labels[node_id] = simple_name

            nx.draw_networkx_labels(
                self.graph, pos, labels, ax=ax, font_size=8, font_weight="bold"
            )

        # Create legend
        legend_elements = [
            mpatches.Patch(color=color, label=node_type)
            for node_type, color in type_colors.items()
            if any(
                data.get("type", "Unknown") == node_type
                for _, data in self.graph.nodes(data=True)
            )
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

        ax.set_title("Dependency Graph", fontsize=16, fontweight="bold")
        ax.axis("off")

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_interactive(self, output_path: str, height: str = "800px") -> None:
        """
        Create an interactive HTML visualization using pyvis.

        Args:
            output_path: Path to save the HTML file.
            height: Height of the visualization.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError(
                "pyvis is required for interactive visualization. Install it with: pip install pyvis"
            )

        # Color mapping for node types
        type_colors = {
            "Entity": "#FF6B6B",
            "Service": "#4ECDC4",
            "Controller": "#95E1D3",
            "Repository": "#F38181",
            "Utility": "#AA96DA",
            "Unknown": "#C7C7C7",
        }

        net = Network(height=height, directed=True, notebook=False)
        net.set_options(
            """
        {
          "nodes": {
            "font": {
              "size": 12
            }
          },
          "edges": {
            "arrows": {
              "to": {
                "enabled": true
              }
            },
            "smooth": {
              "type": "continuous"
            }
          },
          "physics": {
            "hierarchicalRepulsion": {
              "centralGravity": 0.0,
              "springLength": 200,
              "springConstant": 0.01,
              "nodeDistance": 100,
              "damping": 0.09
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "hierarchicalRepulsion",
            "stabilization": {
              "enabled": true,
              "iterations": 1000
            }
          }
        }
        """
        )

        # Add nodes
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            simple_name = data.get("simple_name", node_id.split(".")[-1])
            color = type_colors.get(node_type, "#C7C7C7")

            title = f"{node_id}\nType: {node_type}\nMethods: {data.get('method_count', 0)}"
            if data.get("file_path"):
                title += f"\nFile: {data.get('file_path')}"

            net.add_node(
                node_id,
                label=simple_name,
                color=color,
                title=title,
                size=data.get("complexity_score", 1) * 2 + 10,
            )

        # Add edges
        for source, target, data in self.graph.edges(data=True):
            edge_type = data.get("type", "IMPORTS")
            weight = data.get("weight", 1.0)
            title = f"{edge_type}\nWeight: {weight}"

            net.add_edge(source, target, title=title, width=min(weight / 2, 5))

        net.save_graph(output_path)

