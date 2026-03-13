"""Migration plan reporter for generating JSON and Markdown reports."""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from lovelace.core.clustering import BoundaryEdge, ClusterInfo
from lovelace.core.graph import DependencyGraph

logger = logging.getLogger(__name__)


class MigrationReporter:
    """Generate migration plan reports."""

    def __init__(
        self,
        graph: DependencyGraph,
        clusters: Dict[str, int],
        cluster_info: List[ClusterInfo],
        boundary_calls: List[BoundaryEdge],
        project_name: str,
        project_metadata: Dict = None,
        excluded_classes: List[str] = None,
    ):
        """
        Initialize the reporter with analysis results.

        Args:
            graph: Dependency graph.
            clusters: Dictionary mapping class_id -> cluster_id.
            cluster_info: List of ClusterInfo objects.
            boundary_calls: List of BoundaryEdge objects.
            project_name: Name of the project.
            project_metadata: Optional project metadata (version, framework, etc.)
            excluded_classes: Optional list of classes filtered out during domain analysis.
        """
        self.graph = graph
        self.clusters = clusters
        self.cluster_info = cluster_info
        self.boundary_calls = boundary_calls
        self.project_name = project_name
        self.project_metadata = project_metadata or {}
        self.excluded_classes = excluded_classes or []

    def generate_json_report(self, output_path: Path) -> None:
        """
        Generate detailed JSON report.

        Args:
            output_path: Path to save the JSON file.
        """
        logger.info(f"Generating JSON report to {output_path}")

        # Build report structure
        report = {
            "project": self.project_name,
            "analysis_date": datetime.utcnow().isoformat() + "Z",
            "total_classes": len(self.clusters),
            "suggested_services": len(self.cluster_info),
            "project_metadata": self.project_metadata,
            "excluded_classes": self.excluded_classes,
            "clusters": [],
            "boundary_calls": [],
            "warnings": [],
        }

        # Add cluster information
        for info in self.cluster_info:
            cluster_data = {
                "id": info.id,
                "suggested_name": info.suggested_name,
                "description": info.description,
                "rationale": info.rationale,
                "classes": info.classes,
                "metrics": {
                    "class_count": info.class_count,
                    "internal_cohesion": round(info.internal_cohesion, 3),
                    "external_coupling": round(info.external_coupling, 3),
                    "complexity_score": info.complexity_score,
                },
                "dominant_type": info.dominant_type,
                "entities": info.entities,
                "entry_points": info.entry_points,
            }
            report["clusters"].append(cluster_data)

        # Add boundary calls
        for boundary in self.boundary_calls:
            boundary_data = {
                "from_service": boundary.from_service,
                "to_service": boundary.to_service,
                "from_class": boundary.from_class,
                "to_class": boundary.to_class,
                "method": boundary.method,
                "edge_type": boundary.edge_type,
                "weight": boundary.weight,
                "refactoring_required": True,
            }
            report["boundary_calls"].append(boundary_data)

        # Generate warnings
        warnings = self._generate_warnings()
        report["warnings"] = warnings

        # Write JSON file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"JSON report written to {output_path}")

    def generate_markdown_report(self, output_path: Path) -> None:
        """
        Generate human-readable Markdown report.

        Args:
            output_path: Path to save the Markdown file.
        """
        logger.info(f"Generating Markdown report to {output_path}")

        lines = []
        lines.append(f"# Migration Plan: {self.project_name}\n")
        lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Executive Summary
        lines.append("## Executive Summary\n")
        lines.append(f"- **Total Classes Analyzed:** {len(self.clusters)}")
        lines.append(f"- **Suggested Microservices:** {len(self.cluster_info)}")
        lines.append(f"- **Estimated Boundary Calls to Refactor:** {len(self.boundary_calls)}\n")

        # Suggested Service Decomposition
        lines.append("## Suggested Service Decomposition\n")

        for info in sorted(self.cluster_info, key=lambda x: x.id):
            cohesion_pct = int(info.internal_cohesion * 100)
            coupling_pct = int(info.external_coupling * 100)

            lines.append(f"### {info.id + 1}. {info.suggested_name}\n")
            
            # Add description and rationale if available
            if info.description:
                lines.append(f"*{info.description}*\n")
            if info.rationale:
                lines.append(f"**Rationale:** {info.rationale}\n")
            
            lines.append(
                f"**Classes:** {info.class_count} | **Cohesion:** {cohesion_pct}% | **Coupling:** {coupling_pct}%\n"
            )

            # Classes table
            lines.append("| Class          | Type       | Complexity |")
            lines.append("| -------------- | ---------- | ---------- |")
            for class_id in info.classes:
                node_data = self.graph.graph.nodes[class_id]
                simple_name = node_data.get("simple_name", class_id.split(".")[-1])
                node_type = node_data.get("type", "Unknown")
                complexity = node_data.get("complexity_score", 0)
                lines.append(f"| {simple_name} | {node_type} | {complexity} |")
            lines.append("")

            # Entities owned
            if info.entities:
                lines.append("**Entities owned:** " + ", ".join([self._get_simple_name(e) for e in info.entities]))
                lines.append("")

            # Entry points
            if info.entry_points:
                lines.append("**Entry points:**")
                for entry_point in info.entry_points:
                    simple_name = self._get_simple_name(entry_point)
                    lines.append(f"- `{simple_name}`")
                lines.append("")

            lines.append("---\n")

        # Cross-Service Dependencies
        if self.boundary_calls:
            lines.append("## Cross-Service Dependencies\n")
            lines.append(
                "These method calls will need HTTP/gRPC clients after extraction:\n"
            )
            lines.append("| From         | To          | Method        | Priority |")
            lines.append("| ------------ | ----------- | ------------- | -------- |")

            # Group by service pair
            service_pairs: Dict[tuple, List[BoundaryEdge]] = {}
            for boundary in self.boundary_calls:
                key = (boundary.from_service, boundary.to_service)
                if key not in service_pairs:
                    service_pairs[key] = []
                service_pairs[key].append(boundary)

            for (from_svc, to_svc), boundaries in sorted(service_pairs.items()):
                # Use first boundary for the row
                boundary = boundaries[0]
                method = boundary.method or boundary.edge_type
                priority = "HIGH" if len(boundaries) > 3 else "MEDIUM"
                lines.append(f"| {from_svc} | {to_svc} | {method} | {priority} |")

            lines.append("")

        # Recommendations
        lines.append("## Recommendations\n")
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. **{rec['title']}** - {rec['description']}")

        # Warnings
        warnings = self._generate_warnings()
        if warnings:
            lines.append("\n## Warnings\n")
            for warning in warnings:
                lines.append(f"- **{warning['type']}:** {warning['message']}")
                if "suggestion" in warning:
                    lines.append(f"  - Suggestion: {warning['suggestion']}")

        # Write Markdown file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown report written to {output_path}")

    def _get_simple_name(self, class_id: str) -> str:
        """Get simple name from class ID."""
        node_data = self.graph.graph.nodes.get(class_id, {})
        return node_data.get("simple_name", class_id.split(".")[-1])

    def _generate_warnings(self) -> List[Dict]:
        """Generate warnings based on analysis."""
        warnings = []

        # Check for high coupling
        for info in self.cluster_info:
            if info.external_coupling > 5.0:
                warnings.append(
                    {
                        "type": "HIGH_COUPLING",
                        "message": f"{info.suggested_name} has high external coupling ({info.external_coupling:.1f} dependencies per class)",
                        "suggestion": "Consider creating shared DTOs or reducing dependencies",
                    }
                )

            if info.internal_cohesion < 0.3:
                warnings.append(
                    {
                        "type": "LOW_COHESION",
                        "message": f"{info.suggested_name} has low internal cohesion ({info.internal_cohesion:.1%})",
                        "suggestion": "Review if classes in this cluster should be split further",
                    }
                )

        # Check for services with many boundary calls
        boundary_counts: Dict[str, int] = {}
        for boundary in self.boundary_calls:
            boundary_counts[boundary.from_service] = boundary_counts.get(boundary.from_service, 0) + 1

        for service, count in boundary_counts.items():
            if count > 10:
                warnings.append(
                    {
                        "type": "MANY_BOUNDARY_CALLS",
                        "message": f"{service} has {count} cross-service dependencies",
                        "suggestion": "Consider creating a shared library or API gateway",
                    }
                )

        return warnings

    def _generate_recommendations(self) -> List[Dict]:
        """Generate migration recommendations."""
        recommendations = []

        # Sort clusters by coupling (lowest first - easiest to extract)
        sorted_clusters = sorted(self.cluster_info, key=lambda x: x.external_coupling)

        if sorted_clusters:
            first_service = sorted_clusters[0]
            recommendations.append(
                {
                    "title": f"Start with {first_service.suggested_name}",
                    "description": f"Lowest coupling ({first_service.external_coupling:.1f}), clear boundaries",
                }
            )

        # Find services that depend on the first one
        if len(sorted_clusters) > 1:
            first_service_classes = set(sorted_clusters[0].classes)
            for cluster in sorted_clusters[1:]:
                # Check if this cluster depends on the first one
                depends_on_first = False
                for class_id in cluster.classes:
                    for neighbor in self.graph.graph.successors(class_id):
                        if neighbor in first_service_classes:
                            depends_on_first = True
                            break
                    if depends_on_first:
                        break

                if depends_on_first:
                    recommendations.append(
                        {
                            "title": f"Extract {cluster.suggested_name} second",
                            "description": f"Depends on {first_service.suggested_name}",
                        }
                    )
                    break

        # DTO recommendation
        if self.boundary_calls:
            recommendations.append(
                {
                    "title": "Create shared DTOs",
                    "description": "Define common data transfer objects for cross-service communication",
                }
            )

        return recommendations

