"""Main analyzer orchestrator for Lovelace."""

import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from lovelace.agents.scribe import ScribeAgent
from lovelace.core.cache import CacheManager
from lovelace.core.clustering import ClusterEngine, ClusterInfo
from lovelace.core.config import LovelaceConfig, load_config, get_project_root
from lovelace.core.graph import DependencyGraph
from lovelace.core.llm import LLMClient
from lovelace.core.parser import ClassMetadata, JavaParser
from lovelace.core.reporter import MigrationReporter
from lovelace.core.vector import VectorEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LovelaceAnalyzer:
    """Main analyzer class that orchestrates the analysis workflow."""

    def __init__(self, config: Optional[LovelaceConfig] = None, config_path: Optional[Path] = None):
        """
        Initialize the analyzer.

        Args:
            config: Pre-loaded configuration. If None, will load from config_path.
            config_path: Path to lovelace.yaml. If None, will search for it.
        """
        if config is None:
            self.config = load_config(config_path)
        else:
            self.config = config

        self.project_root = get_project_root(config_path)
        self.parser = JavaParser(ignore_paths=self.config.analysis.ignore_paths)
        self.graph = DependencyGraph()
        self.parsed_classes: List[ClassMetadata] = []
        self.analyzed_source_dir: Optional[Path] = None  # Stores the directory passed to analyze()

        # Initialize LLM client if API key is available
        self.llm_client: Optional[LLMClient] = None
        try:
            self.llm_client = LLMClient(self.config.llm)
            logger.info("LLM client initialized")
        except ValueError as e:
            logger.warning(f"LLM client not available: {e}")

    def analyze(self, source_dir: Optional[Path] = None, use_cache: bool = True) -> DependencyGraph:
        """
        Perform the full analysis workflow (Workflow 1: The Scan).

        This implements:
        1. Load configuration
        2. Check cache for unchanged files
        3. Scan and parse Java files (or load from cache)
        4. Build dependency graph
        5. Return graph summary statistics

        Args:
            source_dir: Directory to analyze. If None, uses project root.
            use_cache: Whether to use cache for faster re-scans. Default True.

        Returns:
            DependencyGraph object with all parsed classes and dependencies.
        """
        if source_dir is None:
            source_dir = self.project_root

        # Store the source directory for later use (e.g., metadata detection)
        self.analyzed_source_dir = source_dir

        logger.info(f"Starting analysis of {source_dir}")

        # Initialize cache manager
        cache_dir = self.project_root / ".lovelace"
        cache = CacheManager(cache_dir)

        # Check if config has changed (invalidates cache)
        config_path = self.project_root / "lovelace.yaml"
        if config_path.exists() and cache.update_config_hash(config_path):
            logger.info("Configuration changed, invalidating cache")
            cache.invalidate()

        # Step 1: Scan for Java files
        logger.info("Scanning for Java files...")
        java_files = self.parser.scan_directory(source_dir)
        logger.info(f"Found {len(java_files)} Java files")

        if not java_files:
            logger.warning("No Java files found to analyze")
            return self.graph

        # Step 2: Check cache if enabled
        if use_cache:
            changed_files, unchanged_files = cache.get_changed_files(java_files)
            
            # If no files changed, try loading cached graph
            if not changed_files:
                cached_graph = cache.load_graph()
                if cached_graph:
                    logger.info("Using cached graph (no files changed)")
                    self.graph = cached_graph
                    # Note: parsed_classes will be empty when using cache
                    # This is acceptable for graph operations, but may need
                    # to be repopulated for operations requiring ClassMetadata
                    stats = self.graph.get_statistics()
                    logger.info(f"Graph statistics: {stats}")
                    return self.graph
                else:
                    logger.info("Cache miss, rebuilding graph")
            else:
                logger.info(f"Found {len(changed_files)} changed files, {len(unchanged_files)} unchanged")

        # Step 3: Parse Java files
        logger.info("Parsing Java files...")
        self.parsed_classes = []
        failed_count = 0

        for java_file in java_files:
            logger.debug(f"Parsing {java_file}")
            class_metadata = self.parser.parse_java_file(java_file)
            if class_metadata:
                self.parsed_classes.append(class_metadata)
            else:
                failed_count += 1

        logger.info(f"Successfully parsed {len(self.parsed_classes)} classes")
        if failed_count > 0:
            logger.warning(f"Failed to parse {failed_count} files")

        # Step 4: Build dependency graph
        logger.info("Building dependency graph...")

        # Reset graph
        self.graph = DependencyGraph()

        # Add all class nodes
        for class_metadata in self.parsed_classes:
            self.graph.add_class_node(class_metadata)

        # Add all dependency edges
        for class_metadata in self.parsed_classes:
            self.graph.add_class_dependencies(class_metadata)

        # Step 5: Save to cache
        if use_cache:
            cache.update_file_hashes(java_files)
            cache.save_graph(self.graph, len(java_files))
            logger.debug("Graph saved to cache")

        # Step 6: Return graph summary
        stats = self.graph.get_statistics()
        logger.info("Analysis complete!")
        logger.info(f"Graph statistics: {stats}")

        return self.graph

    def export_graph(self, output_path: Path, format: str = "json") -> None:
        """
        Export the dependency graph to a file.

        Args:
            output_path: Path to save the exported graph.
            format: Export format ("json" or "graphml").
        """
        if format == "json":
            import json

            graph_data = self.graph.to_json()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, indent=2, default=str)
            logger.info(f"Graph exported to {output_path} (JSON)")

        elif format == "graphml":
            self.graph.to_graphml(str(output_path))
            logger.info(f"Graph exported to {output_path} (GraphML)")

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'graphml'")

    def get_summary(self) -> dict:
        """
        Get a summary of the analysis results.

        Returns:
            Dictionary with summary statistics.
        """
        stats = self.graph.get_statistics()
        return {
            "project_name": self.config.project.name,
            "java_version": self.config.project.java_version,
            "graph_statistics": stats,
        }

    def _find_build_file_root(self, start_dir: Path) -> Path:
        """
        Find the project root by looking for pom.xml or build.gradle in parent directories.
        
        Args:
            start_dir: Directory to start searching from (typically the analyzed source dir).
            
        Returns:
            Path to the directory containing build files, or project_root as fallback.
        """
        current = start_dir
        
        # Walk up the directory tree looking for build files
        for _ in range(10):  # Limit depth to prevent infinite loops
            if (current / "pom.xml").exists() or (current / "build.gradle").exists():
                logger.debug(f"Found build file at {current}")
                return current
            
            parent = current.parent
            if parent == current:  # Reached filesystem root
                break
            current = parent
        
        # Fallback to project_root
        logger.debug(f"No build file found, using project_root: {self.project_root}")
        return self.project_root

    def _detect_project_metadata(self) -> dict:
        """
        Detect project metadata using LLM to analyze build files.
        
        Returns:
            Dictionary with project metadata including Spring Boot version,
            Java version, framework, and build tool.
        """
        # Default metadata
        metadata = {
            "framework": "unknown",
            "build_tool": "unknown",
            "spring_boot_version": None,
            "java_version": self.config.project.java_version,
        }
        
        # Determine the root directory for build files
        # When analyzing a subdirectory (e.g., src/main/java), find the actual project root
        # by looking for pom.xml or build.gradle in parent directories
        search_root = self.analyzed_source_dir if self.analyzed_source_dir else self.project_root
        build_file_root = self._find_build_file_root(search_root)
        
        # Check for build files
        pom_path = build_file_root / "pom.xml"
        gradle_path = build_file_root / "build.gradle"
        
        build_file_content = None
        build_file_type = None
        
        if pom_path.exists():
            build_file_type = "pom.xml"
            try:
                with open(pom_path, 'r', encoding='utf-8') as f:
                    build_file_content = f.read()
            except Exception as e:
                logger.warning(f"Error reading pom.xml: {e}")
        elif gradle_path.exists():
            build_file_type = "build.gradle"
            try:
                with open(gradle_path, 'r', encoding='utf-8') as f:
                    build_file_content = f.read()
            except Exception as e:
                logger.warning(f"Error reading build.gradle: {e}")
        
        # If we have a build file and LLM is available, analyze it
        if build_file_content and self.llm_client:
            try:
                # Truncate content if too large (keep first 3000 chars which includes parent and properties)
                if len(build_file_content) > 3000:
                    build_file_content = build_file_content[:3000] + "\n... (truncated)"
                
                prompt = f"""Analyze this {build_file_type} file and extract project metadata.

Build File Content:
```
{build_file_content}
```

Extract the following information and return ONLY valid JSON (no explanation):

{{
  "framework": "spring-boot | quarkus | micronaut | plain-java | unknown",
  "build_tool": "maven | gradle | unknown",
  "spring_boot_version": "x.y.z or null if not Spring Boot",
  "java_version": "version number as string"
}}

Guidelines:
- For Maven: check <parent> for spring-boot-starter-parent version
- For Maven: check <properties> for spring-boot.version or java.version
- For Gradle: check dependencies for springBootVersion
- Return null for spring_boot_version if it's not a Spring Boot project
- Return version numbers as strings (e.g., "17", "3.2.0")
"""

                messages = [
                    {"role": "system", "content": "You are an expert at analyzing Java build files. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ]
                
                response = self.llm_client.chat(messages, temperature=0.1)
                
                # Parse JSON response
                content = response.content.strip()
                if content.startswith('```'):
                    content = content.split('```')[1]
                    if content.startswith('json'):
                        content = content[4:]
                    content = content.strip()
                
                llm_metadata = json.loads(content)
                
                # Merge with defaults, preferring LLM results
                metadata.update({k: v for k, v in llm_metadata.items() if v is not None})
                
                logger.info(f"LLM detected project metadata: {metadata}")
                
            except Exception as e:
                logger.warning(f"Error using LLM to detect metadata: {e}, using defaults")
        else:
            # Fallback: basic detection without LLM
            if pom_path.exists():
                metadata["build_tool"] = "maven"
            elif gradle_path.exists():
                metadata["build_tool"] = "gradle"
        
        logger.info(f"Project metadata detected: {metadata}")
        return metadata

    def plan(self, output_dir: Optional[Path] = None) -> dict:
        """
        Generate migration plan (Workflow 2: The Plan).

        This implements:
        1. Generate embeddings for all classes
        2. Detect communities using hybrid clustering
        3. Generate migration plan reports (JSON + Markdown)

        Args:
            output_dir: Directory to save reports. If None, uses project_root/output.

        Returns:
            Dictionary with plan summary.
        """
        if not self.parsed_classes:
            raise ValueError("Must run analyze() before plan()")

        if output_dir is None:
            output_dir = self.project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting migration planning...")

        # Step 1: Initialize vector engine
        logger.info("Initializing vector engine...")
        vectors_dir = self.project_root / ".lovelace" / "vectors"
        vector_engine = VectorEngine(persist_directory=vectors_dir)

        # Step 2: Generate embeddings
        logger.info("Generating embeddings for classes...")
        vector_engine.add_classes(self.parsed_classes)

        # Step 3: Initialize cluster engine
        logger.info("Initializing cluster engine...")
        cluster_config = self.config.analysis.clustering
        constraints = self.config.analysis.constraints
        cluster_engine = ClusterEngine(
            graph=self.graph,
            vector_engine=vector_engine,
            config=cluster_config,
            constraints=constraints,
        )

        # Step 4: Detect communities
        logger.info("Detecting communities...")
        clusters = cluster_engine.detect_communities()
        cluster_info = cluster_engine.get_cluster_report()
        boundary_edges = cluster_engine.get_boundary_edges()

        # Step 4.5: Detect project metadata
        logger.info("Detecting project metadata...")
        project_metadata = self._detect_project_metadata()

        # Step 5: Generate reports
        logger.info("Generating reports...")
        reporter = MigrationReporter(
            graph=self.graph,
            clusters=clusters,
            cluster_info=cluster_info,
            boundary_calls=boundary_edges,
            project_name=self.config.project.name,
            project_metadata=project_metadata,
        )

        # Generate JSON report
        json_path = output_dir / "migration-plan.json"
        reporter.generate_json_report(json_path)

        # Generate Markdown report
        md_path = output_dir / "migration-plan.md"
        reporter.generate_markdown_report(md_path)

        logger.info("Migration planning complete!")
        logger.info(f"Reports saved to:")
        logger.info(f"  - {json_path}")
        logger.info(f"  - {md_path}")

        return {
            "clusters": len(cluster_info),
            "boundary_calls": len(boundary_edges),
            "json_report": str(json_path),
            "markdown_report": str(md_path),
        }

    def plan_with_llm(self, output_dir: Optional[Path] = None) -> dict:
        """
        Generate migration plan using LLM-first clustering.

        This is an alternative to plan() that uses LLM to:
        1. Filter domain classes from external dependencies
        2. Propose service boundaries based on domain understanding
        3. Assign classes to services

        Args:
            output_dir: Directory to save reports. If None, uses project_root/output.

        Returns:
            Dictionary with plan summary.
        """
        if not self.parsed_classes:
            raise ValueError("Must run analyze() before plan_with_llm()")

        if self.llm_client is None:
            raise ValueError("LLM client required for LLM-based planning. Set OPENAI_API_KEY.")

        if output_dir is None:
            output_dir = self.project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting LLM-first migration planning...")

        # Initialize LLM cluster engine
        from lovelace.core.llm_clustering import LLMClusterEngine
        cluster_engine = LLMClusterEngine(
            graph=self.graph,
            parsed_classes=self.parsed_classes,
            llm_client=self.llm_client,
        )

        # Run LLM clustering
        logger.info("Running LLM-based clustering...")
        clusters = cluster_engine.detect_communities()
        cluster_info = cluster_engine.get_cluster_report()
        boundary_edges = cluster_engine.get_boundary_edges()

        # Detect project metadata
        logger.info("Detecting project metadata...")
        project_metadata = self._detect_project_metadata()

        # Generate reports
        logger.info("Generating reports...")
        reporter = MigrationReporter(
            graph=self.graph,
            clusters=clusters,
            cluster_info=cluster_info,
            boundary_calls=boundary_edges,
            project_name=self.config.project.name,
            project_metadata=project_metadata,
            excluded_classes=cluster_engine.excluded_classes,
        )

        # Generate JSON report
        json_path = output_dir / "migration-plan.json"
        reporter.generate_json_report(json_path)

        # Generate Markdown report
        md_path = output_dir / "migration-plan.md"
        reporter.generate_markdown_report(md_path)

        # Print cost report
        cost_report = self.llm_client.get_cost_report()
        logger.info(f"LLM Usage: ${cost_report['total_cost_usd']:.4f} / ${cost_report['cost_limit_usd']:.2f}")

        logger.info("LLM-first migration planning complete!")
        logger.info(f"Reports saved to:")
        logger.info(f"  - {json_path}")
        logger.info(f"  - {md_path}")

        return {
            "clusters": len(cluster_info),
            "boundary_calls": len(boundary_edges),
            "json_report": str(json_path),
            "markdown_report": str(md_path),
            "domain_classes": len(cluster_engine.domain_classes),
            "llm_cost_usd": cost_report["total_cost_usd"],
        }


    def generate_documentation(
        self, cluster_id: Optional[int] = None, output_dir: Optional[Path] = None
    ) -> dict:
        """
        Generate OpenAPI documentation for clusters (Workflow 3: The Scribe).

        Also automatically generates implementation guide (Workflow 3.5) with LLM enhancement.

        Args:
            cluster_id: Specific cluster ID to document. If None, documents all clusters.
            output_dir: Directory to save documentation. If None, uses project_root/output/services.

        Returns:
            Dictionary with documentation paths and implementation guide path
        """
        if self.llm_client is None:
            raise ValueError("LLM client not available. Set OPENAI_API_KEY environment variable.")

        if not self.parsed_classes:
            raise ValueError("Must run analyze() before generate_documentation()")

        if output_dir is None:
            output_dir = self.project_root / "output" / "services"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prefer in-memory plan if available
        migration_plan = self.migration_plan
        
        # Otherwise load from disk
        if not migration_plan:
            # Load migration plan to get cluster info
            # Prefer enhanced plan if it exists (has improved names)
            enhanced_plan_path = self.project_root / "output" / f"{self.config.project.name}_migration-plan-enhanced.json"
            plan_json_path = self.project_root / "output" / "migration-plan.json"
            
            if enhanced_plan_path.exists():
                logger.info(f"Using enhanced migration plan: {enhanced_plan_path}")
                plan_json_path = enhanced_plan_path
            elif not plan_json_path.exists():
                raise FileNotFoundError(
                    "Migration plan not found. Run plan_with_llm() first."
                )

            with open(plan_json_path, "r", encoding="utf-8") as f:
                migration_plan = json.load(f)

        clusters = migration_plan.get("clusters", [])
        if cluster_id is not None:
            clusters = [c for c in clusters if c.get("id") == cluster_id]
            if not clusters:
                raise ValueError(f"Cluster {cluster_id} not found")

        logger.info(f"Generating documentation for {len(clusters)} cluster(s)...")

        scribe = ScribeAgent(self.llm_client, self.graph, self.parser)
        documentation_paths = []

        for cluster_data in clusters:
            # Convert dict to ClusterInfo
            cluster = ClusterInfo(
                id=cluster_data["id"],
                suggested_name=cluster_data["suggested_name"],
                classes=cluster_data["classes"],
                class_count=cluster_data["metrics"]["class_count"],
                internal_cohesion=cluster_data["metrics"]["internal_cohesion"],
                external_coupling=cluster_data["metrics"]["external_coupling"],
                complexity_score=cluster_data["metrics"]["complexity_score"],
                dominant_type=cluster_data["dominant_type"],
                entities=cluster_data["entities"],
                entry_points=cluster_data["entry_points"],
            )

            # Check if name was improved (from enhanced plan)
            original_name = cluster_data.get("original_name")
            if original_name and original_name != cluster.suggested_name:
                # Clean up old folder if it exists
                old_service_dir = output_dir / original_name
                if old_service_dir.exists():
                    logger.info(
                        f"Removing old service folder '{original_name}' (renamed to '{cluster.suggested_name}')"
                    )
                    shutil.rmtree(old_service_dir)

            service_dir = output_dir / cluster.suggested_name
            service_dir.mkdir(parents=True, exist_ok=True)

            # Generate OpenAPI spec
            openapi_spec = scribe.generate_openapi(cluster, self.project_root)
            openapi_path = service_dir / "openapi.yaml"
            with open(openapi_path, "w", encoding="utf-8") as f:
                yaml.dump(openapi_spec, f, default_flow_style=False, sort_keys=False)

            # Generate PlantUML diagram
            diagram = scribe.generate_diagram(cluster)
            diagram_path = service_dir / "diagrams" / "api-flow.puml"
            diagram_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diagram_path, "w", encoding="utf-8") as f:
                f.write(diagram)

            documentation_paths.append(
                {
                    "cluster_id": cluster.id,
                    "service_name": cluster.suggested_name,
                    "openapi_path": str(openapi_path),
                    "diagram_path": str(diagram_path),
                }
            )

            logger.info(f"Documentation generated for {cluster.suggested_name}")

        # Generate implementation guide (Workflow 3.5)
        logger.info("Generating implementation guide...")
        guide_output_dir = output_dir.parent if output_dir.name == "services" else output_dir
        guide_result = self.generate_implementation_guide(
            output_dir=guide_output_dir, 
            enhance_with_llm=True,
            migration_plan=migration_plan
        )
        logger.info(f"Implementation guide generated: {guide_result['guide_path']}")

        # Print cost report
        cost_report = self.llm_client.get_cost_report()
        logger.info(f"LLM Usage: ${cost_report['total_cost_usd']:.4f} / ${cost_report['cost_limit_usd']:.2f}")

        return {
            "documentation_paths": documentation_paths,
            "implementation_guide": guide_result,
            "llm_cost_usd": cost_report["total_cost_usd"],
            "llm_usage": cost_report,
        }

    def generate_implementation_guide(
        self, 
        output_dir: Optional[Path] = None, 
        enhance_with_llm: bool = True,
        migration_plan: Optional[dict] = None
    ) -> dict:
        """
        Generate AI-ready implementation guide for Workflow 4 (The Extract).

        Combines enhanced migration plan, OpenAPI specs, and step-by-step
        instructions into a single markdown document for AI-assisted implementation.

        Args:
            output_dir: Directory to save the guide. Defaults to project_root/output.
            enhance_with_llm: If True (default), use LLM to polish specific sections of the guide.
                             Since LLM costs are negligible, enhancement is enabled by default.
            migration_plan: Optional migration plan dict. If None, tries to load from self or disk.

        Returns:
            Dictionary with guide path and metadata.
        """
        if output_dir is None:
            output_dir = self.project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Use provided plan
        # 2. Use self.migration_plan
        # 3. Load from disk
        if not migration_plan:
            migration_plan = self.migration_plan
            
        if not migration_plan:
            # Load enhanced migration plan from disk
            enhanced_plan_path = output_dir / f"{self.config.project.name}_migration-plan-enhanced.json"
            plan_json_path = output_dir / "migration-plan.json"
            
            if enhanced_plan_path.exists():
                logger.info(f"Using enhanced migration plan: {enhanced_plan_path}")
                plan_json_path = enhanced_plan_path
            elif not plan_json_path.exists():
                raise FileNotFoundError("Migration plan not found. Run plan_with_llm() first.")

            with open(plan_json_path, "r", encoding="utf-8") as f:
                migration_plan = json.load(f)

        # Load OpenAPI specs
        services_dir = output_dir / "services"
        openapi_specs = {}
        if services_dir.exists():
            for service_dir in services_dir.iterdir():
                if service_dir.is_dir():
                    openapi_path = service_dir / "openapi.yaml"
                    if openapi_path.exists():
                        try:
                            with open(openapi_path, "r", encoding="utf-8") as f:
                                openapi_specs[service_dir.name] = yaml.safe_load(f)
                        except Exception as e:
                            logger.warning(f"Failed to load OpenAPI spec for {service_dir.name}: {e}")

        # Generate the implementation guide programmatically
        guide_content = self._build_implementation_guide(migration_plan, openapi_specs)
        
        # Optionally enhance with LLM
        llm_cost = 0.0
        if enhance_with_llm:
            if self.llm_client is None:
                logger.warning("LLM client not available, skipping enhancement")
            else:
                logger.info("Enhancing guide with LLM...")
                guide_content, enhancement_cost = self._enhance_guide_with_llm(guide_content, migration_plan)
                llm_cost = enhancement_cost
        
        # Write the guide
        guide_path = output_dir / f"{self.config.project.name}_implementation-guide.md"
        with open(guide_path, "w", encoding="utf-8") as f:
            f.write(guide_content)

        logger.info(f"Implementation guide written to {guide_path}")

        return {
            "guide_path": str(guide_path),
            "services_count": len(migration_plan.get("clusters", [])),
            "openapi_specs_loaded": len(openapi_specs),
            "llm_cost_usd": llm_cost,
        }

    def _build_implementation_guide(
        self, migration_plan: dict, openapi_specs: dict
    ) -> str:
        """Build the implementation guide markdown content."""
        lines = []
        
        # Header
        project_name = migration_plan.get("project", "Unknown")
        lines.append(f"# Implementation Guide: {project_name}")
        lines.append(f"\n> Generated for Workflow 4: The Extract (Implementation)")
        lines.append(f"> Based on enhanced migration plan from {migration_plan.get('analysis_date', 'N/A')}\n")
        
        # Table of Contents
        lines.append("## Table of Contents\n")
        lines.append("1. [Executive Summary](#executive-summary)")
        lines.append("2. [Extraction Order](#extraction-order)")
        lines.append("3. [Service Implementation Details](#service-implementation-details)")
        lines.append("4. [Cross-Service Refactoring](#cross-service-refactoring)")
        lines.append("5. [Step-by-Step Instructions](#step-by-step-instructions)")
        lines.append("6. [Database Migration Scripts](#database-migration-scripts)\n")

        # Executive Summary
        lines.append("---\n## Executive Summary\n")
        clusters = migration_plan.get("clusters", [])
        boundary_calls = migration_plan.get("boundary_calls", [])
        llm_review = migration_plan.get("llm_review", {})
        
        lines.append(f"- **Total Classes:** {migration_plan.get('total_classes', 0)}")
        lines.append(f"- **Suggested Services:** {migration_plan.get('suggested_services', 0)}")
        lines.append(f"- **Boundary Calls to Refactor:** {len(boundary_calls)}")
        lines.append(f"- **Overall Assessment Score:** {llm_review.get('overall_score', 'N/A')}")
        
        if llm_review.get("assessment"):
            lines.append(f"\n**Assessment:** {llm_review['assessment']}\n")

        # Extraction Order
        lines.append("---\n## Extraction Order\n")
        extraction_order = llm_review.get("extraction_order", [])
        if extraction_order:
            lines.append("The following extraction order is recommended based on dependency analysis:\n")
            lines.append("| Priority | Service | Rationale |")
            lines.append("|----------|---------|-----------|")
            for item in extraction_order:
                lines.append(f"| {item.get('priority', '-')} | **{item.get('service_name', '-')}** | {item.get('rationale', '-')} |")
        else:
            # Generate default order based on coupling
            sorted_clusters = sorted(clusters, key=lambda x: x.get("metrics", {}).get("external_coupling", 0))
            lines.append("Recommended order (lowest coupling first):\n")
            for i, cluster in enumerate(sorted_clusters, 1):
                coupling = cluster.get("metrics", {}).get("external_coupling", 0)
                lines.append(f"{i}. **{cluster['suggested_name']}** (coupling: {coupling:.1%})")
        lines.append("")

        # Service Implementation Details
        lines.append("---\n## Service Implementation Details\n")
        
        for cluster in clusters:
            service_name = cluster.get("suggested_name", f"service-{cluster['id']}")
            lines.append(f"### {service_name}\n")
            
            # Service metadata
            metrics = cluster.get("metrics", {})
            lines.append(f"**Cluster ID:** {cluster['id']}  ")
            lines.append(f"**Classes:** {metrics.get('class_count', 0)}  ")
            lines.append(f"**Cohesion:** {metrics.get('internal_cohesion', 0):.1%}  ")
            lines.append(f"**Coupling:** {metrics.get('external_coupling', 0):.1%}\n")
            
            # Name improvement rationale if present
            if cluster.get("name_improvement_rationale"):
                lines.append(f"*Name rationale: {cluster['name_improvement_rationale']}*\n")

            # Classes to migrate
            lines.append("#### Classes to Migrate\n")
            lines.append("```")
            for class_name in cluster.get("classes", []):
                lines.append(class_name)
            lines.append("```\n")

            # Entities owned
            if cluster.get("entities"):
                lines.append("#### Domain Entities\n")
                for entity in cluster["entities"]:
                    simple_name = entity.split(".")[-1]
                    lines.append(f"- `{simple_name}` ({entity})")
                lines.append("")

            # Entry points
            if cluster.get("entry_points"):
                lines.append("#### Entry Points (Controllers)\n")
                for entry in cluster["entry_points"]:
                    simple_name = entry.split(".")[-1]
                    lines.append(f"- `{simple_name}`")
                lines.append("")

            # OpenAPI Documentation
            if service_name in openapi_specs:
                spec = openapi_specs[service_name]
                lines.append("#### API Endpoints\n")
                paths = spec.get("paths", {})
                if paths:
                    lines.append("| Method | Path | Operation | Description |")
                    lines.append("|--------|------|-----------|-------------|")
                    for path, methods in paths.items():
                        for method, details in methods.items():
                            if isinstance(details, dict):
                                op_id = details.get("operationId", "-")
                                desc = details.get("summary", details.get("description", "-"))
                                if len(desc) > 50:
                                    desc = desc[:47] + "..."
                                lines.append(f"| `{method.upper()}` | `{path}` | {op_id} | {desc} |")
                    lines.append("")
                
                # Schemas
                schemas = spec.get("components", {}).get("schemas", {})
                if schemas:
                    lines.append("#### Data Schemas\n")
                    for schema_name, schema_def in schemas.items():
                        if schema_name not in ("Error", "String"):  # Skip generic schemas
                            props = schema_def.get("properties", {})
                            if props:
                                lines.append(f"**{schema_name}:**")
                                lines.append("```yaml")
                                for prop_name, prop_def in list(props.items())[:10]:  # Limit properties
                                    prop_type = prop_def.get("type", "object")
                                    lines.append(f"  {prop_name}: {prop_type}")
                                if len(props) > 10:
                                    lines.append(f"  # ... {len(props) - 10} more properties")
                                lines.append("```")
                    lines.append("")
            
            lines.append("---\n")

        # Cross-Service Refactoring
        lines.append("## Cross-Service Refactoring\n")
        lines.append("The following calls cross service boundaries and require refactoring:\n")
        
        if boundary_calls:
            # Group by from_service -> to_service
            service_pairs = {}
            for bc in boundary_calls:
                key = (bc.get("from_service"), bc.get("to_service"))
                if key not in service_pairs:
                    service_pairs[key] = []
                service_pairs[key].append(bc)
            
            for (from_svc, to_svc), calls in sorted(service_pairs.items()):
                lines.append(f"### {from_svc} → {to_svc}\n")
                lines.append("| From Class | To Class | Type | Action Required |")
                lines.append("|------------|----------|------|-----------------|")
                for call in calls:
                    from_class = call.get("from_class", "-").split(".")[-1]
                    to_class = call.get("to_class", "-").split(".")[-1]
                    edge_type = call.get("edge_type", "UNKNOWN")
                    action = "Replace with Feign/WebClient call" if call.get("refactoring_required") else "Review"
                    lines.append(f"| `{from_class}` | `{to_class}` | {edge_type} | {action} |")
                lines.append("")
        else:
            lines.append("*No cross-service boundary calls detected.*\n")

        # Step-by-Step Instructions
        lines.append("---\n## Step-by-Step Instructions\n")
        lines.append("Follow these steps for each service extraction:\n")
        
        lines.append("### Phase 1: Scaffold New Service\n")
        lines.append("```bash")
        lines.append("# For each service, create a new Spring Boot project:")
        lines.append("# Option A: Spring Initializr")
        lines.append("curl https://start.spring.io/starter.zip \\")
        lines.append("  -d dependencies=web,data-jpa,validation \\")
        lines.append("  -d name={service-name} \\")
        lines.append("  -d packageName=com.example.{service} \\")
        lines.append("  -o {service-name}.zip")
        lines.append("")
        lines.append("# Option B: Quarkus")
        lines.append("mvn io.quarkus:quarkus-maven-plugin:create \\")
        lines.append("  -DprojectGroupId=com.example \\")
        lines.append("  -DprojectArtifactId={service-name}")
        lines.append("```\n")

        lines.append("### Phase 2: Migrate Domain Classes\n")
        lines.append("For each service, copy the following class categories:\n")
        lines.append("1. **Entities** - JPA entity classes with `@Entity` annotation")
        lines.append("2. **Repositories** - Spring Data repositories")  
        lines.append("3. **Services** - Business logic classes")
        lines.append("4. **Controllers** - REST controllers (entry points)")
        lines.append("5. **DTOs** - Data transfer objects\n")
        lines.append("```bash")
        lines.append("# Example: Copy entity files")
        lines.append("cp src/main/java/org/example/monolith/entity/Pet.java \\")
        lines.append("   services/pet-service/src/main/java/com/example/pet/entity/")
        lines.append("```\n")

        lines.append("### Phase 3: Refactor Cross-Service Calls\n")
        lines.append("Replace direct method calls with HTTP clients:\n")
        lines.append("```java")
        lines.append("// BEFORE: Direct call in monolith")
        lines.append("@Autowired")
        lines.append("private UserService userService;")
        lines.append("")
        lines.append("public void doSomething() {")
        lines.append("    User user = userService.getUser(id);")
        lines.append("}")
        lines.append("")
        lines.append("// AFTER: Feign client call")
        lines.append("@FeignClient(name = \"user-service\")")
        lines.append("public interface UserClient {")
        lines.append("    @GetMapping(\"/users/{id}\")")
        lines.append("    User getUser(@PathVariable Long id);")
        lines.append("}")
        lines.append("")
        lines.append("@Autowired")
        lines.append("private UserClient userClient;")
        lines.append("")
        lines.append("public void doSomething() {")
        lines.append("    User user = userClient.getUser(id);")
        lines.append("}")
        lines.append("```\n")

        lines.append("### Phase 4: Database Migration\n")
        lines.append("Generate database migration scripts for each service:\n")
        lines.append("```sql")
        lines.append("-- Flyway migration: V1__create_service_tables.sql")
        lines.append("-- Extract only tables owned by this service")
        lines.append("CREATE TABLE IF NOT EXISTS entity_name (")
        lines.append("    id BIGINT PRIMARY KEY AUTO_INCREMENT,")
        lines.append("    -- columns from original table")
        lines.append(");")
        lines.append("```\n")

        lines.append("### Phase 5: Configure Service Communication\n")
        lines.append("```yaml")
        lines.append("# application.yml for each service")
        lines.append("spring:")
        lines.append("  application:")
        lines.append("    name: {service-name}")
        lines.append("  cloud:")
        lines.append("    openfeign:")
        lines.append("      client:")
        lines.append("        config:")
        lines.append("          default:")
        lines.append("            connectTimeout: 5000")
        lines.append("            readTimeout: 5000")
        lines.append("```\n")

        # Database Migration Scripts section
        lines.append("---\n## Database Migration Scripts\n")
        lines.append("Generate Flyway/Liquibase scripts for each service's entities:\n")
        
        for cluster in clusters:
            if cluster.get("entities"):
                service_name = cluster.get("suggested_name")
                lines.append(f"### {service_name}\n")
                lines.append("Entities requiring table extraction:")
                for entity in cluster["entities"]:
                    simple_name = entity.split(".")[-1]
                    lines.append(f"- `{simple_name}` → `{simple_name.lower()}s` table")
                lines.append("")

        # Warnings and Suggestions
        if llm_review.get("warnings") or llm_review.get("suggestions"):
            lines.append("---\n## Warnings and Recommendations\n")
            
            if llm_review.get("warnings"):
                lines.append("### ⚠️ Warnings\n")
                for warning in llm_review["warnings"]:
                    lines.append(f"- **{warning.get('type', 'WARNING')}**: {warning.get('message', '')}")
                    if warning.get("affected_services"):
                        lines.append(f"  - Affected: {', '.join(warning['affected_services'])}")
                lines.append("")
            
            if llm_review.get("suggestions"):
                lines.append("### 💡 Suggestions\n")
                for suggestion in llm_review["suggestions"]:
                    lines.append(f"**{suggestion.get('cluster_name', 'Service')}** (Cluster {suggestion.get('cluster_id', '-')})")
                    lines.append(f"- Issue: {suggestion.get('issue', '-')}")
                    lines.append(f"- Recommendation: {suggestion.get('recommendation', '-')}")
                    lines.append("")

        # Footer
        lines.append("---\n")
        lines.append("*This guide was auto-generated by Lovelace. Review each step before implementation.*")
        
        return "\n".join(lines)

    def _enhance_guide_with_llm(self, guide_content: str, migration_plan: dict) -> Tuple[str, float]:
        """
        Enhance the implementation guide with LLM-generated polish.

        Args:
            guide_content: The programmatically generated guide content.
            migration_plan: The migration plan dictionary.

        Returns:
            Tuple of (enhanced_content, cost_usd).
        """
        if self.llm_client is None:
            return guide_content, 0.0

        llm_review = migration_plan.get("llm_review", {})
        project_name = migration_plan.get("project", "Unknown")
        
        prompt = f"""You are an expert technical writer specializing in microservice migration guides.

Given the following implementation guide for migrating {project_name} to microservices, enhance it by:

1. Adding a brief, project-specific introduction paragraph (2-3 sentences) after the header
2. Polishing the Executive Summary section with more natural language while keeping all facts
3. Adding context-specific migration tips based on the assessment and warnings
4. Improving the step-by-step instructions with project-relevant examples where appropriate

## Current Guide:
{guide_content}  # Limit to avoid token limits

## LLM Review Context:
{json.dumps(llm_review, indent=2)[:2000]}

Return the complete enhanced guide in markdown format. Do not remove any sections or technical details. Only enhance the prose and add helpful context.
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert technical writer who creates clear, actionable migration guides. "
                "Enhance the provided guide with better prose and context-specific tips while preserving all technical details.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.llm_client.chat(messages, temperature=0.7)
            enhanced_content = response.content.strip()
            
            # Extract cost
            cost_report = self.llm_client.get_cost_report()
            cost = cost_report.get("total_cost_usd", 0.0) - cost_report.get("previous_total_cost_usd", 0.0)
            
            return enhanced_content, cost
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}, using original content")
            return guide_content, 0.0

