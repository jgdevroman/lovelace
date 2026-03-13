"""LLM-first pipeline with gateway-first strangler fig pattern and checkpointing."""

import json
import logging
from pathlib import Path
from typing import List, Optional

import yaml

from lovelace.agents.gateway import GatewayAgent
from lovelace.agents.service_generator import ServiceGeneratorAgent
from lovelace.core.checkpoint import PipelineCheckpoint
from lovelace.core.clustering import BoundaryEdge, ClusterInfo
from lovelace.core.service_spec import ServiceResult
from lovelace.core.spec_builder import SpecBuilder
from lovelace.core.service_processor import process_service
from lovelace.core.gateway_processor import process_gateway
from lovelace.core.compose_processor import process_compose

logger = logging.getLogger(__name__)


def run_llm_first_pipeline_v2(
    analyzer,
    source_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    gateway_url: str = "http://localhost:8080",
    monolith_url: str = "http://localhost:8081",
    cost_limit_per_service: float = 0.50,

    resume: bool = True,
    clear_checkpoints: bool = False,
    auto_approve: bool = False,
) -> dict:
    """
    V2 LLM-first microservice decomposition with gateway-first strangler fig.
    
    Key features:
    - **Gateway-first**: Build gateway that proxies to monolith first
    - **Checkpointing**: Resume from where we left off
    - **Agentic generation**: LLM uses tools iteratively
    - **Per-service cost budget**: Default $0.50/service
    
    Step order:
    1. Analyze codebase
    2. LLM clustering
    3. Generate OpenAPI docs
    4. Build API Gateway (proxies to monolith)
    5. Extract shared-service
    6. Extract domain services (one at a time)
    7. Generate and validate docker-compose
    8. Summary and completion
    
    Args:
        analyzer: LovelaceAnalyzer instance with LLM client.
        source_dir: Directory to analyze.
        output_dir: Directory for output.
        gateway_url: Base URL for API gateway.
        monolith_url: URL of original monolith (for gateway proxy).
        cost_limit_per_service: Max LLM cost per service in USD.
        resume: If True, resume from last checkpoint.
        clear_checkpoints: If True, clear all checkpoints and start fresh.
        
    Returns:
        Dictionary with pipeline results.
    """
    if analyzer.llm_client is None:
        raise ValueError("LLM client required. Set OPENAI_API_KEY.")
    
    if output_dir is None:
        output_dir = analyzer.project_root / "output_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize checkpointing
    checkpoint = PipelineCheckpoint(output_dir)
    if clear_checkpoints:
        checkpoint.clear()
    
    results = {
        "gateway_url": gateway_url,
        "monolith_url": monolith_url,
        "version": "v2-gateway-first-strangler-fig",
        "steps": {},
        "service_results": [],
        "resumed_from": None,
    }
    
    logger.info("=" * 60)
    logger.info("Starting Lovelace Pipeline")
    logger.info("=" * 60)
    
    # Determine what to resume from
    if resume:
        completed = checkpoint.get_completed_steps()
        if completed:
            logger.info(f"Found checkpoints: {completed}")
            results["resumed_from"] = completed
    
    # =========================================================================
    # Step 1: Analyze codebase
    # =========================================================================
    if resume and checkpoint.exists("analysis"):
        logger.info("\n[Step 1/8] Loading cached analysis...")
        analysis_data = checkpoint.load("analysis")
        # Rebuild graph from saved data
        graph = analyzer.analyze(source_dir=source_dir)  # Fast if cached
    else:
        logger.info("\n[Step 1/8] Analyzing codebase...")
        graph = analyzer.analyze(source_dir=source_dir)
        stats = graph.get_statistics()
        checkpoint.save("analysis", {
            "node_count": stats["node_count"],
            "edge_count": stats["edge_count"],
            "source_dir": str(source_dir or analyzer.project_root),
        })
    
    stats = graph.get_statistics()
    results["steps"]["analyze"] = {
        "node_count": stats["node_count"],
        "edge_count": stats["edge_count"],
    }
    logger.info(f"✓ Found {stats['node_count']} classes, {stats['edge_count']} dependencies")
    
    # =========================================================================
    # Step 2: LLM clustering
    # =========================================================================
    if resume and checkpoint.exists("clustering"):
        logger.info("\n[Step 2/8] Loading cached clustering...")
        clustering_data = checkpoint.load("clustering")
        plan_path = Path(clustering_data["json_report"])
    else:
        logger.info("\n[Step 2/8] Running LLM clustering...")
        plan_result = analyzer.plan_with_llm(output_dir=output_dir)
        checkpoint.save("clustering", plan_result)
        plan_path = Path(plan_result["json_report"])
    
    # Load migration plan
    with open(plan_path, "r", encoding="utf-8") as f:
        migration_plan = json.load(f)
        # Fix for resumption: Ensure analyzer has the plan loaded
        analyzer.migration_plan = migration_plan
    
    # Convert to ClusterInfo objects
    clusters = _load_clusters(migration_plan)
    boundaries = _load_boundaries(migration_plan)
    
    results["steps"]["plan"] = {"clusters": len(clusters)}
    logger.info(f"✓ Loaded {len(clusters)} clusters")
    
    # =========================================================================
    # Step 3: Generate OpenAPI documentation
    # =========================================================================
    services_dir = output_dir / "services"
    services_dir.mkdir(parents=True, exist_ok=True)
    
    if resume and checkpoint.exists("documentation"):
        logger.info("\n[Step 3/8] Loading cached documentation...")
        doc_data = checkpoint.load("documentation")
    else:
        logger.info("\n[Step 3/8] Generating OpenAPI documentation...")
        doc_result = analyzer.generate_documentation(output_dir=services_dir)
        doc_data = {
            "services_documented": len(doc_result["documentation_paths"]),
            "paths": doc_result["documentation_paths"],
        }
        checkpoint.save("documentation", doc_data)
    
    results["steps"]["documentation"] = doc_data
    logger.info(f"✓ Generated documentation for {doc_data['services_documented']} services")
    
    # =========================================================================
    # Step 4: Generate API Gateway (proxies to monolith)
    # =========================================================================
    if resume and checkpoint.exists("gateway"):
        logger.info("\n[Step 4/8] Loading cached gateway...")
        gateway_data = checkpoint.load("gateway")
    else:
        logger.info("\n[Step 4/8] Generating API Gateway (proxies to monolith)...")
        
        gateway_agent = GatewayAgent(
            llm_client=analyzer.llm_client,
            graph=graph
        )
        
        # Generate Gateway via Processor (handles retry loop and validation)
        gateway_data = process_gateway(
            gateway_agent=gateway_agent,
            services_dir=services_dir,
            clusters=clusters,
            project_metadata=migration_plan.get("project_metadata", {}),
            monolith_url=monolith_url,
            llm_client=analyzer.llm_client,
            checkpoint=checkpoint,
        )
        
        if not gateway_data:
            raise RuntimeError("Failed to generate API Gateway")
            
        checkpoint.save("gateway", gateway_data)
    
    results["steps"]["gateway"] = gateway_data
    logger.info(f"✓ Gateway ready at {gateway_data['path']}")
    
    # =========================================================================
    # Step 5: Extract shared-service first (if exists)
    # =========================================================================
    logger.info("\n[Step 5/8] Extracting services (Strangler Fig Pattern)...")
    
    # Initialize generator
    generator = ServiceGeneratorAgent(
        llm_client=analyzer.llm_client,
        graph=graph,
        monolith_root=source_dir or analyzer.project_root,
        cost_limit_per_service=cost_limit_per_service,
    )
    
    # Initialize spec builder
    spec_builder = SpecBuilder(
        graph=graph,
        monolith_root=source_dir or analyzer.project_root,
        all_clusters=clusters,
    )
    
    # Find shared-service
    shared_cluster = _find_shared_cluster(clusters)
    
    if shared_cluster:
        # Process shared service with validation
        process_service(
            cluster=shared_cluster, 
            boundaries=boundaries, 
            checkpoint=checkpoint, 
            results=results, 
            generator=generator, 
            spec_builder=spec_builder, 
            services_dir=services_dir,
            gateway_data=gateway_data,
            resume=resume,
            auto_approve=auto_approve,
            llm_client=analyzer.llm_client,
        )

    
    # =========================================================================
    # Step 6: Extract domain services one at a time
    # =========================================================================
    domain_clusters = [c for c in clusters if c.suggested_name not in ("shared-service", "shared")]
    
    for i, cluster in enumerate(domain_clusters, 1):
        should_continue = process_service(
            cluster=cluster,
            boundaries=boundaries,
            checkpoint=checkpoint,
            results=results,
            generator=generator,
            spec_builder=spec_builder,
            services_dir=services_dir,
            gateway_data=gateway_data,
            resume=resume,
            auto_approve=auto_approve,
            llm_client=analyzer.llm_client,
            index=i
        )
        
        if not should_continue:
            logger.info("Pipeline paused for manual approval. Exiting.")
            return results
    
    # =========================================================================
    # Step 7: Generate and validate docker-compose
    # =========================================================================
    logger.info("\n[Step 7/8] Generating and validating docker-compose...")
    
    if resume and checkpoint.exists("compose_validated"):
        logger.info("Loading cached docker-compose validation...")
        compose_data = checkpoint.load("compose_validated")
    else:
        compose_data = process_compose(
            services_dir=services_dir,
            output_dir=output_dir,
            llm_client=analyzer.llm_client,
            checkpoint=checkpoint,
        )
        
        if compose_data:
            logger.info(f"✓ Docker-compose validated: {compose_data['path']}")
        else:
            logger.warning("Docker-compose validation failed, but continuing...")
    
    results["steps"]["compose"] = compose_data

    
    # =========================================================================
    # Summary
    # =========================================================================
    cost_report = analyzer.llm_client.get_cost_report()
    results["total_llm_cost_usd"] = cost_report["total_cost_usd"]
    
    successful = sum(1 for r in results["service_results"] if r.get("success"))
    total = len(results["service_results"])
    
    logger.info("\n" + "=" * 60)
    logger.info("V2 Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"  Services: {successful}/{total} successful")
    logger.info(f"  Total LLM cost: ${cost_report['total_cost_usd']:.4f}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Gateway: {gateway_data['path']}")
    
    return results


def _load_clusters(migration_plan: dict) -> List[ClusterInfo]:
    """Load ClusterInfo objects from migration plan."""
    clusters = []
    for cluster_data in migration_plan.get("clusters", []):
        cluster = ClusterInfo(
            id=cluster_data["id"],
            suggested_name=cluster_data["suggested_name"],
            classes=cluster_data["classes"],
            class_count=cluster_data["metrics"]["class_count"],
            internal_cohesion=cluster_data["metrics"]["internal_cohesion"],
            external_coupling=cluster_data["metrics"]["external_coupling"],
            complexity_score=cluster_data["metrics"]["complexity_score"],
            dominant_type=cluster_data["dominant_type"],
            entities=cluster_data.get("entities", []),
            entry_points=cluster_data.get("entry_points", []),
        )
        clusters.append(cluster)
    return clusters


def _load_boundaries(migration_plan: dict) -> List[BoundaryEdge]:
    """Load BoundaryEdge objects from migration plan."""
    boundaries = []
    for bc in migration_plan.get("boundary_calls", []):
        boundary = BoundaryEdge(
            from_service=bc["from_service"],
            to_service=bc["to_service"],
            from_class=bc["from_class"],
            to_class=bc["to_class"],
            method=bc.get("method"),
            edge_type=bc["edge_type"],
            weight=bc["weight"],
        )
        boundaries.append(boundary)
    return boundaries


def _find_shared_cluster(clusters: List[ClusterInfo]) -> Optional[ClusterInfo]:
    """Find the shared-service cluster if it exists."""
    for cluster in clusters:
        if cluster.suggested_name in ("shared-service", "common-service", "shared"):
            return cluster
    return None






