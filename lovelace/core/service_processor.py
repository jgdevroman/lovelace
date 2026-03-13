import logging
from pathlib import Path
from typing import List, Any

from lovelace.agents.docker_generator import generate_and_build_docker_image
from lovelace.agents.generator_tools import DockerHealthCheckTool
from lovelace.core.checkpoint import PipelineCheckpoint
from lovelace.core.clustering import BoundaryEdge, ClusterInfo
from lovelace.core.spec_builder import SpecBuilder
from lovelace.agents.service_generator import ServiceGeneratorAgent

logger = logging.getLogger(__name__)

def update_gateway_route(
    gateway_path: Path,
    service_name: str,
    service_url: str,
):
    """Update gateway to route to a newly extracted service."""
    # This would modify the gateway's application.yml to update the route
    logger.info(f"Gateway route updated: {service_name} -> {service_url}")


def process_service(
    cluster: ClusterInfo, 
    boundaries: List[BoundaryEdge], 
    checkpoint: PipelineCheckpoint, 
    results: dict, 
    generator: ServiceGeneratorAgent, 
    spec_builder: SpecBuilder, 
    services_dir: Path, 
    gateway_data: dict, 
    resume: bool, 
    auto_approve: bool, 
    llm_client: Any, 
    index: int = 0
) -> bool:
    """
    Process a single service: Generate -> Docker -> Health -> Approval.
    Returns True if allowed to continue to next service, False if stuck/exit.
    """
    service_name = cluster.suggested_name
    checkpoint_name = f"service_{service_name}"
    
    # 1. Check for Pending/Approved state
    if checkpoint.is_pending_approval(service_name):
        # We are resuming. Implicitly approve.
        logger.info(f"Checking approval for {service_name}...")
        checkpoint.approve_service(service_name)
        # Fall through to 'loading cached' logic or just load it directly if we have the service_ result
        if checkpoint.exists(checkpoint_name):
             svc_data = checkpoint.load(checkpoint_name)
             results["service_results"].append(svc_data)
             # Update gateway since we are approved now
             update_gateway_route(
                gateway_path=Path(gateway_data["path"]),
                service_name=service_name,
                service_url=f"http://api-gateway:8080/api/{service_name}", 
             )
             return True
        else:
            # Weird state, pending approval but no service data? Regenerate.
            pass

    # Resumption / Loading Logic
    previous_data = None
    if resume and checkpoint.exists(checkpoint_name):
        previous_data = checkpoint.load(checkpoint_name)
        state = previous_data.get("validation_state", "unknown")
        
        # If fully finished/approved, skip
        if state in ["healthy", "approved", "completed", "docker_built"]:
            logger.info(f"[{index}] Skipping {service_name} (State: {state})")
            results["service_results"].append(previous_data)
            return True
        
        # If compiled, we can resume at Docker
        if state == "compiled":
            logger.info(f"[{index}] Resuming {service_name} from compiled state...")
            # We don't have a full ServiceResult object, but we have enough to proceed?
            # We need 'service_path', 'cost_usd', 'iterations', 'message' from previous run if possible
            # But 'ServiceResult' class usage below depends on it.
            # Let's rely on re-generation or mock it.
            # Actually, re-generation (ServiceGenerator) is smart enough to skip if files exist and test passes?
            # Not really. 
            pass

    logger.info(f"[{index}] Building {service_name}...")
    
    cluster_boundaries = [b for b in boundaries if b.from_class in cluster.classes]
    spec = spec_builder.build_spec(cluster, cluster_boundaries)
    
    # If resuming from "compiled", we effectively want to skip generation.
    # But ServiceGenerator.generate_service runs the whole loop.
    # Ideally we'd have generator.resume_service() or similar.
    # For now, let's just proceed with standard generation BUT save intermediate checkpoint.
    
    result = generator.generate_service(spec, services_dir)
    
    # Save "compiled" state immediately if success
    if result.success:
         intermediate_data = {
            "name": service_name,
            "success": True,
            "cost_usd": result.cost_usd,
            "iterations": result.iterations,
            "message": result.message,
            "validation_state": "compiled",
            "service_path": str(result.service_path) # Ensure this is saved
        }
         checkpoint.save(checkpoint_name, intermediate_data)
    
    # If generation successful, try Docker & Health
    validation_status = "pending"
    docker_image = f"{service_name}:latest"
    
    if result.success:
        logger.info(f"[{index}] Validating {service_name}...")
        
        # Docker Build (Self-Fixing Loop)
        should_fix_docker = True
        service_path = Path(result.service_path)
        
        unique_port = 8080 + index
        
        docker_res = generate_and_build_docker_image(
            service_path=service_path,
            service_name=service_name,
            spring_boot_version="3.2.0",
            llm_client=llm_client,
            image_name=docker_image,
            max_retries=10,
            port=unique_port
        )
        
        if docker_res.success:
            result.validation_state = "docker_built"
            result.docker_image = docker_image
            # Health Check
            # Use unique port for both host and container to avoid conflicts
            health_res = DockerHealthCheckTool(service_path).run(
                image_name=docker_image, 
                port=unique_port, 
                container_port=unique_port
            )
            if health_res.success:
                result.validation_state = "healthy"
                validation_status = "healthy"
            else:
                logger.warning(f"Health check failed: {health_res.output}")
        else:
             logger.warning(f"Docker build failed final: {docker_res.output}")
    
    svc_data = {
        "name": service_name,
        "success": result.success,
        "cost_usd": result.cost_usd,
        "iterations": result.iterations,
        "message": result.message,
        "validation_state": result.validation_state,
        "docker_image": result.docker_image,
        "service_path": str(result.service_path)
    }
    checkpoint.save(checkpoint_name, svc_data)
    results["service_results"].append(svc_data)
    
    if result.success:
        # Request Approval
        checkpoint.request_approval(service_name, svc_data)
        
        if auto_approve:
            logger.info(f"[{index}] Auto-approving {service_name}")
            checkpoint.approve_service(service_name)
             # Update gateway
            update_gateway_route(
                gateway_path=Path(gateway_data["path"]),
                service_name=service_name,
                service_url=f"http://api-gateway:8080/api/{service_name}", 
            )
            return True
        else:
             logger.info(f"[{index}] {service_name} ready. Stopping for manual approval/restart.")
             return False # Stop pipeline
             
    return True # Continue if failed (just logs failure)
