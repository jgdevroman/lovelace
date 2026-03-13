"""Gateway processor - orchestrates gateway generation and Docker build."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from lovelace.agents.docker_generator import generate_and_build_docker_image
from lovelace.agents.generator_tools import CompileTool, GetCompileErrorsTool
from lovelace.core.clustering import ClusterInfo

logger = logging.getLogger(__name__)

MAX_FIX_ITERATIONS = 10


def process_gateway(
    gateway_agent: Any,
    services_dir: Path,
    clusters: List[ClusterInfo],
    project_metadata: Dict,
    monolith_url: str,
    llm_client: Any,
    checkpoint: Any,
    max_fix_iterations: int = MAX_FIX_ITERATIONS,
) -> Optional[Dict]:
    """
    Generate and validate the API Gateway.
    
    The gateway agent uses a tool-based agentic approach internally.
    This processor handles checkpointing and Docker build orchestration.
    """
    gateway_path = None
    routes = {}

    # 1. Try Load Existing Compiled State
    if checkpoint.exists("gateway_compiled"):
        data = checkpoint.load("gateway_compiled")
        path_str = data.get("path")
        if path_str:
            candidate_path = Path(path_str)
            if candidate_path.exists():
                # Verify it still compiles
                if CompileTool(candidate_path).run().success:
                    logger.info("Resuming with existing compiled Gateway.")
                    gateway_path = candidate_path
                    routes = data.get("routes", {})

    # 2. Generate Gateway if not already compiled
    if not gateway_path:
        logger.info("Generating API Gateway using tool-based agent...")
        
        # The agent handles generation AND fixing internally using tools
        gateway_result = gateway_agent.generate_gateway(
            output_dir=services_dir,
            extracted_services=clusters,
            openapi_specs={},  # No specs yet
            project_metadata=project_metadata,
        )
        
        gateway_path = gateway_result.get("path")
        routes = gateway_result.get("routes", {})
        success = gateway_result.get("success", False)
        
        if not success:
            # Generation failed - try additional fix attempts
            logger.warning("Initial generation did not compile. Attempting additional fixes...")
            
            for attempt in range(max_fix_iterations):
                # Get current compile errors
                compile_res = CompileTool(gateway_path).run()
                if compile_res.success:
                    logger.info("Gateway fixed successfully!")
                    success = True
                    break
                
                logger.warning(f"Fix attempt {attempt + 1}/{max_fix_iterations}...")
                
                # Get detailed errors
                error_result = GetCompileErrorsTool(gateway_path).run()
                error_message = error_result.output if error_result else compile_res.output
                
                # Use tool-based fix approach
                fix_success = gateway_agent.fix_gateway(
                    gateway_path=gateway_path,
                    error_message=error_message,
                )
                
                if not fix_success:
                    logger.warning(f"Fix attempt {attempt + 1} could not apply fixes")
            
            # Final check
            compile_res = CompileTool(gateway_path).run()
            success = compile_res.success
        
        if not success:
            logger.error("Failed to generate compiling Gateway. Terminating.")
            return None
        
        # Save checkpoint
        checkpoint.save("gateway_compiled", {
            "path": str(gateway_path),
            "routes": routes
        })
        logger.info(f"✓ Gateway compiled successfully at {gateway_path}")

    # 3. Docker Build (Self-Fixing Loop via Helper)
    logger.info("Building Docker image for Gateway...")
    
    docker_res = generate_and_build_docker_image(
        service_path=gateway_path,
        service_name="api-gateway",
        spring_boot_version="3.2.0",
        llm_client=llm_client,
        image_name="api-gateway",
        max_retries=10,
        port=8080
    )
    
    if docker_res.success:
        logger.info("Gateway built and dockerized successfully.")
        return {
            "path": str(services_dir / "api-gateway"),
            "routes": routes,
        }
    else:
        logger.error(f"Gateway Docker build failed: {docker_res.output}")
        return None
