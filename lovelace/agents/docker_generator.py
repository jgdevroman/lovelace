import logging
import re
from pathlib import Path
from typing import Any

from lovelace.agents.generator_tools import WriteDockerfileTool, DockerBuildTool, ToolResult

logger = logging.getLogger(__name__)

from typing import Any, Optional


def generate_and_build_docker_image(
    service_path: Path, 
    service_name: str, 
    spring_boot_version: str, 
    llm_client: Any,
    image_name: str,
    max_retries: int = 3,
    port: int = 8080
) -> ToolResult:
    """
    Generate Dockerfile and build image with self-fixing retry loop.
    Returns the final result of the build.
    """
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        logger.info(f"Docker build attempt {attempt}/{max_retries} for {service_name}...")
        
        # 1. Generate/Fix Dockerfile
        success = _generate_dockerfile_content(
            service_path, 
            service_name, 
            spring_boot_version, 
            llm_client, 
            error_message=last_error,
            port=port
        )
        
        if not success:
            logger.warning(f"Failed to generate Dockerfile for {service_name}")
            continue # Or break?
            
        # 2. Build
        build_tool = DockerBuildTool(service_path)
        res = build_tool.run(image_name=image_name)
        
        if res.success:
            logger.info(f"Successfully built Docker image: {image_name}")
            return res
        else:
            logger.warning(f"Docker build failed: {res.output}")
            last_error = res.output
            
    return ToolResult(success=False, output=f"Failed to build Docker image after {max_retries} attempts. Last error: {last_error}")


def _generate_dockerfile_content(service_path: Path, service_name: str, spring_boot_version: str, llm_client: Any, error_message: Optional[str] = None, port: int = 8080) -> bool:
    """Generate Dockerfile content using LLM (Internal helper)."""
    
    if error_message:
        current_dockerfile = ""
        dockerfile_path = service_path / "Dockerfile"
        if dockerfile_path.exists():
            try:
                current_dockerfile = dockerfile_path.read_text()
            except Exception as e:
                logger.warning(f"Failed to read existing Dockerfile: {e}")

        prompt = f"""You are a DevOps expert. The previous Docker build failed. Fix the Dockerfile.
        
        Service: {service_name} (Spring Boot {spring_boot_version})
        
        Error:
        {error_message}
        
        Current Dockerfile:
        ```dockerfile
        {current_dockerfile}
        ```
        
        Return ONLY the fixed Dockerfile content in a markdown code block.
        """
    else:
        prompt = f"""You are a DevOps expert. Generate a multi-stage Dockerfile for a Spring Boot microservice named '{service_name}'.
        
        Requirements:
        1. Create a multi-stage build for a Spring Boot {spring_boot_version} application.
        2. Detect and use the appropriate Java version for the base image (e.g. eclipse-temurin:21-jdk-jammy for Spring Boot 3+).
        3. Use the provided './mvnw' wrapper for building. Do NOT install system maven.
        4. Expose port {port}.
        5. Entrypoint should run the jar with '-Dserver.port={port}'.
        
        Return ONLY the Dockerfile content in a markdown code block.
        """
    
    try:
        messages = [
            {"role": "system", "content": "You are a DevOps expert."},
            {"role": "user", "content": prompt}
        ]
        
        response = llm_client.chat(messages, temperature=0.1)
        content = response.content
        
        # Extract content from code block
        match = re.search(r"```(?:dockerfile)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if match:
            dockerfile_content = match.group(1).strip()
        else:
            dockerfile_content = content.replace("```", "").strip()
            
        tool = WriteDockerfileTool(service_path)
        res = tool.run(content=dockerfile_content)
        return res.success
    except Exception as e:
        logger.error(f"Failed to generate Dockerfile with LLM: {e}")
        return False
