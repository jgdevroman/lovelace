"""Docker Compose processor - generates and validates docker-compose with self-fixing loop."""

import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from lovelace.agents.generator_tools import ToolResult

logger = logging.getLogger(__name__)

MAX_FIX_ITERATIONS = 10
HEALTH_CHECK_TIMEOUT = 120  # seconds
SERVICE_START_WAIT = 30  # seconds to wait for services to start


def process_compose(
    services_dir: Path,
    output_dir: Path,
    llm_client: Any,
    checkpoint: Any,
    max_iterations: int = MAX_FIX_ITERATIONS,
) -> Optional[Dict]:
    """
    Generate and validate docker-compose.yml with self-fixing loop.
    
    Args:
        services_dir: Directory containing generated services
        output_dir: Output directory for docker-compose.yml
        llm_client: LLM client for generation and fixing
        checkpoint: Checkpoint manager
        max_iterations: Maximum fix attempts
        
    Returns:
        Dict with compose path and validation status, or None on failure
    """
    compose_path = output_dir / "docker-compose.yml"
    
    # Check for existing valid compose
    if checkpoint.exists("compose_validated"):
        data = checkpoint.load("compose_validated")
        if Path(data.get("path", "")).exists():
            logger.info("Resuming with validated docker-compose")
            return data
    
    # Discover services with their ports
    services = _discover_services(services_dir)
    if not services:
        logger.error("No services found to compose")
        return None
    
    logger.info(f"Discovered {len(services)} services for docker-compose")
    
    # Generate initial compose (deterministic, not LLM)
    if not checkpoint.exists("compose_generated"):
        logger.info("Generating docker-compose.yml...")
        compose_content = _generate_compose_deterministic(services)
        compose_path.write_text(compose_content, encoding="utf-8")
        checkpoint.save("compose_generated", {"path": str(compose_path)})
    
    # Validation loop with self-fixing
    for attempt in range(1, max_iterations + 1):
        logger.info(f"Docker-compose validation attempt {attempt}/{max_iterations}")

        # Best-effort cleanup of stale containers that still hold compose ports.
        _cleanup_conflicting_containers(compose_path, list(services.keys()))
        
        # Bring up services
        up_result = _docker_compose_up(output_dir)
        if not up_result.success:
            logger.warning(f"docker-compose up failed: {up_result.output}")

            failure_type = _classify_compose_failure(up_result.output)
            logger.info(f"Compose failure type classified as: {failure_type}")

            if failure_type == "environment_conflict":
                # Retry after cleanup; this class of error is usually transient.
                _cleanup_conflicting_containers(compose_path, list(services.keys()))
                _docker_compose_down(output_dir)
                continue

            # Try to fix compose file itself
            fixed = _fix_compose_file(
                compose_path,
                up_result.output,
                llm_client,
                expected_services=list(services.keys()),
            )
            if fixed:
                continue
            else:
                logger.error("Could not fix docker-compose.yml")
                return None
        
        # Wait for services to start
        logger.info(f"Waiting {SERVICE_START_WAIT}s for services to start...")
        time.sleep(SERVICE_START_WAIT)
        
        # Check health using docker-compose ps
        health_result = _check_all_health_via_compose(output_dir, services)
        
        if health_result["all_healthy"]:
            logger.info("✓ All services healthy!")
            _docker_compose_down(output_dir)
            
            result = {
                "path": str(compose_path),
                "services": list(services.keys()),
                "validated": True,
            }
            checkpoint.save("compose_validated", result)
            return result
        
        # Get logs from unhealthy services
        unhealthy = health_result["unhealthy"]
        logger.warning(f"Unhealthy services: {unhealthy}")
        
        # Collect error logs
        error_logs = {}
        for svc_name in unhealthy:
            logs = _get_container_logs(svc_name, output_dir)
            error_logs[svc_name] = logs
        
        # Bring down before fixing
        _docker_compose_down(output_dir)
        
        # Policy: do not mutate generated services during compose validation.
        logger.warning("Service auto-fixes disabled; attempting docker-compose.yml-only fix")
        _fix_compose_file(
            compose_path,
            str(error_logs),
            llm_client,
            expected_services=list(services.keys()),
        )
    
    logger.error(f"Failed to validate compose after {max_iterations} attempts")
    _docker_compose_down(output_dir)
    return None


def _discover_services(services_dir: Path) -> Dict[str, Dict]:
    """Discover services with Docker images and their ports."""
    services = {}
    for svc_dir in services_dir.iterdir():
        if svc_dir.is_dir():
            # Check if it has a Dockerfile
            if (svc_dir / "Dockerfile").exists():
                port = _get_service_port(svc_dir)
                services[svc_dir.name] = {
                    "path": svc_dir,
                    "port": port,
                }
    return services


def _get_service_port(service_path: Path) -> int:
    """Get port from application.yml or use default."""
    app_yml = service_path / "src" / "main" / "resources" / "application.yml"
    if app_yml.exists():
        content = app_yml.read_text()
        match = re.search(r"port:\s*(\d+)", content)
        if match:
            return int(match.group(1))
    return 8080  # default


def _generate_compose_deterministic(services: Dict[str, Dict]) -> str:
    """Generate docker-compose.yml deterministically (no LLM)."""
    # Assign unique host ports using a high range to avoid local collisions.
    host_port_base = 18080
    service_configs = []
    
    # Sort services: api-gateway first, then others
    sorted_names = sorted(services.keys(), key=lambda x: (0 if x == "api-gateway" else 1, x))
    
    other_services = [n for n in sorted_names if n != "api-gateway"]
    
    for i, name in enumerate(sorted_names):
        svc = services[name]
        container_port = svc["port"]
        host_port = host_port_base + i
        
        config = f"""  {name}:
    build: ./services/{name}
    ports:
      - "{host_port}:{container_port}"
    environment:
      - SERVER_PORT={container_port}
    networks:
      - petclinic-net
    healthcheck:
      test: ["CMD-SHELL", "wget --spider -q http://localhost:{container_port}/actuator/health || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 10
      start_period: 60s"""
        
        # Add depends_on for api-gateway
        if name == "api-gateway" and other_services:
            depends = "\n    depends_on:\n" + "\n".join(
                f"      {svc}:\n        condition: service_started"
                for svc in other_services
            )
            config += depends
        
        service_configs.append(config)
    
    compose = f"""services:
{chr(10).join(service_configs)}

networks:
  petclinic-net:
    driver: bridge
"""
    return compose


def _docker_compose_up(output_dir: Path) -> ToolResult:
    """Run docker-compose up -d."""
    try:
        result = subprocess.run(
            ["docker-compose", "up", "-d", "--build"],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0:
            return ToolResult(success=True, output="Services started")
        return ToolResult(success=False, output=result.stderr or result.stdout)
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="docker-compose up timed out after 180s")
    except Exception as e:
        return ToolResult(success=False, output=str(e))


def _docker_compose_down(output_dir: Path) -> ToolResult:
    """Run docker-compose down."""
    try:
        subprocess.run(
            ["docker-compose", "down", "--remove-orphans"],
            cwd=output_dir,
            capture_output=True,
            timeout=60,
        )
        return ToolResult(success=True, output="Services stopped")
    except Exception as e:
        return ToolResult(success=False, output=str(e))


def _check_all_health_via_compose(output_dir: Path, services: Dict[str, Dict]) -> Dict:
    """Check health of all services using docker-compose ps."""
    healthy = []
    unhealthy = []
    
    try:
        # Use docker-compose ps to check status
        result = subprocess.run(
            ["docker-compose", "ps", "--format", "json"],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # Parse JSON output (one JSON object per line)
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        container = json.loads(line)
                        name = container.get("Service", container.get("Name", ""))
                        state = container.get("State", "").lower()
                        health = container.get("Health", "").lower()
                        
                        # Check if running and healthy
                        if state == "running" and (health == "healthy" or health == ""):
                            healthy.append(name)
                        else:
                            unhealthy.append(name)
                    except json.JSONDecodeError:
                        pass
            
            # Add any services not in output as unhealthy
            for name in services:
                if name not in healthy and name not in unhealthy:
                    unhealthy.append(name)
        else:
            # Fallback: check each service manually
            for name, info in services.items():
                port = info["port"]
                check = _check_single_service_health(name, port)
                if check:
                    healthy.append(name)
                else:
                    unhealthy.append(name)
                    
    except Exception as e:
        logger.warning(f"Health check error: {e}")
        # Assume all unhealthy on error
        unhealthy = list(services.keys())
    
    return {
        "all_healthy": len(unhealthy) == 0 and len(healthy) > 0,
        "healthy": healthy,
        "unhealthy": unhealthy,
    }


def _check_single_service_health(container_name: str, port: int) -> bool:
    """Check health of a single service by hitting its actuator endpoint."""
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "wget", "-q", "-O", "-", 
             f"http://localhost:{port}/actuator/health"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0 and "UP" in result.stdout
    except Exception:
        return False


def _get_container_logs(service_name: str, output_dir: Path) -> str:
    """Get logs from a container."""
    try:
        result = subprocess.run(
            ["docker-compose", "logs", "--tail=100", service_name],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)


def _fix_compose_file(
    compose_path: Path,
    error: str,
    llm_client: Any,
    expected_services: Optional[List[str]] = None,
) -> bool:
    """Fix docker-compose.yml based on error."""
    current_content = compose_path.read_text()
    
    prompt = f"""The docker-compose failed with this error:
```
{error[:2000]}
```

Current docker-compose.yml:
```yaml
{current_content}
```

Fix the docker-compose.yml to resolve this error.
Return ONLY the fixed YAML content, no explanation."""

    messages = [
        {"role": "system", "content": "You are a DevOps expert. Fix docker-compose issues. Return only valid YAML."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = llm_client.chat(messages, temperature=0.1)
        content = response.content.strip()
        
        # Extract YAML
        if "```" in content:
            match = re.search(r"```(?:yaml|yml)?\s*(.*?)```", content, re.DOTALL)
            if match:
                content = match.group(1).strip()
        
        if not content or content == current_content:
            return False

        if not _validate_compose_structure(content, expected_services or []):
            logger.warning("Rejected compose fix: structural validation failed")
            return False

        if not _validate_compose_config(content, compose_path.parent):
            logger.warning("Rejected compose fix: 'docker-compose config' failed")
            return False

        compose_path.write_text(content, encoding="utf-8")
        logger.info("Applied fix to docker-compose.yml")
        return True
    except Exception as e:
        logger.error(f"Failed to fix compose: {e}")
    
    return False


def _fix_service(service_path: Path, error_logs: str, llm_client: Any) -> bool:
    """Fix a failing service based on error logs."""
    # Look for common error patterns and fix the source files
    
    # JPA/Hibernate errors - fix entity relationships
    if "AnnotationException" in error_logs or "mappedBy" in error_logs:
        return _fix_jpa_entities(service_path, error_logs, llm_client)
    
    # Bean creation errors
    if "BeanCreationException" in error_logs or "NoSuchBeanDefinitionException" in error_logs:
        return _fix_spring_config(service_path, error_logs, llm_client)
    
    return False


def _fix_jpa_entities(service_path: Path, error: str, llm_client: Any) -> bool:
    """Fix JPA entity relationship errors."""
    java_dir = service_path / "src" / "main" / "java"
    if not java_dir.exists():
        return False
    
    # Find all entity files
    entity_files = {}
    for java_file in java_dir.rglob("*.java"):
        content = java_file.read_text()
        if "@Entity" in content:
            entity_files[java_file.name] = {
                "path": java_file,
                "content": content
            }
    
    if not entity_files:
        return False
    
    # Build prompt with all entities
    entities_context = "\n\n".join([
        f"### {name}\n```java\n{info['content']}\n```"
        for name, info in entity_files.items()
    ])
    
    prompt = f"""The service failed with this JPA error:
```
{error[:1500]}
```

## Entity files:
{entities_context}

Fix the JPA entity relationships. For each file that needs fixing, output:

### FIXED: <filename>
```java
<fixed content>
```

Return ONLY the fixed files."""

    messages = [
        {"role": "system", "content": "You are a Java/JPA expert. Fix entity relationship issues."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = llm_client.chat(messages, temperature=0.1)
        
        # Parse fixed files
        pattern = r"###\s*FIXED:\s*(\S+)\s*\n```(?:java)?\s*(.*?)```"
        matches = re.findall(pattern, response.content, re.DOTALL | re.IGNORECASE)
        
        any_fixed = False
        for filename, content in matches:
            filename = filename.strip()
            if filename in entity_files:
                entity_files[filename]["path"].write_text(content.strip(), encoding="utf-8")
                logger.info(f"Fixed entity: {filename}")
                any_fixed = True
        
        return any_fixed
    except Exception as e:
        logger.error(f"Failed to fix entities: {e}")
        return False


def _fix_spring_config(service_path: Path, error: str, llm_client: Any) -> bool:
    """Fix Spring configuration errors."""
    # For now, just log - this is a more complex fix
    logger.warning(f"Spring config error detected, manual fix may be needed")
    return False


def _rebuild_service_image(service_path: Path, service_name: str) -> bool:
    """Rebuild Docker image for a service."""
    try:
        logger.info(f"Rebuilding {service_name}...")
        
        # First recompile
        compile_result = subprocess.run(
            ["mvn", "clean", "package", "-DskipTests", "-q"],
            cwd=service_path,
            capture_output=True,
            timeout=180,
        )
        
        if compile_result.returncode != 0:
            logger.warning(f"Maven build failed for {service_name}")
            return False
        
        # Then rebuild image
        result = subprocess.run(
            ["docker", "build", "-t", f"{service_name}:latest", "."],
            cwd=service_path,
            capture_output=True,
            timeout=300,
        )
        
        if result.returncode == 0:
            logger.info(f"Rebuilt Docker image: {service_name}:latest")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to rebuild {service_name}: {e}")
        return False


def _classify_compose_failure(error_output: str) -> str:
    """Classify docker-compose up failure for targeted recovery."""
    text = (error_output or "").lower()

    if "yaml:" in text or "cannot start any token" in text or "did not find expected" in text:
        return "compose_syntax"
    if "undefined service" in text or "depends on undefined service" in text:
        return "compose_semantic"
    if "failed to solve" in text or "dockerfile:" in text or "target " in text:
        return "service_build"
    if "port is already allocated" in text or "already in use by container" in text:
        return "environment_conflict"
    if "unhealthy" in text or "health" in text:
        return "runtime_health"
    return "unknown"



def _validate_compose_structure(content: str, expected_services: List[str]) -> bool:
    """Validate basic compose structure and service dependencies."""
    try:
        parsed = yaml.safe_load(content)
    except Exception as e:
        logger.warning(f"Compose YAML parse failed: {e}")
        return False

    if not isinstance(parsed, dict):
        return False

    services = parsed.get("services")
    if not isinstance(services, dict) or not services:
        return False

    # Guardrail: never accept a fix that drops discovered services.
    missing = [name for name in expected_services if name not in services]
    if missing:
        logger.warning(f"Compose fix removed required services: {missing}")
        return False

    service_names = set(services.keys())
    for name, definition in services.items():
        if not isinstance(definition, dict):
            return False

        depends_on = definition.get("depends_on")
        if isinstance(depends_on, list):
            if any(dep not in service_names for dep in depends_on):
                return False
        elif isinstance(depends_on, dict):
            if any(dep not in service_names for dep in depends_on.keys()):
                return False

    return True


def _validate_compose_config(content: str, output_dir: Path) -> bool:
    """Validate candidate compose with docker-compose config before applying."""
    temp_path = output_dir / ".docker-compose.candidate.yml"
    try:
        temp_path.write_text(content, encoding="utf-8")
        result = subprocess.run(
            ["docker-compose", "-f", str(temp_path), "config"],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"Compose config validation failed: {e}")
        return False
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


def _cleanup_conflicting_containers(compose_path: Path, service_names: List[str]) -> None:
    """Stop/remove stale containers that occupy host ports used by compose services."""
    host_ports = _extract_host_ports(compose_path)
    if not host_ports:
        return

    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.ID}}|{{.Image}}|{{.Ports}}|{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode != 0:
            return

        for line in result.stdout.splitlines():
            parts = line.split("|", 3)
            if len(parts) != 4:
                continue
            container_id, image, ports_text, name = parts
            if not any(f":{port}->" in ports_text for port in host_ports):
                continue

            # Only clean up likely pipeline-created containers.
            is_generated = (
                image.endswith(":latest")
                or name.startswith("output-")
                or any(svc in name for svc in service_names)
                or any(svc in image for svc in service_names)
            )
            if not is_generated:
                continue

            logger.warning(
                f"Removing stale container '{name}' using compose port(s): {ports_text}"
            )
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=20)
    except Exception as e:
        logger.warning(f"Failed to clean conflicting containers: {e}")


def _extract_host_ports(compose_path: Path) -> List[int]:
    """Extract host ports from compose services. Supports short and long ports syntax."""
    try:
        parsed = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    services = parsed.get("services") if isinstance(parsed, dict) else None
    if not isinstance(services, dict):
        return []

    host_ports: List[int] = []
    for definition in services.values():
        if not isinstance(definition, dict):
            continue
        ports = definition.get("ports", [])
        if not isinstance(ports, list):
            continue

        for entry in ports:
            if isinstance(entry, str):
                # e.g. "18080:8080"
                left = entry.split(":", 1)[0].strip().strip('"').strip("'")
                if left.isdigit():
                    host_ports.append(int(left))
            elif isinstance(entry, dict):
                published = entry.get("published")
                if isinstance(published, int):
                    host_ports.append(published)
                elif isinstance(published, str) and published.isdigit():
                    host_ports.append(int(published))

    return sorted(set(host_ports))
