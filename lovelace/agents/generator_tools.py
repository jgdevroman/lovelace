"""Tools for the agentic service generator.

These tools allow the LLM agent to generate and validate microservices iteratively.
"""

import logging
import subprocess
import re
import time
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: str = ""
    data: Any = None


@dataclass
class CompileError:
    """A single compilation error."""
    file_path: str
    line: int
    column: int
    message: str
    error_type: str = "error"  # error, warning

    def __str__(self):
        return f"{self.file_path}:{self.line}:{self.column}: {self.error_type}: {self.message}"


class BaseTool(ABC):
    """Base class for generator tools."""
    
    name: str = "base_tool"
    description: str = "Base tool"
    
    def __init__(self, service_path: Path):
        self.service_path = service_path
    
    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass
    
    def to_schema(self) -> dict:
        """Return OpenAI function schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema()
        }
    
    def get_parameters_schema(self) -> dict:
        """Override to define tool parameters."""
        return {"type": "object", "properties": {}}


class WriteJavaFileTool(BaseTool):
    """Write a Java source file."""
    
    name = "write_java_file"
    description = "Write a Java source file to the service. Path should be relative to src/main/java/"
    
    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": "Path relative to src/main/java/, e.g. 'com/example/User.java'"
                },
                "content": {
                    "type": "string", 
                    "description": "Complete Java source code"
                }
            },
            "required": ["relative_path", "content"]
        }
    
    def run(self, relative_path: str = None, content: str = None, path: str = None, **kwargs) -> ToolResult:
        # Accept both 'path' and 'relative_path' for robustness
        actual_path = relative_path or path or kwargs.get('file_path')
        if not actual_path:
            return ToolResult(success=False, output="Missing required argument: relative_path or path")
        if not content:
            return ToolResult(success=False, output="Missing required argument: content")
        try:
            file_path = self.service_path / "src" / "main" / "java" / actual_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            logger.debug(f"Wrote Java file: {file_path}")
            return ToolResult(success=True, output=f"Created {actual_path}")
        except Exception as e:
            logger.error(f"Failed to write Java file: {e}")
            return ToolResult(success=False, output=str(e))


class WritePomTool(BaseTool):
    """Write pom.xml file."""
    
    name = "write_pom"
    description = "Write the Maven pom.xml file for the service"
    
    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Complete pom.xml content"
                }
            },
            "required": ["content"]
        }
    
    def run(self, content: str = None, **kwargs) -> ToolResult:
        # Ignore extra args (e.g. path) and support alternate content key.
        actual_content = content or kwargs.get("pom_content")
        if not actual_content:
            return ToolResult(success=False, output="Missing required argument: content")
        try:
            pom_path = self.service_path / "pom.xml"
            pom_path.write_text(actual_content, encoding="utf-8")
            logger.debug(f"Wrote pom.xml: {pom_path}")
            return ToolResult(success=True, output="Created pom.xml")
        except Exception as e:
            logger.error(f"Failed to write pom.xml: {e}")
            return ToolResult(success=False, output=str(e))


class WriteApplicationYmlTool(BaseTool):
    """Write application.yml file."""
    
    name = "write_application_yml"
    description = "Write the Spring application.yml configuration file"
    
    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Complete application.yml content"
                }
            },
            "required": ["content"]
        }
    
    def run(self, content: str) -> ToolResult:
        try:
            yml_path = self.service_path / "src" / "main" / "resources" / "application.yml"
            yml_path.parent.mkdir(parents=True, exist_ok=True)
            yml_path.write_text(content, encoding="utf-8")
            logger.debug(f"Wrote application.yml: {yml_path}")
            return ToolResult(success=True, output="Created application.yml")
        except Exception as e:
            logger.error(f"Failed to write application.yml: {e}")
            return ToolResult(success=False, output=str(e))


class CompileTool(BaseTool):
    """Compile the service with Maven."""
    
    name = "compile"
    description = "Run 'mvn compile' to check if the service compiles. Returns success/failure."
    
    def get_parameters_schema(self) -> dict:
        return {"type": "object", "properties": {}}
    
    def run(self) -> ToolResult:
        try:
            result = subprocess.run(
                ["mvn", "compile", "-q", "-B"],
                cwd=self.service_path,
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            if result.returncode == 0:
                logger.info(f"Compile succeeded for {self.service_path.name}")
                return ToolResult(success=True, output="BUILD SUCCESS")
            else:
                logger.warning(f"Compile failed for {self.service_path.name}")
                return ToolResult(
                    success=False, 
                    output=result.stdout + result.stderr
                )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="Compile timeout (120s)")
        except FileNotFoundError:
            return ToolResult(success=False, output="Maven not found. Install Maven first.")
        except Exception as e:
            return ToolResult(success=False, output=str(e))


class GetCompileErrorsTool(BaseTool):
    """Get structured compilation errors."""
    
    name = "get_compile_errors"
    description = "Get detailed, structured compilation errors from the last compile attempt"
    
    def get_parameters_schema(self) -> dict:
        return {"type": "object", "properties": {}}
    
    def run(self) -> ToolResult:
        try:
            result = subprocess.run(
                ["mvn", "compile", "-B"],
                cwd=self.service_path,
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            if result.returncode == 0:
                return ToolResult(success=True, output="No errors", data=[])
            
            errors = self._parse_maven_errors(result.stdout + result.stderr)
            
            if errors:
                error_str = "\n".join(str(e) for e in errors[:10])  # Limit to 10
                return ToolResult(success=True, output=error_str, data=errors)
            else:
                return ToolResult(success=False, output=result.stderr[:2000])
                
        except Exception as e:
            return ToolResult(success=False, output=str(e))
    
    def _parse_maven_errors(self, output: str) -> List[CompileError]:
        """Parse Maven compiler output into structured errors."""
        errors = []
        
        # Pattern: [ERROR] /path/to/File.java:[line,col] error: message
        pattern = r'\[ERROR\]\s+([^:]+):?\[(\d+),(\d+)\]\s*(?:error:\s*)?(.*?)(?=\[ERROR\]|\[INFO\]|$)'
        
        for match in re.finditer(pattern, output, re.DOTALL):
            file_path = match.group(1).strip()
            line = int(match.group(2))
            col = int(match.group(3))
            message = match.group(4).strip()
            
            # Clean up message
            message = re.sub(r'\s+', ' ', message)[:200]
            
            errors.append(CompileError(
                file_path=file_path,
                line=line,
                column=col,
                message=message
            ))
        
        # Also try simpler pattern for "cannot find symbol" type errors
        if not errors:
            simple_pattern = r'\[ERROR\].*?\.java:\[(\d+),(\d+)\]\s*(.*)'
            for match in re.finditer(simple_pattern, output):
                errors.append(CompileError(
                    file_path="unknown",
                    line=int(match.group(1)),
                    column=int(match.group(2)),
                    message=match.group(3).strip()[:200]
                ))
        
        return errors


class ReadFileTool(BaseTool):
    """Read a file from the service."""
    
    name = "read_file"
    description = "Read a file from the generated service"
    
    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": "Path relative to service root, e.g. 'src/main/java/com/example/User.java'"
                }
            },
            "required": ["relative_path"]
        }
    
    def run(self, relative_path: str = None, path: str = None, **kwargs) -> ToolResult:
        # Accept both 'path' and 'relative_path' for robustness
        actual_path = relative_path or path or kwargs.get('file_path')
        if not actual_path:
            return ToolResult(success=False, output="Missing required argument: relative_path or path")
        try:
            file_path = self.service_path / actual_path
            if not file_path.exists():
                return ToolResult(success=False, output=f"File not found: {actual_path}")
            
            content = file_path.read_text(encoding="utf-8")
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output=str(e))


class ReadMonolithSourceTool(BaseTool):
    """Read source code from the original monolith."""
    
    name = "read_monolith_source"
    description = "Read the original source code of a class from the monolith"
    
    def __init__(self, service_path: Path, monolith_root: Path, graph):
        super().__init__(service_path)
        self.monolith_root = monolith_root
        self.graph = graph
    
    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "class_fqn": {
                    "type": "string",
                    "description": "Fully qualified class name, e.g. 'com.example.User'"
                }
            },
            "required": ["class_fqn"]
        }
    
    def run(self, class_fqn: str = None, fqn: str = None, **kwargs) -> ToolResult:
        actual_fqn = class_fqn or fqn or kwargs.get("fully_qualified_name")
        if not actual_fqn:
            return ToolResult(success=False, output="Missing required argument: class_fqn")
        try:
            if actual_fqn not in self.graph.graph.nodes:
                return ToolResult(success=False, output=f"Class not found in graph: {actual_fqn}")
            
            node_data = self.graph.graph.nodes[actual_fqn]
            file_path = node_data.get("file_path", "")
            
            if not file_path:
                return ToolResult(success=False, output=f"No file path for: {actual_fqn}")
            
            source_path = Path(file_path)
            if not source_path.is_absolute():
                source_path = self.monolith_root / source_path
            
            if not source_path.exists():
                return ToolResult(success=False, output=f"Source file not found: {source_path}")
            
            content = source_path.read_text(encoding="utf-8")
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output=str(e))

class WriteDockerfileTool(BaseTool):
    """Write Dockerfile."""
    
    name = "write_dockerfile"
    description = "Write the Dockerfile for the service"
    
    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Complete Dockerfile content"
                }
            },
            "required": ["content"]
        }
    
    def run(self, content: str) -> ToolResult:
        try:
            docker_path = self.service_path / "Dockerfile"
            docker_path.write_text(content, encoding="utf-8")
            logger.debug(f"Wrote Dockerfile: {docker_path}")
            return ToolResult(success=True, output="Created Dockerfile")
        except Exception as e:
            logger.error(f"Failed to write Dockerfile: {e}")
            return ToolResult(success=False, output=str(e))


class DockerBuildTool(BaseTool):
    """Build Docker image."""
    
    name = "docker_build"
    description = "Build Docker image from the service"
    
    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "image_name": {
                    "type": "string",
                    "description": "Name/tag for the image (e.g., 'vet-service:latest')"
                }
            },
            "required": ["image_name"]
        }
    
    def run(self, image_name: str) -> ToolResult:
        try:
            # First ensure compiled artifact exists, but we assume mvn compile was run
            # Run docker build
            result = subprocess.run(
                ["docker", "build", "-t", image_name, "."],
                cwd=self.service_path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if result.returncode == 0:
                logger.info(f"Docker build success: {image_name}")
                return ToolResult(success=True, output=f"Built image {image_name}")
            else:
                logger.warning(f"Docker build failed: {image_name}")
                return ToolResult(
                    success=False, 
                    output=result.stdout + result.stderr
                )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="Docker build timeout (300s)")
        except FileNotFoundError:
            return ToolResult(success=False, output="Docker not found")
        except Exception as e:
            return ToolResult(success=False, output=str(e))


class DockerHealthCheckTool(BaseTool):
    """Check service health in Docker."""
    
    name = "docker_health_check"
    description = "Run container and check health/connectivity"
    
    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "image_name": {
                    "type": "string",
                    "description": "Image to run"
                },
                "port": {
                    "type": "integer",
                    "description": "Host port to map to 8080"
                },
                "health_endpoint": {
                    "type": "string",
                    "description": "Health check path (default /actuator/health)"
                },
                "container_port": {
                    "type": "integer",
                    "description": "Port exposed by the container (default 8080)"
                }
            },
            "required": ["image_name", "port"]
        }
    
    def run(self, image_name: str, port: int, health_endpoint: str = "/actuator/health", container_port: int = 8080) -> ToolResult:
        container_id = None
        try:
            # 1. Run container
            run_cmd = [
                "docker", "run", "-d", "--rm",
                "-p", f"{port}:{container_port}",
                image_name
            ]
            result = subprocess.run(run_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return ToolResult(success=False, output=f"Failed to start container: {result.stderr}")
            
            container_id = result.stdout.strip()
            
            # 2. Wait and poll health
            retries = 30
            url = f"http://localhost:{port}{health_endpoint}"
            
            for i in range(retries):
                try:
                    time.sleep(2)
                    response = requests.get(url, timeout=2)
                    if response.status_code == 200:
                        # Success!
                        # Extract explicit status if possible
                        status = "UP"
                        try:
                            json_body = response.json()
                            status = json_body.get("status", "UP")
                        except:
                            pass
                            
                        return ToolResult(
                            success=True, 
                            output=f"Health check passed. Status: {status}",
                            data={"container_id": container_id}
                        )
                except Exception:
                    pass
            
            # Failed
            try:
                logs = subprocess.run(
                    ["docker", "logs", "--tail", "50", container_id],
                    capture_output=True, text=True
                ).stdout
            except:
                logs = "Could not get logs"
                
            return ToolResult(success=False, output=f"Health check failed after 60s. Logs:\n{logs}")
            
        except Exception as e:
            return ToolResult(success=False, output=str(e))
        finally:
            if container_id:
                subprocess.run(["docker", "stop", container_id], capture_output=True)
class DoneTool(BaseTool):
    """Signal that service generation is complete."""
    
    name = "done"
    description = "Signal that you have finished generating the service and it compiles successfully"
    
    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether generation was successful"
                },
                "message": {
                    "type": "string",
                    "description": "Summary of what was generated"
                }
            },
            "required": ["success", "message"]
        }
    
    def run(self, success: bool = True, message: str = "Done", **kwargs) -> ToolResult:
        # Be permissive to reduce orchestration dead-ends from partial tool calls.
        if "success" in kwargs:
            success = kwargs["success"]
        if "message" in kwargs:
            message = kwargs["message"]
        return ToolResult(success=bool(success), output=str(message))


def create_tool_set(
    service_path: Path,
    monolith_root: Optional[Path] = None,
    graph=None
) -> List[BaseTool]:
    """Create the complete set of tools for service generation."""
    tools = [
        WriteJavaFileTool(service_path),
        WritePomTool(service_path),
        WriteApplicationYmlTool(service_path),
        CompileTool(service_path),
        GetCompileErrorsTool(service_path),
        ReadFileTool(service_path),
        WriteDockerfileTool(service_path),
        DockerBuildTool(service_path),
        DockerHealthCheckTool(service_path),
        DoneTool(service_path),
    ]
    
    if monolith_root and graph:
        tools.append(ReadMonolithSourceTool(service_path, monolith_root, graph))
    
    return tools
