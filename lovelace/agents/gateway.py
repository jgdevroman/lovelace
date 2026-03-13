"""Gateway Agent - LLM-first Spring Cloud Gateway generation.

Uses batch generation for initial creation, and tool-based iterative approach for fixing.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from lovelace.agents.base import BaseAgent
from lovelace.agents.generator_tools import (
    BaseTool,
    ToolResult,
    WritePomTool,
    WriteApplicationYmlTool,
    WriteJavaFileTool,
    CompileTool,
    GetCompileErrorsTool,
    ReadFileTool,
)
from lovelace.core.clustering import ClusterInfo
from lovelace.core.graph import DependencyGraph
from lovelace.core.llm import LLMClient

logger = logging.getLogger(__name__)

MAX_FIX_ITERATIONS = 10


class GatewayAgent(BaseAgent):
    """
    LLM-powered agent for generating Spring Cloud Gateway services.
    
    Uses batch generation for initial creation (one-shot),
    and tool-based iterative approach for fixing compilation errors.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        graph: DependencyGraph,
        monolith_base_url: str = "http://localhost:9090"
    ):
        """
        Initialize Gateway Agent.

        Args:
            llm_client: LLM for intelligent gateway generation
            graph: Dependency graph (kept for BaseAgent compatibility)
            monolith_base_url: Where monolith will run after gateway deployment
        """
        super().__init__(llm_client, graph)
        self.monolith_base_url = monolith_base_url
        self.tools: List[BaseTool] = []

    def generate_gateway(
        self,
        output_dir: Path,
        extracted_services: List[ClusterInfo],
        openapi_specs: Dict[str, dict],
        project_metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate complete Spring Cloud Gateway service using BATCH approach.

        All files are generated in one LLM call (pom.xml, application.yml, Java files).
        
        Args:
            output_dir: Base output directory
            extracted_services: List of services to be extracted
            openapi_specs: OpenAPI specs for each service
            project_metadata: Project metadata dict from migration plan

        Returns:
            Dict with path, routes, and success status
        """
        logger.info(f"Gateway Agent: Analyzing {len(extracted_services)} services")

        # Get Spring Boot version with validation
        spring_boot_version = self._get_valid_spring_boot_version(project_metadata)
        logger.info(f"Using Spring Boot version: {spring_boot_version}")

        # Setup gateway path
        gateway_path = output_dir / "api-gateway"
        gateway_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate routing plan
        routing_plan = self._generate_routing_plan(extracted_services, openapi_specs)
        logger.info(f"Generated routing plan with {len(routing_plan.get('routes', []))} routes")

        # Step 2: BATCH generate all files
        self._generate_spring_cloud_gateway(
            gateway_path=gateway_path,
            routing_plan=routing_plan,
            monolith_url=self.monolith_base_url,
            spring_boot_version=spring_boot_version,
        )

        # Build route patterns
        route_patterns = {}
        for route in routing_plan.get("routes", []):
            service_name = route.get("service_name")
            if service_name:
                route_patterns[service_name] = f"/api/{service_name}/**"

        logger.info(f"✓ Gateway generated at {gateway_path}")

        # Check if it compiles
        compile_result = CompileTool(gateway_path).run()

        return {
            "path": gateway_path,
            "routes": route_patterns,
            "routing_plan": routing_plan,
            "success": compile_result.success,
        }

    def fix_gateway(
        self,
        gateway_path: Path,
        error_message: str,
        max_iterations: int = MAX_FIX_ITERATIONS,
    ) -> bool:
        """
        Fix gateway code using ITERATIVE tool-based approach.
        
        Args:
            gateway_path: Path to the gateway service directory
            error_message: The compile error output
            max_iterations: Maximum fix iterations
            
        Returns:
            True if gateway now compiles, False otherwise
        """
        logger.info("Fixing gateway using iterative tool-based approach...")

        # Create tools for the gateway
        self.tools = self._create_gateway_tools(gateway_path)

        # Read current files for context
        files_context = self._read_current_files(gateway_path)

        system_prompt = self._build_fix_system_prompt()
        user_prompt = f"""The gateway compilation failed. Please fix the issues using the tools.

## Compile Error:
```
{error_message}
```

## Current Gateway Files:
{files_context}

Analyze the error and use the appropriate tools to fix the files. After fixing, use the compile tool to verify."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            logger.debug(f"Fix iteration {iterations}/{max_iterations}")

            try:
                response = self.llm.chat(messages=messages, temperature=0.1)
                tool_calls = self._extract_tool_calls(response.content)

                if not tool_calls:
                    # No tool calls - check compile status
                    compile_result = CompileTool(gateway_path).run()
                    if compile_result.success:
                        logger.info("Gateway fixed successfully!")
                        return True
                    
                    # Prompt for action
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": f"Still not compiling. Error:\n{compile_result.output}\n\nPlease use tools to fix."
                    })
                    continue

                # Execute tool calls
                tool_results = []
                compile_succeeded = False
                
                for tool_call in tool_calls:
                    result = self._execute_tool(tool_call)
                    tool_results.append({
                        "tool": tool_call["name"],
                        "result": result.output if result else "Tool not found"
                    })

                    # Check if compile succeeded
                    if tool_call["name"] == "compile" and result and result.success:
                        compile_succeeded = True

                if compile_succeeded:
                    logger.info("Gateway fixed successfully!")
                    return True

                # Add results to conversation
                messages.append({"role": "assistant", "content": response.content})
                tool_output = "\n".join(
                    f"[{r['tool']}]: {r['result']}" for r in tool_results
                )
                messages.append({
                    "role": "user",
                    "content": f"Tool results:\n{tool_output}\n\nContinue fixing if needed, then compile to verify."
                })

            except Exception as e:
                logger.error(f"Error in fix loop: {e}")
                continue

        # Final check
        compile_result = CompileTool(gateway_path).run()
        return compile_result.success

    def _get_valid_spring_boot_version(self, project_metadata: Optional[Dict]) -> str:
        """Get and validate Spring Boot version."""
        default_version = "3.2.0"
        
        if not project_metadata:
            return default_version

        # .get(default) does not apply when the key exists with value None.
        raw_version = project_metadata.get("spring_boot_version")
        if raw_version is None:
            logger.info(f"No Spring Boot version detected, using {default_version}")
            return default_version

        version = str(raw_version).strip()
        if not version:
            logger.info(f"Empty Spring Boot version detected, using {default_version}")
            return default_version
        
        # Validate - reject invalid versions
        if version.startswith("4.") or not version[0].isdigit():
            logger.warning(f"Invalid Spring Boot version '{version}', using {default_version}")
            return default_version
            
        return version

    def _generate_routing_plan(
        self,
        extracted_services: List[ClusterInfo],
        openapi_specs: Dict[str, dict]
    ) -> Dict:
        """Generate intelligent routing plan using LLM."""
        service_endpoints = {}
        for service in extracted_services:
            service_name = service.suggested_name
            if service_name in openapi_specs:
                spec = openapi_specs[service_name]
                paths = list(spec.get('paths', {}).keys())
                service_endpoints[service_name] = {
                    'paths': paths,
                    'class_count': service.class_count,
                    'entities': service.entities,
                    'entry_points': service.entry_points
                }

        context = f"""Design routing configuration for Spring Cloud Gateway API Gateway.

## Extracted Services:
```json
{json.dumps(service_endpoints, indent=2)}
```

## Requirements:
1. **Route extracted services** - Higher priority (order: 100-200)
2. **Monolith fallback** - Lowest priority (order: 999)
3. **Path grouping** - Combine related paths
4. **Port assignments** - Each service gets a port starting from 8081

## Output Format:
```json
{{
  "routes": [
    {{
      "id": "service-name-route",
      "service_name": "service-name",
      "path_pattern": "/path/**",
      "target_url": "http://localhost:8081",
      "priority": 100
    }}
  ],
  "port_assignments": {{
    "service-name": 8081
  }}
}}
```

Return ONLY valid JSON, no explanation."""

        messages = [
            {"role": "system", "content": "You are an expert in Spring Cloud Gateway configuration."},
            {"role": "user", "content": context}
        ]

        response = self.llm.chat(messages, temperature=0.2)
        content = response.content.strip()
        
        # Extract JSON from response
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse routing plan, using defaults")
            return {"routes": [], "port_assignments": {}}

    def _generate_spring_cloud_gateway(
        self,
        gateway_path: Path,
        routing_plan: Dict,
        monolith_url: str,
        spring_boot_version: str
    ):
        """Generate Spring Cloud Gateway files using BATCH LLM calls."""
        
        # Generate pom.xml
        pom_content = self._llm_generate_maven_pom(routing_plan, spring_boot_version)
        (gateway_path / "pom.xml").write_text(pom_content, encoding="utf-8")

        # Generate application.yml
        yml_content = self._llm_generate_application_yml(routing_plan, monolith_url)
        resources_dir = gateway_path / "src" / "main" / "resources"
        resources_dir.mkdir(parents=True, exist_ok=True)
        (resources_dir / "application.yml").write_text(yml_content, encoding="utf-8")

        # Generate main application class
        app_class = self._llm_generate_spring_main_class()
        java_dir = gateway_path / "src" / "main" / "java" / "com" / "example" / "gateway"
        java_dir.mkdir(parents=True, exist_ok=True)
        (java_dir / "ApiGatewayApplication.java").write_text(app_class, encoding="utf-8")

        # Generate toggle controller
        controller = self._llm_generate_toggle_controller(routing_plan)
        (java_dir / "RouteToggleController.java").write_text(controller, encoding="utf-8")

    def _llm_generate_maven_pom(self, routing_plan: Dict, spring_boot_version: str) -> str:
        """Generate pom.xml using LLM."""
        prompt = f"""Generate a Maven pom.xml for Spring Cloud Gateway API Gateway.

## Requirements:
1. Spring Boot {spring_boot_version}
2. Spring Cloud Gateway dependency (compatible version)
3. Actuator for health checks
4. Reactive web (required for Gateway)

## Routing needs:
- {len(routing_plan.get('routes', []))} routes configured

Generate ONLY the pom.xml content, no explanation."""

        messages = [
            {"role": "system", "content": "You are a Spring Boot expert. Generate production-ready Maven configurations."},
            {"role": "user", "content": prompt}
        ]

        response = self.llm.chat(messages, temperature=0.1)
        return self._extract_xml_code(response.content)

    def _llm_generate_application_yml(self, routing_plan: Dict, monolith_url: str) -> str:
        """Generate application.yml using LLM."""
        prompt = f"""Generate application.yml for Spring Cloud Gateway.

## Routing Plan:
```json
{json.dumps(routing_plan, indent=2)}
```

## Requirements:
1. Configure all routes from the plan
2. Add monolith fallback route with Path=/** and priority 999, target: {monolith_url}
3. Configure actuator endpoints
4. Set gateway port to 8080
5. Add logging for debugging

Generate ONLY the YAML content, no explanation."""

        messages = [
            {"role": "system", "content": "You are a Spring Cloud Gateway expert."},
            {"role": "user", "content": prompt}
        ]

        response = self.llm.chat(messages, temperature=0.1)
        return self._extract_yaml_code(response.content)

    def _llm_generate_spring_main_class(self) -> str:
        """Generate Spring Boot main application class."""
        prompt = """Generate a Spring Boot main application class for API Gateway.

## Requirements:
1. Package: com.example.gateway
2. Class name: ApiGatewayApplication
3. @SpringBootApplication annotation
4. Main method to run the app
5. Include a /fallback endpoint that returns graceful error message
6. Add JavaDoc explaining the strangler fig pattern

Generate ONLY the Java code, no explanation."""

        messages = [
            {"role": "system", "content": "You are a Spring Boot expert. Generate clean, production-ready code."},
            {"role": "user", "content": prompt}
        ]

        response = self.llm.chat(messages, temperature=0.1)
        return self._extract_java_code(response.content)

    def _llm_generate_toggle_controller(self, routing_plan: Dict) -> str:
        """Generate feature toggle controller."""
        services = list(routing_plan.get('port_assignments', {}).keys())

        prompt = f"""Generate a REST controller for managing gateway route toggles.

## Services to manage:
{', '.join(services) if services else 'owner-service, vet-service'}

## Requirements:
1. Package: com.example.gateway
2. Class name: RouteToggleController
3. Endpoints:
   - POST /admin/routes/{{serviceName}}/enable
   - POST /admin/routes/{{serviceName}}/disable
   - GET /admin/routes/status
4. In-memory toggle state (Map<String, Boolean>)
5. Return JSON responses
6. Add JavaDoc

Generate ONLY the Java code, no explanation."""

        messages = [
            {"role": "system", "content": "You are a Spring Boot expert. Generate clean REST controllers."},
            {"role": "user", "content": prompt}
        ]

        response = self.llm.chat(messages, temperature=0.1)
        return self._extract_java_code(response.content)

    # =========================================================================
    # Tool-based fix methods
    # =========================================================================

    def _create_gateway_tools(self, gateway_path: Path) -> List[BaseTool]:
        """Create tools for gateway fixing."""
        return [
            WritePomTool(gateway_path),
            WriteApplicationYmlTool(gateway_path),
            WriteJavaFileTool(gateway_path),
            CompileTool(gateway_path),
            GetCompileErrorsTool(gateway_path),
            ReadFileTool(gateway_path),
        ]

    def _build_fix_system_prompt(self) -> str:
        """Build system prompt for fix mode."""
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        )

        return f"""You are an expert Java developer fixing compilation errors in a Spring Cloud Gateway.

You have access to these tools:
{tool_descriptions}

WORKFLOW:
1. Analyze the compile errors carefully
2. Use write_pom, write_application_yml, or write_java_file to fix the issues
3. Use compile to verify the fix worked
4. Repeat until compilation succeeds

IMPORTANT:
- Fix one issue at a time
- Always compile after making changes
- Use the read_file tool if you need to see current file contents

To call a tool, use this format:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>"""

    def _read_current_files(self, gateway_path: Path) -> str:
        """Read current gateway files for context."""
        files_content = []
        
        pom_path = gateway_path / "pom.xml"
        if pom_path.exists():
            files_content.append(f"### pom.xml\n```xml\n{pom_path.read_text()}\n```")
        
        yml_path = gateway_path / "src" / "main" / "resources" / "application.yml"
        if yml_path.exists():
            files_content.append(f"### application.yml\n```yaml\n{yml_path.read_text()}\n```")
        
        java_dir = gateway_path / "src" / "main" / "java" / "com" / "example" / "gateway"
        if java_dir.exists():
            for java_file in java_dir.glob("*.java"):
                files_content.append(f"### {java_file.name}\n```java\n{java_file.read_text()}\n```")
        
        return "\n\n".join(files_content) if files_content else "No files found."

    def _extract_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response."""
        tool_calls = []
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                if "name" in call_data:
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {match.group(1)}")
        
        return tool_calls

    def _execute_tool(self, tool_call: Dict[str, Any]) -> Optional[ToolResult]:
        """Execute a tool call and return the result."""
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    return tool.run(**arguments)
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    return ToolResult(success=False, output=str(e))
        
        logger.warning(f"Unknown tool: {tool_name}")
        return None

    # =========================================================================
    # Code extraction helpers
    # =========================================================================

    def _extract_yaml_code(self, llm_output: str) -> str:
        """Extract YAML code from LLM response."""
        pattern = r"```(?:yaml|yml)?\s+(.*?)```"
        matches = re.findall(pattern, llm_output, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
        return llm_output.strip().replace("```", "").strip()

    def _extract_java_code(self, llm_output: str) -> str:
        """Extract Java code from LLM response."""
        pattern = r"```(?:java)?\s+(.*?)```"
        matches = re.findall(pattern, llm_output, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
        return llm_output.strip().replace("```", "").strip()

    def _extract_xml_code(self, llm_output: str) -> str:
        """Extract XML code from LLM response."""
        pattern = r"```(?:xml)?\s+(.*?)```"
        matches = re.findall(pattern, llm_output, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
        return llm_output.strip().replace("```", "").strip()
