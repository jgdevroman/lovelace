"""Agentic service generator with tool-use capabilities.

This agent iteratively generates microservices using LLM with access to tools
for writing files, compiling, and fixing errors.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from lovelace.agents.base import BaseAgent
from lovelace.agents.generator_tools import (
    BaseTool,
    ToolResult,
    create_tool_set,
    CompileTool,
    GetCompileErrorsTool,
)
from lovelace.core.graph import DependencyGraph
from lovelace.core.llm import LLMClient
from lovelace.core.service_spec import ServiceSpec, ServiceResult

logger = logging.getLogger(__name__)

# Default token budget per service in USD
DEFAULT_COST_LIMIT_PER_SERVICE = 0.50
DEFAULT_MAX_ITERATIONS = 15
# Legacy alias kept for external callers
MAX_ITERATIONS = DEFAULT_MAX_ITERATIONS


class ServiceGeneratorAgent(BaseAgent):
    """
    Agentic LLM that generates microservices with tool access.
    
    The agent iteratively uses tools to:
    1. Write pom.xml and application.yml
    2. Write Java source files (entities, repositories, services, controllers)
    3. Compile to check for errors
    4. Fix errors and retry until successful
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        graph: DependencyGraph,
        monolith_root: Optional[Path] = None,
        cost_limit_per_service: float = DEFAULT_COST_LIMIT_PER_SERVICE,
    ):
        super().__init__(llm_client, graph)
        self.monolith_root = monolith_root or Path.cwd()
        self.cost_limit_per_service = cost_limit_per_service
        self.tools: List[BaseTool] = []
        self._current_cost = 0.0
    
    def generate_service(
        self,
        spec: ServiceSpec,
        output_dir: Path,
        max_iterations: int = 0,
    ) -> ServiceResult:
        """
        Generate a complete microservice from specification.
        
        Args:
            spec: Service specification with entities, repositories, etc.
            output_dir: Base directory for output (service will be created as subdirectory).
            max_iterations: Maximum tool-use iterations. 0 = auto-calculate from spec size.
            
        Returns:
            ServiceResult with success status and path.
        """
        service_path = output_dir / spec.name
        service_path.mkdir(parents=True, exist_ok=True)
        
        # Auto-calculate iteration budget based on spec complexity
        if max_iterations <= 0:
            expected_files = (
                1  # pom.xml
                + 1  # application.yml
                + 1  # Dockerfile
                + 1  # Application.java
                + len(spec.entities)
                + len(spec.repositories)
                + len(spec.services)
                + len(spec.controllers)
            )
            # 2x multiplier gives room for compile/fix cycles
            max_iterations = max(DEFAULT_MAX_ITERATIONS, expected_files * 2)
            logger.info(f"Auto-calculated max_iterations={max_iterations} for {expected_files} expected files")
        
        logger.info(f"Generating service: {spec.name} at {service_path}")
        
        # Create tools for this service
        self.tools = create_tool_set(
            service_path=service_path,
            monolith_root=self.monolith_root,
            graph=self.graph,
        )
        
        # Build tool schemas for LLM
        tool_schemas = [tool.to_schema() for tool in self.tools]
        
        # Build initial prompt with spec
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(spec)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Agentic loop
        self._current_cost = 0.0
        iterations = 0
        last_tool_result = None
        
        while iterations < max_iterations:
            iterations += 1
            
            # Check cost limit
            if self._current_cost >= self.cost_limit_per_service:
                logger.warning(f"Cost limit reached for {spec.name}: ${self._current_cost:.4f}")
                return ServiceResult(
                    success=False,
                    service_path=str(service_path),
                    message=f"Cost limit reached: ${self._current_cost:.4f}",
                    cost_usd=self._current_cost,
                    iterations=iterations,
                )
            
            try:
                # Call LLM with tools
                response = self.llm.chat(
                    messages=messages,
                    temperature=0.2,
                )
                self._current_cost += response.cost_usd
                
                # Check if response contains tool calls
                tool_calls = self._extract_tool_calls(response.content)
                
                if not tool_calls:
                    # No tool calls - check if agent signaled done
                    if self._is_done_signal(response.content):
                        # Verify compilation
                        compile_tool = CompileTool(service_path)
                        compile_result = compile_tool.run()
                        
                        if compile_result.success:
                            logger.info(f"Service {spec.name} generated successfully!")
                            return ServiceResult(
                                success=True,
                                service_path=str(service_path),
                                message=f"Generated {spec.name} with {iterations} iterations",
                                cost_usd=self._current_cost,
                                iterations=iterations,
                            )
                        else:
                            # Add compile errors to context for retry
                            messages.append({"role": "assistant", "content": response.content})
                            messages.append({
                                "role": "user",
                                "content": f"The service does not compile yet. Errors:\n{compile_result.output}\n\nPlease fix these errors using the tools."
                            })
                            continue
                    else:
                        # No tool calls and no done signal - prompt for action
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({
                            "role": "user",
                            "content": "Please use the tools to generate the service. Start with write_pom, then write the Java files."
                        })
                        continue
                
                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    result = self._execute_tool(tool_call)
                    tool_results.append({
                        "tool": tool_call["name"],
                        "result": result.output if result else "Tool not found"
                    })
                    
                    # Check for done signal
                    if tool_call["name"] == "done":
                        if result and result.success:
                            # Final compile check
                            compile_tool = CompileTool(service_path)
                            compile_result = compile_tool.run()
                            
                            if compile_result.success:
                                return ServiceResult(
                                    success=True,
                                    service_path=str(service_path),
                                    message=result.output,
                                    cost_usd=self._current_cost,
                                    iterations=iterations,
                                )
                
                # Add results to conversation
                messages.append({"role": "assistant", "content": response.content})
                tool_output = "\n".join(
                    f"[{r['tool']}]: {r['result']}" for r in tool_results
                )
                messages.append({
                    "role": "user",
                    "content": f"Tool results:\n{tool_output}\n\nContinue with the next step."
                })
                
            except Exception as e:
                logger.error(f"Error in agent loop: {e}", exc_info=True)
                messages.append({
                    "role": "user",
                    "content": f"An error occurred: {e}. Please continue."
                })
        
        # Max iterations reached
        logger.warning(f"Max iterations reached for {spec.name}")
        
        # Final compile check
        compile_tool = CompileTool(service_path)
        compile_result = compile_tool.run()
        
        # Completeness check — compilation alone is not enough
        missing = self._check_completeness(spec, service_path)
        if missing:
            logger.warning(f"Service {spec.name} is incomplete. Missing: {missing}")
            return ServiceResult(
                success=False,
                service_path=str(service_path),
                message=f"Max iterations reached. Missing files: {', '.join(missing)}",
                compile_errors=[compile_result.output] if not compile_result.success else [],
                cost_usd=self._current_cost,
                iterations=iterations,
            )
        
        return ServiceResult(
            success=compile_result.success,
            service_path=str(service_path),
            message=f"Max iterations reached. Compile: {'SUCCESS' if compile_result.success else 'FAILED'}",
            compile_errors=[compile_result.output] if not compile_result.success else [],
            cost_usd=self._current_cost,
            iterations=iterations,
        )
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        )
        
        return f"""You are an expert Java developer generating Spring Boot microservices.

You have access to the following tools:
{tool_descriptions}

WORKFLOW:
1. First, use write_pom to create the Maven pom.xml
2. Use write_application_yml to create the Spring configuration
3. Write Java files in this priority order:
   a. Entities (needed for compilation of everything else)
   b. Repositories
   c. Controllers — these are the MOST IMPORTANT business-logic files, write them BEFORE any optional files
   d. Service classes (if any in the spec)
   e. Application main class
4. Use compile to check if the service builds
5. If compilation fails, use get_compile_errors and fix the issues
6. Write Dockerfile last
7. When everything compiles, use the done tool to signal completion

CRITICAL: You MUST generate ALL controllers listed in the spec with full method implementations.
Do not stop or call done until every controller has been written.

IMPORTANT RULES:
- Entities MUST have @Id annotation
- Repositories MUST extend JpaRepository<EntityType, IdType>
- Include spring-boot-starter-validation for @Valid, @NotBlank, etc.
- All fields inherited from base classes must be included directly (no extends to missing classes)
- Use proper package names based on the service specification

CONTROLLER IMPLEMENTATION RULES:
- Each controller entry in the spec includes a 'source_code' field with the original Java source.
- Use that source as the authoritative reference for method signatures, request mappings, and business logic.
- Implement every handler method with real logic — do NOT leave bodies empty or as stubs.
- Translate Spring MVC patterns (Model, ModelAndView, Thymeleaf redirects) to REST equivalents:
  * Return ResponseEntity<T> instead of String view names
  * Use ResponseEntity.ok(result) for 200 responses
  * Use ResponseEntity.notFound().build() for missing resources
  * Use ResponseEntity.created(uri).body(saved) for POST endpoints
- Inject repositories directly if there is no intermediate service class in the spec.
- When the original source references classes that are not in this service (e.g. Thymeleaf Model),
  omit those parameters and adapt the logic accordingly.

To call a tool, use this exact format:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

You can make multiple tool calls in sequence."""
    
    def _build_user_prompt(self, spec: ServiceSpec) -> str:
        """Build the initial user prompt with service specification."""
        return f"""Generate the following microservice:

SERVICE SPECIFICATION:
{spec.to_json()}

Start by creating the pom.xml with these dependencies:
- spring-boot-starter-web
- spring-boot-starter-data-jpa
- spring-boot-starter-validation
- h2 (runtime scope)
- flyway-core
- spring-boot-starter-test (test scope)

Then create the Java files in this order: entities, repositories, controllers, service classes, main Application class.
Make sure all entities have @Id and proper JPA annotations.
Repositories must extend JpaRepository.

IMPORTANT — for each controller in the spec, you will find a 'source_code' field
containing the full original Java source. You MUST implement every handler method
with real logic based on that source. Do not generate empty method bodies or TODO stubs.
Translate any Spring MVC patterns (Model, ModelAndView, view-name Strings) into
ResponseEntity<T>-based REST responses.

CHECKLIST — do NOT call the done tool until ALL of these exist:
- pom.xml
- application.yml
- Every entity listed in the spec
- Every repository listed in the spec
- Every controller listed in the spec (with real method bodies)

Begin now with write_pom."""
    
    def _extract_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response."""
        import re
        
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
    
    def _is_done_signal(self, content: str) -> bool:
        """Check if the response signals completion."""
        done_keywords = [
            "service is complete",
            "generation complete",
            "successfully generated",
            "all files created",
            "compilation successful",
        ]
        content_lower = content.lower()
        return any(kw in content_lower for kw in done_keywords)

    def _check_completeness(self, spec: ServiceSpec, service_path: Path) -> List[str]:
        """Verify that all expected files from the spec were actually generated.
        
        Returns a list of missing component descriptions. Empty list = complete.
        """
        missing = []
        
        # Check pom.xml
        if not (service_path / "pom.xml").exists():
            missing.append("pom.xml")
        
        # Helper: recursively find .java files by simple name
        java_files = {p.stem: p for p in service_path.rglob("*.java")}
        
        for entity in spec.entities:
            if entity.simple_name not in java_files:
                missing.append(f"Entity:{entity.simple_name}")
        
        for repo in spec.repositories:
            if repo.interface_name not in java_files:
                missing.append(f"Repository:{repo.interface_name}")
        
        for ctrl in spec.controllers:
            if ctrl.simple_name not in java_files:
                missing.append(f"Controller:{ctrl.simple_name}")
            else:
                # Check for empty-body controllers (class with no methods)
                content = java_files[ctrl.simple_name].read_text(encoding="utf-8")
                # Strip comments/whitespace; an empty body has only annotations/imports and {}
                import re
                body_match = re.search(r'public\s+class\s+\w+[^{]*\{(.*)\}', content, re.DOTALL)
                if body_match and body_match.group(1).strip() == "":
                    missing.append(f"Controller:{ctrl.simple_name} (empty body)")
        
        for svc in spec.services:
            if svc.simple_name not in java_files:
                missing.append(f"Service:{svc.simple_name}")
        
        return missing
