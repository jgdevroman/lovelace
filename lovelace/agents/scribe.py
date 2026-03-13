"""Scribe Agent for generating OpenAPI documentation and diagrams."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import javalang
import yaml

from lovelace.agents.base import BaseAgent
from lovelace.core.clustering import ClusterInfo
from lovelace.core.graph import DependencyGraph
from lovelace.core.llm import LLMClient
from lovelace.core.parser import JavaParser

logger = logging.getLogger(__name__)

# Mapping of Spring annotations to HTTP methods
HTTP_METHOD_ANNOTATIONS = {
    "GetMapping": "get",
    "PostMapping": "post",
    "PutMapping": "put",
    "DeleteMapping": "delete",
    "PatchMapping": "patch",
}

# Java type to OpenAPI type mapping
JAVA_TO_OPENAPI_TYPES = {
    "long": "integer",
    "int": "integer",
    "integer": "integer",
    "string": "string",
    "boolean": "boolean",
    "bigdecimal": "number",
    "double": "number",
    "float": "number",
    "list": "array",
}


class ScribeAgent(BaseAgent):
    """
    Scribe Agent generates OpenAPI specifications and PlantUML diagrams.

    This agent combines static analysis (extracting REST endpoints) with
    AI enhancement (adding descriptions, examples, error codes).
    """

    def __init__(self, llm_client: LLMClient, graph: DependencyGraph, parser: Optional[JavaParser] = None):
        """
        Initialize the Scribe Agent.

        Args:
            llm_client: LLM client for API calls
            graph: Dependency graph
            parser: Java parser instance (creates new one if not provided)
        """
        super().__init__(llm_client, graph)
        self.parser = parser or JavaParser()

    def generate_openapi(self, cluster: ClusterInfo, project_root: Path) -> Dict[str, Any]:
        """
        Generate OpenAPI spec for a cluster.

        Args:
            cluster: Cluster information
            project_root: Root directory of the project

        Returns:
            OpenAPI 3.0 spec as dictionary
        """
        logger.info(f"Scribe Agent: Generating OpenAPI for {cluster.suggested_name}...")

        # Phase 1: Static extraction
        skeleton = self._extract_skeleton(cluster, project_root)

        # Phase 2: AI enhancement
        try:
            enhanced = self._enhance_with_llm(skeleton, cluster)
            logger.info(f"Scribe Agent: OpenAPI generation complete for {cluster.suggested_name}")
            return enhanced
        except Exception as e:
            logger.warning(f"Scribe Agent: LLM enhancement failed, using skeleton: {e}")
            return skeleton

    def generate_diagram(self, cluster: ClusterInfo) -> str:
        """
        Generate PlantUML sequence diagram for a cluster.

        Args:
            cluster: Cluster information

        Returns:
            PlantUML diagram as string
        """
        logger.info(f"Scribe Agent: Generating diagram for {cluster.suggested_name}...")

        # Build diagram from entry points and their dependencies
        lines = ["@startuml", f"title {cluster.suggested_name} API Flow", ""]

        # Add participants
        participants = set()
        has_entry_points = False
        
        if cluster.entry_points:
            has_entry_points = True
            for entry_point in cluster.entry_points:
                if entry_point in self.graph.graph.nodes:
                    node_data = self.graph.graph.nodes[entry_point]
                    simple_name = node_data.get("simple_name", entry_point.split(".")[-1])
                    participants.add(('Client', 'actor'))
                    participants.add((simple_name, 'participant'))

                    # Find dependencies
                    for neighbor in self.graph.graph.successors(entry_point):
                        if neighbor in self.graph.graph.nodes:
                            neighbor_data = self.graph.graph.nodes[neighbor]
                            neighbor_simple = neighbor_data.get("simple_name", neighbor.split(".")[-1])
                            neighbor_type = neighbor_data.get("type", "Unknown")
                            if neighbor_type in ["Service", "Repository"]:
                                participants.add((neighbor_simple, 'participant'))
        else:
            # No entry points - this might be an entity-only cluster
            # Add entities as participants
            for entity_fqn in cluster.entities:
                if entity_fqn in self.graph.graph.nodes:
                    node_data = self.graph.graph.nodes[entity_fqn]
                    simple_name = node_data.get("simple_name", entity_fqn.split(".")[-1])
                    participants.add((simple_name, 'entity'))

        if not participants:
            # Fallback: add any class from the cluster
            for class_fqn in cluster.classes[:3]:  # Limit to first 3
                if class_fqn in self.graph.graph.nodes:
                    node_data = self.graph.graph.nodes[class_fqn]
                    simple_name = node_data.get("simple_name", class_fqn.split(".")[-1])
                    node_type = node_data.get("type", "Unknown")
                    ptype = 'participant' if node_type == "Controller" else 'entity'
                    participants.add((simple_name, ptype))

        for name, ptype in sorted(participants):
            lines.append(f"{ptype} \"{name}\"")

        lines.append("")
        
        if has_entry_points and cluster.entry_points:
            first_entry = cluster.entry_points[0].split(".")[-1]
            lines.append(f"Client -> {first_entry}: API Request")
        elif participants:
            # Use first participant if no entry points
            first_participant = sorted(participants)[0][0]
            lines.append(f"note over {first_participant}: Entity-only cluster")
        
        lines.append("@enduml")

        return "\n".join(lines)

    def _extract_skeleton(self, cluster: ClusterInfo, project_root: Path) -> Dict[str, Any]:
        """Extract OpenAPI skeleton from controllers and entities."""
        openapi_spec = {
            "openapi": "3.0.3",
            "info": {
                "title": f"{cluster.suggested_name} API",
                "version": "1.0.0",
            },
            "paths": {},
            "components": {"schemas": {}},
        }

        # Generate schemas from entities
        for entity_fqn in cluster.entities:
            schema = self._generate_schema(entity_fqn, project_root)
            if schema:
                simple_name = entity_fqn.split(".")[-1]
                openapi_spec["components"]["schemas"][simple_name] = schema

        # Generate paths from controllers
        for controller_fqn in cluster.entry_points:
            paths = self._extract_endpoints(controller_fqn, project_root)
            openapi_spec["paths"].update(paths)

        return openapi_spec

    def _extract_endpoints(self, controller_fqn: str, project_root: Path) -> Dict[str, Dict]:
        """Extract REST endpoints from a controller file."""
        if controller_fqn not in self.graph.graph.nodes:
            return {}

        node_data = self.graph.graph.nodes[controller_fqn]
        file_path = Path(node_data.get("file_path", ""))

        # Resolve relative paths
        if not file_path.is_absolute():
            file_path = project_root / file_path

        if not file_path.exists():
            logger.warning(f"Controller file not found: {file_path}")
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = javalang.parse.parse(content)
            paths = {}

            # Get base path from class-level @RequestMapping
            base_path = ""
            for type_decl in tree.types:
                if hasattr(type_decl, "annotations"):
                    for ann in type_decl.annotations:
                        if isinstance(ann, javalang.tree.Annotation):
                            ann_name = ".".join(ann.name) if isinstance(ann.name, list) else ann.name
                            if ann_name == "RequestMapping":
                                base_path = self._extract_annotation_value(ann) or ""

            # Extract methods with HTTP method annotations
            for type_decl in tree.types:
                if hasattr(type_decl, "methods"):
                    for method in type_decl.methods:
                        endpoint = self._method_to_endpoint(method, base_path)
                        if endpoint:
                            full_path = endpoint["path"]
                            http_method = endpoint["http_method"]

                            if full_path not in paths:
                                paths[full_path] = {}

                            paths[full_path][http_method] = {
                                "operationId": endpoint["operation_id"],
                                "parameters": endpoint["parameters"],
                                "responses": {
                                    "200": {
                                        "description": "Successful response",
                                        "content": {
                                            "application/json": {
                                                "schema": endpoint.get("response_schema", {})
                                            }
                                        },
                                    }
                                },
                            }

                            if endpoint.get("request_body"):
                                paths[full_path][http_method]["requestBody"] = endpoint["request_body"]

            return paths

        except Exception as e:
            logger.warning(f"Failed to parse controller {controller_fqn}: {e}")
            return {}

    def _method_to_endpoint(self, method, base_path: str) -> Optional[Dict[str, Any]]:
        """Convert a Java method to endpoint info."""
        if not hasattr(method, "annotations"):
            return None

        http_method = None
        path_suffix = ""

        # Find HTTP method annotation
        for ann in method.annotations:
            if isinstance(ann, javalang.tree.Annotation):
                ann_name = ".".join(ann.name) if isinstance(ann.name, list) else ann.name
                if ann_name in HTTP_METHOD_ANNOTATIONS:
                    http_method = HTTP_METHOD_ANNOTATIONS[ann_name]
                    path_suffix = self._extract_annotation_value(ann) or ""
                    break

        if not http_method:
            return None

        # Build parameters from method params
        parameters = []
        request_body = None

        if hasattr(method, "parameters") and method.parameters:
            for param in method.parameters:
                param_annotations = []
                if hasattr(param, "annotations") and param.annotations:
                    for ann in param.annotations:
                        if isinstance(ann, javalang.tree.Annotation):
                            ann_name = ".".join(ann.name) if isinstance(ann.name, list) else ann.name
                            param_annotations.append(ann_name)

                param_type_str = self._get_type_string(param.type)
                param_name = param.name

                if "PathVariable" in param_annotations:
                    parameters.append(
                        {
                            "name": param_name,
                            "in": "path",
                            "required": True,
                            "schema": {"type": self._java_to_openapi_type(param_type_str)},
                        }
                    )
                elif "RequestParam" in param_annotations:
                    parameters.append(
                        {
                            "name": param_name,
                            "in": "query",
                            "required": True,
                            "schema": {"type": self._java_to_openapi_type(param_type_str)},
                        }
                    )
                elif "RequestBody" in param_annotations:
                    type_name = param_type_str.split(".")[-1].split("<")[0]  # Handle generics
                    request_body = {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": f"#/components/schemas/{type_name}"}
                            }
                        },
                    }

        # Build response schema from return type
        response_schema = {}
        if hasattr(method, "return_type") and method.return_type:
            return_type_str = self._get_type_string(method.return_type)
            if return_type_str and return_type_str.lower() not in ("void", "none"):
                # Handle List<T> and other generics
                if "list" in return_type_str.lower() or "collection" in return_type_str.lower():
                    # Extract generic type
                    generic_type = return_type_str.split("<")[-1].split(">")[0].strip()
                    generic_simple = generic_type.split(".")[-1]
                    response_schema = {
                        "type": "array",
                        "items": {"$ref": f"#/components/schemas/{generic_simple}"},
                    }
                else:
                    type_simple = return_type_str.split(".")[-1]
                    response_schema = {"$ref": f"#/components/schemas/{type_simple}"}

        full_path = base_path + path_suffix
        if not full_path.startswith("/"):
            full_path = "/" + full_path

        return {
            "http_method": http_method,
            "path": full_path,
            "operation_id": method.name,
            "parameters": parameters,
            "request_body": request_body,
            "response_schema": response_schema,
        }

    def _generate_schema(self, entity_fqn: str, project_root: Path) -> Dict[str, Any]:
        """Generate OpenAPI schema from an entity class."""
        if entity_fqn not in self.graph.graph.nodes:
            return {}

        node_data = self.graph.graph.nodes[entity_fqn]
        file_path = Path(node_data.get("file_path", ""))

        if not file_path.is_absolute():
            file_path = project_root / file_path

        if not file_path.exists():
            return {}

        class_meta = self.parser.parse_java_file(file_path)
        if not class_meta:
            return {}

        properties = {}
        for field_type, field_name in class_meta.fields:
            # Skip JPA relationships (they're references, not properties)
            if any(
                rel.field_name == field_name
                for rel in (class_meta.jpa_relationships or [])
            ):
                continue

            properties[field_name] = {
                "type": self._java_to_openapi_type(field_type),
            }

        return {
            "type": "object",
            "properties": properties,
        }

    def _java_to_openapi_type(self, java_type: str) -> str:
        """Map Java types to OpenAPI types."""
        type_str = str(java_type).lower()
        for java, openapi in JAVA_TO_OPENAPI_TYPES.items():
            if java in type_str:
                return openapi
        return "object"  # Default for complex types

    def _get_type_string(self, type_node) -> str:
        """Extract type string from javalang type node."""
        if hasattr(type_node, "name"):
            if isinstance(type_node.name, str):
                return type_node.name
            elif isinstance(type_node.name, list):
                return ".".join(type_node.name)
        return str(type_node)

    def _extract_annotation_value(self, annotation) -> Optional[str]:
        """Extract the value from an annotation like @GetMapping("/{id}")."""
        if not hasattr(annotation, "element"):
            return None

        element = annotation.element
        if isinstance(element, javalang.tree.Literal):
            return element.value.strip('"').strip("'")
        elif isinstance(element, list):
            for elem in element:
                if isinstance(elem, javalang.tree.ElementValuePair):
                    if isinstance(elem.value, javalang.tree.Literal):
                        return elem.value.value.strip('"').strip("'")
                elif isinstance(elem, javalang.tree.Literal):
                    return elem.value.strip('"').strip("'")

        return None

    def _enhance_with_llm(self, skeleton: Dict[str, Any], cluster: ClusterInfo) -> Dict[str, Any]:
        """Enhance OpenAPI skeleton with LLM-generated descriptions and examples."""
        prompt = f"""You are an API documentation expert. Given the following OpenAPI skeleton for a microservice, enhance it with:

1. Clear summaries and descriptions for all endpoints
2. Example request/response bodies
3. Proper error response codes (400, 404, 500, etc.)
4. Schema descriptions and examples
5. Tags for grouping endpoints

## Service: {cluster.suggested_name}

## Current OpenAPI Skeleton:
```json
{json.dumps(skeleton, indent=2)}
```

Enhance this OpenAPI spec and return ONLY the complete, valid OpenAPI 3.0 JSON. Do not include markdown code blocks or explanations.
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert at writing OpenAPI 3.0 specifications. "
                "Generate production-ready API documentation with clear descriptions, examples, and error handling.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.llm.chat(messages, temperature=0.7)
            enhanced = json.loads(response.content.strip().strip("```json").strip("```"))
            return enhanced
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON, using skeleton")
            return skeleton

