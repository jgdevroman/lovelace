"""Build ServiceSpec from ClusterInfo and dependency graph."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from lovelace.core.clustering import BoundaryEdge, ClusterInfo
from lovelace.core.graph import DependencyGraph
from lovelace.core.service_spec import (
    ControllerSpec,
    EntitySpec,
    FieldSpec,
    GatewayClientSpec,
    MethodSpec,
    RelationshipSpec,
    RepositorySpec,
    ServiceClassSpec,
    ServiceSpec,
)

logger = logging.getLogger(__name__)


class SpecBuilder:
    """Builds ServiceSpec from ClusterInfo."""
    
    DEFAULT_MAVEN_DEPENDENCIES = [
        "spring-boot-starter-web",
        "spring-boot-starter-data-jpa",
        "spring-boot-starter-validation",
        "h2",
        "flyway-core",
    ]
    
    def __init__(
        self,
        graph: DependencyGraph,
        monolith_root: Path,
        all_clusters: List[ClusterInfo],
    ):
        self.graph = graph
        self.monolith_root = monolith_root
        self.all_clusters = all_clusters
        self._class_to_cluster = self._build_class_cluster_map()
    
    def _build_class_cluster_map(self) -> Dict[str, str]:
        """Map class FQN to cluster name."""
        mapping = {}
        for cluster in self.all_clusters:
            for class_fqn in cluster.classes:
                mapping[class_fqn] = cluster.suggested_name
        return mapping
    
    def build_spec(
        self,
        cluster: ClusterInfo,
        boundary_edges: List[BoundaryEdge],
        spring_boot_version: str = "3.2.0",
        java_version: int = 17,
    ) -> ServiceSpec:
        """
        Build a ServiceSpec from a ClusterInfo.
        
        Args:
            cluster: Cluster to build spec for.
            boundary_edges: Boundary edges for this cluster.
            spring_boot_version: Spring Boot version.
            java_version: Java version.
            
        Returns:
            Complete ServiceSpec.
        """
        logger.info(f"Building spec for {cluster.suggested_name}")
        
        # Determine base package
        base_package = self._derive_base_package(cluster)
        
        # Build component specs
        entities = self._build_entity_specs(cluster)
        repositories = self._build_repository_specs(cluster, entities)
        services = self._build_service_specs(cluster)
        controllers = self._build_controller_specs(cluster)
        gateway_clients = self._build_gateway_client_specs(cluster, boundary_edges)
        
        # Check if depends on shared service
        depends_on_shared = self._check_shared_dependency(cluster)
        
        return ServiceSpec(
            name=cluster.suggested_name,
            base_package=base_package,
            spring_boot_version=spring_boot_version,
            java_version=java_version,
            entities=entities,
            repositories=repositories,
            services=services,
            controllers=controllers,
            gateway_clients=gateway_clients,
            maven_dependencies=self.DEFAULT_MAVEN_DEPENDENCIES.copy(),
            depends_on_shared=depends_on_shared,
        )
    
    def _derive_base_package(self, cluster: ClusterInfo) -> str:
        """Derive base package from cluster classes."""
        if not cluster.classes:
            return f"com.example.{cluster.suggested_name.replace('-', '')}"
        
        first_class = cluster.classes[0]
        if first_class in self.graph.graph.nodes:
            package = self.graph.graph.nodes[first_class].get("package", "")
            if package:
                return package
        
        return f"com.example.{cluster.suggested_name.replace('-', '')}"
    
    def _build_entity_specs(self, cluster: ClusterInfo) -> List[EntitySpec]:
        """Build EntitySpec for each entity in the cluster."""
        entities = []
        
        for entity_fqn in cluster.entities:
            if entity_fqn not in self.graph.graph.nodes:
                continue
            
            node_data = self.graph.graph.nodes[entity_fqn]
            simple_name = entity_fqn.split(".")[-1]
            
            # Read source code
            source_code = self._read_source(node_data.get("file_path", ""))
            
            # Build field specs (including inherited)
            fields = self._extract_fields(entity_fqn, node_data)
            
            # Build relationship specs
            relationships = self._extract_relationships(entity_fqn, source_code)
            
            # Derive table name
            table_name = self._derive_table_name(simple_name, source_code)
            
            # Check for external superclass
            superclass = self._get_external_superclass(entity_fqn, cluster)
            
            entities.append(EntitySpec(
                fqn=entity_fqn,
                simple_name=simple_name,
                table_name=table_name,
                fields=fields,
                relationships=relationships,
                source_code=source_code,
                superclass=superclass,
            ))
        
        return entities
    
    def _extract_fields(self, class_fqn: str, node_data: dict) -> List[FieldSpec]:
        """Extract fields from a class, including inherited ones."""
        fields = []
        seen_names = set()
        
        # Get fields from node data
        raw_fields = node_data.get("fields", [])
        for field_info in raw_fields:
            if len(field_info) >= 2:
                field_type = field_info[0]
                field_name = field_info[1]
                
                if field_name not in seen_names:
                    seen_names.add(field_name)
                    annotations = self._infer_field_annotations(field_name, field_type)
                    fields.append(FieldSpec(
                        name=field_name,
                        type=field_type,
                        annotations=annotations,
                        is_inherited=False,
                    ))
        
        # Add inherited fields from superclass hierarchy
        inherited = self._get_inherited_fields(class_fqn)
        for field in inherited:
            if field.name not in seen_names:
                seen_names.add(field.name)
                fields.append(field)
        
        return fields
    
    def _get_inherited_fields(self, class_fqn: str) -> List[FieldSpec]:
        """Get fields inherited from superclass hierarchy."""
        inherited = []
        
        if class_fqn not in self.graph.graph.nodes:
            return inherited
        
        node_data = self.graph.graph.nodes[class_fqn]
        superclass = node_data.get("superclass")
        
        if not superclass:
            return inherited
        
        # Resolve superclass FQN
        superclass_fqn = self._resolve_superclass_fqn(class_fqn, superclass)
        
        if superclass_fqn and superclass_fqn in self.graph.graph.nodes:
            super_node = self.graph.graph.nodes[superclass_fqn]
            raw_fields = super_node.get("fields", [])
            
            for field_info in raw_fields:
                if len(field_info) >= 2:
                    field_type = field_info[0]
                    field_name = field_info[1]
                    annotations = self._infer_field_annotations(field_name, field_type)
                    
                    inherited.append(FieldSpec(
                        name=field_name,
                        type=field_type,
                        annotations=annotations,
                        is_inherited=True,
                    ))
            
            # Recurse up hierarchy
            inherited.extend(self._get_inherited_fields(superclass_fqn))
        
        return inherited
    
    def _resolve_superclass_fqn(self, class_fqn: str, superclass: str) -> Optional[str]:
        """Resolve simple superclass name to FQN."""
        if "." in superclass:
            return superclass
        
        node_data = self.graph.graph.nodes.get(class_fqn, {})
        package = node_data.get("package", "")
        
        # Check same package
        candidate = f"{package}.{superclass}"
        if candidate in self.graph.graph.nodes:
            return candidate
        
        # Search all nodes
        for node_id in self.graph.graph.nodes:
            if node_id.endswith(f".{superclass}"):
                return node_id
        
        return None
    
    def _infer_field_annotations(self, field_name: str, field_type: str) -> List[str]:
        """Infer JPA annotations for a field."""
        annotations = []
        
        if field_name == "id":
            annotations.append("@Id")
            annotations.append("@GeneratedValue(strategy = GenerationType.IDENTITY)")
        
        # Column name mapping
        if field_name in ("firstName", "lastName"):
            snake = re.sub(r'(?<!^)(?=[A-Z])', '_', field_name).lower()
            annotations.append(f'@Column(name = "{snake}")')
        
        return annotations
    
    def _extract_relationships(self, class_fqn: str, source_code: str) -> List[RelationshipSpec]:
        """Extract JPA relationship annotations from source."""
        relationships = []
        
        patterns = [
            (r'@OneToMany.*?private\s+\w+<(\w+)>\s+(\w+)', 'OneToMany'),
            (r'@ManyToOne.*?private\s+(\w+)\s+(\w+)', 'ManyToOne'),
            (r'@OneToOne.*?private\s+(\w+)\s+(\w+)', 'OneToOne'),
            (r'@ManyToMany.*?private\s+\w+<(\w+)>\s+(\w+)', 'ManyToMany'),
        ]
        
        for pattern, rel_type in patterns:
            for match in re.finditer(pattern, source_code, re.DOTALL):
                target = match.group(1)
                field = match.group(2)
                relationships.append(RelationshipSpec(
                    type=rel_type,
                    target_entity=target,
                    field_name=field,
                ))
        
        return relationships
    
    def _derive_table_name(self, simple_name: str, source_code: str) -> str:
        """Derive table name from @Table annotation or class name."""
        match = re.search(r'@Table\s*\(\s*name\s*=\s*"(\w+)"', source_code)
        if match:
            return match.group(1)
        
        # Convert PascalCase to snake_case and pluralize
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', simple_name).lower()
        return f"{snake}s"
    
    def _get_external_superclass(self, class_fqn: str, cluster: ClusterInfo) -> Optional[str]:
        """Get superclass if it's outside this cluster."""
        if class_fqn not in self.graph.graph.nodes:
            return None
        
        node_data = self.graph.graph.nodes[class_fqn]
        superclass = node_data.get("superclass")
        
        if not superclass:
            return None
        
        superclass_fqn = self._resolve_superclass_fqn(class_fqn, superclass)
        
        if superclass_fqn and superclass_fqn not in cluster.classes:
            return superclass
        
        return None
    
    def _build_repository_specs(
        self, cluster: ClusterInfo, entities: List[EntitySpec]
    ) -> List[RepositorySpec]:
        """Build RepositorySpec for each entity."""
        repositories = []
        
        for entity in entities:
            repo_name = f"{entity.simple_name}Repository"
            
            # Check if repository exists in cluster
            repo_fqn = None
            for class_fqn in cluster.classes:
                if class_fqn.endswith(repo_name):
                    repo_fqn = class_fqn
                    break
            
            # Determine ID type from entity fields
            id_type = "Long"
            for field in entity.fields:
                if field.name == "id":
                    id_type = self._java_type_to_wrapper(field.type)
                    break
            
            repositories.append(RepositorySpec(
                interface_name=repo_name,
                entity_fqn=entity.fqn,
                entity_simple_name=entity.simple_name,
                id_type=id_type,
            ))
        
        return repositories
    
    def _java_type_to_wrapper(self, java_type: str) -> str:
        """Convert primitive types to wrapper types."""
        primitives = {
            "int": "Integer",
            "long": "Long",
            "boolean": "Boolean",
            "double": "Double",
            "float": "Float",
        }
        return primitives.get(java_type, java_type)
    
    def _build_service_specs(self, cluster: ClusterInfo) -> List[ServiceClassSpec]:
        """Build ServiceClassSpec for service classes."""
        services = []
        
        for class_fqn in cluster.classes:
            if class_fqn not in self.graph.graph.nodes:
                continue
            
            node_data = self.graph.graph.nodes[class_fqn]
            node_type = node_data.get("type", "")
            
            if node_type == "Service":
                simple_name = class_fqn.split(".")[-1]
                source_code = self._read_source(node_data.get("file_path", ""))
                
                services.append(ServiceClassSpec(
                    fqn=class_fqn,
                    simple_name=simple_name,
                    source_code=source_code,
                ))
        
        return services
    
    def _build_controller_specs(self, cluster: ClusterInfo) -> List[ControllerSpec]:
        """Build ControllerSpec for controllers."""
        controllers = []
        
        for class_fqn in cluster.entry_points:
            if class_fqn not in self.graph.graph.nodes:
                continue
            
            node_data = self.graph.graph.nodes[class_fqn]
            simple_name = class_fqn.split(".")[-1]
            source_code = self._read_source(node_data.get("file_path", ""))
            
            # Extract base path from @RequestMapping
            base_path = ""
            match = re.search(r'@RequestMapping\s*\(\s*"([^"]+)"', source_code)
            if match:
                base_path = match.group(1)
            
            controllers.append(ControllerSpec(
                fqn=class_fqn,
                simple_name=simple_name,
                base_path=base_path,
                source_code=source_code,
            ))
        
        return controllers
    
    def _build_gateway_client_specs(
        self, cluster: ClusterInfo, boundary_edges: List[BoundaryEdge]
    ) -> List[GatewayClientSpec]:
        """Build GatewayClientSpec for cross-service calls."""
        clients: Dict[str, List[str]] = {}
        
        for edge in boundary_edges:
            if edge.from_class in cluster.classes:
                target_service = edge.to_service
                if target_service not in clients:
                    clients[target_service] = []
                if edge.method:
                    clients[target_service].append(edge.method)
        
        return [
            GatewayClientSpec(target_service=svc, methods_needed=methods)
            for svc, methods in clients.items()
        ]
    
    def _check_shared_dependency(self, cluster: ClusterInfo) -> bool:
        """Check if this cluster depends on shared-service."""
        for class_fqn in cluster.classes:
            if class_fqn not in self.graph.graph.nodes:
                continue
            
            node_data = self.graph.graph.nodes[class_fqn]
            superclass = node_data.get("superclass")
            
            if superclass:
                superclass_fqn = self._resolve_superclass_fqn(class_fqn, superclass)
                if superclass_fqn:
                    superclass_cluster = self._class_to_cluster.get(superclass_fqn)
                    if superclass_cluster == "shared-service":
                        return True
        
        return False
    
    def _read_source(self, file_path: str) -> str:
        """Read source code from file."""
        if not file_path:
            return ""
        
        path = Path(file_path)
        if not path.is_absolute():
            path = self.monolith_root / path
        
        try:
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read source: {e}")
        
        return ""
