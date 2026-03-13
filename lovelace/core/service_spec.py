"""Service specification data classes.

These classes define the structure of a microservice to be generated.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class FieldSpec:
    """Specification for a Java field."""
    name: str
    type: str
    annotations: List[str] = field(default_factory=list)
    is_inherited: bool = False
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "annotations": self.annotations,
            "is_inherited": self.is_inherited,
        }


@dataclass
class RelationshipSpec:
    """Specification for an entity relationship."""
    type: str  # OneToMany, ManyToOne, ManyToMany, OneToOne
    target_entity: str  # Simple name or FQN
    field_name: str
    mapped_by: Optional[str] = None
    cascade: Optional[str] = None
    fetch: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "target_entity": self.target_entity,
            "field_name": self.field_name,
            "mapped_by": self.mapped_by,
            "cascade": self.cascade,
            "fetch": self.fetch,
        }


@dataclass
class MethodSpec:
    """Specification for a method."""
    name: str
    return_type: str
    parameters: List[str] = field(default_factory=list)  # ["String id", "int page"]
    annotations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "return_type": self.return_type,
            "parameters": self.parameters,
            "annotations": self.annotations,
        }


@dataclass
class EntitySpec:
    """Specification for a JPA entity."""
    fqn: str
    simple_name: str
    table_name: str
    fields: List[FieldSpec] = field(default_factory=list)
    relationships: List[RelationshipSpec] = field(default_factory=list)
    methods: List[MethodSpec] = field(default_factory=list)
    source_code: str = ""  # Original source for reference
    superclass: Optional[str] = None  # If extends something outside cluster
    
    def to_dict(self) -> dict:
        return {
            "fqn": self.fqn,
            "simple_name": self.simple_name,
            "table_name": self.table_name,
            "fields": [f.to_dict() for f in self.fields],
            "relationships": [r.to_dict() for r in self.relationships],
            "methods": [m.to_dict() for m in self.methods],
            "superclass": self.superclass,
            "source_code": self.source_code,
        }


@dataclass
class RepositorySpec:
    """Specification for a Spring Data repository."""
    interface_name: str
    entity_fqn: str
    entity_simple_name: str
    id_type: str = "Long"
    custom_methods: List[MethodSpec] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "interface_name": self.interface_name,
            "entity_fqn": self.entity_fqn,
            "entity_simple_name": self.entity_simple_name,
            "id_type": self.id_type,
            "custom_methods": [m.to_dict() for m in self.custom_methods],
        }


@dataclass
class ServiceClassSpec:
    """Specification for a service class."""
    fqn: str
    simple_name: str
    dependencies: List[str] = field(default_factory=list)  # Injected dependencies
    methods: List[MethodSpec] = field(default_factory=list)
    source_code: str = ""  # Original source
    
    def to_dict(self) -> dict:
        return {
            "fqn": self.fqn,
            "simple_name": self.simple_name,
            "dependencies": self.dependencies,
            "methods": [m.to_dict() for m in self.methods],
            "source_code": self.source_code,
        }


@dataclass
class ControllerSpec:
    """Specification for a REST controller."""
    fqn: str
    simple_name: str
    base_path: str = ""  # e.g., "/api/users"
    dependencies: List[str] = field(default_factory=list)
    endpoints: List[MethodSpec] = field(default_factory=list)
    source_code: str = ""
    
    def to_dict(self) -> dict:
        return {
            "fqn": self.fqn,
            "simple_name": self.simple_name,
            "base_path": self.base_path,
            "dependencies": self.dependencies,
            "endpoints": [e.to_dict() for e in self.endpoints],
            "source_code": self.source_code,
        }


@dataclass
class GatewayClientSpec:
    """Specification for a gateway HTTP client."""
    target_service: str
    methods_needed: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "target_service": self.target_service,
            "methods_needed": self.methods_needed,
        }


@dataclass
class ServiceSpec:
    """Complete specification for a microservice."""
    name: str
    base_package: str
    spring_boot_version: str = "3.2.0"
    java_version: int = 17
    
    # Components
    entities: List[EntitySpec] = field(default_factory=list)
    repositories: List[RepositorySpec] = field(default_factory=list)
    services: List[ServiceClassSpec] = field(default_factory=list)
    controllers: List[ControllerSpec] = field(default_factory=list)
    gateway_clients: List[GatewayClientSpec] = field(default_factory=list)
    
    # Dependencies
    maven_dependencies: List[str] = field(default_factory=list)
    
    # Cross-service info
    depends_on_shared: bool = False
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "base_package": self.base_package,
            "spring_boot_version": self.spring_boot_version,
            "java_version": self.java_version,
            "entities": [e.to_dict() for e in self.entities],
            "repositories": [r.to_dict() for r in self.repositories],
            "services": [s.to_dict() for s in self.services],
            "controllers": [c.to_dict() for c in self.controllers],
            "gateway_clients": [g.to_dict() for g in self.gateway_clients],
            "maven_dependencies": self.maven_dependencies,
            "depends_on_shared": self.depends_on_shared,
        }
    
    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ServiceResult:
    """Result from service generation."""
    success: bool
    service_path: Optional[str] = None
    message: str = ""
    compile_errors: List[str] = field(default_factory=list)
    cost_usd: float = 0.0
    iterations: int = 0
    # Validation state
    validation_state: str = "pending"  # pending, compiled, docker_built, healthy, approved
    docker_image: Optional[str] = None
    docker_build_output: str = ""
