"""Java code parser for static analysis."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

import javalang


@dataclass
class MethodMetadata:
    """Metadata for a Java method."""

    name: str
    return_type: Optional[str]
    parameters: List[tuple[str, str]]  # (type, name)
    annotations: List[str] = field(default_factory=list)
    is_public: bool = False
    is_static: bool = False


@dataclass
class DependencyInfo:
    """Information about a dependency relationship."""

    target_class: str
    dependency_type: str  # CALLS, INHERITS, IMPORTS
    method_name: Optional[str] = None  # For CALLS type


@dataclass
class JPARelationship:
    """Information about a JPA relationship between entities."""

    target_class: str  # The related entity class
    relationship_type: str  # ManyToOne, OneToMany, OneToOne, ManyToMany
    field_name: str  # The field name in the source class


@dataclass
class ClassMetadata:
    """Metadata for a Java class or interface."""

    fully_qualified_name: str
    simple_name: str
    package_name: str
    file_path: Path
    is_interface: bool = False
    is_abstract: bool = False
    superclass: Optional[str] = None
    interfaces: List[str] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    methods: List[MethodMetadata] = field(default_factory=list)
    fields: List[tuple[str, str]] = field(default_factory=list)  # (type, name)
    dependencies: List[DependencyInfo] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)
    jpa_relationships: List[JPARelationship] = field(default_factory=list)  # JPA entity relationships


class JavaParser:
    """Parser for Java source files."""

    def __init__(self, ignore_paths: Optional[List[str]] = None):
        """
        Initialize the Java parser.

        Args:
            ignore_paths: List of glob patterns for paths to ignore.
        """
        self.ignore_paths = ignore_paths or []

    def _should_ignore(self, file_path: Path) -> bool:
        """Check if a file path should be ignored."""
        path_str = str(file_path)
        for pattern in self.ignore_paths:
            # Simple glob matching - convert ** to match any directory
            pattern_regex = pattern.replace("**/", "").replace("**", "")
            if pattern_regex in path_str or path_str.endswith(pattern_regex):
                return True
        return False

    def scan_directory(self, root_dir: Path) -> List[Path]:
        """
        Recursively scan directory for Java files.

        Args:
            root_dir: Root directory to scan.

        Returns:
            List of Path objects for .java files.
        """
        java_files = []
        for path in root_dir.rglob("*.java"):
            if not self._should_ignore(path):
                java_files.append(path)
        return java_files

    def _extract_package(self, tree: javalang.tree.CompilationUnit) -> str:
        """Extract package name from AST."""
        if tree.package:
            if isinstance(tree.package.name, str):
                return tree.package.name
            return ".".join(tree.package.name)
        return ""

    def _extract_imports(self, tree: javalang.tree.CompilationUnit) -> Set[str]:
        """Extract import statements from AST."""
        imports = set()
        if tree.imports:
            for imp in tree.imports:
                if isinstance(imp, javalang.tree.Import):
                    if isinstance(imp.path, str):
                        imports.add(imp.path)
                    else:
                        imports.add(".".join(imp.path))
        return imports

    def _extract_annotations(self, node) -> List[str]:
        """Extract annotation names from a node."""
        annotations = []
        if hasattr(node, "annotations") and node.annotations:
            for ann in node.annotations:
                if isinstance(ann, javalang.tree.Annotation):
                    ann_name = ".".join(ann.name) if isinstance(ann.name, list) else ann.name
                    annotations.append(ann_name)
        return annotations

    def _extract_method_signature(self, method: javalang.tree.MethodDeclaration) -> MethodMetadata:
        """Extract method signature information."""
        return_type = None
        if method.return_type:
            if isinstance(method.return_type, javalang.tree.ReferenceType):
                if isinstance(method.return_type.name, str):
                    return_type = method.return_type.name
                else:
                    return_type = ".".join(method.return_type.name)
            elif isinstance(method.return_type, str):
                return_type = method.return_type

        parameters = []
        if method.parameters:
            for param in method.parameters:
                if hasattr(param.type, "name"):
                    if isinstance(param.type.name, str):
                        param_type = param.type.name
                    else:
                        param_type = ".".join(param.type.name)
                else:
                    param_type = str(param.type)
                param_name = param.name
                parameters.append((param_type, param_name))

        annotations = self._extract_annotations(method)

        modifiers = set(method.modifiers) if hasattr(method, "modifiers") else set()

        return MethodMetadata(
            name=method.name,
            return_type=return_type,
            parameters=parameters,
            annotations=annotations,
            is_public="public" in modifiers,
            is_static="static" in modifiers,
        )

    def _extract_fields(self, class_node) -> List[tuple[str, str]]:
        """Extract field declarations from a class."""
        fields = []
        if hasattr(class_node, "fields"):
            for field_decl in class_node.fields:
                if hasattr(field_decl.type, "name"):
                    if isinstance(field_decl.type.name, str):
                        field_type = field_decl.type.name
                    else:
                        field_type = ".".join(field_decl.type.name)
                else:
                    field_type = str(field_decl.type)
                for declarator in field_decl.declarators:
                    fields.append((field_type, declarator.name))
        return fields

    def _extract_jpa_relationships(self, class_node, package_name: str) -> List[JPARelationship]:
        """
        Extract JPA relationship annotations from fields.

        Args:
            class_node: AST node for the class.
            package_name: Package name for resolving relative class names.

        Returns:
            List of JPARelationship objects.
        """
        relationships = []
        if not hasattr(class_node, "fields"):
            return relationships

        # JPA relationship annotations
        jpa_annotations = {
            "ManyToOne",
            "OneToMany",
            "OneToOne",
            "ManyToMany",
            "JoinColumn",
            "JoinTable",
        }

        for field_decl in class_node.fields:
            # Extract field annotations
            field_annotations = self._extract_annotations(field_decl)
            field_annotations_lower = [ann.lower() for ann in field_annotations]

            # Check for JPA relationship annotations
            relationship_type = None
            for ann in field_annotations:
                ann_lower = ann.lower()
                if "manytoone" in ann_lower:
                    relationship_type = "ManyToOne"
                elif "onetomany" in ann_lower:
                    relationship_type = "OneToMany"
                elif "onetoone" in ann_lower:
                    relationship_type = "OneToOne"
                elif "manytomany" in ann_lower:
                    relationship_type = "ManyToMany"

            if relationship_type:
                # Extract field type (target entity)
                field_type = None
                if hasattr(field_decl, "type") and hasattr(field_decl.type, "name"):
                    if isinstance(field_decl.type.name, str):
                        field_type = field_decl.type.name
                    elif isinstance(field_decl.type.name, list):
                        field_type = ".".join(field_decl.type.name)
                    else:
                        field_type = str(field_decl.type.name)

                # Handle generic types (e.g., List<User>)
                if field_type and "<" in field_type:
                    # Extract the generic type parameter
                    generic_start = field_type.find("<")
                    generic_end = field_type.rfind(">")
                    if generic_start < generic_end:
                        generic_type = field_type[generic_start + 1 : generic_end].strip()
                        # Remove package if it's a simple name
                        if "." not in generic_type and package_name:
                            field_type = f"{package_name}.{generic_type}"
                        else:
                            field_type = generic_type

                # Resolve relative class names
                if field_type and "." not in field_type and package_name:
                    field_type = f"{package_name}.{field_type}"

                if field_type:
                    # Extract field name
                    field_name = None
                    if hasattr(field_decl, "declarators") and field_decl.declarators:
                        field_name = field_decl.declarators[0].name

                    if field_name:
                        relationships.append(
                            JPARelationship(
                                target_class=field_type,
                                relationship_type=relationship_type,
                                field_name=field_name,
                            )
                        )

        return relationships

    def _extract_dependencies_from_methods(
        self, methods: List[MethodMetadata], imports: Set[str]
    ) -> List[DependencyInfo]:
        """Extract dependency information from method calls and imports."""
        dependencies = []
        # For now, we'll extract dependencies from imports
        # Method call analysis would require more sophisticated AST traversal
        for imp in imports:
            # Filter out java.* and javax.* standard library imports
            if not (imp.startswith("java.") or imp.startswith("javax.")):
                dependencies.append(DependencyInfo(target_class=imp, dependency_type="IMPORTS"))
        return dependencies

    def parse_java_file(self, file_path: Path) -> Optional[ClassMetadata]:
        """
        Parse a single Java file and extract metadata.

        Args:
            file_path: Path to the Java file.

        Returns:
            ClassMetadata object or None if parsing fails.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = javalang.parse.parse(content)

            package_name = self._extract_package(tree)
            imports = self._extract_imports(tree)

            # Extract class/interface declarations
            if not tree.types:
                return None

            # For simplicity, handle the first top-level type
            # Real monoliths might have multiple classes per file, but that's less common
            type_decl = tree.types[0]

            if not isinstance(type_decl, (javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration)):
                return None

            is_interface = isinstance(type_decl, javalang.tree.InterfaceDeclaration)
            simple_name = type_decl.name

            # Build fully qualified name
            if package_name:
                fully_qualified_name = f"{package_name}.{simple_name}"
            else:
                fully_qualified_name = simple_name

            # Extract annotations
            annotations = self._extract_annotations(type_decl)

            # Extract superclass and interfaces
            superclass = None
            if hasattr(type_decl, "extends") and type_decl.extends:
                if isinstance(type_decl.extends, list) and len(type_decl.extends) > 0:
                    if isinstance(type_decl.extends[0].name, str):
                        superclass = type_decl.extends[0].name
                    else:
                        superclass = ".".join(type_decl.extends[0].name)
                elif hasattr(type_decl.extends, "name"):
                    if isinstance(type_decl.extends.name, str):
                        superclass = type_decl.extends.name
                    else:
                        superclass = ".".join(type_decl.extends.name)

            interfaces = []
            if hasattr(type_decl, "implements") and type_decl.implements:
                for impl in type_decl.implements:
                    if isinstance(impl.name, str):
                        interfaces.append(impl.name)
                    else:
                        interfaces.append(".".join(impl.name))

            # Extract methods
            methods = []
            if hasattr(type_decl, "methods") and type_decl.methods:
                for method in type_decl.methods:
                    methods.append(self._extract_method_signature(method))

            # Extract fields
            fields = self._extract_fields(type_decl)

            # Extract JPA relationships
            jpa_relationships = self._extract_jpa_relationships(type_decl, package_name)

            # Extract dependencies
            dependencies = self._extract_dependencies_from_methods(methods, imports)

            # Check for abstract modifier
            modifiers = set(type_decl.modifiers) if hasattr(type_decl, "modifiers") else set()
            is_abstract = "abstract" in modifiers

            # Add inheritance dependencies
            if superclass:
                dependencies.append(DependencyInfo(target_class=superclass, dependency_type="INHERITS"))

            for interface in interfaces:
                dependencies.append(DependencyInfo(target_class=interface, dependency_type="INHERITS"))

            return ClassMetadata(
                fully_qualified_name=fully_qualified_name,
                simple_name=simple_name,
                package_name=package_name,
                file_path=file_path,
                is_interface=is_interface,
                is_abstract=is_abstract,
                superclass=superclass,
                interfaces=interfaces,
                annotations=annotations,
                methods=methods,
                fields=fields,
                dependencies=dependencies,
                imports=imports,
                jpa_relationships=jpa_relationships,
            )

        except javalang.parser.JavaSyntaxError as e:
            # Skip files with syntax errors
            return None
        except Exception as e:
            # Log error but continue processing
            return None

