"""Semantic vector store for class embeddings using ChromaDB."""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from lovelace.core.parser import ClassMetadata

logger = logging.getLogger(__name__)


class VectorEngine:
    """Semantic vector store for class embeddings."""

    def __init__(self, persist_directory: Optional[Path] = None):
        """
        Initialize the vector engine.

        Args:
            persist_directory: Path to .lovelace/ cache directory.
                              If None, uses in-memory storage.
        """
        if chromadb is None:
            raise ImportError(
                "chromadb is required for VectorEngine. Install it with: pip install chromadb"
            )

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for VectorEngine. Install it with: pip install sentence-transformers"
            )

        # Initialize embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2

        # Initialize ChromaDB
        if persist_directory:
            persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(persist_directory / "chromadb"), settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(settings=Settings(anonymized_telemetry=False))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="class_embeddings", metadata={"hnsw:space": "cosine"}
        )

        # Cache for similarity lookups
        self._similarity_cache: Dict[Tuple[str, str], float] = {}

    def _create_text_representation(self, class_metadata: ClassMetadata) -> str:
        """
        Create a text representation of a class for embedding.

        Args:
            class_metadata: ClassMetadata object.

        Returns:
            Text string representing the class.
        """
        parts = [f"CLASS: {class_metadata.fully_qualified_name}"]

        # Add type information
        if class_metadata.is_interface:
            parts.append("TYPE: Interface")
        elif class_metadata.is_abstract:
            parts.append("TYPE: Abstract")
        else:
            parts.append("TYPE: Class")

        # Add annotations
        if class_metadata.annotations:
            annotations_str = ", ".join([f"@{ann}" for ann in class_metadata.annotations])
            parts.append(f"ANNOTATIONS: {annotations_str}")

        # Add method signatures (not bodies - saves tokens)
        if class_metadata.methods:
            parts.append("METHODS:")
            for method in class_metadata.methods:
                params = ", ".join([f"{ptype} {pname}" for ptype, pname in method.parameters])
                return_type = method.return_type or "void"
                method_sig = f"  - {method.name}({params}) -> {return_type}"
                parts.append(method_sig)

        # Add fields
        if class_metadata.fields:
            parts.append("FIELDS:")
            for field_type, field_name in class_metadata.fields:
                parts.append(f"  - {field_type} {field_name}")

        return "\n".join(parts)

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file content for caching."""
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def embed_class(self, class_metadata: ClassMetadata) -> List[float]:
        """
        Generate embedding for a single class.

        Args:
            class_metadata: ClassMetadata object.

        Returns:
            Vector embedding (list of floats).
        """
        class_id = class_metadata.fully_qualified_name

        # Check if already embedded
        existing = self.collection.get(ids=[class_id])
        if existing["ids"]:
            # Return existing embedding
            return existing["embeddings"][0]

        # Create text representation
        text = self._create_text_representation(class_metadata)

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True).tolist()

        # Store in ChromaDB
        file_hash = self._get_file_hash(class_metadata.file_path)
        self.collection.add(
            ids=[class_id],
            embeddings=[embedding],
            metadatas=[{"file_path": str(class_metadata.file_path), "file_hash": file_hash}],
        )

        return embedding

    def add_classes(self, classes: List[ClassMetadata]) -> None:
        """
        Batch add classes to the vector store.

        Args:
            classes: List of ClassMetadata objects.
        """
        logger.info(f"Embedding {len(classes)} classes...")

        # Filter out classes that are already embedded (by checking file hash)
        to_embed = []
        existing_ids = set(self.collection.get()["ids"])

        for class_meta in classes:
            class_id = class_meta.fully_qualified_name
            if class_id not in existing_ids:
                to_embed.append(class_meta)
            else:
                # Check if file has changed
                existing_meta = self.collection.get(ids=[class_id])["metadatas"][0]
                current_hash = self._get_file_hash(class_meta.file_path)
                if existing_meta.get("file_hash") != current_hash:
                    # File changed, re-embed
                    to_embed.append(class_meta)
                    # Remove old embedding
                    self.collection.delete(ids=[class_id])

        if not to_embed:
            logger.info("All classes already embedded and up-to-date")
            return

        # Batch embed
        texts = [self._create_text_representation(cm) for cm in to_embed]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        # Prepare metadata
        ids = [cm.fully_qualified_name for cm in to_embed]
        metadatas = [
            {
                "file_path": str(cm.file_path),
                "file_hash": self._get_file_hash(cm.file_path),
            }
            for cm in to_embed
        ]

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        logger.info(f"Successfully embedded {len(to_embed)} classes")

    def get_similarity(self, class_a: str, class_b: str) -> float:
        """
        Calculate cosine similarity between two classes.

        Args:
            class_a: Fully qualified class name.
            class_b: Fully qualified class name.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        # Check cache
        cache_key = tuple(sorted([class_a, class_b]))
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        # Get embeddings
        try:
            results = self.collection.get(ids=[class_a, class_b], include=["embeddings"])
            if not results or "ids" not in results or len(results["ids"]) != 2:
                logger.debug(f"Could not find embeddings for {class_a} and/or {class_b}")
                return 0.0

            embeddings = results.get("embeddings")
            if embeddings is None or len(embeddings) != 2:
                logger.debug(f"Could not get embeddings for {class_a} and/or {class_b}")
                return 0.0

            emb_a = embeddings[0]
            emb_b = embeddings[1]

            if emb_a is None or emb_b is None:
                logger.debug(f"One or both embeddings are None for {class_a} and {class_b}")
                return 0.0
            
            # Convert to numpy arrays if needed
            import numpy as np
            if not isinstance(emb_a, np.ndarray):
                emb_a = np.array(emb_a)
            if not isinstance(emb_b, np.ndarray):
                emb_b = np.array(emb_b)

            # Calculate cosine similarity
            similarity = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))

            # Cache result
            self._similarity_cache[cache_key] = similarity
            return similarity

        except Exception as e:
            logger.warning(f"Error calculating similarity between {class_a} and {class_b}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return 0.0

    def find_similar(self, class_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find most semantically similar classes.

        Args:
            class_id: Fully qualified class name.
            top_k: Number of similar classes to return.

        Returns:
            List of (class_id, similarity_score) tuples, sorted by similarity descending.
        """
        try:
            # Get embedding for the query class
            results = self.collection.get(ids=[class_id])
            if not results["ids"]:
                return []

            query_embedding = results["embeddings"][0]

            # Query ChromaDB for similar classes
            similar = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k + 1,  # +1 because it will include itself
            )

            # Filter out the query class itself and format results
            results_list = []
            for i, similar_id in enumerate(similar["ids"][0]):
                if similar_id != class_id:
                    similarity = 1.0 - similar["distances"][0][i]  # Convert distance to similarity
                    results_list.append((similar_id, similarity))

            return results_list

        except Exception as e:
            logger.warning(f"Error finding similar classes for {class_id}: {e}")
            return []

    def get_all_similarities(self) -> Dict[Tuple[str, str], float]:
        """
        Get pairwise similarity matrix for all classes.

        Returns:
            Dictionary mapping (class_a, class_b) -> similarity score.
        """
        all_ids = self.collection.get()["ids"]
        similarities = {}

        for i, class_a in enumerate(all_ids):
            for class_b in all_ids[i + 1 :]:
                similarity = self.get_similarity(class_a, class_b)
                similarities[(class_a, class_b)] = similarity

        return similarities

