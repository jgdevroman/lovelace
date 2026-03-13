"""Cache manager for AST parse results and dependency graph using SQLite."""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from lovelace.core.graph import DependencyGraph

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of parsed files and dependency graph using SQLite."""

    def __init__(self, cache_dir: Path):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory where cache files are stored (typically .lovelace/).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table for tracking file hashes
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                last_modified REAL NOT NULL
            )
            """
        )

        # Table for cached graph
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_cache (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                graph_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                file_count INTEGER NOT NULL
            )
            """
        )

        # Table for tracking config hash (to invalidate on config changes)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS config_cache (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                config_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        conn.commit()
        conn.close()

    def _compute_hash(self, file_path: Path) -> str:
        """
        Compute MD5 hash of file content.

        Args:
            file_path: Path to the file.

        Returns:
            MD5 hash hex string.
        """
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute hash for {file_path}: {e}")
            return ""

    def get_changed_files(
        self, java_files: List[Path]
    ) -> Tuple[List[Path], List[Path]]:
        """
        Identify which files have changed since last cache.

        Args:
            java_files: List of Java file paths to check.

        Returns:
            Tuple of (changed_files, unchanged_files).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        changed_files = []
        unchanged_files = []

        for file_path in java_files:
            file_path_str = str(file_path)
            current_hash = self._compute_hash(file_path)

            if not current_hash:
                # If we can't compute hash, treat as changed
                changed_files.append(file_path)
                continue

            cursor.execute(
                "SELECT content_hash FROM file_hashes WHERE file_path = ?",
                (file_path_str,),
            )
            result = cursor.fetchone()

            if result is None:
                # New file
                changed_files.append(file_path)
            elif result[0] != current_hash:
                # File changed
                changed_files.append(file_path)
            else:
                # File unchanged
                unchanged_files.append(file_path)

        conn.close()
        return changed_files, unchanged_files

    def update_file_hashes(self, files: List[Path]) -> None:
        """
        Update file hashes in cache after successful parsing.

        Args:
            files: List of file paths that were successfully parsed.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for file_path in files:
            file_path_str = str(file_path)
            content_hash = self._compute_hash(file_path)
            last_modified = Path(file_path).stat().st_mtime if file_path.exists() else 0.0

            cursor.execute(
                """
                INSERT OR REPLACE INTO file_hashes (file_path, content_hash, last_modified)
                VALUES (?, ?, ?)
                """,
                (file_path_str, content_hash, last_modified),
            )

        conn.commit()
        conn.close()

    def remove_file_hash(self, file_path: Path) -> None:
        """
        Remove a file hash from cache (when file is deleted).

        Args:
            file_path: Path to the file that was deleted.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM file_hashes WHERE file_path = ?", (str(file_path),))

        conn.commit()
        conn.close()

    def get_cached_file_count(self) -> int:
        """
        Get the number of files currently cached.

        Returns:
            Number of cached files.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM file_hashes")
        count = cursor.fetchone()[0]

        conn.close()
        return count

    def save_graph(self, graph: DependencyGraph, file_count: int) -> None:
        """
        Save dependency graph to cache.

        Args:
            graph: DependencyGraph instance to cache.
            file_count: Number of files that were parsed to build this graph.
        """
        graph_json = json.dumps(graph.to_json(), default=str)
        created_at = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO graph_cache (id, graph_json, created_at, file_count)
            VALUES (1, ?, ?, ?)
            """,
            (graph_json, created_at, file_count),
        )

        conn.commit()
        conn.close()

        logger.debug(f"Saved graph to cache ({file_count} files)")

    def load_graph(self) -> Optional[DependencyGraph]:
        """
        Load cached dependency graph if available and valid.

        Returns:
            DependencyGraph instance if cache exists, None otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT graph_json, file_count FROM graph_cache WHERE id = 1"
        )
        result = cursor.fetchone()

        conn.close()

        if result is None:
            return None

        try:
            graph_data = json.loads(result[0])
            graph = DependencyGraph.from_json(graph_data)
            logger.debug(f"Loaded cached graph ({result[1]} files)")
            return graph
        except Exception as e:
            logger.warning(f"Failed to load cached graph: {e}")
            return None

    def update_config_hash(self, config_path: Path) -> bool:
        """
        Update config hash and check if config has changed.

        Args:
            config_path: Path to lovelace.yaml config file.

        Returns:
            True if config changed (or is new), False if unchanged.
        """
        if not config_path.exists():
            return True  # Treat missing config as changed

        config_hash = self._compute_hash(config_path)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT config_hash FROM config_cache WHERE id = 1")
        result = cursor.fetchone()

        if result is None:
            # New config
            cursor.execute(
                """
                INSERT INTO config_cache (id, config_hash, updated_at)
                VALUES (1, ?, ?)
                """,
                (config_hash, datetime.now().isoformat()),
            )
            conn.commit()
            conn.close()
            return True

        if result[0] != config_hash:
            # Config changed
            cursor.execute(
                """
                UPDATE config_cache SET config_hash = ?, updated_at = ?
                WHERE id = 1
                """,
                (config_hash, datetime.now().isoformat()),
            )
            conn.commit()
            conn.close()
            return True

        conn.close()
        return False

    def invalidate(self) -> None:
        """Clear all cached data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM file_hashes")
        cursor.execute("DELETE FROM graph_cache")
        cursor.execute("DELETE FROM config_cache")

        conn.commit()
        conn.close()

        logger.info("Cache invalidated")

    def get_cache_info(self) -> dict:
        """
        Get information about the current cache state.

        Returns:
            Dictionary with cache statistics.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Count cached files
        cursor.execute("SELECT COUNT(*) FROM file_hashes")
        file_count = cursor.fetchone()[0]

        # Get graph cache info
        cursor.execute("SELECT created_at, file_count FROM graph_cache WHERE id = 1")
        graph_result = cursor.fetchone()

        conn.close()

        info = {
            "cached_files": file_count,
            "has_graph_cache": graph_result is not None,
        }

        if graph_result:
            info["graph_created_at"] = graph_result[0]
            info["graph_file_count"] = graph_result[1]

        return info
