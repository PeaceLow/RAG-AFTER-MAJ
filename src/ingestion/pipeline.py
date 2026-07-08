import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from tqdm import tqdm

from src.models import Chunk
from src.ingestion.chunker import chunk_python, chunk_markdown

HASH_FILE = "file_hashes.json"


def _hash_file(path: str) -> str:
    """Compute MD5 hash of a file's content."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _load_hashes(index_dir: str) -> Dict[str, str]:
    """Load previously saved file hashes."""
    path = os.path.join(index_dir, HASH_FILE)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]
    return {}


def save_hashes(
    index_dir: str, hashes: Dict[str, str]
) -> None:
    """Save file hashes to disk."""
    os.makedirs(index_dir, exist_ok=True)
    path = os.path.join(index_dir, HASH_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hashes, f)


def _chunk_file(
    file_path: str, max_chunk_size: int
) -> List[Chunk]:
    """Chunk a single file based on its extension."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (UnicodeDecodeError, Exception):
        return []

    if not content.strip():
        return []

    ext = Path(file_path).suffix.lower()
    if ext == ".py":
        return chunk_python(
            file_path, content, max_chunk_size=max_chunk_size
        )
    elif ext == ".md":
        return chunk_markdown(
            file_path, content, max_chunk_size=max_chunk_size
        )
    return []


def build_pipeline(
    repo_path: str,
    max_chunk_size: int = 2000,
    index_dir: Optional[str] = None,
    old_chunks: Optional[List[Chunk]] = None,
) -> Tuple[List[Chunk], Dict[str, str]]:
    """
    Build the ingestion pipeline with incremental support.

    If index_dir is provided and contains previous hashes,
    only new/modified files are re-chunked. Deleted files'
    chunks are removed.

    Args:
        repo_path: Path to the repository to ingest.
        max_chunk_size: Maximum chunk size in characters.
        index_dir: Path to the index directory (for
                   loading previous hashes).
        old_chunks: Previously indexed chunks (for
                    incremental mode).

    Returns:
        Tuple of (all_chunks, new_file_hashes).
    """
    # 1. Discover all eligible files
    file_paths = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            fp = os.path.join(root, file)
            if ".git" in fp or "__pycache__" in fp:
                continue
            ext = Path(fp).suffix.lower()
            if ext in (".py", ".md"):
                file_paths.append(fp)

    print(
        f"\033[96m\033[1mTrouvé\033[0m {len(file_paths)}"
        f" fichiers dans {repo_path}."
    )

    # 2. Compute current hashes
    current_hashes: Dict[str, str] = {}
    for fp in file_paths:
        try:
            current_hashes[fp] = _hash_file(fp)
        except Exception:
            pass

    # 3. Load old hashes (incremental mode)
    old_hashes: Dict[str, str] = {}
    if index_dir:
        old_hashes = _load_hashes(index_dir)

    incremental = bool(old_hashes and old_chunks)

    if incremental:
        # Determine changed / new / deleted files
        changed: List[str] = []
        unchanged: List[str] = []
        for fp, h in current_hashes.items():
            if fp not in old_hashes or old_hashes[fp] != h:
                changed.append(fp)
            else:
                unchanged.append(fp)

        deleted = set(old_hashes.keys()) - set(
            current_hashes.keys()
        )

        print(
            f"\033[93m\033[1mIncrémental\033[0m : "
            f"{len(changed)} modifiés, "
            f"{len(unchanged)} inchangés, "
            f"{len(deleted)} supprimés."
        )

        # Keep chunks from unchanged files
        all_chunks = [
            c for c in old_chunks  # type: ignore[union-attr]
            if c.file_path not in deleted
            and c.file_path in unchanged
        ]

        # Re-chunk only changed/new files
        for fp in tqdm(changed, desc="Re-chunking"):
            all_chunks.extend(
                _chunk_file(fp, max_chunk_size)
            )
    else:
        # Full indexing
        print("Début de l'ingestion complète...")
        all_chunks: List[Chunk] = []  # type: ignore[no-redef]
        for fp in tqdm(
            file_paths, desc="Traitement des fichiers"
        ):
            all_chunks.extend(
                _chunk_file(fp, max_chunk_size)
            )

    print(
        f"\033[92m\033[1mSuccès\033[0m : "
        f"{len(all_chunks)} chunks générés."
    )
    return all_chunks, current_hashes
