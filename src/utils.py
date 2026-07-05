"""
Utility functions for the safe-guideline-tooling-eval project.
"""

import hashlib
from datetime import datetime, timezone


def sha256_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def timestamp_now() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def timestamp_for_folder() -> str:
    """Get current timestamp formatted for folder names (YYYYMMDD_HHMMSS)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_doc_id(sha256: str) -> str:
    """
    Create a neutral doc_id from SHA256.
    Format: DOC-{first 12 chars of sha256}
    """
    return f"DOC-{sha256[:12]}"


def hash_file(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash_obj = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256_hash_obj.update(chunk)
    return sha256_hash_obj.hexdigest()


def get_git_commit() -> str | None:
    """Try to get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return None
