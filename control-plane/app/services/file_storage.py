import uuid
import logging
from pathlib import Path
from fastapi import UploadFile, HTTPException

from app.config import settings

logger = logging.getLogger(__name__)

ALLOWED_MIME_TYPES = {
    "text/plain", "text/markdown", "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".doc", ".docx"}


async def save_context_file(file: UploadFile) -> dict:
    """Save an uploaded context file to the filesystem. Returns metadata dict."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '{suffix}' not allowed")

    content = await file.read()
    size_bytes = len(content)
    max_bytes = settings.max_context_file_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {settings.max_context_file_size_mb}MB"
        )

    stored_name = f"{uuid.uuid4()}{suffix}"
    dest_path = Path(settings.context_files_path) / stored_name
    dest_path.write_bytes(content)
    logger.info("Saved context file %s (%d bytes)", stored_name, size_bytes)

    return {
        "filename": stored_name,
        "original_name": file.filename,
        "file_path": str(dest_path.resolve()),
        "mime_type": file.content_type or "application/octet-stream",
        "size_bytes": size_bytes,
    }


def delete_context_file(file_path: str) -> None:
    """Delete a file from the filesystem."""
    path = Path(file_path)
    if path.exists():
        path.unlink()
        logger.info("Deleted context file %s", file_path)
    else:
        logger.warning("Context file not found for deletion: %s", file_path)


def read_context_file(file_path: str) -> str:
    """Read text content from a context file."""
    path = Path(file_path)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error("Failed to read context file %s: %s", file_path, e)
        return ""
