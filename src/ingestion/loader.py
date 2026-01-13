"""
Document loader for various file formats.

Loads documents from files and extracts content with metadata.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import hashlib


@dataclass
class Document:
    """Represents a loaded document with metadata."""
    
    document_id: str
    content: str
    source: str
    file_type: str
    title: Optional[str] = None
    
    @classmethod
    def from_file(cls, path: Path) -> "Document":
        """Load a document from a file path."""
        content = path.read_text(encoding="utf-8")
        file_type = path.suffix.lower().lstrip(".")
        
        # Generate document ID from content hash
        doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Extract title from first heading or filename
        title = cls._extract_title(content, file_type) or path.stem
        
        return cls(
            document_id=doc_id,
            content=content,
            source=path.name,
            file_type=file_type,
            title=title,
        )
    
    @staticmethod
    def _extract_title(content: str, file_type: str) -> Optional[str]:
        """Extract title from document content."""
        if file_type == "md":
            # Look for first H1 heading
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("# "):
                    return line[2:].strip()
        return None


def load_document(path: Path) -> Document:
    """
    Load a document from a file path.
    
    Supports: .md, .txt, .markdown
    
    Args:
        path: Path to the document file
        
    Returns:
        Document object with content and metadata
        
    Raises:
        ValueError: If file type is not supported
    """
    supported_extensions = {".md", ".txt", ".markdown"}
    
    if path.suffix.lower() not in supported_extensions:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported: {supported_extensions}"
        )
    
    return Document.from_file(path)


def load_corpus(corpus_dir: Path) -> list[Document]:
    """
    Load all documents from a corpus directory.
    
    Args:
        corpus_dir: Path to directory containing documents
        
    Returns:
        List of Document objects
    """
    documents = []
    supported_extensions = {".md", ".txt", ".markdown"}
    
    for path in corpus_dir.iterdir():
        if path.is_file() and path.suffix.lower() in supported_extensions:
            try:
                doc = load_document(path)
                documents.append(doc)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
    
    return documents
