"""
Document chunker with configurable size and overlap.

Splits documents into smaller chunks for embedding and retrieval.
"""

from dataclasses import dataclass
from typing import Optional
import hashlib
import re

from .loader import Document


@dataclass
class Chunk:
    """Represents a chunk of a document."""
    
    chunk_id: str
    document_id: str
    content: str
    source: str
    section: Optional[str]
    char_start: int
    char_end: int
    chunk_index: int
    
    def to_dict(self) -> dict:
        """Convert chunk to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "source": self.source,
            "section": self.section or "",
            "char_start": self.char_start,
            "char_end": self.char_end,
            "chunk_index": self.chunk_index,
        }


def _generate_chunk_id(document_id: str, content: str, index: int) -> str:
    """Generate a unique chunk ID."""
    hash_input = f"{document_id}:{index}:{content[:100]}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:12]


def _find_current_section(content: str, position: int) -> Optional[str]:
    """Find the section heading that contains the given position."""
    # Look for markdown headings before this position
    heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    
    current_section = None
    for match in heading_pattern.finditer(content):
        if match.start() <= position:
            current_section = match.group(2).strip()
        else:
            break
    
    return current_section


def _find_split_point(text: str, target: int) -> int:
    """
    Find the best split point near the target position.
    
    Prefers splitting at paragraph breaks, then sentences, then words.
    """
    if target >= len(text):
        return len(text)
    
    # Look for paragraph break (double newline) within 100 chars
    search_start = max(0, target - 100)
    search_end = min(len(text), target + 100)
    search_region = text[search_start:search_end]
    
    # Try paragraph break
    para_break = search_region.rfind("\n\n")
    if para_break != -1:
        return search_start + para_break + 2
    
    # Try sentence break
    for punct in [". ", "! ", "? ", ".\n"]:
        sent_break = search_region.rfind(punct)
        if sent_break != -1:
            return search_start + sent_break + len(punct)
    
    # Try word break
    space_break = search_region.rfind(" ")
    if space_break != -1:
        return search_start + space_break + 1
    
    # Fall back to exact position
    return target


def chunk_document(
    document: Document,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """
    Split a document into overlapping chunks.
    
    Args:
        document: The document to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of Chunk objects
    """
    content = document.content
    chunks = []
    
    if len(content) <= chunk_size:
        # Document fits in single chunk
        chunk_id = _generate_chunk_id(document.document_id, content, 0)
        section = _find_current_section(content, 0)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            content=content,
            source=document.source,
            section=section,
            char_start=0,
            char_end=len(content),
            chunk_index=0,
        ))
        return chunks
    
    # Split into overlapping chunks
    position = 0
    chunk_index = 0
    
    while position < len(content):
        # Calculate end position
        end_target = position + chunk_size
        
        if end_target >= len(content):
            # Last chunk
            chunk_content = content[position:]
            end_position = len(content)
        else:
            # Find a good split point
            end_position = _find_split_point(content, end_target)
            chunk_content = content[position:end_position]
        
        # Generate chunk
        chunk_id = _generate_chunk_id(document.document_id, chunk_content, chunk_index)
        section = _find_current_section(content, position)
        
        chunks.append(Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            content=chunk_content.strip(),
            source=document.source,
            section=section,
            char_start=position,
            char_end=end_position,
            chunk_index=chunk_index,
        ))
        
        # Move position with overlap
        position = end_position - chunk_overlap
        chunk_index += 1
        
        # Prevent infinite loop
        if position >= len(content) - chunk_overlap:
            break
    
    return chunks
