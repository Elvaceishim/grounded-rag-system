"""
Tests for document chunking.
"""

import pytest
from src.ingestion.loader import Document
from src.ingestion.chunker import chunk_document, Chunk


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    content = """# Introduction

This is the introduction section with some content.

## Section One

This is section one with detailed content that explains things.

## Section Two

This is section two with more detailed content.
"""
    return Document(
        document_id="test123",
        content=content,
        source="test.md",
        file_type="md",
        title="Test Document",
    )


class TestChunking:
    """Tests for document chunking."""
    
    def test_single_chunk_small_doc(self, sample_document):
        """Small document fits in single chunk."""
        chunks = chunk_document(sample_document, chunk_size=2000, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0].document_id == "test123"
    
    def test_multiple_chunks(self, sample_document):
        """Large document creates multiple chunks."""
        chunks = chunk_document(sample_document, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
    
    def test_chunk_overlap(self, sample_document):
        """Chunks have overlapping content."""
        chunks = chunk_document(sample_document, chunk_size=100, chunk_overlap=30)
        
        if len(chunks) >= 2:
            # Check that chunks overlap
            chunk1_end = chunks[0].content[-30:]
            chunk2_start = chunks[1].content[:30]
            # There should be some common content
            assert len(set(chunk1_end.split()) & set(chunk2_start.split())) > 0
    
    def test_chunk_has_metadata(self, sample_document):
        """Chunks include proper metadata."""
        chunks = chunk_document(sample_document, chunk_size=500, chunk_overlap=50)
        
        for chunk in chunks:
            assert chunk.chunk_id is not None
            assert chunk.document_id == "test123"
            assert chunk.source == "test.md"
            assert chunk.chunk_index >= 0
    
    def test_chunk_positions(self, sample_document):
        """Chunks have valid char positions."""
        chunks = chunk_document(sample_document, chunk_size=100, chunk_overlap=20)
        
        for chunk in chunks:
            assert chunk.char_start >= 0
            assert chunk.char_end > chunk.char_start
            assert chunk.char_end <= len(sample_document.content)
    
    def test_section_detection(self, sample_document):
        """Chunks detect sections from markdown headings."""
        chunks = chunk_document(sample_document, chunk_size=100, chunk_overlap=20)
        
        # At least some chunks should have sections
        sections = [c.section for c in chunks if c.section]
        assert len(sections) > 0


class TestChunkToDictionary:
    """Tests for chunk serialization."""
    
    def test_chunk_to_dict(self):
        """Chunk can be converted to dictionary."""
        chunk = Chunk(
            chunk_id="abc123",
            document_id="doc456",
            content="Test content",
            source="test.md",
            section="Introduction",
            char_start=0,
            char_end=12,
            chunk_index=0,
        )
        
        d = chunk.to_dict()
        
        assert d["chunk_id"] == "abc123"
        assert d["document_id"] == "doc456"
        assert d["content"] == "Test content"
        assert d["source"] == "test.md"
        assert d["section"] == "Introduction"
