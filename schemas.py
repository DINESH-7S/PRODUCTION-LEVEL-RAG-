from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TextSegment(BaseModel):
    """Represents a chunked fragment of a document with source mapping."""
    content: str = Field(..., description="The textual content of the chunk")
    doc_id: str = Field(..., description="Identifier of the source document")
    page_num: int = Field(..., description="The page number in the original PDF")
    seq_index: int = Field(..., description="Sequence number of the chunk in this document")

class ContextPassage(BaseModel):
    """Holds metadata and score for a passage retrieved from the database."""
    content: str
    doc_id: str
    page_num: int
    seq_index: int
    relevance_score: float = 0.0

class RAGSearchResult(BaseModel):
    """Results returned from the vector store search."""
    contexts: List[str]
    sources: List[str]
    passages: List[ContextPassage] = []
    strategy: str

class QueryRequest(BaseModel):
    """Request payload for running the QA pipeline."""
    question: str
    strategy: str = "dense"
    top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    chat_history: List[Dict[str, str]] = []
    enable_crag: bool = True
    answer_length: str = "Detailed Analysis"

class QueryResponse(BaseModel):
    """Response payload containing generated answer and sources."""
    answer: str
    sources: List[str]
    context_count: int
    passages: List[ContextPassage] = []
    match_confidence: float = 1.0
    web_fallback_used: bool = False

class TaskStatusResponse(BaseModel):
    """Progress details for background processing tasks."""
    task_id: str
    status: str  # pending, extracting, vectorizing, completed, failed
    progress: float  # progress percentage (0.0 to 100.0)
    message: str
    processed_count: int = 0
    error: Optional[str] = None
