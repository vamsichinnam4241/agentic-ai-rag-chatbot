from typing import List, TypedDict
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """API request payload for /chat endpoint."""

    query: str = Field(
        ...,
        description="The user's question regarding the Agentic AI eBook",
        examples=["What is Agentic AI according to the eBook?"]
    )


class ChatResponse(BaseModel):
    """API response payload for /chat endpoint."""

    answer: str = Field(
        ...,
        description="The grounded answer from the eBook or fallback message",
        examples=["Agentic AI refers to..."]
    )
    retrieved_chunks: List[str] = Field(
        default_factory=list,
        description="List of relevant text chunks retrieved from Pinecone vector index"
    )
    confidence_score: float = Field(
        ...,
        description="Relevance confidence score calculated from retrieval evidence (0.0 - 1.0)"
    )


class RAGState(TypedDict):
    """State dictionary maintained across the LangGraph execution flow."""

    question: str
    context: str
    retrieved_chunks: List[str]
    scores: List[float]
    answer: str
    confidence_score: float
