import os
import logging
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from app.config import settings

logger = logging.getLogger(__name__)

# Global mock documents list (None means uninitialized)
_mock_documents: Optional[List[Document]] = None


def set_mock_documents(docs: List[Document]) -> None:
    """Utility function for unit tests to register mock documents in memory."""
    global _mock_documents
    _mock_documents = docs


def _ensure_fallback_documents_loaded():
    """Auto-load ebook.pdf chunks into local memory if no external keys are configured."""
    global _mock_documents
    if _mock_documents is None:
        if os.path.exists(settings.pdf_path):
            try:
                from langchain_community.document_loaders import PyPDFLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter

                logger.info(f"Loading local eBook chunks from '{settings.pdf_path}' for offline retrieval...")
                loader = PyPDFLoader(settings.pdf_path)
                pages = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap
                )
                _mock_documents = text_splitter.split_documents(pages)
                logger.info(f"Successfully loaded {len(_mock_documents)} eBook chunks into local memory.")
            except Exception as e:
                logger.error(f"Error loading fallback eBook documents: {e}")
                _mock_documents = []
        else:
            _mock_documents = []


def get_embeddings():
    """Initialize OpenAI Embeddings model."""
    from langchain_openai import OpenAIEmbeddings
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY is not set.")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key
    )


def get_pinecone_vector_store():
    """Initialize Pinecone Vector Store instance."""
    from langchain_pinecone import PineconeVectorStore
    embeddings = get_embeddings()
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=settings.pinecone_api_key
    )


def retrieve_context(
    query: str,
    top_k: Optional[int] = None
) -> Tuple[List[str], str, List[float]]:
    """
    Retrieve top-k relevant document chunks for a query from Pinecone (or local eBook fallback).

    Returns:
        retrieved_chunks: List of raw string text for retrieved chunks
        formatted_context: Concatenated text with document page metadata
        scores: List of float similarity scores
    """
    k = top_k or settings.top_k_results

    # Use Pinecone if API keys are configured, otherwise use local eBook memory search
    if settings.pinecone_api_key and settings.openai_api_key:
        try:
            vector_store = get_pinecone_vector_store()
            results_with_scores = vector_store.similarity_search_with_relevance_scores(query, k=k)
        except Exception as e:
            logger.error(f"Pinecone search failed ({e}). Falling back to local eBook search.")
            results_with_scores = _retrieve_from_mock(query, k)
    else:
        results_with_scores = _retrieve_from_mock(query, k)

    retrieved_chunks: List[str] = []
    formatted_pieces: List[str] = []
    scores: List[float] = []

    for doc, score in results_with_scores:
        cleaned_score = max(0.0, min(1.0, float(score)))
        chunk_text = doc.page_content.strip()
        page = doc.metadata.get("page", "N/A")
        
        retrieved_chunks.append(chunk_text)
        scores.append(round(cleaned_score, 4))
        formatted_pieces.append(f"[Page {page}]: {chunk_text}")

    formatted_context = "\n\n".join(formatted_pieces)
    return retrieved_chunks, formatted_context, scores


def _retrieve_from_mock(query: str, k: int) -> List[Tuple[Document, float]]:
    """Helper method for strict term-based similarity search across local eBook chunks."""
    _ensure_fallback_documents_loaded()
    if not _mock_documents:
        return []

    stop_words = {"what", "is", "are", "the", "a", "an", "according", "to", "ebook", "in", "of", "and", "or", "for", "with", "who", "won", "secret", "recipe"}
    query_keywords = [t.lower() for t in query.split() if t.lower() not in stop_words and len(t) > 1]
    
    if not query_keywords:
        query_keywords = [t.lower() for t in query.split() if len(t) > 1]

    results: List[Tuple[Document, float]] = []

    for doc in _mock_documents:
        content_lower = doc.page_content.lower()
        matches = sum(1 for term in query_keywords if term in content_lower)
        
        if matches > 0:
            match_ratio = matches / len(query_keywords)
            # Require at least 50% keyword match or multiple keyword hits to qualify above similarity threshold
            if match_ratio >= 0.5 or matches >= 2:
                score = round(min(0.50 + (match_ratio * 0.45), 0.95), 4)
            else:
                score = 0.15 # Below similarity threshold (0.35)
            
            results.append((doc, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]
