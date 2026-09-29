import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from app.schemas import RAGState
from app.config import settings
from app.vector_store import retrieve_context

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "I couldn't find enough information in the provided Agentic AI eBook."


def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """
    Node 1: Retrieve context and scores from Pinecone vector store for the question.
    """
    question = state.get("question", "")
    logger.info(f"Retrieving context for question: '{question}'")

    retrieved_chunks, context, scores = retrieve_context(question)

    # Compute candidate confidence score from retrieval evidence
    if scores:
        confidence_score = round(max(scores), 4)
    else:
        confidence_score = 0.0

    return {
        "context": context,
        "retrieved_chunks": retrieved_chunks,
        "scores": scores,
        "confidence_score": confidence_score
    }


def generate_node(state: RAGState) -> Dict[str, Any]:
    """
    Node 2: Generate response using LLM grounded strictly in retrieved context.
    """
    question = state.get("question", "")
    context = state.get("context", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    scores = state.get("scores", [])
    confidence_score = state.get("confidence_score", 0.0)

    # Rule: If no chunks retrieved or retrieval similarity score is below threshold, trigger fallback immediately
    if not retrieved_chunks or not context.strip() or confidence_score < settings.similarity_threshold:
        logger.info(f"Low retrieval confidence ({confidence_score} < {settings.similarity_threshold}). Triggering fallback.")
        return {
            "answer": FALLBACK_MESSAGE,
            "confidence_score": 0.0
        }

    # If OpenAI API Key is present, invoke LLM
    if settings.openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=0,
                openai_api_key=settings.openai_api_key
            )

            system_prompt = (
                "You are an expert AI assistant grounded strictly in the provided Agentic AI eBook context.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Answer the user's question ONLY using the provided eBook context below.\n"
                "2. Do NOT use any general knowledge, prior knowledge, or external information.\n"
                "3. If the provided context does NOT contain enough explicit information to answer the question, "
                f"you MUST reply with EXACTLY: \"{FALLBACK_MESSAGE}\"\n"
                "4. Do not hallucinate, speculate, or invent any details.\n\n"
                f"EBOOK CONTEXT:\n{context}"
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Question: {question}")
            ]

            response = llm.invoke(messages)
            answer_text = response.content.strip()

            # Check if LLM indicates insufficient information
            if (
                FALLBACK_MESSAGE.lower() in answer_text.lower() or
                "couldn't find enough information" in answer_text.lower() or
                "not enough information" in answer_text.lower()
            ):
                return {
                    "answer": FALLBACK_MESSAGE,
                    "confidence_score": 0.0
                }

            return {
                "answer": answer_text,
                "confidence_score": confidence_score
            }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "answer": FALLBACK_MESSAGE,
                "confidence_score": 0.0
            }

    else:
        # Fallback mode for testing when OpenAI key is omitted
        logger.warning("No OPENAI_API_KEY available in .env. Formulating grounded response directly from retrieved eBook context.")
        top_chunk = retrieved_chunks[0]
        lines = [line.strip() for line in top_chunk.split("\n") if line.strip()]
        answer_snippet = " ".join(lines[:5])
        return {
            "answer": answer_snippet,
            "confidence_score": confidence_score
        }


# Build LangGraph workflow: START -> retrieve -> generate -> END
builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

rag_graph = builder.compile()
