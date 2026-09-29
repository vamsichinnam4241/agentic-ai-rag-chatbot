import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse
from app.graph import rag_graph
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="Agentic AI eBook RAG Chatbot API",
    description="Production-grade RAG chatbot orchestration with LangGraph, Pinecone, and OpenAI",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def read_root():
    """Root endpoint returning API status."""
    return {
        "status": "online",
        "service": "Agentic AI eBook RAG Chatbot",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Readiness/Liveness probe endpoint."""
    return {
        "status": "healthy",
        "pinecone_configured": bool(settings.pinecone_api_key),
        "openai_configured": bool(settings.openai_api_key)
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["RAG Chatbot"]
)
async def chat_endpoint(request: ChatRequest):
    """
    RAG Chatbot query endpoint.
    
    Executes the LangGraph workflow:
    START -> retrieve -> generate -> END
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter cannot be empty."
        )

    logger.info(f"Received query: '{request.query}'")

    try:
        initial_state = {
            "question": request.query,
            "context": "",
            "retrieved_chunks": [],
            "scores": [],
            "answer": "",
            "confidence_score": 0.0
        }

        # Invoke LangGraph workflow
        final_state = await rag_graph.ainvoke(initial_state)

        return ChatResponse(
            answer=final_state.get("answer", ""),
            retrieved_chunks=final_state.get("retrieved_chunks", []),
            confidence_score=final_state.get("confidence_score", 0.0)
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
