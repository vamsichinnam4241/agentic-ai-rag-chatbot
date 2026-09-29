import pytest
from app.graph import rag_graph, FALLBACK_MESSAGE
from app.vector_store import set_mock_documents
from langchain_core.documents import Document


@pytest.mark.asyncio
async def test_langgraph_workflow_execution():
    # Register mock documents for state testing
    doc = Document(
        page_content="Agentic AI systems use autonomous LLM agents to accomplish multi-step goals.",
        metadata={"page": 1}
    )
    set_mock_documents([doc])

    initial_state = {
        "question": "What is Agentic AI?",
        "context": "",
        "retrieved_chunks": [],
        "scores": [],
        "answer": "",
        "confidence_score": 0.0
    }

    final_state = await rag_graph.ainvoke(initial_state)

    assert "answer" in final_state
    assert "retrieved_chunks" in final_state
    assert "confidence_score" in final_state
    assert final_state["confidence_score"] >= 0.0


@pytest.mark.asyncio
async def test_langgraph_fallback_on_unsupported_query():
    # Clear mock documents so no matching chunks exist
    set_mock_documents([])

    initial_state = {
        "question": "Who won the 2022 FIFA World Cup?",
        "context": "",
        "retrieved_chunks": [],
        "scores": [],
        "answer": "",
        "confidence_score": 0.0
    }

    final_state = await rag_graph.ainvoke(initial_state)

    assert final_state["answer"] == FALLBACK_MESSAGE
    assert final_state["confidence_score"] == 0.0
    assert final_state["retrieved_chunks"] == []
