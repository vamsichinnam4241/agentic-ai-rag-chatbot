import pytest
from app.graph import rag_graph, FALLBACK_MESSAGE
from app.vector_store import set_mock_documents
from langchain_core.documents import Document


@pytest.mark.asyncio
async def test_grounding_fifa_world_cup_unrelated_question():
    """
    Critical Benchmark Test:
    Ensures that unrelated general knowledge questions (e.g. 2022 FIFA World Cup)
    are strictly rejected with the explicit required fallback message.
    """
    # Even if mock documents exist about AI, querying FIFA World Cup should produce no match or low confidence
    doc = Document(
        page_content="Agentic AI systems orchestrate workflows using tools and memory.",
        metadata={"page": 3}
    )
    set_mock_documents([doc])

    payload = {
        "question": "Who won the 2022 FIFA World Cup?",
        "context": "",
        "retrieved_chunks": [],
        "scores": [],
        "answer": "",
        "confidence_score": 0.0
    }

    state = await rag_graph.ainvoke(payload)

    assert state["answer"] == FALLBACK_MESSAGE, f"Expected exact fallback string, got: {state['answer']}"
    assert state["confidence_score"] == 0.0, f"Expected 0.0 confidence score for unsupported query, got: {state['confidence_score']}"


@pytest.mark.asyncio
async def test_grounding_exact_fallback_string_format():
    """
    Verifies that the exact string required by the specification is returned:
    'I couldn't find enough information in the provided Agentic AI eBook.'
    """
    set_mock_documents([])
    payload = {
        "question": "What is the secret recipe for Coca-Cola?",
        "context": "",
        "retrieved_chunks": [],
        "scores": [],
        "answer": "",
        "confidence_score": 0.0
    }

    state = await rag_graph.ainvoke(payload)
    assert state["answer"] == "I couldn't find enough information in the provided Agentic AI eBook."


@pytest.mark.asyncio
async def test_grounding_supported_question_with_context():
    """
    Verifies that supported questions return non-zero confidence score when matching context is retrieved.
    """
    doc = Document(
        page_content="According to the eBook, Agentic AI refers to AI systems designed to autonomously pursue goals using workflows, tools, and reflection.",
        metadata={"page": 12}
    )
    set_mock_documents([doc])

    payload = {
        "question": "What is Agentic AI according to the eBook?",
        "context": "",
        "retrieved_chunks": [],
        "scores": [],
        "answer": "",
        "confidence_score": 0.0
    }

    state = await rag_graph.ainvoke(payload)

    assert len(state["retrieved_chunks"]) > 0
    assert state["confidence_score"] > 0.0
    assert state["answer"] != ""
