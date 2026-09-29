import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "pinecone_configured" in data
    assert "openai_configured" in data


def test_chat_endpoint_empty_query():
    response = client.post("/chat", json={"query": "   "})
    assert response.status_code == 400


def test_chat_endpoint_valid_schema():
    payload = {"query": "What is Agentic AI according to the eBook?"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "retrieved_chunks" in data
    assert "confidence_score" in data
    assert isinstance(data["retrieved_chunks"], list)
    assert isinstance(data["confidence_score"], float)
