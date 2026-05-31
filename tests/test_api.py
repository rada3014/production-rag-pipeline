"""
Integration tests — real pipeline, real API calls (needs .env with API keys).

Run: pytest tests/ -v
"""

import os
import sys

import pytest
from starlette.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api import app  # noqa: E402

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "BERT_(language_model).pdf")


@pytest.fixture(scope="module")
def client():
    """
    Starts the real server once for all tests.
    scope="module" means the PDF is loaded only once — not before every test.
    """
    with TestClient(app) as c:
        with open(PDF_PATH, "rb") as f:
            c.post("/upload", files={"file": ("BERT_(language_model).pdf", f, "application/pdf")})
        yield c


# Test 1: GET /health → 200
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200


# Test 2: POST /query with empty string → 422
def test_query_empty_string(client):
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422


# Test 3: POST /query real question → 200 non-empty answer
def test_query_real_question(client):
    resp = client.post("/query", json={"question": "What is BERT?"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["retrieval_qa"]["answer"]) > 0
    assert len(data["reranked_llm"]["answer"]) > 0
