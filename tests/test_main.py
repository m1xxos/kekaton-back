import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


@patch("httpx.AsyncClient")
def test_list_models(mock_client):
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {"models": ["llama2", "mistral"]}
    mock_response.status_code = 200
    
    # Configure mock
    mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
    
    response = client.get("/models")
    assert response.status_code == 200
    assert "models" in response.json()


@patch("httpx.AsyncClient")
def test_generate(mock_client):
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model": "llama2",
        "created_at": "2023-11-04T12:34:56Z",
        "response": "This is a test response",
        "done": True
    }
    mock_response.status_code = 200
    
    # Configure mock
    mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
    
    test_request = {
        "model": "llama2",
        "prompt": "Tell me a joke"
    }
    
    response = client.post("/generate", json=test_request)
    assert response.status_code == 200
    assert response.json()["response"] == "This is a test response"
