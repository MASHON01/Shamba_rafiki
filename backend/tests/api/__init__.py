"""
API tests: every endpoint via FastAPI's TestClient.

Drives the real FastAPI app through HTTP using the api_client fixture
(dependency-overridden to the fake LLM + real retriever), so the full
request/response envelope, error handling, and routing are exercised
without a live server or llama-server.
"""