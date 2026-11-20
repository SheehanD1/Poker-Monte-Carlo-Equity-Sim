"""Integration tests for the poker-equity FastAPI application scaffold.

Covers the ``/health`` and ``/`` system endpoints plus CORS headers.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from poker_equity.api.app import app, create_app


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a TestClient instance for the FastAPI application."""
    return TestClient(app)


def test_create_app_returns_fastapi_instance() -> None:
    """Factory returns a FastAPI application with the health and root routes."""
    application = create_app()
    routes = {getattr(route, "path", None) for route in application.routes}
    assert "/health" in routes
    assert "/" in routes


class TestHealthAndRootEndpoints:
    """Tests for system health check, root metadata, and CORS headers."""

    def test_root_endpoint(self, client: TestClient) -> None:
        """GET / returns 200 and metadata links."""
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "Poker Monte Carlo Equity API"
        assert data["version"] == "0.1.0"
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"

    def test_health_check_endpoint(self, client: TestClient) -> None:
        """GET /health returns 200 with ok status."""
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert data["service"] == "poker-equity-api"

    def test_cors_headers_present(self, client: TestClient) -> None:
        """Verify CORS headers are returned in API responses."""
        res = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert res.status_code == 200
        assert res.headers.get("access-control-allow-origin") == "*"

    def test_cors_preflight_allows_get(self, client: TestClient) -> None:
        """OPTIONS preflight for /health advertises allowed methods."""
        res = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert res.status_code in (200, 204)
        allow_origin = res.headers.get("access-control-allow-origin")
        assert allow_origin in ("*", "http://localhost:3000")

    def test_openapi_docs_available(self, client: TestClient) -> None:
        """OpenAPI schema and Swagger UI are served."""
        schema = client.get("/openapi.json")
        assert schema.status_code == 200
        body = schema.json()
        assert body["info"]["title"] == "Poker Monte Carlo Equity API"
        assert "/health" in body["paths"]

        docs = client.get("/docs")
        assert docs.status_code == 200
