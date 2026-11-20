"""FastAPI application factory with CORS middleware and health check.

This module scaffolds the REST API surface. Domain endpoints (equity, range,
odds, outs) are added in subsequent commits.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """API health status and service metadata."""

    status: str = Field("ok", description="Service health status")
    version: str = Field("0.1.0", description="API version")
    service: str = Field("poker-equity-api", description="Service identifier")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Returns
    -------
    FastAPI
        Configured FastAPI application with CORS middleware and health route.
    """
    application = FastAPI(
        title="Poker Monte Carlo Equity API",
        description=(
            "High-performance REST API for Texas Hold'em hand evaluation, "
            "range parsing, 13x13 matrix generation, and Monte Carlo equity simulation."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get(
        "/",
        summary="API Root",
        description="Returns API service information and link to documentation.",
        tags=["System"],
    )
    def root() -> dict[str, Any]:
        """Root endpoint returning service status and documentation links."""
        return {
            "name": "Poker Monte Carlo Equity API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }

    @application.get(
        "/health",
        response_model=HealthResponse,
        summary="Service Health Check",
        description="Check if the API service is alive and healthy.",
        tags=["System"],
    )
    def health_check() -> HealthResponse:
        """Return the health status of the API service."""
        return HealthResponse(
            status="ok",
            version="0.1.0",
            service="poker-equity-api",
        )

    return application


app = create_app()
