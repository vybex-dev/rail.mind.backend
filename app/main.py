"""
RailMind AI — FastAPI Application Entry Point.

Start with:
    uvicorn app.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Routers ────────────────────────────────────────────────────────────────
from app.routers.delay import router as delay_router
from app.routers.crowd import router as crowd_router
from app.routers.safety import router as safety_router

# ── Model singletons (for /health status checks) ──────────────────────────
from app.models.delay_model import delay_predictor
from app.models.crowd_model import crowd_forecaster
# NOTE: safety_model is imported lazily — CLIP (~600 MB) loads on first use,
# not at startup, to prevent Railway from OOM-crashing during boot.
from app.models.safety_model import track_safety_detector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Log banner on startup; log shutdown on exit."""
    logger.info("=" * 60)
    logger.info("RailMind AI — starting up")
    logger.info(
        "  delay_predictor   loaded: %s",
        getattr(delay_predictor, "is_loaded", "unknown"),
    )
    logger.info(
        "  crowd_forecaster  loaded: %s",
        getattr(crowd_forecaster, "is_loaded", "unknown"),
    )
    logger.info(
        "  track_safety_detector  loaded (CLIP): %s",
        track_safety_detector.is_loaded,
    )
    logger.info("=" * 60)

    yield  # ← server is live between here and the next line

    logger.info("RailMind AI — shutting down")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RailMind AI",
    description=(
        "AI-powered backend for Indian Railways — "
        "delay prediction, crowd forecasting, and track safety detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (adjust origins for production) ──────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include routers ────────────────────────────────────────────────────────
app.include_router(delay_router)
app.include_router(crowd_router)
app.include_router(safety_router)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@app.get("/", tags=["Meta"], summary="API root")
async def root() -> dict:
    """Welcome message and quick link to the interactive docs."""
    return {
        "message": "Welcome to RailMind AI 🚆",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Meta"], summary="Health check")
async def health_check() -> dict:
    """
    Returns the operational status of all three AI modules.

    Response shape
    --------------
    ```json
    {
      "status": "ok",
      "modules": {
        "delay":  {"loaded": true},
        "crowd":  {"loaded": true},
        "safety": {"loaded": false, "mode": "mock"}
      }
    }
    ```

    The `loaded` flag reflects whether the underlying ML model was
    successfully initialised at startup.  A `false` value means the
    module is running in mock/fallback mode — the API still responds,
    but predictions are simulated rather than model-driven.
    """
    safety_loaded: bool = track_safety_detector.is_loaded

    # `delay_predictor` and `crowd_forecaster` expose `is_loaded` by convention;
    # fall back to True if the attribute is absent (models that don't use the flag).
    delay_loaded: bool = getattr(delay_predictor, "is_loaded", True)
    crowd_loaded: bool = getattr(crowd_forecaster, "is_loaded", True)

    return {
        "status": "ok",
        "modules": {
            "delay": {
                "loaded": delay_loaded,
            },
            "crowd": {
                "loaded": crowd_loaded,
            },
            "safety": {
                "loaded": safety_loaded,
                "mode": "ai" if safety_loaded else "mock",
            },
        },
    }