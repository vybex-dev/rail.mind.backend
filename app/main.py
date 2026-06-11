"""
RailMind AI — FastAPI Application Entry Point.

Start with:
    uvicorn app.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""
import sys

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
# NOTE: CLIP (~600 MB) loads lazily on the FIRST /safety/analyze request —
# NOT at startup — to prevent Railway free tier from OOM-crashing during boot.
from app.models.safety_model import track_safety_detector
sys.dont_write_bytecode = False 
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
    # CLIP is lazy — it won't be loaded yet at startup, that's intentional.
    logger.info(
        "  track_safety_detector  mode: lazy (CLIP loads on first /safety/analyze request)"
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
        "safety": {"loaded": false, "mode": "lazy"}
      }
    }
    ```

    The `loaded` flag for safety reflects whether CLIP has been loaded yet.
    It will be false until the first /safety/analyze request is made —
    this is intentional (lazy loading to avoid OOM on Railway free tier).
    Once loaded it becomes true and stays true.
    """
    safety_loaded: bool = track_safety_detector.is_loaded

    delay_loaded: bool = getattr(delay_predictor, "is_loaded", True)
    crowd_loaded: bool = getattr(crowd_forecaster, "is_loaded", True)

    # Determine safety mode label
    if safety_loaded:
        safety_mode = "ai"
    elif track_safety_detector._load_failed:
        safety_mode = "mock"
    else:
        safety_mode = "lazy"   # not yet loaded, will try on first request

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
                "mode": safety_mode,
            },
        },
    }
