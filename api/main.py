"""
Main Application Entrypoint for USM RESTful API.

Built with FastAPI, featuring CORS protection, request latency profiling,
RFC 7807 structured exception handling, and auto-generated Swagger documentation.
"""

import sys
import os
import time
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure the root project directory is on the sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.routes import router  # noqa: E402

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("USM_API")

# ==============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ==============================================================================
app = FastAPI(
    title="AI-Powered Unified Service Management (USM) API",
    description=(
        "Production-grade RESTful API powering IT Service Management (ITSM), "
        "LangGraph multi-agent triage workflows, ChromaDB vector RAG retrieval, "
        "and real-time operational telemetry."
    ),
    version="1.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "USM IT Operations Team",
        "email": "support@usm-system.internal"
    },
    license_info={
        "name": "Internal Enterprise License",
    }
)

# ==============================================================================
# CORS MIDDLEWARE
# ==============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# REQUEST LATENCY & PROFILING MIDDLEWARE
# ==============================================================================
@app.middleware("http")
async def add_process_time_and_log(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Incoming: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    logger.info(
        f"Completed: {request.method} {request.url.path} "
        f"Status={response.status_code} Latency={process_time:.4f}s"
    )
    return response


# ==============================================================================
# RFC 7807 GLOBAL EXCEPTION HANDLER
# ==============================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception at {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "type": "https://errors.usm.local/internal-server-error",
            "title": "Internal Server Error",
            "status": 500,
            "detail": str(exc),
            "instance": str(request.url.path),
        }
    )


# ==============================================================================
# ROOT SERVICE DISCOVERY ENDPOINT
# ==============================================================================
@app.get("/", tags=["System Information"])
def root_service_info():
    """
    Returns service discovery metadata and API documentation entrypoints.
    """
    return {
        "service": "AI-Powered Unified Service Management API",
        "version": "1.5.0",
        "status": "online",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_schema": "/openapi.json"
        },
        "endpoints": {
            "health": "/health",
            "analytics": "/api/v1/analytics/metrics",
            "tickets": "/api/v1/tickets",
            "agent_chat": "/api/v1/agent/chat",
            "knowledge_query": "/api/v1/knowledge/query",
            "knowledge_documents": "/api/v1/knowledge/documents"
        }
    }


# ==============================================================================
# ATTACH ROUTER & STARTUP
# ==============================================================================
@app.on_event("startup")
def startup_event():
    from database import init_db
    init_db()
    logger.info("Database schema initialized and verified on API startup.")

app.include_router(router)


# ==============================================================================
# DIRECT CLI LAUNCH SUPPORT
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

