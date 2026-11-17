import sys
import time
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("USM_API")

app = FastAPI(
    title="Unified Service Management API",
    description="REST API for IT service management, agent triage, and knowledge retrieval.",
    version="1.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_and_log(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    response.headers["X-Process-Time"] = f"{latency:.4f}s"
    logger.info(f"{request.method} {request.url.path} status={response.status_code} latency={latency:.4f}s")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception at {request.method} {request.url.path}: {exc}", exc_info=True)
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


@app.get("/", tags=["System"])
def root_service_info():
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


@app.on_event("startup")
def startup_event():
    from database import init_db
    init_db()
    logger.info("Database initialized.")


app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
