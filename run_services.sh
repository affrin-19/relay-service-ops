#!/usr/bin/env bash
# ==============================================================================
# AI-Powered Unified Service Management (USM) System - Dual Service Launcher
# ==============================================================================
# Usage:
#   ./run_services.sh          -> Runs both FastAPI (port 8000) & Streamlit (port 8501)
#   ./run_services.sh api      -> Runs only FastAPI Backend
#   ./run_services.sh ui       -> Runs only Streamlit Frontend
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Resolve Virtual Environment Python & Binaries
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
VENV_UVICORN="$SCRIPT_DIR/venv/bin/uvicorn"
VENV_STREAMLIT="$SCRIPT_DIR/venv/bin/streamlit"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Error: Virtual environment not found at $SCRIPT_DIR/venv"
    echo "Please set up your venv first."
    exit 1
fi

MODE="${1:-all}"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export HOME="$SCRIPT_DIR"

cleanup() {
    echo ""
    echo "🛑 Shutting down USM services..."
    if [ ! -z "$API_PID" ]; then
        kill "$API_PID" 2>/dev/null || true
    fi
    if [ ! -z "$UI_PID" ]; then
        kill "$UI_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "✅ All USM services stopped cleanly."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

case "$MODE" in
    api)
        echo "🚀 Starting USM FastAPI REST Backend on http://0.0.0.0:8000..."
        echo "📖 Interactive API Docs (Swagger UI): http://localhost:8000/docs"
        "$VENV_UVICORN" api.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    ui)
        echo "🎨 Starting USM Streamlit Frontend on http://localhost:8501..."
        "$VENV_STREAMLIT" run app.py --server.port 8501 --server.headless false
        ;;
    all)
        echo "======================================================================"
        echo "🌟 AI-Powered Unified Service Management (USM) System"
        echo "======================================================================"
        echo "🌐 Starting FastAPI REST Backend on http://0.0.0.0:8000 ..."
        "$VENV_UVICORN" api.main:app --host 0.0.0.0 --port 8000 &
        API_PID=$!

        sleep 2

        echo "🎨 Starting Streamlit User Portal on http://localhost:8501 ..."
        "$VENV_STREAMLIT" run app.py --server.port 8501 --server.headless false &
        UI_PID=$!

        echo ""
        echo "✅ Services are running:"
        echo "   • FastAPI Backend:       http://localhost:8000"
        echo "   • Interactive API Docs:  http://localhost:8000/docs"
        echo "   • Streamlit Dashboard:   http://localhost:8501"
        echo ""
        echo "Press Ctrl+C to stop all services."
        wait
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [all|api|ui]"
        exit 1
        ;;
esac

