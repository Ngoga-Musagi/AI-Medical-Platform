#!/usr/bin/env bash
# ============================================
# AI Medical Data Platform - Startup Script
# ============================================
# Cross-platform: Linux, macOS, Windows (Git Bash / MSYS2 / WSL)
#
# Ollama runs INSIDE Docker - no local install needed!
# Mistral model is auto-pulled on first startup.
#
# Usage:
#   ./run.sh                    - Start with default (ollama-mistral in Docker)
#   ./run.sh start              - Same as above
#   ./run.sh ollama-mistral     - Start with Mistral via Docker Ollama
#   ./run.sh ollama-llama3      - Start with Llama 3 via Docker Ollama
#   ./run.sh gemini             - Start with Google Gemini (cloud)
#   ./run.sh claude             - Start with Claude (cloud)
#   ./run.sh openai             - Start with OpenAI GPT-4 (cloud)
#   ./run.sh stop               - Stop all services
#   ./run.sh logs               - View logs
#   ./run.sh batch              - Run batch analysis on all 50 notes
#   ./run.sh test               - Test the API endpoints
#   ./run.sh status             - Show container status
#   ./run.sh restart [provider] - Restart with specified provider
#   ./run.sh pull [model]       - Pull a model into Docker Ollama

set -e

# ============================================
# Cross-platform setup
# ============================================
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$PROJECT_DIR"

# Detect OS
detect_os() {
    case "$(uname -s 2>/dev/null || echo Windows)" in
        Linux*)   OS_TYPE="linux" ;;
        Darwin*)  OS_TYPE="macos" ;;
        MINGW*|MSYS*|CYGWIN*) OS_TYPE="windows" ;;
        *)        OS_TYPE="windows" ;;
    esac
}
detect_os

# Detect Python command
detect_python() {
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        PYTHON_CMD=""
    fi
}
detect_python

# Detect docker compose command
detect_docker_compose() {
    if docker compose version &>/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &>/dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD=""
    fi
}
detect_docker_compose

# Colors (works in most terminals including Git Bash on Windows)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}${BOLD}============================================${NC}"
    echo -e "${BLUE}${BOLD}  AI Medical Data Platform                  ${NC}"
    echo -e "${BLUE}${BOLD}  Clinical AI Advisor + Analytics Dashboard ${NC}"
    echo -e "${BLUE}${BOLD}============================================${NC}"
    echo -e "  OS: ${CYAN}${OS_TYPE}${NC} | Docker Compose: ${CYAN}${COMPOSE_CMD:-NOT FOUND}${NC}"
    echo ""
}

ok()   { echo -e "  ${GREEN}[OK]${NC}    $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC}  $1"; }
err()  { echo -e "  ${RED}[ERROR]${NC} $1"; }
info() { echo -e "  ${CYAN}[INFO]${NC}  $1"; }

# ============================================
# Export all variables from .env
# ============================================
# This ensures host env vars always match .env,
# preventing stale Windows/system env vars from
# overriding the values Docker Compose passes to containers.
export_env() {
    if [ -f .env ]; then
        while IFS= read -r line || [ -n "$line" ]; do
            # Strip Windows \r carriage returns
            line="${line//$'\r'/}"
            # Skip comments, blank lines
            [[ -z "$line" || "$line" == \#* ]] && continue
            # Split on first = only
            key="${line%%=*}"
            value="${line#*=}"
            # Skip lines without =
            [[ "$key" == "$line" ]] && continue
            # Trim whitespace from key
            key=$(echo "$key" | xargs)
            [[ -z "$key" ]] && continue
            # Remove surrounding quotes from value
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"
            export "$key=$value"
        done < .env
        ok ".env variables exported to shell"
    fi
}

# ============================================
# Pre-flight checks
# ============================================
preflight() {
    local errors=0

    # Check Docker
    if ! command -v docker &>/dev/null; then
        err "Docker is not installed or not in PATH"
        echo "       Install Docker Desktop: https://docs.docker.com/get-docker/"
        errors=$((errors + 1))
    elif ! docker info &>/dev/null 2>&1; then
        err "Docker daemon is not running"
        echo "       Please start Docker Desktop and try again"
        errors=$((errors + 1))
    else
        ok "Docker is running"
    fi

    # Check docker compose
    if [ -z "$COMPOSE_CMD" ]; then
        err "docker compose not found"
        echo "       Install Docker Desktop (includes Compose V2)"
        errors=$((errors + 1))
    else
        ok "Docker Compose available ($COMPOSE_CMD)"
    fi

    # Check .env
    if [ ! -f .env ]; then
        warn ".env file not found - creating from .env.example"
        if [ -f .env.example ]; then
            cp .env.example .env
            ok "Created .env from .env.example"
        else
            err "No .env or .env.example found!"
            errors=$((errors + 1))
        fi
    else
        ok ".env file found"
    fi

    # Export .env variables (overrides any stale host env vars)
    export_env

    # Ensure outputs directory exists
    mkdir -p outputs/predictions 2>/dev/null || true

    # Ensure models directory exists (for volume mount)
    mkdir -p models 2>/dev/null || true

    if [ $errors -gt 0 ]; then
        echo ""
        err "Pre-flight checks failed ($errors errors). Fix the issues above and try again."
        exit 1
    fi
    echo ""
}

# ============================================
# Start services
# ============================================
start_services() {
    local PROVIDER="${1:-ollama-mistral}"
    print_header
    info "LLM Provider: ${BOLD}${PROVIDER}${NC}"
    echo ""

    # Pre-flight
    preflight

    # Validate provider for cloud APIs
    case "$PROVIDER" in
        gemini)
            if ! grep -qE "^GEMINI_API_KEY=.{10,}" .env 2>/dev/null && \
               ! grep -qE "^GOOGLE_API_KEY=.{10,}" .env 2>/dev/null; then
                warn "GEMINI_API_KEY not set in .env - Gemini may not work"
            fi
            ;;
        claude)
            if ! grep -qE "^ANTHROPIC_API_KEY=sk-ant-.{10,}" .env 2>/dev/null; then
                warn "ANTHROPIC_API_KEY not set in .env - Claude may not work"
            fi
            ;;
        openai)
            if ! grep -qE "^OPENAI_API_KEY=sk-.{10,}" .env 2>/dev/null; then
                warn "OPENAI_API_KEY not set in .env - OpenAI may not work"
            fi
            ;;
        ollama-*)
            info "Ollama will run inside Docker (model auto-pulled on first start)"
            ;;
    esac

    # Clean up old containers (ignore errors)
    info "Cleaning up old containers..."
    docker rm -f medical-neo4j medical-api medical-dashboard medical-ollama medical-ollama-pull 2>/dev/null || true
    echo ""

    # Build and start
    info "Building Docker images (first time may take a few minutes)..."
    LLM_PROVIDER="$PROVIDER" $COMPOSE_CMD build
    echo ""

    info "Starting services..."
    info "  - Ollama (LLM engine)"
    info "  - Neo4j  (Knowledge Graph)"
    info "  - API    (REST + Chatbot)"
    info "  - Dashboard (Analytics + Chat UI)"
    echo ""
    LLM_PROVIDER="$PROVIDER" $COMPOSE_CMD up -d --force-recreate
    echo ""

    # Wait for Ollama model pull
    if [[ "$PROVIDER" == ollama-* ]]; then
        info "Waiting for Ollama to be ready and model to be pulled..."
        info "(First run downloads the model ~4GB - subsequent starts are instant)"
        local pull_wait=0
        local max_pull_wait=300  # 5 minutes max for model pull
        while [ $pull_wait -lt $max_pull_wait ]; do
            # Check if ollama-pull container has finished
            local pull_status
            pull_status=$(docker inspect -f '{{.State.Status}}' medical-ollama-pull 2>/dev/null || echo "unknown")
            if [ "$pull_status" = "exited" ]; then
                local exit_code
                exit_code=$(docker inspect -f '{{.State.ExitCode}}' medical-ollama-pull 2>/dev/null || echo "1")
                if [ "$exit_code" = "0" ]; then
                    ok "Ollama model ready"
                    break
                else
                    warn "Model pull may have had issues (exit code: $exit_code)"
                    warn "Checking if model is already available..."
                    break
                fi
            fi
            sleep 5
            pull_wait=$((pull_wait + 5))
            if [ $((pull_wait % 30)) -eq 0 ]; then
                info "Still pulling model... ($pull_wait/${max_pull_wait}s)"
            fi
        done
        if [ $pull_wait -ge $max_pull_wait ]; then
            warn "Model pull is taking longer than expected. It will continue in the background."
        fi
    fi

    # Wait for API to be healthy
    info "Waiting for API to be ready..."
    local api_wait=0
    local max_api_wait=120
    while [ $api_wait -lt $max_api_wait ]; do
        if curl -sf http://localhost:8000/health 2>/dev/null | grep -q "healthy"; then
            break
        fi
        sleep 3
        api_wait=$((api_wait + 3))
        if [ $((api_wait % 15)) -eq 0 ]; then
            info "Still waiting for API... ($api_wait/${max_api_wait}s)"
        fi
    done

    echo ""
    if curl -sf http://localhost:8000/health 2>/dev/null | grep -q "healthy"; then
        echo -e "${GREEN}${BOLD}============================================${NC}"
        echo -e "${GREEN}${BOLD}  Platform is running!                      ${NC}"
        echo -e "${GREEN}${BOLD}============================================${NC}"
    else
        echo -e "${YELLOW}${BOLD}============================================${NC}"
        echo -e "${YELLOW}${BOLD}  Platform starting (API still loading)...  ${NC}"
        echo -e "${YELLOW}${BOLD}============================================${NC}"
    fi

    echo ""
    ok "Neo4j Browser:      http://localhost:7474"
    ok "API (REST + Chat):  http://localhost:8000"
    ok "API Docs (Swagger): http://localhost:8000/docs"
    ok "Dashboard + Chat:   http://localhost:8050"
    ok "Ollama API:         http://localhost:11434"
    echo ""
    info "Provider: ${BOLD}${PROVIDER}${NC}"

    # Show health
    local HEALTH
    HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null || echo "")
    if [ -n "$HEALTH" ]; then
        info "Health: $HEALTH"
    fi

    echo ""
    echo -e "${CYAN}Quick commands:${NC}"
    echo "  ./run.sh logs       # View service logs"
    echo "  ./run.sh test       # Test API endpoints"
    echo "  ./run.sh batch      # Analyze all 50 clinical notes"
    echo "  ./run.sh stop       # Stop all services"
    echo ""
}

# ============================================
# Stop services
# ============================================
stop_services() {
    print_header
    info "Stopping all services..."
    $COMPOSE_CMD down
    ok "All services stopped"
}

# ============================================
# View logs
# ============================================
show_logs() {
    local service="${1:-}"
    if [ -n "$service" ]; then
        $COMPOSE_CMD logs -f --tail=100 "$service"
    else
        $COMPOSE_CMD logs -f --tail=100
    fi
}

# ============================================
# Pull model into Docker Ollama
# ============================================
pull_model() {
    local MODEL="${1:-mistral}"
    info "Pulling model '$MODEL' into Docker Ollama..."

    # Make sure Ollama is running
    if ! docker ps --format '{{.Names}}' | grep -q medical-ollama; then
        info "Starting Ollama container first..."
        $COMPOSE_CMD up -d ollama
        info "Waiting for Ollama to be ready..."
        sleep 10
    fi

    docker exec medical-ollama ollama pull "$MODEL"
    ok "Model '$MODEL' pulled successfully"
}

# ============================================
# Batch analysis
# ============================================
run_batch() {
    print_header
    info "Running batch analysis on all 50 clinical notes..."
    echo "  This will take a few minutes depending on your LLM provider."
    echo ""

    # Try running via Docker exec first, fall back to local Python
    if docker ps --format '{{.Names}}' | grep -q medical-api; then
        info "Running batch via Docker container..."
        docker exec medical-api python run_batch.py
    elif [ -n "$PYTHON_CMD" ]; then
        info "Running batch locally with $PYTHON_CMD..."
        $PYTHON_CMD run_batch.py
    else
        err "No Python found and API container is not running"
        err "Start the platform first: ./run.sh start"
        exit 1
    fi
}

# ============================================
# Test endpoints
# ============================================
run_tests() {
    print_header
    info "Testing API endpoints..."
    echo ""

    # JSON formatting helper (uses Python if available)
    format_json() {
        if [ -n "$PYTHON_CMD" ]; then
            $PYTHON_CMD -m json.tool 2>/dev/null || cat
        else
            cat
        fi
    }

    echo -e "${BLUE}${BOLD}1. Health Check${NC}"
    curl -s http://localhost:8000/health | format_json
    echo ""

    echo -e "${BLUE}${BOLD}2. Knowledge Graph Stats${NC}"
    curl -s http://localhost:8000/knowledge_graph/stats | format_json
    echo ""

    echo -e "${BLUE}${BOLD}3. Analyze Clinical Note (Pneumonia)${NC}"
    curl -s -X POST http://localhost:8000/analyze_note \
        -H "Content-Type: application/json" \
        -d '{"note_id": "TEST001", "text": "45 year old male with fever and productive cough for 5 days. Diagnosed with pneumonia. Started on amoxicillin 500mg TID."}' \
        | format_json
    echo ""

    echo -e "${BLUE}${BOLD}4. Analyze Note with Contraindication (UTI + Ciprofloxacin)${NC}"
    curl -s -X POST http://localhost:8000/analyze_note \
        -H "Content-Type: application/json" \
        -d '{"note_id": "TEST002", "text": "30 year old female with dysuria and frequency. Diagnosed UTI. Prescribed ciprofloxacin 500mg BID."}' \
        | format_json
    echo ""

    echo -e "${BLUE}${BOLD}5. Medical Advisor Chat${NC}"
    curl -s -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "I have a 60 year old patient with chest pain and elevated troponin. Suspecting MI. What drugs and tests do you recommend?"}' \
        | format_json
    echo ""

    echo -e "${BLUE}${BOLD}6. Performance Metrics${NC}"
    curl -s http://localhost:8000/metrics | format_json
    echo ""

    echo -e "${BLUE}${BOLD}7. List Providers${NC}"
    curl -s http://localhost:8000/providers | format_json
    echo ""

    ok "All tests completed!"
}

# ============================================
# Main entry point
# ============================================
case "${1:-start}" in
    start)
        start_services "ollama-mistral"
        ;;
    ollama-mistral|ollama-llama3|ollama-meditron|gemini|claude|openai)
        start_services "$1"
        ;;
    stop)
        stop_services
        ;;
    logs)
        show_logs "${2:-}"
        ;;
    batch)
        run_batch
        ;;
    test)
        run_tests
        ;;
    restart)
        stop_services
        echo ""
        start_services "${2:-ollama-mistral}"
        ;;
    status)
        print_header
        $COMPOSE_CMD ps
        ;;
    pull)
        pull_model "${2:-mistral}"
        ;;
    *)
        print_header
        echo -e "${BOLD}Usage:${NC} $0 <command> [options]"
        echo ""
        echo -e "${BOLD}Start Commands:${NC}"
        echo "  start               Start with default (ollama-mistral in Docker)"
        echo "  ollama-mistral      Start with Mistral via Docker Ollama (default, free)"
        echo "  ollama-llama3       Start with Llama 3 via Docker Ollama (free)"
        echo "  ollama-meditron     Start with Meditron via Docker Ollama (medical)"
        echo "  gemini              Start with Google Gemini (cloud, free tier)"
        echo "  claude              Start with Claude (cloud, paid)"
        echo "  openai              Start with OpenAI GPT-4 (cloud, paid)"
        echo ""
        echo -e "${BOLD}Management Commands:${NC}"
        echo "  stop                Stop all services"
        echo "  restart [provider]  Restart with specified provider"
        echo "  logs [service]      View logs (optional: api, dashboard, ollama, neo4j)"
        echo "  status              Show container status"
        echo ""
        echo -e "${BOLD}Analysis Commands:${NC}"
        echo "  batch               Run batch analysis on 50 clinical notes"
        echo "  test                Test all API endpoints"
        echo ""
        echo -e "${BOLD}Ollama Commands:${NC}"
        echo "  pull [model]        Pull a model into Docker Ollama (default: mistral)"
        echo ""
        echo -e "${BOLD}Examples:${NC}"
        echo "  $0                  # Start with Mistral (default)"
        echo "  $0 gemini           # Start with Gemini"
        echo "  $0 pull llama3      # Pull Llama 3 model"
        echo "  $0 restart claude   # Restart with Claude"
        echo ""
        exit 1
        ;;
esac
