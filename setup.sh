#!/usr/bin/env bash
#
# setup.sh — first-time setup for the Local Agentic RAG Backend.
#
# What it does:
#   1. Creates .env from .env.example (if missing).
#   2. Reads DOCS_HOST_PATH from .env and creates that documents folder.
#   3. Checks prerequisites (Docker, Docker Compose, Ollama) and pulls the model.
#   4. Optionally builds and launches the stack, then runs ingestion.
#
# Usage:
#   ./setup.sh            # interactive: prompts before launching
#   ./setup.sh --yes      # non-interactive: build, launch and ingest
#   ./setup.sh --no-start # only prepare .env + folders, don't touch Docker
#
# Safe to re-run: every step is idempotent.

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BOLD="$(tput bold 2>/dev/null || true)"
GREEN="$(tput setaf 2 2>/dev/null || true)"
YELLOW="$(tput setaf 3 2>/dev/null || true)"
RED="$(tput setaf 1 2>/dev/null || true)"
RESET="$(tput sgr0 2>/dev/null || true)"

info()  { echo "${GREEN}✔${RESET} $*"; }
warn()  { echo "${YELLOW}!${RESET} $*"; }
error() { echo "${RED}✗${RESET} $*" >&2; }
step()  { echo; echo "${BOLD}==> $*${RESET}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ASSUME_YES=0
START_STACK=1
for arg in "$@"; do
  case "$arg" in
    -y|--yes)      ASSUME_YES=1 ;;
    --no-start)    START_STACK=0 ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) error "Unknown argument: $arg"; exit 1 ;;
  esac
done

confirm() {
  # confirm "Question?"  -> returns 0 for yes
  [ "$ASSUME_YES" -eq 1 ] && return 0
  local reply
  read -r -p "$1 [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

# ---------------------------------------------------------------------------
# 1. .env
# ---------------------------------------------------------------------------
step "Preparing environment file"
if [ -f .env ]; then
  info ".env already exists — leaving it untouched."
else
  cp .env.example .env
  info "Created .env from .env.example."
fi

# Read DOCS_HOST_PATH from .env (fall back to default).
DOCS_HOST_PATH="$(grep -E '^DOCS_HOST_PATH=' .env | tail -n1 | cut -d= -f2- || true)"
DOCS_HOST_PATH="${DOCS_HOST_PATH:-./data}"
MODEL_NAME="$(grep -E '^MODEL_NAME=' .env | tail -n1 | cut -d= -f2- || true)"
MODEL_NAME="${MODEL_NAME:-qwen2.5:3b}"

# ---------------------------------------------------------------------------
# 2. Documents directory
# ---------------------------------------------------------------------------
step "Creating documents directory"
if mkdir -p "$DOCS_HOST_PATH" 2>/dev/null; then
  info "Documents folder ready: $DOCS_HOST_PATH"
else
  error "Could not create $DOCS_HOST_PATH. Create it manually (check the mount/permissions)."
  exit 1
fi
# On exFAT/NTFS, chown is not permitted and not needed — do not attempt it.
FS_TYPE="$(df -T "$DOCS_HOST_PATH" 2>/dev/null | awk 'NR==2 {print $2}')"
if [[ "$FS_TYPE" == "exfat" || "$FS_TYPE" == "ntfs" || "$FS_TYPE" == "fuseblk" ]]; then
  warn "Documents folder is on a $FS_TYPE drive — that's fine for read-only document storage."
  warn "Qdrant's index and the model cache use Docker named volumes (system disk) instead."
fi

file_count="$(find "$DOCS_HOST_PATH" -type f \( -iname '*.pdf' -o -iname '*.txt' -o -iname '*.md' -o -iname '*.json' \) 2>/dev/null | wc -l | tr -d ' ')"
if [ "$file_count" -eq 0 ]; then
  warn "No .pdf/.txt/.md/.json files found in $DOCS_HOST_PATH yet."
  warn "Add your knowledge files there, then run ingestion (this script can do it)."
else
  info "Found $file_count ingestible file(s) in $DOCS_HOST_PATH."
fi

# ---------------------------------------------------------------------------
# 3. Prerequisites
# ---------------------------------------------------------------------------
step "Checking prerequisites"

missing=0
if command -v docker >/dev/null 2>&1; then
  info "Docker: $(docker --version | cut -d, -f1)"
else
  error "Docker not found. Install Docker: https://docs.docker.com/engine/install/"
  missing=1
fi

if docker compose version >/dev/null 2>&1; then
  info "Docker Compose: $(docker compose version --short 2>/dev/null || echo present)"
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  info "Docker Compose (legacy): $(docker-compose version --short 2>/dev/null || echo present)"
  COMPOSE="docker-compose"
else
  error "Docker Compose not found."
  missing=1
fi

if command -v ollama >/dev/null 2>&1; then
  info "Ollama: $(ollama --version 2>/dev/null | head -n1)"
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL_NAME"; then
    info "Model '$MODEL_NAME' already pulled."
  else
    if confirm "Pull Ollama model '$MODEL_NAME' now?"; then
      ollama pull "$MODEL_NAME"
      info "Pulled '$MODEL_NAME'."
    else
      warn "Skipped model pull. Run 'ollama pull $MODEL_NAME' before first use."
    fi
  fi
else
  warn "Ollama not found on host. Install it (https://ollama.com) and run 'ollama pull $MODEL_NAME'."
  warn "The container reaches Ollama at host.docker.internal:11434."
fi

if [ "$missing" -ne 0 ]; then
  error "Please install the missing prerequisites above, then re-run ./setup.sh."
  exit 1
fi

# ---------------------------------------------------------------------------
# 4. Build, launch, ingest
# ---------------------------------------------------------------------------
if [ "$START_STACK" -eq 0 ]; then
  step "Done (prepared .env and folders; skipped Docker due to --no-start)."
  exit 0
fi

step "Build and launch the container stack"
if confirm "Run '$COMPOSE up -d --build' now?"; then
  $COMPOSE up -d --build
  info "Stack is starting."

  step "Waiting for the API to become healthy"
  healthy=0
  for _ in $(seq 1 30); do
    if curl -fsS "http://localhost:8000/health" >/dev/null 2>&1; then
      healthy=1; break
    fi
    sleep 2
  done
  if [ "$healthy" -eq 1 ]; then
    info "API is up: http://localhost:8000/  (docs at /docs)"
  else
    warn "API didn't report healthy in time. Check logs: $COMPOSE logs -f app"
  fi

  if [ "$file_count" -gt 0 ]; then
    step "Ingesting documents"
    if confirm "Run ingestion now (embeds $file_count file(s) into Qdrant)?"; then
      docker exec -it agentic_rag_mcp python -m app.ingest
      info "Ingestion complete."
    else
      warn "Skipped. Run later: docker exec -it agentic_rag_mcp python -m app.ingest"
    fi
  else
    warn "No documents to ingest yet. Add files to $DOCS_HOST_PATH and run:"
    warn "  docker exec -it agentic_rag_mcp python -m app.ingest"
  fi
else
  warn "Skipped launch. When ready, run: $COMPOSE up -d --build"
fi

echo
info "Setup finished. Chat at http://localhost:8000/"
