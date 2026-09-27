# JARVIS API service image.
#
# Builds the FastAPI service surface. The Docker-isolated code-execution
# sandbox runs in SEPARATE one-shot containers (deploy/Dockerfile.sandbox);
# on Linux hosts with the Docker socket mounted, the sandbox container can
# even be a sibling of this one (see compose comments).
#
# Build:
#   docker build -t jarvis-api:0.12.0 .
#
# Run:
#   docker run -p 8000:8000 \
#     -e JARVIS_API_KEY=... \
#     -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
#     -v jarvis-data:/data \
#     jarvis-api:0.12.0
#
FROM python:3.12-slim

# uv for fast, lockfile-exact installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first (cached layer).
COPY pyproject.toml README.md ./
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python \
        litellm pydantic-settings python-dotenv structlog fastapi "uvicorn[standard]" \
        ddgs rich chromadb sentence-transformers cachetools

# Application code.
COPY jarvis/ jarvis/
COPY ui/ ui/
COPY evaluation/ evaluation/
COPY docs/ docs/

# Non-root runtime user; /data is the volume mount point for the DB.
RUN useradd --create-home --uid 1000 jarvis \
    && mkdir -p /data \
    && chown -R jarvis:jarvis /app /data
USER jarvis
ENV PATH="/app/.venv/bin:${PATH}" \
    DB_PATH=/data/jarvis.db \
    VECTOR_DB_PATH=/data/chroma_db

EXPOSE 8000

# Healthcheck against the unauthenticated liveness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "jarvis.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
