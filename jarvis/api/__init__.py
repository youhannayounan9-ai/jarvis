"""
jarvis/api/
───────────
Service layer — separates the *agent runtime* from *interfaces*.

Currently exposed:
  - FastAPI app (app.py): /health, /chat, /sessions, confirmations, /tools

The runtime (jarvis.runtime.JarvisRuntime) is transport-agnostic; HTTP is
just one interface among CLI / dashboard / voice.
"""
