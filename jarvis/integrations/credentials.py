"""
jarvis/integrations/credentials.py
──────────────────────────────────
v0.29 Part 4 — the credential boundary (explicit local-development scope).

HONEST LIMITATION, documented rather than papered over:

  JARVIS v0.29 ships NO encrypted OS-keychain / vault integration (adding a
  credential-vault dependency is out of scope and forbidden by AGENTS.md
  without explicit instruction). The boundary below is therefore a
  LOCAL-DEVELOPMENT credential boundary, explicitly labeled as such:

    - Credential material lives ONLY inside the `integration_accounts`
      SQLite row in the user's own local database file, in a
      lightly-obfuscated (XOR + base64, keyed per-install) form. THIS IS
      NOT ENCRYPTION against an attacker with the file: it prevents
      accidental eyeball/lint/patch-note exposure, nothing more.
    - The system behaves, in every observable way, as if the secret were
      plaintext in a config file — because effectively it is. Do not use
      v0.29 integrations with real high-value accounts. Production-grade
      secret storage is a documented future requirement
      (docs/JARVIS_V029_SECURITY_MODEL.md §2).

  What v0.29 DOES guarantee structurally (independent of storage):
    - secrets never appear in prompts (the model-facing tool surface has
      no parameter and no code path that returns them);
    - secrets never appear in tool output, action descriptions, ledger
      visible fields, logs, or API/dashboard responses (sanitization +
      non-secret projections);
    - secrets are excluded from repr/compare on the account record;
    - raw OAuth codes / refresh tokens never reach the LLM (Part 19 —
      the local-dev connect flow validates the code OUTSIDE the model).
"""

from __future__ import annotations

import base64
import hashlib
import os
import uuid


# Process-lifetime key for ":memory:" databases (tests, ephemeral runs):
# such databases have no directory, so a key FILE would either pollute the
# repo or — worse — be recreated per call. The key lives only in this
# process variable, which exactly matches the ":memory:" lifetime.
_memory_key: bytes | None = None


def _install_key() -> bytes:
    """
    Per-install obfuscation key: derived from a random file created next to
    the database at first use. Removes the "grep the repo for the constant"
    failure mode; provides NO protection against someone who can read both
    files (documented limitation).

    A ":memory:" database (tests, ephemeral runs) has no directory — its key
    is generated once per process and kept in memory only (never written to
    disk anywhere).
    """
    global _memory_key
    from jarvis.config import settings

    if str(settings.db_path).strip() == ":memory:":
        if _memory_key is None:
            _memory_key = os.urandom(32)
        return _memory_key

    base_dir = os.path.dirname(os.path.abspath(settings.db_path)) or "."
    key_path = os.path.join(base_dir, ".jarvis_integration_key")
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            return f.read()
    key = os.urandom(32)
    # Restrictive flags where the OS honors them; failure is non-fatal
    # (Windows ignores POSIX modes) — the limitation is documented anyway.
    fd_flags = getattr(os, "O_WRONLY", 0) | getattr(os, "O_CREAT", 0) | getattr(os, "O_EXCL", 0)
    try:
        fd = os.open(key_path, fd_flags, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(key)
    except FileExistsError:
        with open(key_path, "rb") as f:
            return f.read()
    return key


def new_credential_material() -> str:
    """A random bearer-shaped local-dev token (opaque; provider-shaped)."""
    return f"loc-dev_{uuid.uuid4().hex}_{uuid.uuid4().hex[:12]}"


def obfuscate(secret: str) -> str:
    """XOR + base64 obfuscation (NOT encryption — see module docstring)."""
    raw = secret.encode("utf-8")
    key = _install_key()
    keystream = hashlib.sha512(key).digest()
    repeated = (keystream * (len(raw) // len(keystream) + 1))[: len(raw)]
    xored = bytes(a ^ b for a, b in zip(raw, repeated))
    return base64.urlsafe_b64encode(xored).decode("ascii")


def deobfuscate(stored: str) -> str:
    """Inverse of obfuscate(); corrupt input yields '' (never raises)."""
    try:
        key = _install_key()
        raw = base64.urlsafe_b64decode(stored.encode("ascii"))
        keystream = hashlib.sha512(key).digest()
        repeated = (keystream * (len(raw) // len(keystream) + 1))[: len(raw)]
        return bytes(a ^ b for a, b in zip(raw, repeated)).decode("utf-8")
    except Exception:
        return ""


def credential_fingerprint(secret: str) -> str:
    """
    Non-reversible display fingerprint (first 4 + last 2 of sha256) — the
    ONLY credential-derived value allowed outside the store. Lets a human
    confirm WHICH secret is configured without exposing it.
    """
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    return f"…{digest[:4]}…{digest[-2:]}"
