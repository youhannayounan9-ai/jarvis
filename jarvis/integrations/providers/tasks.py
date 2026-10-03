"""
jarvis/integrations/providers/tasks.py
──────────────────────────────────────
v0.29 Part 10 — the TASKS integration (local dev provider), kept narrow.

Supported: list, get, create, complete (an update that flips status to
"done"). Delete is DISABLED at the provider level (the adapter's inherited
delete_resource raises AUTHORIZATION_DENIED) AND no scope menu path grants
TASKS_DELETE to the reference provider wiring — completing is the terminal
operation. This is the concrete realization of Part 10's "delete disabled
unless there is a concrete need".
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone as _tz  # noqa: F401

from jarvis.config import settings
from jarvis.integrations.base import (
    AuthState,
    ConnectedAccount,
    IntegrationProvider,
    Operation,
    ProviderCapabilities,
    ProviderResource,
    ResourceSpec,
    SideEffectRisk,
)
from jarvis.integrations.errors import ProviderError, ProviderErrorCategory
from jarvis.integrations.oauth import (
    LocalOAuthTokenClient,
    OAuthClientConfig,
    OAuthIntegrationProvider,
    local_authorization_server,
    local_introspection_auth_state,
)
from jarvis.integrations.providers.calendar import (
    PREFIX_AUTHENTICATED,
    PREFIX_EXPIRED,
    PREFIX_REVOKED,
)
from jarvis.integrations.scopes import TASKS_READ, TASKS_WRITE

_STATUS_DONE = "done"
_STATUS_OPEN = "open"


@dataclass
class _StoredTask:
    task_id: str
    account_id: str
    title: str
    notes: str
    status: str = _STATUS_OPEN
    completed_at: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(tz=_tz.utc).isoformat())


class LocalTasksBackend:
    MAX_TASKS = 500

    def __init__(self) -> None:
        self._tasks: dict[str, _StoredTask] = {}
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self._tasks.clear()

    def create(self, account_id: str, title: str, notes: str) -> _StoredTask:
        with self._lock:
            if len(self._tasks) >= self.MAX_TASKS:
                raise ProviderError(
                    ProviderErrorCategory.PROVIDER_OUTAGE,
                    "tasks store is full (bounded development provider)",
                )
            task = _StoredTask(
                task_id=f"task_{uuid.uuid4().hex[:16]}",
                account_id=account_id,
                title=title,
                notes=notes,
            )
            self._tasks[task.task_id] = task
            return task

    def list(self, account_id: str, limit: int) -> list[_StoredTask]:
        with self._lock:
            tasks = [t for t in self._tasks.values() if t.account_id == account_id]
            tasks.sort(key=lambda t: (t.status != _STATUS_OPEN, t.created_at))
            return tasks[: max(1, min(int(limit), 50))]

    def get(self, account_id: str, task_id: str) -> _StoredTask:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.account_id != account_id:
                raise ProviderError(
                    ProviderErrorCategory.NOT_FOUND,
                    f"task {task_id[:24]} not found",
                )
            return task

    def set_status(self, account_id: str, task_id: str, status: str) -> _StoredTask:
        with self._lock:
            task = self.get(account_id, task_id)
            if task.status == _STATUS_DONE and status == _STATUS_DONE:
                return task  # idempotent complete
            task.status = status
            task.completed_at = (
                datetime.now(tz=_tz.utc).isoformat() if status == _STATUS_DONE else None
            )
            return task


_backend = LocalTasksBackend()


def tasks_backend() -> LocalTasksBackend:
    return _backend


def reset_tasks_backend() -> None:
    _backend.reset()


class LocalTasksProvider(OAuthIntegrationProvider, IntegrationProvider):
    """
    The narrow tasks adapter (development-only; Part 10).

    v0.30: OAuth-capable through the SAME provider-neutral mixin as
    calendar — the OAuth implementation is not duplicated per provider.
    """

    def __init__(self) -> None:
        super().__init__(
            ProviderCapabilities(
                provider="tasks",
                display_name="Tasks (local development provider)",
                resource=ResourceSpec(
                    kind="task",
                    operations=frozenset(
                        {Operation.LIST, Operation.GET, Operation.CREATE, Operation.UPDATE}
                    ),
                    required_scopes={
                        Operation.LIST: TASKS_READ,
                        Operation.GET: TASKS_READ,
                        Operation.CREATE: TASKS_WRITE,
                        Operation.UPDATE: TASKS_WRITE,
                    },
                    operation_risk={
                        Operation.LIST: SideEffectRisk.READ_ONLY,
                        Operation.GET: SideEffectRisk.READ_ONLY,
                        Operation.CREATE: SideEffectRisk.LOW_SIDE_EFFECT,
                        Operation.UPDATE: SideEffectRisk.LOW_SIDE_EFFECT,
                    },
                ),
                grantable_scopes=frozenset({TASKS_READ, TASKS_WRITE}),
                production_like=False,
                supports_idempotency_key=False,
                description=(
                    "Deterministic local task list for development and "
                    "tests. Complete-only updates; delete unsupported."
                ),
            )
        )

    # ── v0.30: OAuth capability (same mixin as calendar — not duplicated) ───

    def oauth_config(self) -> OAuthClientConfig:
        provider = self.capabilities.provider
        client_id = str(getattr(settings, "OAUTH_TASKS_CLIENT_ID", "local-dev-tasks") or "")
        client_secret = str(
            getattr(settings, "OAUTH_TASKS_CLIENT_SECRET", "local-dev-tasks-secret") or ""
        )
        local_authorization_server().register_client(client_id, client_secret)
        return OAuthClientConfig(
            provider=provider,
            client_id=client_id,
            client_secret=client_secret,
            authorization_endpoint=f"local-oauth://{provider}/authorize",
            token_endpoint=f"local-oauth://{provider}/token",
            revocation_endpoint=f"local-oauth://{provider}/revoke",
            redirect_uri="",  # derived by the mixin (fixed path + session param)
            scope_map={s: s for s in (TASKS_READ, TASKS_WRITE)},
        )

    def token_client(self) -> LocalOAuthTokenClient:
        return LocalOAuthTokenClient(local_authorization_server())

    def verify_authentication(self, account: ConnectedAccount) -> AuthState:
        if account.is_oauth_account:
            return local_introspection_auth_state(account)
        secret = account._credential_secret
        if not secret:
            return AuthState.REVOKED
        if secret.startswith(PREFIX_REVOKED):
            return AuthState.REVOKED
        if secret.startswith(PREFIX_EXPIRED):
            return AuthState.EXPIRED
        if secret.startswith(PREFIX_AUTHENTICATED):
            return AuthState.AUTHENTICATED
        return AuthState.ERROR

    def is_available(self) -> bool:
        return True

    def list_resources(self, account: ConnectedAccount, limit: int) -> list[ProviderResource]:
        return [_task_to_resource(t) for t in _backend.list(account.account_id, limit)]

    def get_resource(self, account: ConnectedAccount, resource_id: str) -> ProviderResource:
        return _task_to_resource(_backend.get(account.account_id, resource_id))

    def create_resource(
        self,
        account: ConnectedAccount,
        fields: dict[str, str],
        *,
        idempotency_key: str | None = None,
    ) -> ProviderResource:
        title = (fields.get("title") or "").strip()
        if not title:
            raise ProviderError(
                ProviderErrorCategory.VALIDATION_ERROR,
                "task title is required",
            )
        return _task_to_resource(
            _backend.create(account.account_id, title, fields.get("notes", ""))
        )

    def update_resource(
        self, account: ConnectedAccount, resource_id: str, fields: dict[str, str]
    ) -> ProviderResource:
        status = fields.get("status", "").strip().lower()
        if status not in (_STATUS_OPEN, _STATUS_DONE):
            raise ProviderError(
                ProviderErrorCategory.VALIDATION_ERROR,
                "tasks support only status open/done updates",
            )
        return _task_to_resource(
            _backend.set_status(account.account_id, resource_id, status)
        )


def _task_to_resource(t: _StoredTask) -> ProviderResource:
    return ProviderResource(
        resource_id=t.task_id,
        kind="task",
        fields={
            "id": t.task_id,
            "title": t.title,
            "status": t.status,
            "notes": t.notes,
        },
        raw_external={"title": t.title, "notes": t.notes},
    )
