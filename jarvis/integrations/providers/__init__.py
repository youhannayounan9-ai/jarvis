# jarvis/integrations/providers/__init__.py
"""
Provider adapters. v0.29 ships two development-only local providers
(calendar — the reference; tasks — narrow). Production adapters implement
the same IntegrationProvider seam without any runtime change.
"""

from jarvis.integrations.providers.calendar import (
    LocalCalendarProvider,
    calendar_backend,
    reset_calendar_backend,
)
from jarvis.integrations.providers.tasks import (
    LocalTasksProvider,
    reset_tasks_backend,
    tasks_backend,
)

__all__ = [
    "LocalCalendarProvider",
    "calendar_backend",
    "reset_calendar_backend",
    "LocalTasksProvider",
    "tasks_backend",
    "reset_tasks_backend",
]
