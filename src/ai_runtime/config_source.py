"""Where the "which profile does this task use" answer comes from.

Everything behind ``ModelConfigSource`` is swappable: env today, a Postgres
table later, a dict in tests. Call sites ask for a task and never learn which
source answered, so moving the binding into a database is one new class and no
change to any use case.
"""

import os
from dataclasses import dataclass
from typing import Mapping, Protocol

from ai_runtime.catalog import TASK_DEFAULTS, DEFAULT_PROFILE

# One spelling, platform-wide: LLM_PROFILE__<TASK>, e.g.
#   LLM_PROFILE__QUIZ_GENERATION=deepseek-chat
# and LLM_PROFILE=<name> as the fallback for every task.
TASK_ENV_PREFIX = "LLM_PROFILE__"
GLOBAL_ENV_VAR = "LLM_PROFILE"


@dataclass(frozen=True)
class TaskBinding:
    """The resolved answer for one task, plus where it came from."""

    task: str
    profile: str
    source: str  # "env:LLM_PROFILE__X" | "env:LLM_PROFILE" | "task-default" | ...
    temperature: float | None = None  # None keeps the profile's own value


class ModelConfigSource(Protocol):
    def binding_for(self, task: str) -> TaskBinding:
        ...


class StaticConfigSource:
    """Fixed bindings, for tests and for pinning a run during an experiment."""

    def __init__(self, bindings: Mapping[str, str], default_profile: str = DEFAULT_PROFILE) -> None:
        self._bindings = dict(bindings)
        self._default_profile = default_profile

    def binding_for(self, task: str) -> TaskBinding:
        profile = self._bindings.get(task)
        if profile is not None:
            return TaskBinding(task=task, profile=profile, source="static")
        return TaskBinding(task=task, profile=self._default_profile, source="static-default")


class EnvConfigSource:
    """Resolve a task to a profile name from the environment.

    Order: LLM_PROFILE__<TASK>, then LLM_PROFILE, then ``fallback_profile``
    (used by the caller to keep a pre-profile deployment working), then the
    task default from the catalog.
    """

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        fallback_profile: str | None = None,
        task_defaults: Mapping[str, str] | None = None,
        default_profile: str = DEFAULT_PROFILE,
    ) -> None:
        self._env = env if env is not None else os.environ
        self._fallback_profile = fallback_profile
        self._task_defaults = dict(task_defaults if task_defaults is not None else TASK_DEFAULTS)
        self._default_profile = default_profile

    def binding_for(self, task: str) -> TaskBinding:
        task_var = f"{TASK_ENV_PREFIX}{task.upper()}"
        explicit = _non_empty(self._env.get(task_var))
        if explicit:
            return TaskBinding(task=task, profile=explicit, source=f"env:{task_var}")

        global_choice = _non_empty(self._env.get(GLOBAL_ENV_VAR))
        if global_choice:
            return TaskBinding(task=task, profile=global_choice, source=f"env:{GLOBAL_ENV_VAR}")

        if self._fallback_profile:
            return TaskBinding(task=task, profile=self._fallback_profile, source="legacy-env")

        profile = self._task_defaults.get(task, self._default_profile)
        source = "task-default" if task in self._task_defaults else "global-default"
        return TaskBinding(task=task, profile=profile, source=source)


def _non_empty(value: str | None) -> str | None:
    """Compose passes "" for unset .env variables, which is not a choice."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


__all__ = [
    "GLOBAL_ENV_VAR",
    "TASK_ENV_PREFIX",
    "EnvConfigSource",
    "ModelConfigSource",
    "StaticConfigSource",
    "TaskBinding",
]
