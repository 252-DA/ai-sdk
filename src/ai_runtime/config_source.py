"""Where the "which profile does this task use" answer comes from.

Everything behind ``ModelConfigSource`` is swappable: env, a table an admin
edits at runtime (``OverrideConfigSource`` layered over env), a dict in tests.
Call sites ask for a task and never learn which source answered.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Collection, Mapping, Protocol

from ai_runtime.catalog import TASK_DEFAULTS, DEFAULT_PROFILE

logger = logging.getLogger(__name__)

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


class OverrideConfigSource:
    """Bindings that can change while the process runs, layered over ``base``.

    ``fetch`` returns ``{task: profile}`` from a store an admin edits (the
    ``llm_model_bindings`` table); a task it does not name falls through to
    ``base``. The answer is cached for ``ttl_seconds``, so a change takes
    effect within one TTL and a busy worker reads the store once per TTL, not
    once per job.

    The store going down must not take the models down with it: a failed
    fetch keeps the last answer that worked (none, on a cold start) and waits
    one TTL before trying again. A profile the catalog does not know is
    dropped with a warning instead of failing every job that uses the task.
    """

    def __init__(
        self,
        fetch: Callable[[], Mapping[str, str]],
        base: ModelConfigSource,
        *,
        ttl_seconds: float = 30.0,
        known_profiles: Collection[str] | None = None,
        source_label: str = "db",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch
        self._base = base
        self._ttl = ttl_seconds
        self._known = frozenset(known_profiles) if known_profiles is not None else None
        self._label = source_label
        self._clock = clock
        self._lock = threading.Lock()
        self._overrides: dict[str, str] = {}
        self._loaded_at: float | None = None

    def binding_for(self, task: str) -> TaskBinding:
        profile = self._current().get(task)
        if profile:
            return TaskBinding(task=task, profile=profile, source=self._label)
        return self._base.binding_for(task)

    def _current(self) -> dict[str, str]:
        with self._lock:
            now = self._clock()
            if self._loaded_at is not None and now - self._loaded_at < self._ttl:
                return self._overrides
            self._loaded_at = now
            try:
                raw = dict(self._fetch())
            except Exception as exc:  # noqa: BLE001 -- any store error means "keep going"
                logger.warning(
                    "model binding fetch failed; keeping %d previous binding(s): %s",
                    len(self._overrides),
                    exc,
                )
                return self._overrides
            self._overrides = self._valid(raw)
            return self._overrides

    def _valid(self, raw: Mapping[str, str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for task, profile in raw.items():
            name = _non_empty(profile)
            if name is None:
                continue
            if self._known is not None and name not in self._known:
                logger.warning(
                    "ignoring model binding %s=%r: profile is not in the catalog", task, name
                )
                continue
            out[task] = name
        return out


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
    "OverrideConfigSource",
    "StaticConfigSource",
    "TaskBinding",
]
