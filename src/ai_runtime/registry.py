"""Resolve a task to a model client, and cache clients by configuration.

This replaces "one client per process, built at boot". A registry holds a
client per distinct configuration, so one process can run several models at
once (writer and judge, for instance), and a source that changes its answer
later hands back a different client without a restart.
"""

import os
from dataclasses import dataclass
from typing import Mapping

from ai_runtime.catalog import (
    PROFILES,
    ModelProfile,
    known_profile_names,
)
from ai_runtime.config_source import EnvConfigSource, ModelConfigSource, TaskBinding
from ai_runtime.errors import ModelConfigurationError
from ai_runtime.models import ModelClient, ModelConfig, build_model_client
from ai_runtime.runtime import AIRuntime


@dataclass(frozen=True)
class ResolvedModel:
    """What a task resolved to: the client plus the labels worth logging."""

    task: str
    profile: ModelProfile
    binding: TaskBinding
    client: ModelClient

    @property
    def provider(self) -> str:
        return self.profile.provider

    @property
    def model_id(self) -> str:
        return self.profile.model

    def runtime(self) -> AIRuntime:
        return AIRuntime(self.client)


class ModelRegistry:
    def __init__(
        self,
        source: ModelConfigSource | None = None,
        profiles: Mapping[str, ModelProfile] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._source = source if source is not None else EnvConfigSource(env=env)
        self._profiles = dict(profiles if profiles is not None else PROFILES)
        self._env = env if env is not None else os.environ
        self._clients: dict[tuple, ModelClient] = {}

    @property
    def profiles(self) -> Mapping[str, ModelProfile]:
        return self._profiles

    def binding_for(self, task: str) -> TaskBinding:
        return self._source.binding_for(task)

    def profile_for(self, task: str) -> ModelProfile:
        return self._profile(self.binding_for(task))

    def config_for(self, task: str) -> ModelConfig:
        binding = self.binding_for(task)
        return self._config(self._profile(binding), binding)

    def resolve(self, task: str) -> ResolvedModel:
        """Resolve once per unit of work, then reuse within it.

        Resolving again mid-request would let a config change split one request
        across two models, which makes its cost row a lie.
        """
        binding = self.binding_for(task)
        profile = self._profile(binding)
        config = self._config(profile, binding)
        return ResolvedModel(
            task=task,
            profile=profile,
            binding=binding,
            client=self._client(config),
        )

    def runtime_for(self, task: str) -> AIRuntime:
        return self.resolve(task).runtime()

    # ------------------------------------------------------------------

    def _profile(self, binding: TaskBinding) -> ModelProfile:
        profile = self._profiles.get(binding.profile)
        if profile is None:
            raise ModelConfigurationError(
                f"Model profile {binding.profile!r} for task {binding.task!r} "
                f"(from {binding.source}) is not in the catalog. "
                f"Known profiles: {known_profile_names(self._profiles)}"
            )
        return profile

    def _config(self, profile: ModelProfile, binding: TaskBinding) -> ModelConfig:
        return ModelConfig(
            provider=profile.provider,
            model=profile.model,
            # Missing keys stay missing here on purpose: the provider clients
            # build lazily, so a worker with no key for an unused profile still
            # starts, and the error names the provider at first real call.
            api_key=self._api_key(profile),
            base_url=profile.base_url,
            temperature=(
                binding.temperature if binding.temperature is not None else profile.temperature
            ),
            timeout_seconds=profile.timeout_seconds,
            max_retries=profile.max_retries,
        )

    def _api_key(self, profile: ModelProfile) -> str | None:
        for name in profile.key_env:
            value = (self._env.get(name) or "").strip()
            if value:
                return value
        return None

    def _client(self, config: ModelConfig) -> ModelClient:
        key = (config.provider, config.model, config.base_url, config.temperature)
        client = self._clients.get(key)
        if client is None:
            client = build_model_client(config)
            self._clients[key] = client
        return client


__all__ = ["ModelRegistry", "ResolvedModel"]
