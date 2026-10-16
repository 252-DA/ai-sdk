import pytest

from ai_runtime.catalog import PROFILES, TASK_DEFAULTS, ModelProfile
from ai_runtime.config_source import EnvConfigSource, OverrideConfigSource, StaticConfigSource
from ai_runtime.errors import ModelConfigurationError
from ai_runtime.providers.deepseek import DeepSeekModelClient
from ai_runtime.registry import ModelRegistry


def _registry(env: dict[str, str], **kwargs) -> ModelRegistry:
    return ModelRegistry(source=EnvConfigSource(env=env, **kwargs), env=env)


class TestEnvConfigSource:
    def test_task_variable_wins(self):
        env = {"LLM_PROFILE__QUIZ_GENERATION": "gemini-pro", "LLM_PROFILE": "deepseek-chat"}
        binding = EnvConfigSource(env=env).binding_for("quiz_generation")

        assert binding.profile == "gemini-pro"
        assert binding.source == "env:LLM_PROFILE__QUIZ_GENERATION"

    def test_global_variable_applies_to_every_task(self):
        env = {"LLM_PROFILE": "gemini-pro"}
        source = EnvConfigSource(env=env)

        assert source.binding_for("enrichment").profile == "gemini-pro"
        assert source.binding_for("tutor").profile == "gemini-pro"

    def test_empty_string_is_not_a_choice(self):
        # Compose expands an unset .env variable to "", which must not win over
        # the task default.
        env = {"LLM_PROFILE__ENRICHMENT": "", "LLM_PROFILE": "  "}
        binding = EnvConfigSource(env=env).binding_for("enrichment")

        assert binding.profile == TASK_DEFAULTS["enrichment"]
        assert binding.source == "task-default"

    def test_fallback_profile_beats_task_default(self):
        binding = EnvConfigSource(env={}, fallback_profile="legacy").binding_for("enrichment")

        assert binding.profile == "legacy"
        assert binding.source == "legacy-env"

    def test_task_default_is_used_last(self):
        binding = EnvConfigSource(env={}).binding_for("quiz_verification")

        assert binding.profile == TASK_DEFAULTS["quiz_verification"]


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Store:
    """Bảng binding giả: đổi ``rows`` hoặc ``error`` giữa các lần đọc."""

    def __init__(self, rows: dict[str, str] | None = None) -> None:
        self.rows = dict(rows or {})
        self.error: Exception | None = None
        self.reads = 0

    def __call__(self) -> dict[str, str]:
        self.reads += 1
        if self.error is not None:
            raise self.error
        return dict(self.rows)


def _override(store: _Store, clock: _Clock, env: dict[str, str] | None = None, **kwargs):
    return OverrideConfigSource(
        store, EnvConfigSource(env=env or {}), ttl_seconds=30, clock=clock,
        known_profiles=PROFILES, **kwargs,
    )


class TestOverrideConfigSource:
    def test_store_binding_beats_the_env_variable(self):
        store = _Store({"quiz_generation": "gemini-pro"})
        source = _override(store, _Clock(), env={"LLM_PROFILE__QUIZ_GENERATION": "deepseek-chat"})
        binding = source.binding_for("quiz_generation")

        assert binding.profile == "gemini-pro"
        assert binding.source == "db"

    def test_task_missing_from_the_store_falls_through_to_env(self):
        source = _override(_Store({}), _Clock(), env={"LLM_PROFILE__ENRICHMENT": "gemini-pro"})
        binding = source.binding_for("enrichment")

        assert binding.profile == "gemini-pro"
        assert binding.source == "env:LLM_PROFILE__ENRICHMENT"

    def test_reads_the_store_once_per_ttl(self):
        store, clock = _Store({"enrichment": "gemini-pro"}), _Clock()
        source = _override(store, clock)
        for _ in range(5):
            source.binding_for("enrichment")
        assert store.reads == 1

        clock.now = 31
        source.binding_for("enrichment")
        assert store.reads == 2

    def test_a_change_takes_effect_after_the_ttl_without_a_restart(self):
        store, clock = _Store({"enrichment": "gemini-flash"}), _Clock()
        source = _override(store, clock)
        assert source.binding_for("enrichment").profile == "gemini-flash"

        store.rows["enrichment"] = "gemini-pro"
        clock.now = 10
        assert source.binding_for("enrichment").profile == "gemini-flash"
        clock.now = 31
        assert source.binding_for("enrichment").profile == "gemini-pro"

    def test_removing_the_row_returns_the_task_to_env(self):
        store, clock = _Store({"enrichment": "gemini-pro"}), _Clock()
        source = _override(store, clock)
        source.binding_for("enrichment")

        del store.rows["enrichment"]
        clock.now = 31
        assert source.binding_for("enrichment").source == "task-default"

    def test_store_failure_keeps_the_last_good_answer(self):
        store, clock = _Store({"enrichment": "gemini-pro"}), _Clock()
        source = _override(store, clock)
        source.binding_for("enrichment")

        store.error = RuntimeError("db down")
        clock.now = 31
        assert source.binding_for("enrichment").profile == "gemini-pro"

    def test_store_failure_on_cold_start_falls_back_to_env(self):
        store = _Store()
        store.error = RuntimeError("relation does not exist")
        source = _override(store, _Clock())

        assert source.binding_for("quiz_verification").profile == TASK_DEFAULTS["quiz_verification"]

    def test_failed_read_waits_a_ttl_before_retrying(self):
        store, clock = _Store(), _Clock()
        store.error = RuntimeError("db down")
        source = _override(store, clock)
        source.binding_for("enrichment")
        source.binding_for("enrichment")

        assert store.reads == 1

    def test_unknown_profile_in_the_store_is_ignored(self):
        source = _override(_Store({"enrichment": "gpt-99"}), _Clock())
        binding = source.binding_for("enrichment")

        assert binding.profile == TASK_DEFAULTS["enrichment"]

    def test_empty_profile_in_the_store_is_not_a_choice(self):
        source = _override(_Store({"enrichment": "  "}), _Clock())

        assert source.binding_for("enrichment").source == "task-default"

    def test_registry_resolves_through_the_override(self):
        env = {"GEMINI_API_KEY": "gk"}
        source = _override(_Store({"quiz_generation": "gemini-pro"}), _Clock(), env=env)
        resolved = ModelRegistry(source=source, env=env).resolve("quiz_generation")

        assert resolved.profile.name == "gemini-pro"
        assert resolved.binding.source == "db"


class TestModelRegistry:
    def test_resolves_profile_to_configured_client(self):
        registry = _registry({"LLM_PROFILE__QUIZ_GENERATION": "deepseek-chat",
                              "DEEPSEEK_API_KEY": "dk"})
        resolved = registry.resolve("quiz_generation")

        assert isinstance(resolved.client, DeepSeekModelClient)
        assert resolved.provider == "deepseek"
        assert resolved.model_id == PROFILES["deepseek-chat"].model

    def test_api_key_comes_from_the_first_non_empty_key_env(self):
        env = {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "second", "LLM_PROFILE": "gemini-flash"}
        assert _registry(env).config_for("tutor").api_key == "second"

    def test_missing_key_does_not_block_resolution(self):
        # A worker must still start when a profile it never calls has no key;
        # the provider client raises at first use instead.
        assert _registry({"LLM_PROFILE": "gemini-flash"}).config_for("tutor").api_key is None

    def test_same_configuration_reuses_one_client(self):
        registry = _registry({"LLM_PROFILE": "gemini-flash", "GEMINI_API_KEY": "k"})

        assert registry.resolve("enrichment").client is registry.resolve("tutor").client

    def test_two_tasks_can_run_different_models_in_one_process(self):
        registry = _registry({
            "LLM_PROFILE__QUIZ_GENERATION": "deepseek-chat",
            "LLM_PROFILE__QUIZ_VERIFICATION": "gemini-flash",
            "DEEPSEEK_API_KEY": "dk",
            "GEMINI_API_KEY": "gk",
        })

        writer = registry.resolve("quiz_generation")
        judge = registry.resolve("quiz_verification")

        assert writer.client is not judge.client
        assert writer.provider != judge.provider

    def test_unknown_profile_names_the_source_and_the_alternatives(self):
        registry = _registry({"LLM_PROFILE__ENRICHMENT": "gemini-ultra"})

        with pytest.raises(ModelConfigurationError) as excinfo:
            registry.resolve("enrichment")

        message = str(excinfo.value)
        assert "gemini-ultra" in message
        assert "LLM_PROFILE__ENRICHMENT" in message
        assert "deepseek-chat" in message  # the list of what is valid

    def test_binding_temperature_overrides_the_profile(self):
        registry = ModelRegistry(
            source=StaticConfigSource({}, default_profile="gemini-flash"),
            env={"GEMINI_API_KEY": "k"},
        )
        assert registry.config_for("tutor").temperature == PROFILES["gemini-flash"].temperature

    def test_extra_profiles_are_resolvable(self):
        extra = ModelProfile(name="local", provider="openai-compatible", model="qwen",
                             key_env=("LOCAL_KEY",), base_url="http://localhost:11434/v1")
        registry = ModelRegistry(
            source=StaticConfigSource({"tutor": "local"}),
            profiles={**PROFILES, "local": extra},
            env={"LOCAL_KEY": "k"},
        )
        config = registry.config_for("tutor")

        assert config.model == "qwen"
        assert config.base_url == "http://localhost:11434/v1"


class TestCatalog:
    def test_every_task_default_points_at_a_real_profile(self):
        for task, profile in TASK_DEFAULTS.items():
            assert profile in PROFILES, f"{task} points at unknown profile {profile}"

    def test_a_priced_profile_records_when_it_was_priced(self):
        # An undated price silently goes stale and then lies in llm_usage_logs.
        for name, profile in PROFILES.items():
            if profile.is_priced:
                assert profile.priced_on, f"{name} has prices but no priced_on date"

    def test_unpriced_profile_reports_no_cost_rather_than_zero(self):
        profile = ModelProfile(name="x", provider="gemini", model="m")
        assert profile.cost_usd(1000, 1000) is None

    def test_cost_is_per_million_tokens(self):
        profile = ModelProfile(name="x", provider="gemini", model="m",
                               price_in_per_1m=2.0, price_out_per_1m=10.0, priced_on="2026-09-28")
        assert profile.cost_usd(1_000_000, 500_000) == pytest.approx(2.0 + 5.0)

    def test_profile_names_match_their_keys(self):
        for key, profile in PROFILES.items():
            assert key == profile.name


class TestInternalProfiles:
    def test_internal_profile_is_resolvable_but_not_suggested(self):
        from ai_runtime.catalog import known_profile_names

        shim = ModelProfile(name="env-llm", provider="gemini", model="m", internal=True)
        profiles = {**PROFILES, "env-llm": shim}
        registry = ModelRegistry(
            source=StaticConfigSource({"tutor": "env-llm"}), profiles=profiles, env={}
        )

        assert registry.profile_for("tutor").name == "env-llm"
        assert "env-llm" not in known_profile_names(profiles)
        assert "gemini-flash" in known_profile_names(profiles)

    def test_no_catalog_profile_is_internal(self):
        # Người dùng phải chọn được mọi profile có trong catalog.
        assert [name for name, p in PROFILES.items() if p.internal] == []
