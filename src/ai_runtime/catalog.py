"""Model catalog: every model the platform may use is declared here.

A profile is the only unit anything outside this module is allowed to name.
Config sources, job payloads and logs carry a profile *name*; provider ids,
model ids and base URLs never leave this file. That keeps one allowlist for
"which models exist", and keeps an endpoint override from ever arriving with
the platform's own API key attached.

Secrets are not stored here either -- ``key_env`` holds the *names* of the
environment variables that may carry the key, in priority order.
"""

from dataclasses import dataclass
from typing import Literal

Provider = Literal["gemini", "openai-compatible", "deepseek"]

# Tasks are keyed by the same strings that already go into
# llm_usage_logs.use_case, so a binding and its cost rows join on one column.
QUIZ_GENERATION = "quiz_generation"
QUIZ_VERIFICATION = "quiz_verification"
QUIZ_REGENERATION = "quiz_regeneration"
ENRICHMENT = "enrichment"
TUTOR = "tutor"


@dataclass(frozen=True)
class ModelProfile:
    """One named, reviewed way to call a model."""

    name: str
    provider: Provider
    model: str
    key_env: tuple[str, ...] = ()
    base_url: str | None = None
    temperature: float = 0.2
    timeout_seconds: float = 60
    max_retries: int = 2
    # Price is deliberately None until someone reads it off the provider's rate
    # card and records the date. A wrong price is worse than a missing one: it
    # looks like a real number in llm_usage_logs and nothing flags it.
    price_in_per_1m: float | None = None
    price_out_per_1m: float | None = None
    priced_on: str | None = None  # ISO date the prices above were checked
    # Internal profiles exist to keep a migration working; they are resolvable
    # but never offered as a choice, so nobody puts one in a .env on purpose.
    internal: bool = False

    @property
    def is_priced(self) -> bool:
        return self.price_in_per_1m is not None and self.price_out_per_1m is not None

    def cost_usd(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        """USD for one call, or None when this profile has no rate card yet."""
        if not self.is_priced:
            return None
        return (
            (input_tokens or 0) / 1_000_000 * self.price_in_per_1m
            + (output_tokens or 0) / 1_000_000 * self.price_out_per_1m
        )


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------
# To add a model: add one entry here. Nothing else in the platform needs to
# learn about it -- env, DB bindings and payloads only ever say the name.
#
# price_*: fill from the provider rate card and set priced_on to that date.
# Until then cost_usd stays 0 in llm_usage_logs and the worker logs
# llm.cost.unpriced once per profile, which is the honest state.

PROFILES: dict[str, ModelProfile] = {
    "gemini-flash": ModelProfile(
        name="gemini-flash",
        provider="gemini",
        model="gemini-3-flash-preview",
        key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
    "gemini-pro": ModelProfile(
        name="gemini-pro",
        provider="gemini",
        model="gemini-3-pro-preview",
        key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
    "gemini-flash-2": ModelProfile(
        # Kept because it is the model every default in the repo used to name;
        # gives a one-word way back if a newer preview starts misbehaving.
        name="gemini-flash-2",
        provider="gemini",
        model="gemini-2.0-flash",
        key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
    "deepseek-chat": ModelProfile(
        name="deepseek-chat",
        provider="deepseek",
        model="deepseek-chat",
        key_env=("DEEPSEEK_API_KEY",),
    ),
}

# Which profile a task uses when nothing overrides it.
TASK_DEFAULTS: dict[str, str] = {
    # Mirrors what the deployment already runs: quiz generation on DeepSeek
    # (QUIZ_LLM_* in compose), everything else on Gemini.
    QUIZ_GENERATION: "deepseek-chat",
    # The judge must not be the model under judgement -- a model that wrote a
    # bad question tends to accept it. See docs/quiz-generation-redesign.md.
    QUIZ_VERIFICATION: "gemini-flash",
    QUIZ_REGENERATION: "deepseek-chat",
    ENRICHMENT: "gemini-flash",
    TUTOR: "gemini-flash",
}

DEFAULT_PROFILE = "gemini-flash"


def known_profile_names(profiles: dict[str, ModelProfile] | None = None) -> str:
    """Comma-separated selectable names, for error messages worth reading."""
    catalog = profiles if profiles is not None else PROFILES
    return ", ".join(sorted(name for name, p in catalog.items() if not p.internal))


__all__ = [
    "DEFAULT_PROFILE",
    "ENRICHMENT",
    "PROFILES",
    "Provider",
    "QUIZ_GENERATION",
    "QUIZ_REGENERATION",
    "QUIZ_VERIFICATION",
    "TASK_DEFAULTS",
    "TUTOR",
    "ModelProfile",
    "known_profile_names",
]
