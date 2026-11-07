"""Feature flags through OpenFeature.

The Vrixel SDLC tool gives each feature a flag (a key like `emr.emr-feat-3.referral-triage`) in
the product's Flipt when its design is approved: on in dev and staging, off in production until
the feature is released. Put the feature's new behavior behind it:

    from agent import flags

    if flags.enabled("emr.emr-feat-3.referral-triage", user=claims.get("sub", "")):
        ...  # the new behavior
    else:
        ...  # today's behavior

Flags are read from the product's Flipt over OFREP (the OpenFeature Remote Evaluation
Protocol). With no flag server configured every flag reads as its default, so a new feature stays
off. Tests can swap in an in-memory provider with `use_provider`.
"""

from __future__ import annotations

import logging
from typing import Any

from openfeature import api
from openfeature.evaluation_context import EvaluationContext
from openfeature.provider import FeatureProvider

from . import cohorts
from .settings import Settings

log = logging.getLogger("agent.flags")


def configure(settings: Settings) -> None:
    """Point OpenFeature at the product's Flipt, or leave flags at their defaults."""
    if not settings.flags_url:
        log.info("AGENT_FLAGS_URL is not set: every feature flag reads as its default (off)")
        return
    from openfeature.contrib.provider.ofrep import OFREPProvider

    headers = {}
    if settings.flags_environment:
        headers["x-flipt-environment"] = settings.flags_environment
    if settings.flags_namespace:
        headers["x-flipt-namespace"] = settings.flags_namespace
    if settings.flags_token:
        headers["Authorization"] = f"Bearer {settings.flags_token}"
    use_provider(OFREPProvider(settings.flags_url, headers_factory=lambda: dict(headers), timeout=2.0))


def use_provider(provider: FeatureProvider) -> None:
    api.set_provider(provider)


def enabled(key: str, default: bool = False, *, user: str = "", **attributes: Any) -> bool:
    """The flag's value for this caller. Any error (server down, unknown flag) gives `default`.
    Pass the user: a gradual release turns a feature on for a share of users, and each keeps the
    same answer. The request counts toward the flag's on or off side (`cohorts`)."""
    context = EvaluationContext(targeting_key=user or None, attributes=attributes)
    on = bool(api.get_client().get_boolean_value(key, default, context))
    cohorts.seen(key, on)
    return on
