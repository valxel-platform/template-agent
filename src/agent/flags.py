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

Each answer is reused for `flags_cache_s` (30 s) per flag, caller and attributes, and a call to
Flipt waits at most `flags_timeout_s` (250 ms): flags never add a network round trip to most
steps, and a slow Flipt costs a step at most a quarter of a second (it then reads the default).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from openfeature import api
from openfeature.evaluation_context import EvaluationContext
from openfeature.provider import FeatureProvider

from . import cohorts
from .settings import Settings

log = logging.getLogger("agent.flags")

_CACHE_MAX = 10_000


class _Answers:
    """Recent answers, per flag, caller and attributes, for a few seconds (bounded, oldest out)."""

    def __init__(self) -> None:
        self.ttl = 0.0
        self.items: OrderedDict[tuple[Any, ...], tuple[bool, float]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: tuple[Any, ...]) -> bool | None:
        if self.ttl <= 0:
            return None
        with self.lock:
            hit = self.items.get(key)
            if hit is None or hit[1] < time.monotonic():
                return None
            return hit[0]

    def put(self, key: tuple[Any, ...], value: bool) -> None:
        if self.ttl <= 0:
            return
        with self.lock:
            self.items[key] = (value, time.monotonic() + self.ttl)
            self.items.move_to_end(key)
            while len(self.items) > _CACHE_MAX:
                self.items.popitem(last=False)

    def clear(self) -> None:
        with self.lock:
            self.items.clear()


_answers = _Answers()


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
    use_provider(
        OFREPProvider(
            settings.flags_url, headers_factory=lambda: dict(headers), timeout=settings.flags_timeout_s
        ),
        cache_s=settings.flags_cache_s,
    )


def use_provider(provider: FeatureProvider, *, cache_s: float = 0.0) -> None:
    """Use this provider; reuse its answers for `cache_s` seconds (0: ask it every time)."""
    _answers.clear()
    _answers.ttl = cache_s
    api.set_provider(provider)


def enabled(key: str, default: bool = False, *, user: str = "", **attributes: Any) -> bool:
    """The flag's value for this caller. Any error (server down, unknown flag) gives `default`.
    Pass the user: a gradual release turns a feature on for a share of users, and each keeps the
    same answer. The request counts toward the flag's on or off side (`cohorts`)."""
    cached_key = (key, default, user, tuple(sorted((k, repr(v)) for k, v in attributes.items())))
    on = _answers.get(cached_key)
    if on is None:
        context = EvaluationContext(targeting_key=user or None, attributes=attributes)
        on = bool(api.get_client().get_boolean_value(key, default, context))
        _answers.put(cached_key, on)
    cohorts.seen(key, on)
    return on
