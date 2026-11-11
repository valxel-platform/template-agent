"""Product events: the things a feature's goal counts, for the SDLC tool's outcome loop.

The feature's PRD names them (its Instrumentation section) and the HLD says which repo records
each. Record them where they happen:

    from agent import events

    events.record("referral_triaged")
    events.record("referral_triaged", urgency="routine")  # categories only

Each event goes three places, all counts:

- the Prometheus counter `vrixel_events_total{event="referral_triaged"}` on /metrics, which a
  goal's metric can read (`sum(increase(vrixel_events_total{event="referral_triaged"}[$window]))`);
- a summary line once a minute, which the tool counts in staging to prove every event fires
  before release: `vrixel.events {"referral_triaged":12}`;
- the current request's flag cohorts (`cohorts.event`), so a gradual release can compare how
  often users with a feature on do it against users with it off.

Events carry counts and categories, never people: a name that isn't lowercase words, or an
attribute that names or identifies a person (a name, contact details, an id), raises ValueError.
"""

from __future__ import annotations

import atexit
import json
import logging
import re
import threading
from collections.abc import Iterator
from typing import Any

from . import cohorts

MARKER = "vrixel.events"
METRIC = "vrixel_events"
log = logging.getLogger("agent.events")
NAME = re.compile(r"[a-z][a-z0-9_]*(\.[a-z0-9_]+)*")
_ATTRIBUTE = re.compile(r"[a-z][a-z0-9_]*")
PERSONAL = frozenset(
    {
        "name",
        "first_name",
        "last_name",
        "full_name",
        "display_name",
        "username",
        "user_name",
        "email",
        "email_address",
        "phone",
        "phone_number",
        "mobile",
        "dob",
        "date_of_birth",
        "birth_date",
        "birthdate",
        "ssn",
        "social_security_number",
        "mrn",
        "address",
        "street",
        "street_address",
        "zip",
        "zipcode",
        "zip_code",
        "postcode",
        "postal_code",
        "ip",
        "ip_address",
        "card",
        "card_number",
        "pan",
        "cvv",
        "cvc",
        "iban",
        "account_number",
        "routing_number",
        "passport",
        "passport_number",
        "license_number",
        "drivers_license",
        "lat",
        "lng",
        "latitude",
        "longitude",
        "geo",
        "location",
    }
)
PERSON_IDS = (
    "patient",
    "member",
    "beneficiary",
    "subscriber",
    "cardholder",
    "user",
    "customer",
    "guest",
    "client",
    "person",
    "account",
)
_lock = threading.Lock()
_counts: dict[str, int] = {}  # since the last summary line
_totals: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}  # for /metrics, never reset
_reporter: threading.Thread | None = None


def personal(key: str) -> bool:
    if key in PERSONAL or key.endswith(("_email", "_phone", "_dob", "_ssn", "_address", "_card_number")):
        return True
    if key.endswith("_name") and (
        key.startswith(("first", "last", "full")) or any(key.startswith(w + "_") for w in PERSON_IDS)
    ):
        return True
    return any(key in (w, f"{w}_id", f"{w}_ids", f"{w}_number") for w in PERSON_IDS)


def check(event: str, attributes: dict[str, str]) -> None:
    if not isinstance(event, str) or len(event) > 100 or not NAME.fullmatch(event):
        raise ValueError(
            "Event names are lowercase words joined by _ or . (referral_triaged), as the PRD's "
            f"Instrumentation section names them: {event!r}"
        )
    for key, value in attributes.items():
        if not _ATTRIBUTE.fullmatch(key) or key == "event":
            raise ValueError(f"{event}: attribute {key!r} must be a plain name")
        if personal(key):
            raise ValueError(
                f"{event}: attribute {key} names or identifies a person; events carry counts and "
                "categories, never people"
            )
        if not isinstance(value, str) or not value or len(value) > 64:
            raise ValueError(f"{event}: attribute {key} needs a short category value")


def record(event: str, **attributes: str) -> None:
    """Something happened that the feature's goal counts."""
    check(event, attributes)
    with _lock:
        _counts[event] = _counts.get(event, 0) + 1
        k = (event, tuple(sorted(attributes.items())))
        _totals[k] = _totals.get(k, 0) + 1
    cohorts.event(event)


def drain() -> list[str]:
    """The summary line since the last call (none when nothing happened), and the counts start again."""
    with _lock:
        got = dict(sorted(_counts.items()))
        _counts.clear()
    return [f"{MARKER} {json.dumps(got, separators=(',', ':'))}"] if got else []


def report() -> None:
    for line in drain():
        log.info(line)


def start_reporting(period_s: float = 60.0) -> None:
    """Log the summary line once a minute, on a daemon thread, and once more at exit."""
    global _reporter
    if _reporter is not None:
        return
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(period_s):
            try:
                report()
            except Exception as exc:  # noqa: BLE001 - a report must never stop the reporter thread
                log.warning("product events not reported: %s", exc)

    _reporter = threading.Thread(target=loop, name="product-events", daemon=True)
    _reporter.start()
    atexit.register(report)


class Collector:
    """The counter on /metrics: one sample per event and set of categories."""

    def collect(self) -> Iterator[Any]:
        from prometheus_client.core import Metric

        m = Metric(METRIC, "Product events the feature goals count", "counter")
        with _lock:
            totals = dict(_totals)
        for (event, attrs), n in sorted(totals.items()):
            m.add_sample(f"{METRIC}_total", {"event": event, **dict(attrs)}, float(n))
        yield m

    def describe(self) -> list[Any]:
        return []


def register(registry: Any) -> None:
    """Expose the counter on a prometheus_client registry (the app's /metrics)."""
    registry.register(Collector())


def reset() -> None:
    """Tests: forget every count."""
    with _lock:
        _counts.clear()
        _totals.clear()
