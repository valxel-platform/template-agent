"""Prompt and model variants, rolled out through the product's Flipt instead of a redeploy.

`variants.toml` lists the variants the repo has reviewed (a model and the prompts for runs and
conversations). Which one a run uses is a flag: `<agent name>.variant`, a Flipt variant flag whose
variants are named like the tables in `variants.toml`. Flipt's rules decide who gets which, and the
answer depends only on the person (or the triggering event), so each keeps theirs. A run picks its
variant once, when it starts, and keeps it to the end, across replicas and restarts.

Every model call carries the variant: on the `chat` span (`vrixel.variant`), in the gateway's labels
(so its ledger and spans show what each variant costs), and in the run's trace, so the evals, the
post-deploy evals and production traces compare variants on the same terms.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from importlib.resources import files

from . import flags

log = logging.getLogger("agent.variants")

DEFAULT = "default"


@dataclass(frozen=True)
class Variant:
    name: str
    model: str = ""  # "" = the agent's configured model
    system: str = "system"
    chat: str = "chat"

    def prompt(self, mode: str) -> str:
        return self.chat if mode == "conversation" else self.system


def load(text: str | None = None) -> dict[str, Variant]:
    """The reviewed variants; `default` must be there, and every prompt they name must exist."""
    raw = tomllib.loads(text if text is not None else files("agent").joinpath("variants.toml").read_text())
    out = {
        name: Variant(
            name=name,
            model=str(v.get("model", "")),
            system=str(v.get("system", "system")),
            chat=str(v.get("chat", "chat")),
        )
        for name, v in raw.items()
        if isinstance(v, dict)
    }
    if DEFAULT not in out:
        raise ValueError("variants.toml needs a [default] variant")
    prompts = files("agent").joinpath("prompts")
    for v in out.values():
        for p in (v.system, v.chat):
            if not prompts.joinpath(f"{p}.md").is_file():
                raise ValueError(f"variant {v.name} names prompts/{p}.md, which doesn't exist")
    return out


def pick(agent: str, known: dict[str, Variant], *, user: str = "", **attributes: object) -> str:
    """The variant Flipt gives this caller; `default` when there's no flag or it names an unknown one."""
    name = flags.variant(f"{agent}.variant", DEFAULT, user=user, **attributes)
    if name not in known:
        if name != DEFAULT:
            log.warning(
                "flag %s.variant names %r, which variants.toml doesn't have: using default", agent, name
            )
        return DEFAULT
    return name
