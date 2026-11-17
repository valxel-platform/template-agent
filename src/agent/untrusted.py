"""Outside text in the model's context, marked as data.

Tool results (records, search passages from the knowledge service, a server's error text) are
written by other people and other systems. They go to the model inside an <untrusted> block whose
source names where they came from; the system prompt says text inside such a block is data to use,
never instructions to follow. Tags inside the text are defused, so it can't close the block early or
open one that looks like ours.
"""

from __future__ import annotations

import re

_OPEN = re.compile(r"<\s*untrusted", re.I)
_CLOSE = re.compile(r"<\s*/\s*untrusted", re.I)


def wrap(source: str, text: str) -> str:
    body = _OPEN.sub("&lt;untrusted", _CLOSE.sub("&lt;/untrusted", str(text or "")))
    src = re.sub(r"[^a-z0-9_.:-]", "-", source.lower())
    return f'<untrusted source="{src}">\n{body}\n</untrusted>'
