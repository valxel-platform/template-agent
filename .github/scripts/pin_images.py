#!/usr/bin/env python3
"""Pin every image a Dockerfile pulls to the digest its tag points at today (FROM image:tag ->
FROM image:tag@sha256:..., and the same for COPY --from=image:tag). Dependabot then moves tag and
digest together, by pull request. Stage names (FROM build), ARG-built names and scratch are left
alone. Only the repository's own Dockerfiles are read or written."""

import json
import subprocess
import sys
from pathlib import Path


def digest(image: str) -> str:
    out = subprocess.run(["docker", "buildx", "imagetools", "inspect", image, "--format", "{{json .Manifest}}"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)["digest"]


def _image_at(line: str, keyword: str) -> tuple[int, int] | None:
    """Where the image is in a FROM or COPY line: (start, end) of its text, or None. Words are read
    one by one (no regular expression, so no backtracking on long lines)."""
    words, pos = [], 0
    for word in line.split():
        start = line.index(word, pos)
        words.append((word, start))
        pos = start + len(word)
    if not words or words[0][0].upper() != keyword:
        return None
    if keyword == "FROM":
        rest = [(w, s) for w, s in words[1:] if not w.startswith("--")]
        return (rest[0][1], rest[0][1] + len(rest[0][0])) if rest else None
    for w, s in words[1:]:  # COPY --from=<image>
        if w.lower().startswith("--from="):
            return s + len("--from="), s + len(w)
        if not w.startswith("--"):
            return None
    return None


def pin(path: Path) -> bool:
    lines = path.read_text().splitlines(keepends=True)
    stages: set[str] = set()
    changed = False
    for i, raw in enumerate(lines):
        line = raw.rstrip("\n")
        is_from = _image_at(line, "FROM")
        at = is_from or _image_at(line, "COPY")
        if at is None:
            continue
        image = line[at[0]:at[1]]
        pinned = "@sha256:" in image or "$" in image or image.lower() in stages or image == "scratch"
        if not is_from and (image.isdigit() or (":" not in image and "/" not in image)):
            pinned = True  # COPY --from=<stage> or --from=<index>
        if not pinned:
            lines[i] = f"{line[:at[0]]}{image}@{digest(image)}{line[at[1]:]}\n"
            changed = True
        if is_from:
            words = line[at[1]:].split()
            if len(words) >= 2 and words[0].upper() == "AS":
                stages.add(words[1].lower())
    if changed:
        # NOSONAR: path is a Dockerfile found by walking this repository's own checkout (dockerfiles()), in CI
        path.write_text("".join(lines))  # NOSONAR
    return changed


SKIP = {".git", "node_modules", ".venv", "venv", "build", "dist", ".gradle"}


def dockerfiles(root: Path) -> list[Path]:
    """The repository's Dockerfiles, found by walking it (dependency and build folders skipped)."""
    return sorted(p for p in root.rglob("Dockerfile*")
                  if p.is_file() and not SKIP.intersection(p.relative_to(root).parts[:-1]))


if __name__ == "__main__":
    root = Path.cwd().resolve()
    done = [p.relative_to(root) for p in dockerfiles(root) if pin(p)]
    print("\n".join(f"pinned {p}" for p in done) or "every base image is pinned")
    sys.exit(0)
