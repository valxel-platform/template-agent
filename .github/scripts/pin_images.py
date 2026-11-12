#!/usr/bin/env python3
"""Pin every Dockerfile's base images to the digest their tag points at today (FROM image:tag ->
FROM image:tag@sha256:...). Dependabot then moves tag and digest together, by pull request.
Stage names (FROM build), ARG-built names and scratch are left alone."""

import json
import re
import subprocess
import sys
from pathlib import Path

FROM = re.compile(r"^(FROM\s+(?:--platform=\S+\s+)?)(\S+)(.*)$", re.I)


def digest(image: str) -> str:
    out = subprocess.run(["docker", "buildx", "imagetools", "inspect", image, "--format", "{{json .Manifest}}"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)["digest"]


def pin(path: Path) -> bool:
    lines = path.read_text().splitlines(keepends=True)
    stages: set[str] = set()
    changed = False
    for i, line in enumerate(lines):
        m = FROM.match(line.rstrip("\n"))
        if not m:
            continue
        image, rest = m.group(2), m.group(3)
        alias = re.search(r"\bAS\s+(\S+)", rest, re.I)
        if image.lower() not in stages and image != "scratch" and "$" not in image and "@sha256:" not in image:
            lines[i] = f"{m.group(1)}{image}@{digest(image)}{rest}\n"
            changed = True
        if alias:
            stages.add(alias.group(1).lower())
    if changed:
        path.write_text("".join(lines))
    return changed


if __name__ == "__main__":
    files = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True).stdout.split()
    done = [f for f in files if re.search(r"(^|/)Dockerfile[^/]*$", f) and pin(Path(f))]
    print("\n".join(f"pinned {f}" for f in done) or "every base image is pinned")
    sys.exit(0)
