# syntax=docker/dockerfile:1
# uv builds a locked virtualenv; the runtime image has Python and the venv only, as a non-root user.
# Both stages use the same Python image: the venv is built for that exact Python, so the build and
# runtime images must move together (Dependabot bumps both FROM lines in one pull request, and CI
# starts the image to prove it runs).
FROM python:3.13-slim-bookworm@sha256:f0fa218d5b06f9b9235024fa061cde2cb141f726de52134471053d2d9b5324a3 AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.32@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PYTHON=/usr/local/bin/python3
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src src
COPY README.md ./
RUN uv sync --locked --no-dev --no-editable

FROM python:3.13-slim-bookworm@sha256:f0fa218d5b06f9b9235024fa061cde2cb141f726de52134471053d2d9b5324a3
# libpam's CVE-2025-6020 fix (Debian 1.5.2-6+deb12u3) isn't in the upstream image yet: take it here
# until Dependabot's next base-image bump includes it, then drop this line.
RUN apt-get update && apt-get install -y --no-install-recommends --only-upgrade \
        libpam0g libpam-modules libpam-modules-bin libpam-runtime \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER 10001
EXPOSE 8080 8081
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/healthz')"
CMD ["agent"]
