# Multi-stage build. The builder resolves dependencies into a virtualenv and
# the final image copies only that venv, so pip, build tooling, and the wheel
# cache never reach the shipped image.

FROM python:3.12-slim AS builder

# Compilers and headers live only in this stage. scikit-learn and pandas ship
# manylinux wheels for 3.12, so nothing should need to build from source, but a
# transitive dependency without a wheel would otherwise fail the build here
# rather than silently producing a broken image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Only what the install needs. The dev extra is deliberately omitted: pytest,
# ruff and httpx exist for CI against the source tree, not for the runtime.
COPY pyproject.toml ./
COPY api ./api
COPY model ./model

RUN pip install .


FROM python:3.12-slim AS runtime

# Unbuffered so log lines reach the collector immediately instead of sitting in
# a buffer, which matters when the process is killed mid-request.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Runs unprivileged. A container that only reads its own weights and answers
# HTTP has no reason to be root, and this is the cheapest possible mitigation
# if the process is ever compromised.
RUN useradd --create-home --uid 10001 clutch

COPY --from=builder /opt/venv /opt/venv

USER clutch
WORKDIR /home/clutch

EXPOSE 8000

# Uses the same /health endpoint the load balancer will target, so a container
# that cannot load its model fails here rather than being handed traffic.
# urllib avoids installing curl purely to make a health check possible.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", \
         "import urllib.request as u, sys; sys.exit(0 if u.urlopen('http://localhost:8000/health').status == 200 else 1)"]

# Single worker per container. Fargate scales by running more tasks, so adding
# in-container workers would just make one task's resource usage harder to
# reason about.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
