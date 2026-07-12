# tripps — cheapest multimodal travel routing within Sweden.
#
# Installed EDITABLE on purpose: config.PROJECT_ROOT is `parents[2]` of the package, so an
# editable install keeps it resolving to /app (and the data dir to /app/data). A regular
# `pip install .` would relocate the package into site-packages and break that path.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first (their own layer), then the source. The `[flights]` extra pulls
# fast-flights, which transitively provides `primp` — needed for the Tora rail adapter's
# TLS-fingerprint-impersonating client, so without it Tora and flights are down.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[flights]" \
    && useradd --create-home app \
    && mkdir -p /app/data \
    && chown -R app /app

COPY --chown=app:app docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

USER app
EXPOSE 8000

# The lifespan warmup parses the ~500 MB feed (~15 s) and warms Freerider + the SJ key, so
# give the container a generous start period before the healthcheck counts failures.
HEALTHCHECK --start-period=120s --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status == 200 else 1)"

ENTRYPOINT ["./docker-entrypoint.sh"]
