# tripps — cheapest multimodal travel routing within Sweden.
#
# The data dir is chosen by config._default_data_dir(), and TRIPPS_DATA_DIR below settles it
# outright — so nothing depends on where the package itself ends up. The editable install is
# kept only because it makes the layer cheap to rebuild during development.
FROM python:3.12-slim

ENV TRIPPS_DATA_DIR=/app/data \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first (their own layer), then the source. `primp` — the Tora rail adapter's
# TLS-fingerprint-impersonating client — is a core dependency, so the base install already
# prices rail; the `[flights]` extra adds only the Google Flights scrape.
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
