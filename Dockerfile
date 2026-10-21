FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RECSYS_CONFIG=/app/config/project.toml

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY config ./config
COPY artifacts/m8 ./artifacts/m8

RUN python -m pip install --upgrade pip && \
    python -m pip install . && \
    test -f /app/artifacts/m8/serving_bundle.npz && \
    test -f /app/artifacts/m8/m8_manifest.json && \
    useradd --create-home --uid 10001 recsys && \
    mkdir -p /app/data/online && chown -R recsys:recsys /app/data

USER recsys
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).read()" || exit 1

CMD ["python", "-m", "recsys_lab", "serve", "--host", "0.0.0.0", "--port", "8000"]
