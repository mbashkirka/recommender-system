# M9 — Production Reliability

M9 hardens the frozen M8 serving path without retraining or retuning the recommender.

## Reliability contract

The production service fails closed when the bytes of `artifacts/m8/serving_bundle.npz` do not
match the SHA-256 recorded in `artifacts/m8/m8_manifest.json`. `/health` remains a liveness probe,
while `/ready` becomes a strict readiness probe and returns HTTP 503 on integrity failure.

The service still consumes train-derived serving artifacts only. M9 does not access validation or
test data.

## Observability

`GET /metrics` exposes a dependency-free Prometheus-compatible surface containing:

- recommendation request/error counters;
- p50/p95/p99/mean recent recommendation latency;
- route counts (`warm_user_hybrid`, `sparse_user_hybrid`, `cold_user_popularity`);
- assigned-to-served A/B routing counts;
- accepted/rejected outcome counters;
- artifact-integrity and challenger-eligibility gauges.

`GET /v1/ops` returns the same runtime state as structured JSON together with the integrity report.

## Operational preflight

`python -m recsys_lab m9` performs:

1. M8 artifact SHA-256 verification;
2. train-only provenance checks from the M8 manifest;
3. mixed warm/cold in-process inference benchmark;
4. p95/p99 latency gates;
5. zero-error and non-empty-response gates;
6. minimum throughput gate;
7. local online-event-store writability check;
8. Dockerfile and CI workflow presence checks.

Outputs are written to `reports/m9/` and `artifacts/m9/m9_manifest.json`.

## HTTP load test

With the API already running:

```bash
python scripts/load_test.py --requests 500 --concurrency 20
```

This is deliberately separate from the deterministic M9 preflight: HTTP load depends on the host,
BLAS implementation, container/runtime configuration, and concurrent traffic.

## Container

Build M8 first, then build the image so the frozen serving bundle is copied into it:

```bash
python -m recsys_lab m8
docker build -t recsys-lab:m9 .
docker run --rm -p 8000:8000 -v "$PWD/data/online:/app/data/online" recsys-lab:m9
```

Or use `docker compose up --build`.

## CI

`.github/workflows/ci.yml` installs the package on Python 3.12, runs Ruff, the full unit/integration
test suite, and byte-compiles the source tree. It intentionally does not require MovieLens or model
artifacts, so pull-request CI is deterministic and lightweight.
