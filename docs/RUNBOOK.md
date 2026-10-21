# Operations Runbook

## Local service

```bash
source .venv/bin/activate
python -m recsys_lab serve --host 127.0.0.1 --port 8000
```

Validate:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/v1/model-info
curl http://127.0.0.1:8000/v1/ops
```

## Recommendation smoke

```bash
curl -X POST http://127.0.0.1:8000/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 635, "k": 10}'
```

Inspect exposure logging:

```bash
tail -5 data/online/exposures.jsonl
```

## Reliability preflight

```bash
python -m recsys_lab m9
```

A failure is release-blocking. Investigate the specific check in `reports/m9/operational_preflight.json`.

## Load test

With the API running:

```bash
python scripts/load_test.py --requests 500 --concurrency 20
```

Do not compare the in-process M9 benchmark directly with the HTTP benchmark. The latter includes transport, serialization, event logging and concurrency scheduling.

## Docker

```bash
docker build -t recsys-lab:1.0.2 .
docker run --rm \
  -p 8000:8000 \
  -v "$PWD/data/online:/app/data/online" \
  recsys-lab:1.0.2
```

If startup fails, check these first:

1. `RECSYS_CONFIG=/app/config/project.toml` is present in the image environment.
2. `/app/artifacts/m8/serving_bundle.npz` exists.
3. `/app/artifacts/m8/m8_manifest.json` exists.
4. Bundle SHA-256 equals the manifest fingerprint.

## Incident: `/health` is 200 but `/ready` is 503

This is expected fail-closed behavior. The process is alive but must not receive recommendation traffic. Inspect `/v1/ops` or logs for artifact integrity/provenance failure.

## Incident: assigned treatment but served control

Check `challenger_offline_eligible`. This is expected when the offline deployment gate rejected the treatment. Never modify assignment logs to hide this distinction.

## Outcome errors

- `404 unknown_impression`: no matching exposure exists.
- `409 user_mismatch`: the outcome user does not match the exposure owner.

Both protect experiment integrity.
