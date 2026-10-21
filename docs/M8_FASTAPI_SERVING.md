# M8 — FastAPI Serving

M8 converts the frozen offline pipeline into an inference-only online service.

## Artifact rule

`python -m recsys_lab m8` builds `artifacts/m8/serving_bundle.npz` from frozen M4/M5/M6 artifacts, static movie metadata, and **train-only** popularity counts. It does not read validation or test and does not refit/tune any model.

The API loads only the M8 serving bundle for inference. The bundle has a SHA-256 fingerprint exposed by `/ready` and `/v1/model-info` and logged on every exposure.

## Online path

1. sticky SHA-256 user assignment;
2. offline safety gate decides whether treatment is actually eligible;
3. M4 candidate generation (Top-100);
4. M5 selected LTR ordering;
5. M6 deployed or eligible treatment re-ranking;
6. Top-K response with movie title/genres;
7. append-only exposure log;
8. `/v1/outcomes` records click/conversion outcomes.

If the M6 challenger did not pass its offline deployment gate, a user may still have a deterministic `assigned_variant=treatment` for auditability, but `served_variant=control`. This prevents an offline-ineligible policy from leaking into online traffic.

## Endpoints

- `GET /health`
- `GET /ready`
- `GET /v1/model-info`
- `POST /v1/recommend`
- `POST /v1/outcomes`
- FastAPI docs: `/docs`

Local online logs are stored in `data/online/exposures.jsonl` and `data/online/outcomes.jsonl`. M9 will replace this development-oriented persistence layer with monitoring/operational controls.
