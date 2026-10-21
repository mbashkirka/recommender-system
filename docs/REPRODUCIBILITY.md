# Reproducibility

## Data

MovieLens 1M is downloaded locally from the configured GroupLens URL and checked against the pinned MD5. The dataset is intentionally not redistributed in release archives.

## Evaluation discipline

- M1 creates a global chronological 80/10/10 split.
- M2–M6 use train/validation only for development and model selection.
- M7 is a synthetic online-experiment harness and does not use test.
- M8/M9 serving artifacts use train-derived state and frozen validation-selected parameters.
- M10 unseals test only after the serving policy is fixed.

## Frozen serving artifact

M8 records the SHA-256 of `artifacts/m8/serving_bundle.npz`. M9 verifies that fingerprint before readiness. M10 evaluates this same fingerprint on test; it does not silently rebuild a different model.

## Recommended reproduction sequence

```bash
python -m recsys_lab smoke
python -m recsys_lab m1
python -m recsys_lab m2
python -m recsys_lab m3
python -m recsys_lab m4
python -m recsys_lab m5
python -m recsys_lab m6
python -m recsys_lab m7
python -m recsys_lab m8
python -m recsys_lab m9
python -m recsys_lab m10
```

After the first successful M10 run, rerunning M10 uses the cached final-test result in `artifacts/m10/m10_manifest.json` instead of reopening test.
