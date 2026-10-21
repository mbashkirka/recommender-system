# M9.1 — Docker configuration-path hotfix

M9.1 fixes a packaging/runtime-layout bug discovered by running the built Docker image.

## Root cause

The original `project_root()` inferred the repository root from `recsys_lab/config.py`. That is valid
for an editable source checkout, but after `pip install .` inside the image the module lives under
`/usr/local/lib/python3.12/site-packages/recsys_lab/`. The inferred root therefore became
`/usr/local/lib/python3.12`, while the image intentionally stores runtime configuration under
`/app/config/project.toml`.

## Resolution contract

Configuration is now resolved in this order:

1. `RECSYS_CONFIG` — explicit production/container path;
2. `./config/project.toml` — current working directory, useful for installed wheels launched from a project root;
3. source-checkout fallback relative to the Python module.

The Docker image explicitly sets:

```text
RECSYS_CONFIG=/app/config/project.toml
```

All data/artifact/report paths are rooted at the directory containing that `config/` directory, so a
container resolves artifacts at `/app/artifacts` and online events at `/app/data/online`.

## Regression coverage

Tests cover both explicit environment-based resolution and an installed-wheel style current-working-directory layout.
The M9 preflight also verifies that the Dockerfile contains the explicit runtime config contract.
