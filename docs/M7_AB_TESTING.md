# M7 — A/B Experimentation

## Why M7 is not an offline MovieLens A/B test

MovieLens 1M contains historical user ratings. It does not contain randomized recommendation
exposures, treatment assignment, impression IDs, clicks caused by a served ranking, service latency,
or experiment guardrails. Treating ratings as if they were randomized online outcomes would create
an invalid causal claim.

M7 therefore has two separate contracts:

1. **Production experiment contract** — defines assignment, variants and the currently deployable
   recommendation policy. It can read the local M6 deployment-gate manifest.
2. **Deterministic synthetic experiment** — exercises the logging and statistics code end to end.
   It is always marked `demo_only=true`; its lift is not a recommender-quality result.

## Randomization and serving contract

Assignment happens at user level using the existing deterministic SHA-256 bucket. Repeated
impressions for the same `(experiment, user_id)` must therefore carry the same variant. M7 validates
logged assignments against the hash function itself; a modified or inconsistent variant fails the
event contract.

In M8 the recommendation service will use the assignment before policy selection:

```text
request(user_id)
  -> assign experiment arm
  -> choose control/treatment policy
  -> generate/rank/rerank recommendations
  -> write exposure event with experiment + variant + model_version
```

## Event contract

Exposure rows contain:

- globally unique event ID;
- unique impression ID;
- experiment and user ID;
- variant and model version;
- timestamp;
- latency and error indicator;
- pre-period activity covariate for the demo.

Outcome rows contain:

- globally unique event ID;
- impression ID that joins to exactly one exposure;
- experiment, user and variant;
- timestamp no earlier than the exposure;
- click, conversion and non-negative value.

The validator also checks sticky assignment, one model version per arm, binary outcomes and
non-negative service/business fields.

## Statistical unit

The user is the randomization unit, so primary inference is also performed at user level. Multiple
impressions are first aggregated per user. This avoids pretending that repeated impressions from the
same randomized user are independent samples.

The primary metric in the demo is:

```text
engaged_user_rate = users with >= 1 click / exposed users
```

Impression-level CTR is still useful operationally and is reported descriptively, but it is not used
as the naive sample size for the primary hypothesis test.

## SRM

Sample-ratio mismatch compares observed control/treatment user counts with the configured assignment
weights. The default SRM threshold is deliberately stricter than the primary hypothesis threshold:

```text
srm_alpha = 0.001
```

An SRM failure is a data-quality blocker and produces `no_ship` regardless of apparent treatment
lift.

## Confidence intervals and practical significance

M7 reports treatment minus control absolute lift, relative lift, z statistic, p-value and a two-sided
confidence interval. Shipping requires both statistical evidence and a configured minimum practical
relative lift; a tiny statistically significant change is not automatically accepted.

## Power and MDE

Before an experiment, the baseline rate and target relative MDE determine the approximate required
users per arm for the configured alpha and power. After the run, M7 also reports the relative MDE
supported by the achieved sample size.

`reports/m7/power_curve.csv` and `.png` show required users per arm for several relative MDEs. This
makes the experiment-size trade-off visible rather than hiding it in a single test result.

## CUPED

M7 includes CUPED as a secondary variance-reduction example. A pre-period activity covariate is used
to adjust clicks per user:

```text
Y_cuped = Y - theta * (X - mean(X))
theta   = Cov(Y, X) / Var(X)
```

The report includes theta and observed variance reduction. CUPED does not replace randomization and
is not used to manufacture a treatment effect; it is a precision technique when a valid pre-period
covariate is available.

## Guardrails

The demo tracks two service guardrails:

- mean latency increase in milliseconds;
- absolute error-rate increase.

A positive primary metric cannot ship if either configured guardrail is violated.

## Decision state machine

```text
Data/event contract invalid -> fail pipeline
SRM failure                -> no_ship
Guardrail failure          -> no_ship
Significant harmful effect -> no_ship
Significant + practical + sufficiently powered positive effect -> ship
Otherwise                  -> inconclusive
```

This decision is intentionally conservative. In a real product additional guardrails and sequential
monitoring rules would normally be pre-registered before launch.

## Relation to the current M6 result

M6 has an offline deployment gate. If its diversification challenger fails that gate, M7 records the
challenger as offline-ineligible for a real treatment launch. The synthetic M7 treatment remains
available only because its purpose is to test experiment infrastructure, not to bypass the M6 gate.
