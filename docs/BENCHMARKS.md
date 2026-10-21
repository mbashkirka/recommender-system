# Benchmarks

## Confirmed validation development metrics

| Metric | Popularity | M3 PureSVD | Lift |
|---|---:|---:|---:|
| NDCG@10 | 0.126124 | 0.134151 | +6.36% |
| Recall@10 | 0.047749 | 0.072699 | +52.25% |
| Candidate Recall@100 | 0.295857 | 0.324470 | +9.67% |

M4 selected metadata blend `0.0`. M5 selected LTR blend `0.0`. M6's development diversity challenger failed the later relevance deployment gate and was rolled back to the exact anchor.

## Local HTTP load test

Completed local run, 500 requests at concurrency 20:

| Metric | Result |
|---|---:|
| Success rate | 100% |
| Throughput | 431.6 req/s |
| Mean HTTP latency | 44.5 ms |
| p50 | 36.4 ms |
| p95 | 105.1 ms |
| p99 | 144.3 ms |
| Max | 173.8 ms |

These figures are environment-specific and should be presented as a local laptop benchmark, not a universal service SLO.
