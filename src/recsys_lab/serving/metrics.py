from __future__ import annotations

from collections import Counter, deque
from threading import Lock
from time import monotonic
from typing import Any


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


class RuntimeMetrics:
    """Thread-safe in-process metrics for a single API worker.

    M9 intentionally avoids an external metrics dependency. The /metrics endpoint emits a small
    Prometheus-compatible text surface that can later be scraped by Prometheus/OpenTelemetry.
    """

    def __init__(self, latency_window: int = 5000) -> None:
        if latency_window <= 0:
            raise ValueError("latency_window must be > 0")
        self.started = monotonic()
        self._lock = Lock()
        self._recommend_total = 0
        self._recommend_errors = 0
        self._outcomes = Counter()
        self._routes = Counter()
        self._variant_pairs = Counter()
        self._latencies: deque[float] = deque(maxlen=int(latency_window))

    def observe_recommend(
        self,
        latency_ms: float,
        route: str,
        assigned_variant: str,
        served_variant: str,
        error: bool = False,
    ) -> None:
        with self._lock:
            self._recommend_total += 1
            self._recommend_errors += int(bool(error))
            self._routes[str(route)] += 1
            self._variant_pairs[(str(assigned_variant), str(served_variant))] += 1
            self._latencies.append(float(latency_ms))

    def observe_outcome(self, status: str, reason: str | None = None) -> None:
        key = (str(status), str(reason or "none"))
        with self._lock:
            self._outcomes[key] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = list(self._latencies)
            total = self._recommend_total
            errors = self._recommend_errors
            routes = dict(self._routes)
            variants = {f"{a}->{s}": n for (a, s), n in self._variant_pairs.items()}
            outcomes = {f"{s}:{r}": n for (s, r), n in self._outcomes.items()}
        return {
            "uptime_seconds": max(0.0, monotonic() - self.started),
            "recommend_requests_total": total,
            "recommend_errors_total": errors,
            "recommend_error_rate": (errors / total) if total else 0.0,
            "latency_window_count": len(latencies),
            "latency_ms": {
                "p50": _quantile(latencies, 0.50),
                "p95": _quantile(latencies, 0.95),
                "p99": _quantile(latencies, 0.99),
                "mean": (sum(latencies) / len(latencies)) if latencies else 0.0,
            },
            "routes": routes,
            "variant_pairs": variants,
            "outcomes": outcomes,
        }

    def prometheus_text(self, integrity_ok: bool, challenger_eligible: bool) -> str:
        snap = self.snapshot()
        lines = [
            "# HELP recsys_up Service uptime in seconds.",
            "# TYPE recsys_up gauge",
            "recsys_up 1",
            "# HELP recsys_uptime_seconds Service uptime in seconds.",
            "# TYPE recsys_uptime_seconds gauge",
            f"recsys_uptime_seconds {snap['uptime_seconds']:.6f}",
            "# HELP recsys_artifact_integrity_ok Whether the serving artifact passed integrity checks.",
            "# TYPE recsys_artifact_integrity_ok gauge",
            f"recsys_artifact_integrity_ok {1 if integrity_ok else 0}",
            "# HELP recsys_challenger_offline_eligible Whether treatment may be served.",
            "# TYPE recsys_challenger_offline_eligible gauge",
            f"recsys_challenger_offline_eligible {1 if challenger_eligible else 0}",
            "# HELP recsys_recommend_requests_total Recommendation requests observed.",
            "# TYPE recsys_recommend_requests_total counter",
            f"recsys_recommend_requests_total {snap['recommend_requests_total']}",
            "# HELP recsys_recommend_errors_total Recommendation requests that raised an error.",
            "# TYPE recsys_recommend_errors_total counter",
            f"recsys_recommend_errors_total {snap['recommend_errors_total']}",
            "# HELP recsys_recommend_latency_ms Recent recommendation latency quantiles.",
            "# TYPE recsys_recommend_latency_ms gauge",
        ]
        for q in ("p50", "p95", "p99", "mean"):
            lines.append(f'recsys_recommend_latency_ms{{quantile="{q}"}} {snap["latency_ms"][q]:.6f}')
        for route, count in sorted(snap["routes"].items()):
            safe = str(route).replace('"', "'")
            lines.append(f'recsys_recommend_route_total{{route="{safe}"}} {count}')
        for pair, count in sorted(snap["variant_pairs"].items()):
            assigned, served = pair.split("->", 1)
            lines.append(
                f'recsys_variant_routing_total{{assigned="{assigned}",served="{served}"}} {count}'
            )
        for key, count in sorted(snap["outcomes"].items()):
            status, reason = key.split(":", 1)
            lines.append(f'recsys_outcomes_total{{status="{status}",reason="{reason}"}} {count}')
        return "\n".join(lines) + "\n"
