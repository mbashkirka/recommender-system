#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from time import perf_counter

import httpx


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


async def main_async(args: argparse.Namespace) -> int:
    semaphore = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    status_counts: dict[int, int] = {}
    routes: dict[str, int] = {}

    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout) as client:
        async def one(index: int) -> None:
            user_id = args.user_start + (index % args.user_span)
            async with semaphore:
                started = perf_counter()
                try:
                    response = await client.post(
                        "/v1/recommend",
                        json={"user_id": user_id, "k": args.k},
                    )
                    elapsed = (perf_counter() - started) * 1000.0
                    latencies.append(elapsed)
                    status_counts[response.status_code] = status_counts.get(response.status_code, 0) + 1
                    if response.status_code == 200:
                        route = str(response.json().get("route", "unknown"))
                        routes[route] = routes.get(route, 0) + 1
                except httpx.HTTPError:
                    elapsed = (perf_counter() - started) * 1000.0
                    latencies.append(elapsed)
                    status_counts[0] = status_counts.get(0, 0) + 1

        wall_start = perf_counter()
        await asyncio.gather(*(one(i) for i in range(args.requests)))
        wall = max(perf_counter() - wall_start, 1e-12)

    ok = status_counts.get(200, 0)
    summary = {
        "url": args.url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "top_k": args.k,
        "wall_seconds": wall,
        "throughput_rps": args.requests / wall,
        "success_rate": ok / args.requests,
        "status_counts": status_counts,
        "routes": routes,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p50": quantile(latencies, 0.50),
            "p95": quantile(latencies, 0.95),
            "p99": quantile(latencies, 0.99),
            "max": max(latencies),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0 if ok == args.requests else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent HTTP load test for Recsys Lab M9")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--user-start", type=int, default=1)
    parser.add_argument("--user-span", type=int, default=7000)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0 or args.k <= 0 or args.user_span <= 0:
        parser.error("requests, concurrency, k and user-span must be > 0")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
