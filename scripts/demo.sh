#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

echo "== ready =="
curl -fsS "$BASE_URL/ready"
echo

echo "== model info =="
curl -fsS "$BASE_URL/v1/model-info"
echo

echo "== warm-user recommendation =="
curl -fsS -X POST "$BASE_URL/v1/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id":635,"k":10}'
echo

echo "== metrics =="
curl -fsS "$BASE_URL/metrics"
