#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"

curl --fail --silent --show-error "$BASE_URL/health"
printf '\n'
curl --fail --silent --show-error "$BASE_URL/ready"
printf '\n'
curl --fail --silent --show-error "$BASE_URL/v1/model-info"
printf '\n'
curl --fail --silent --show-error -X POST "$BASE_URL/v1/recommend" \
  -H 'Content-Type: application/json' \
  -d '{"user_id":635,"k":10}'
printf '\n'
curl --fail --silent --show-error "$BASE_URL/metrics" | head -30
