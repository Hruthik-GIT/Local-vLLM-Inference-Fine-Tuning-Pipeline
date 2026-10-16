#!/usr/bin/env bash
# Runs the Locust load test at 5, 10, and 20 concurrent users against the
# local vLLM (or LM Studio) OpenAI-compatible endpoint, and writes CSV
# results per concurrency level to benchmarks/results/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${TARGET_HOST:-http://localhost:8000}"
DURATION="${BENCH_DURATION:-60s}"
CONCURRENCY_LEVELS=(5 10 20)
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

echo "Target host: $HOST"
echo "Duration per level: $DURATION"

for USERS in "${CONCURRENCY_LEVELS[@]}"; do
  echo ""
  echo "=== Running benchmark at concurrency=$USERS ==="
  locust -f locustfile.py \
    --host "$HOST" \
    --headless \
    -u "$USERS" \
    -r "$USERS" \
    -t "$DURATION" \
    --csv="$RESULTS_DIR/run_${USERS}users" \
    --csv-full-history \
    --only-summary

  echo "Results written to $RESULTS_DIR/run_${USERS}users_stats.csv"
done

echo ""
echo "All runs complete. Summarize with:"
echo "  python summarize_results.py $RESULTS_DIR"
