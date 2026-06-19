#!/usr/bin/env bash
# benchmark.sh — Measure /predict inference latency
# Usage: bash scripts/benchmark.sh [N]
#   N = number of requests (default 100)

set -euo pipefail

N="${1:-100}"
HOST="${API_HOST:-http://127.0.0.1:5000}"
PAYLOAD='{"text":"This is a wonderful video that I really enjoyed watching."}'

echo "[benchmark] Target: $HOST"
echo "[benchmark] Sending $N requests..."

# Use python instead of curl-loop for accurate per-request timing.
python3 - <<PY
import time, json, statistics, urllib.request, os

host = os.environ.get("API_HOST", "$HOST")
n = int(os.environ.get("N", "$N"))
payload = json.dumps({"text": "This is a wonderful video that I really enjoyed watching."}).encode()

latencies_ms = []
failures = 0
for i in range(n):
    req = urllib.request.Request(
        host + "/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            _ = r.read()
            if r.status != 200:
                failures += 1
    except Exception:
        failures += 1
    t1 = time.perf_counter()
    latencies_ms.append((t1 - t0) * 1000.0)

if not latencies_ms:
    print("[benchmark] No successful requests — is the API running?")
    raise SystemExit(1)

latencies_ms.sort()
def pct(p):
    return latencies_ms[min(len(latencies_ms) - 1, int(len(latencies_ms) * p / 100))]

print(f"requests:    {n}")
print(f"failures:    {failures}")
print(f"mean (ms):   {statistics.mean(latencies_ms):.2f}")
print(f"median (ms): {statistics.median(latencies_ms):.2f}")
print(f"p95 (ms):    {pct(95):.2f}")
print(f"p99 (ms):    {pct(99):.2f}")
print(f"max (ms):    {max(latencies_ms):.2f}")
PY
