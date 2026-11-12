### Table 2a: Latency, one row per load scenario (at least: steady, burst, sustained overload, per-minute limit halved, daily budget exhausted)

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| steady_stand_in | stand-in | 30 | 75.0 | 24.0 | 64.4 / 3286.6 / 3771.2 | 65.5 / 7132.6 / 8558.3 |
| sudden_burst | stand-in | 40 | 15.0 | 160.0 | 64.4 / 2680.3 / 4247.6 | 66.1 / 7265.6 / 8556.6 |
| sustained_overload | stand-in | 50 | 50.0 | 60.0 | 60.4 / 2130.5 / 3461.2 | 60.7 / 4675.6 / 9387.0 |
| per_minute_limit_halved | stand-in | 30 | 45.0 | 40.0 | 58.4 / 67.5 / 755.6 | 60.9 / 5241.1 / 6191.3 |
| daily_budget_exhausted | stand-in | 20 | 15.0 | 80.0 | 57.1 / 65.7 / 66.2 | 57.3 / 65.9 / 66.5 |

### Table 2b: Outcomes, same rows

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| steady_stand_in | 16.7% | 83.3% | 0.0% | 0.33 | 0 | 1226 | $0.858 |
| sudden_burst | 2.5% | 97.5% | 0.0% | 0.4 | 0 | 696 | $0.487 |
| sustained_overload | 4.0% | 96.0% | 0.0% | 0.12 | 0 | 1283 | $0.898 |
| per_minute_limit_halved | 3.3% | 96.7% | 0.0% | 0.27 | 0 | 653 | $0.457 |
| daily_budget_exhausted | 0.0% | 100.0% | 0.0% | 0.0 | 0 | 0 | $0.000 |

### Table 3: Retrieval settings (at least three)

| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Piece 256, fetched 3, re-rank NO | 18 | 61.1% | 11.1% | 27.8% | 1321 | 1297.7 / 3238.5 | 4175.9 / 4921.7 |
| Piece 512, fetched 5, re-rank YES | 18 | 61.1% | 11.1% | 27.8% | 1542 | 1719.7 / 4029.0 | 4441.8 / 5618.3 |
| Piece 1024, fetched 8, re-rank NO | 18 | 50.0% | 22.2% | 27.8% | 1921 | 1436.0 / 3448.0 | 4330.0 / 5642.2 |