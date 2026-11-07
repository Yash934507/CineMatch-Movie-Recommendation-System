# Improvement 2 verification

Executed on November 5, 2026, from the CineMatch project root in PowerShell.
Python commands used the existing project `.venv` (Python 3.12.14).

## Commands and observed results

Baseline, before implementation:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
```

Result: **53 passed**, one existing Starlette/httpx deprecation warning.

Focused checks after the refactor:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_onboarding.py tests/test_api.py -q -p no:cacheprovider
```

Result: **38 passed**, the same warning.

Evaluation (also available as the VS Code task `CineMatch: Evaluate onboarding`):

```powershell
& '.\.venv\Scripts\python.exe' scripts/run_onboarding_evaluation.py
```

Result: completed successfully, wrote the metrics CSV, user-details CSV, summary
JSON, report, chart, and generated README section. All reported ranking results
come from this command. See [evaluation report](improvement2_onboarding_evaluation.md).
The chart was visually inspected for readable labels and complete rendering.
The script was rerun after adding the generated README section and produced the
same printed metrics and selection. An independent audit recomputed aggregates
from all 1,400 per-strategy detail rows and matched the metrics CSV within 1e-12.
It also verified strict timestamp ordering, exactly 100 sampled negatives in this
run, identical candidate hashes between strategies, disjoint validation/test users,
and exactly one generated README section.

Frontend production build (VS Code task `CineMatch: Build frontend`):

```powershell
.\scripts\run_frontend.ps1 -Build
```

Result: **Vite 8.3.5 succeeded**, 17 modules transformed, build time 1.89 seconds.
Output: HTML 0.61 kB, CSS 17.16 kB, JavaScript 236.70 kB
(gzip: 0.37, 4.67, and 74.00 kB respectively). The existing launcher uses its
configured Node executable and local Vite installation. No dependencies were upgraded.

Complete Python suite after refactor and frontend build:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
```

Result: **77 passed**, one warning, 6.92 seconds. All 53 pre-existing tests remain
in the suite; 24 added cases cover temporal isolation, eligibility, candidate
parity, math, deterministic orchestration, and API compatibility.

The warning is `StarletteDeprecationWarning`: the installed Starlette TestClient
deprecates its httpx integration in favor of httpx2. This predates Improvement 2;
tests pass and dependency changes were outside this refactor.

Editor diagnostics for the changed Python implementation, evaluation script,
and new tests reported no errors.

## Live HTTP verification

Started an isolated verification server, leaving any existing server untouched:

```powershell
& '.\.venv\Scripts\python.exe' -m uvicorn cinematch.api:app --app-dir src --host 127.0.0.1 --port 8765
```

Sent a real HTTP request (not TestClient):

```powershell
$onboardingBody = '{"ratings":[{"movie_id":1,"rating":4},{"movie_id":260,"rating":5},{"movie_id":296,"rating":3}],"limit":5}'
$onboardingResponse = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8765/api/v1/onboarding/recommendations' -Method Post -ContentType 'application/json' -Body $onboardingBody
$onboardingResponse.StatusCode
$onboardingResponse.Content
$websiteResponse = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8765/'
$websiteResponse.StatusCode
$websiteResponse.Content
```

POST result: **HTTP 200**, `strategy: svd-fold-in`, `ratings_used: 3`, five
recommendations, excluding IDs 1, 260, and 296. Observed results:

| Movie ID | Title | Predicted rating |
| --- | --- | --- |
| 105504 | Captain Phillips (2013) | 4.8116 |
| 4967 | No Man's Land (2001) | 4.7834 |
| 1250 | Bridge on the River Kwai, The (1957) | 4.7584 |
| 750 | Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) | 4.7489 |
| 1203 | 12 Angry Men (1957) | 4.7395 |

These are serving smoke-test predictions from the existing production model,
not evaluation metrics. GET `/` also returned **HTTP 200** with the React root
and the built JavaScript/CSS asset references. No browser interaction was used.
The temporary verification server was stopped after the checks.

## Changed and generated files

- [Shared implementation](../src/cinematch/onboarding.py) — new.
- [API](../src/cinematch/api.py) — delegates to shared fold-in; same contract.
- [Evaluation runner](../scripts/run_onboarding_evaluation.py) — new.
- [Onboarding tests](../tests/test_onboarding.py) — new.
- [API tests](../tests/test_api.py) — added compatibility and validation cases.
- [VS Code tasks](../.vscode/tasks.json) — evaluation and frontend build tasks.
- [README](../README.md) — generated Improvement 2 section only.
- [Metrics](../data/processed/improvement2_onboarding_metrics.csv),
  [user details](../data/processed/improvement2_user_details.csv),
  [summary](../data/processed/improvement2_summary.json),
  [report](improvement2_onboarding_evaluation.md),
  [chart](figures/improvement2_onboarding_comparison.png), and this verification record.
- Frontend build output regenerated using the existing build command; React source
  and minimum-rating requirement were not changed.

The checkout has no Git metadata, so a Git diff was unavailable. Existing files
were inspected and edits were scoped to the listed implementation and documentation.
Week 1 outputs and the saved production SVD model were not regenerated.

The sandbox helper initially failed before launching commands. Shell inspection
and verification used the approved elevated execution path; no package install
or external service deployment was needed.
