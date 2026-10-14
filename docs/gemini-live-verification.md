# Google Gemini Live Provider Verification Guide

This guide describes how to manually execute and verify live Gemini evaluation runs using Google AI Studio API credentials.

> [!WARNING]
> Automated continuous integration (CI) tests run 100% offline using mock clients. Live provider calls should only be executed manually by developers with verified Google AI Studio API credentials and quotas.

---

## 1. Prerequisites & Test Environments

The repository defines two distinct test and development profiles:

1. **Base Offline Test Environment (without Gemini SDK):**
   ```bash
   pip install -e ".[dev]"
   pytest
   ```
   Runs the full base test suite offline. Gemini SDK-dependent tests are skipped cleanly without collection errors.

2. **Full Offline Test Environment (with Gemini SDK Extra):**
   ```bash
   pip install -e ".[dev,gemini]"
   pytest
   ```
   Runs 100% of the offline test suite, including all mock-based Gemini adapter unit tests.

3. **Live Verification Prerequisites:**
   - Install the optional `gemini` dependency extra: `pip install -e ".[gemini]"`
   - Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
   - Check active quotas and rate limits (RPM, TPM, RPD) in your Google AI Studio project console. Do not assume free access is unlimited.

---

## 2. Set Up API Credentials

Set the `GEMINI_API_KEY` environment variable in your terminal session. Never hardcode credentials, commit `.env` files, or paste keys into repository code.

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY = "your-actual-api-key"
```

**Linux / macOS (Bash/Zsh):**
```bash
export GEMINI_API_KEY="your-actual-api-key"
```

---

## 3. Evaluation Commands

### Offline Evaluation (Stub Client)
If API credentials or quota are unavailable, developers can test the complete evaluation pipeline offline using the deterministic stub client:

```powershell
.venv\Scripts\python -m structured_output_agent --dataset-path evals/data/dev.jsonl --output evals/results/report.json --client stub
```
The stub client produces deterministic responses to verify harness functionality and writes report provenance with `client_type: "stub"`. Stub accuracy is non-evaluative and does not represent real LLM performance.

### Live Gemini Development Evaluation
To evaluate live Gemini models against the development dataset:

```powershell
.venv\Scripts\python -m structured_output_agent --dataset-path evals/data/dev.jsonl --output evals/results/gemini_dev_report.json --client gemini --model gemini-3.1-flash-lite
```

### Model Selection Precedence
The model identifier resolves using this precedence order:
1. **CLI Flag (`--model`):** Highest precedence. For example, `--model gemini-3.1-flash-lite`.
2. **Environment Variable (`GEMINI_MODEL`):** Used if `--model` is omitted.
3. **Default Constant:** `DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"`.

### Request Pacing & Runtime
- **13-Second Interval:** To respect free-tier rate limits (5 RPM), the Gemini adapter enforces default client-side pacing of 13.0 seconds between request starts (`DEFAULT_REQUEST_INTERVAL = 13.0`).
- **First Request Immediate:** The first ticket starts immediately. Subsequent tickets wait for the remainder of the 13-second interval from the previous start time.
- **Estimated Runtime:** For the 20 development examples in `evals/data/dev.jsonl`, execution requires 1 immediate start plus 19 intervals of 13 seconds (247 seconds ≈ 4m 7s), plus network latency. Total execution typically requires **four to five minutes**.
- **External Dependencies:** Quotas (RPM, TPM, RPD) and provider availability depend on external Google AI Studio service conditions. Do not assume `gemini-3.1-flash-lite` is always available or that its quota is guaranteed.

### Request Policy & Retries
The evaluation harness implements a controlled retry architecture:
- **Model-Output Failures (O1–O3):** If a response fails JSON extraction, Pydantic schema validation (e.g. summary > 280 characters), or deterministic C1 policy rules, the evaluation harness executes up to 2 retries per ticket (`retry_budget=2`, allowing up to 3 total attempts per ticket) with sanitized feedback and structured failure logging. Valid outputs (O4, O5) terminate attempts immediately.
- **SDK Operational Retries:** For each model call, the adapter configures `HttpRetryOptions(attempts=1)` to explicitly disable SDK automatic retries on transport or quota failures. Transient operational errors (such as HTTP 503 or 429) fail fast and immediately abort the evaluation run with exit code 1.

> [!NOTE]
> Setting `attempts=1` explicitly disables SDK-level automatic exponential backoff retries. However, developers should note that one application-level SDK call may involve internal SDK operations (such as token authorization), so an application-level call does not strictly equate to exactly one network attempt under all conditions.

---

## 4. Provider Failure Recovery

- **Fail-Fast Exit Code 1:** On operational failures—including HTTP 503 (`UNAVAILABLE`), HTTP 429 (`RESOURCE_EXHAUSTED`), HTTP 400/404 model errors, network timeouts, or connection failures—the harness aborts immediately with exit code `1`.
- **No Partial Report Generation:** An aborted run never produces or writes a final evaluation report. Any pre-existing report at the `--output` path is preserved completely intact.
- **Verify Report Existence:** Always check that the command exited with code `0` and verify the report file exists and has a fresh timestamp before assuming evaluation completed.
- **Do Not Blindly Repeat:** Avoid repeatedly re-running evaluations immediately after provider errors, as this burns quota and can prolong rate-limiting penalties.
- **Transient 503 Errors:** For HTTP 503 service congestion, wait before attempting another run. Do not promise a fixed recovery time, as upstream service load varies.
- **Controlled Retry vs. No Heuristic Fallbacks:** The evaluation runner executes retries exclusively on model-output failures (O1–O3) within the configured `retry_budget=2` limit. Operational and infrastructure failures never retry. No heuristic or rule-based fallback predictions (such as defaulting to `needs_review`) are ever substituted for model decisions.

---

## 5. Safe Evaluation Practices

- **Development Split Only:** Live evaluation must use `evals/data/dev.jsonl`.
- **Protected Test Split:** The held-out dataset `evals/data/test.jsonl` is reserved and must never be accessed, read, statted, or evaluated during these workflows.
- **Fair Comparisons:** When comparing models, evaluate against the exact same development split using compatible evaluation settings.
- **Distinct Report Paths:** Always specify distinct `--output` paths (e.g. `evals/results/report.json` for stub vs `evals/results/gemini_dev_report.json` for live Gemini) to prevent overwriting results.
- **Clean Git State:** Do not stage or commit evaluation reports, temporary files, credentials, or unrelated changes.

---

## 6. Verification Checklist

After the evaluation completes:
1. **Exit Code:** Verify the command exited with code `0`.
2. **Report Generation:** Confirm `evals/results/gemini_dev_report.json` was generated with valid `1.1` schema version.
3. **Outcome Metrics:** Verify outcome counts (O1 through O5) and strict accuracy reflect genuine model responses.
4. **Credential Redaction:** Verify the generated JSON report contains zero credentials or secret tokens.
5. **Rate Limiting:** If `Operational failure (exit 1)` occurs due to HTTP 429 (`RESOURCE_EXHAUSTED`), review your Google AI Studio quota limits.

