# Engineering Dialogue & Collaboration Log: The Relay

**Engineer:** Staff / Senior Backend & AI Systems Engineer (7+ Years Experience)  
**Assistant:** Google Antigravity (AI Coding Assistant)  
**Project:** The Relay — High-Throughput Cited Research Paper Q&A System  
**Repository:** [https://github.com/AnmolRajpoot25/relay-research-assistant](https://github.com/AnmolRajpoot25/relay-research-assistant)  

---

## 1. High-Level Collaboration Narrative

Building a production-ready RAG system with upstream rate-limiting resilience requires careful architectural direction. Generative assistants write code quickly, but they frequently make subtle assumptions—defaulting to brittle external APIs, approximating metrics, or conflating pipeline state transitions. 

My role throughout this project was that of the **lead systems architect**:
1. I laid down the architectural blueprint before any code was written.
2. I defined the component boundaries, state machines, and resilience contracts.
3. When automated test runs and load benchmarks failed or produced skewed data, I diagnosed the theoretical root causes and gave the assistant explicit instructions on how to refactor the implementation.
4. I ensured that all reported figures were mathematically grounded in raw telemetry (`results.csv`).

---

## 2. Phase 1: Architecture Definition & Guiding the Initial Build

Before writing code, I set the core design principles and boundaries:

### My Architectural Directives:
* **Humanized & Modular Architecture:** Decouple the codebase into clean single-responsibility packages (`indexing`, `pipeline`, `models`, `resilience`, `mock_service`, `load_testing`, `evaluation`). No monolithic files or opaque variable names.
* **Deterministic Offline Retrieval:** Overrule heavy online dependencies. Use a section-aware PDF parser and a sublinear TF-IDF Vector Space Model (unigram + bigram) with cosine similarity and two-stage lexical reranking. Retrieval must run in under 5ms completely offline.
* **5-Stage Pipeline Graph:**
  1. *QuestionAnalyzer:* Keyphrase extraction and early out-of-scope guardrail.
  2. *PassageFinder:* Vector retrieval tool annotating chunks with immutable paper title, section (§), and page ($p.$) metadata.
  3. *ResponseDrafter:* Synthesis strictly grounded in retrieved evidence.
  4. *QualityChecker:* An automated audit loop verifying citations and claims against source context, redrafting up to 2 times on critique failure before failing closed.
  5. *CitationFormatter:* Canonical bracketed citation formatting.
* **Proactive Token Budgeting & Multi-Tier Fallback Router:** Rather than waiting for HTTP 429 rejections, track sliding-window usage (RPM/TPM) and switch tiers proactively:
  $$\text{Primary (Groq 70B)} \xrightarrow{80\% \text{ quota}} \text{Fallback (Groq 8B)} \xrightarrow{95\% \text{ quota}} \text{Calibrated Stand-In Service}$$
* **State Checkpointing:** Implement an atomic snapshot manager (`backup_system/snapshot_manager.py`) to guarantee zero request state loss during mid-pipeline model failovers.

---

## 3. Phase 2: Technical Errors Diagnosed & Theoretical Solutions Directed

During development and benchmark execution, several complex failure modes emerged. In each case, I analyzed the failure theoretically and directed the assistant on the exact fix.

---

### Challenge 1: The Pipeline Status Lifecycle Bug (All Requests Finishing with `status="error"`)
* **What Happened:** During our first full load run across 200+ requests, the summary exporter reported `Succeeded %: 0.0%` and `Timed out or given up %: 100.0%`, even though logs showed answers being generated and quality audits passing.
* **My Theoretical Diagnosis:**  
  I inspected `pipeline/orchestrator.py` and traced the state transitions:
  ```python
  execution_record = PipelineExecutionResult(..., status="error")
  ...
  if execution_record.status not in ("rate_limited", "refused", "error"):
      execution_record.status = "ok"
  ```
  The assistant had initialized `status` to `"error"` as a defensive default. However, because line 334 checked `status not in ("rate_limited", "refused", "error")`, the condition evaluated to `False` on every run. As a result, successful requests never transitioned to `"ok"`.
* **My Instruction to the Assistant:**  
  *"Status must be modeled as a strict state machine. Initialize the pipeline record with `status='in_progress'`. If an upstream 429 occurs, set `status='rate_limited'` and exit. If the question is outside library scope or fails the maximum audit loops, set `status='refused'`. In the finalization block, preserve any terminal status (`rate_limited`, `refused`, `error`, `timeout`), and transition to `status='ok'` only when a draft passes verification. Refactor `pipeline/orchestrator.py` and `models/fallback_router.py` to preserve rate-limit telemetry."*
* **The Fix:** The assistant implemented the state machine, and subsequent test runs cleanly separated successful queries ($100\%$ on nominal load) from rate-limited rejections.

---

### Challenge 2: Mock Service JSON Crashes Causing Quality Loop Fail-Closed Cascades
* **What Happened:** In `mock_service/mock_server.py`, all requests generated via `/v1/chat/completions` were picking random sentences from a hardcoded list of paper summaries. Under load, `QuestionAnalyzer` and `QualityChecker` threw `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)` on every call. Because our safety policy mandates failing closed on unparseable audit output, every request was rejected.
* **My Theoretical Diagnosis:**  
  An LLM endpoint in an agentic pipeline serves multiple roles: some calls require structured JSON schema output (decomposition, audit verification), while others require natural prose (drafting). A naive mock returning unstructured text breaks downstream JSON parsers.
* **My Instruction to the Assistant:**  
  *"The stand-in service must be context-aware. In `mock_service/mock_server.py`, inspect the incoming payload messages:
  1. If the prompt contains audit markers (`passes_audit`, `QualityChecker`), return valid JSON with `passes_audit: true`, confidence score, and citation counts.
  2. If the prompt asks for decomposition (`sub_queries`, `QuestionAnalyzer`), return schema-valid JSON with sub-queries and answerability booleans.
  3. If the prompt is drafting an answer, return technically accurate answers containing verifiable paper and section citations for our 13 factual questions, and honest refusal text for the 5 out-of-scope questions.
  Replace the random choice logic with this schema-aware router."*
* **The Fix:** The assistant updated `mock_server.py` with `generate_mock_model_response()`. Audit parsing errors dropped to zero, and the quality checker accurately passed valid drafts and rejected ungrounded claims.

---

### Challenge 3: Synthetic Telemetry in Table 2b (The "Peak Queue" Heuristic)
* **What Happened:** Reviewing the code in `load_testing/metrics_collector.py`, I discovered the following fallback line:
  ```python
  if peak_queue == 0 and total_count > 0:
      peak_queue = max(1, int(total_count * 0.15))
  ```
* **My Theoretical Diagnosis:**  
  The assistant had implemented a synthetic estimation formula to avoid reporting zero queue depth. In a rigorous benchmark submitted to hiring teams, synthetic numbers masquerading as observed telemetry are unacceptable.
* **My Instruction to the Assistant:**  
  *"Delete the 0.15 heuristic immediately. Table 2b asks for observed peak queue length. We must sample the real waiting queue depth from the executor (`executor._work_queue.qsize()`) as tasks are submitted during burst and sustained overload. If workers keep up and no tasks wait in queue, report `0` honestly. Never fabricate telemetry."*
* **The Fix:** The assistant removed the heuristic from `metrics_collector.py` and wired `_work_queue.qsize()` sampling into `load_testing/test_runner.py`.

---

### Challenge 4: Upstream Rate-Limiting Corrupting the Retrieval Ablation (Table 3)
* **What Happened:** When executing the 3-setting retrieval ablation (256 tokens vs 512 tokens vs 1024 tokens across 18 questions = 54 requests), Setting 1 completed, but Settings 2 and 3 reported $0.0\%$ correct answers and 0 tokens.
* **My Theoretical Diagnosis:**  
  The 54 queries were dispatched in rapid succession into the stand-in service. Because the mock server enforces a 30 RPM sliding window, the quota was depleted by question 19. All subsequent requests in Settings 2 and 3 received HTTP 429s, completely masking their retrieval performance.
* **My Instruction to the Assistant:**  
  *"Table 3 is designed to evaluate retrieval quality and chunk-size trade-offs (piece size, top-k, reranking), not rate-limiting. Before running the retrieval experiment in `run_retrieval_experiment()`, send an admin request to `/admin/configure` setting `rate_limiting_active=False` and resetting counters. This isolates the retrieval experiment from quota exhaustion and allows clean measurement of context window economics and citation accuracy."*
* **The Fix:** The assistant added the dynamic configuration call. All 18 questions across all 3 settings executed cleanly, demonstrating that 512-token chunks with reranking achieved the optimal quality-to-cost ratio ($61.1\%$ precision at $1,542$ tokens/answer).

---

## 4. Phase 3: Final Verification & Results Reconciliation

Once the implementation was stabilized, I supervised the execution of the full test campaign:
1. Ran `python -m pytest tests/ -v` — verified all 16 automated tests passed.
2. Ran `python run_load_tests.py` — executed all 5 workload scenarios and the 3 retrieval settings.
3. Verified that `results/results.csv` was generated with all 224 per-request rows and the exact 12 columns mandated by Page 4 of the brief.
4. Cross-checked `results/load_test_summary.md` to ensure that every percentile ($p50/p95/p99$), percentage, and token count reconciled mathematically with `results.csv`.
5. Authored the comprehensive 2-page write-up (`writeup/writeup.md`) addressing all 10 prescribed sections.

---

## 5. Summary of Roles: Engineer vs. Assistant

| Aspect | Staff Engineer Role | AI Assistant Role |
| :--- | :--- | :--- |
| **System Architecture** | Designed 5-stage RAG, resilience layer, and 3-tier fallback strategy | Generated initial boilerplate and scaffolding |
| **Retrieval Strategy** | Overruled heavy online embeddings in favor of deterministic offline TF-IDF | Implemented tokenization and vector indexing logic |
| **Debugging & Diagnostics** | Traced state lifecycle bugs, mock JSON parse failures, and quota masking | Wrote patches and updated targeted functions |
| **Data Integrity** | Rejected synthetic heuristics; enforced mathematical reconciliation | Computed numpy percentiles and formatted CSV/Markdown |
| **Submission Packaging** | Defined commit boundaries and clean deliverable structure | Executed git operations and packaged clean archives |
