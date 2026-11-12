# The Relay — High-Throughput Research Q&A Pipeline

An enterprise-grade, four-stage research paper question-answering pipeline architected with strict citation grounding, sliding-window rate limiting, proactive token budgeting, automated quality verification loops, and a calibrated stand-in inference service.

Built for the **Vaani Senior Engineering Assignment: "The Relay"**.

---

## 1. System Architecture

```
                          ┌───────────────────────────┐
                          │   Incoming User Inquiry   │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
      ┌───────────────────────────────────────────────────────────────────┐
      │  Stage 1: Question Decomposition & Guardrail (QuestionAnalyzer)   │
      │  - Extracts technical keyphrases                                 │
      │  - Flags out-of-scope inquiries to avoid wasteful model calls     │
      └─────────────────────────────────┬─────────────────────────────────┘
                                        │
                                        ▼
      ┌───────────────────────────────────────────────────────────────────┐
      │  Stage 2: Passage Retrieval Tool (PassageFinder)                  │
      │  - Deterministic vector cosine search over chunked arXiv corpus   │
      │  - Enriches passages with section and page provenance metadata    │
      └─────────────────────────────────┬─────────────────────────────────┘
                                        │
                                        ▼
 ┌───►┌───────────────────────────────────────────────────────────────────┐
 │    │  Stage 3: Cited Response Synthesis (ResponseDrafter)              │
 │    │  - Grounded prompt enforcing bracketed source citations           │
 │    │  - Incorporates critique feedback on re-draft iterations          │
 │    └─────────────────────────────────┬─────────────────────────────────┘
 │                                      │
 │                                      ▼
 │    ┌───────────────────────────────────────────────────────────────────┐
 │    │  Stage 4: Automated Quality Audit [Branch/Loop] (QualityChecker)  │
 └───-┤  - Verifies cited pages/sections match retrieved context          │
(Fail)│  - Passes clean drafts; loops back to Stage 3 if critique fails   │
      └─────────────────────────────────┬─────────────────────────────────┘
                                        │ (Pass)
                                        ▼
      ┌───────────────────────────────────────────────────────────────────┐
      │  Stage 5: Canonical Citation Formatter & References Compilation   │
      └─────────────────────────────────┬─────────────────────────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │   Final Cited Response    │
                          └───────────────────────────┘
```

---

## 2. Directory Structure

```
d:\Vaani_Assignment\
├── README.md                          # Production architecture & operational manual
├── .env.example                       # Environment secrets template
├── requirements.txt                   # Production dependencies
├── config.py                          # Unified system settings & quota thresholds
│
├── papers/                            # Curated arXiv research library
│   ├── __init__.py
│   ├── download_papers.py             # Pinned version PDF downloader
│   ├── attention_is_all_you_need_v7.pdf
│   ├── moshi_speech_text_v1.pdf
│   └── dynamic_chunking_v2.pdf
│
├── indexing/                          # Document parsing & vector search
│   ├── __init__.py
│   ├── paper_parser.py                # Academic PDF section and page extractor
│   ├── text_chunker.py                # Token-aware sliding chunker (tiktoken)
│   └── passage_retriever.py           # Sublinear TF-IDF vector space model with 2-stage reranking
│
├── pipeline/                          # 4-Stage Q&A orchestration
│   ├── __init__.py
│   ├── question_analyzer.py           # Stage 1: Decomposition & guardrails
│   ├── passage_finder.py              # Stage 2: Vector retrieval tool
│   ├── response_drafter.py            # Stage 3: Cited answer synthesis
│   ├── quality_checker.py             # Stage 4: Audit & re-draft loop controller
│   ├── citation_formatter.py          # Post-processing & provenance indexing
│   └── orchestrator.py                # End-to-end pipeline coordinator
│
├── models/                            # Model access & proactive token budgeting
│   ├── __init__.py
│   ├── groq_client.py                 # Groq LPU client with header introspection
│   ├── token_budget_tracker.py        # Sliding-window RPM/TPM usage tracker
│   ├── fallback_router.py             # Seamless primary -> fallback -> stand-in router
│   └── model_client.py                # Unified caller interface
│
├── resilience/                        # Fault tolerance & traffic shaping
│   ├── __init__.py
│   ├── rate_limiter.py                # Token-bucket ingress rate limiter
│   ├── retry_policy.py                # Exponential backoff with full jitter
│   ├── request_queue.py               # Priority queue with user patience timeouts
│   └── circuit_breaker.py             # Three-state (closed/open/half-open) breaker
│
├── mock_service/                      # Calibrated stand-in model service
│   ├── __init__.py
│   ├── calibration_data.py            # Statistical profiles from real Groq measurements
│   ├── latency_simulator.py           # Calibrated log-normal TTFT & generation delays
│   ├── load_degradation.py            # Concurrency contention & quadratic load engine
│   └── mock_server.py                 # FastAPI service emulating /v1/chat/completions
│
├── load_testing/                      # High-throughput benchmark engine
│   ├── __init__.py
│   ├── scenarios.py                   # Steady, burst, overload & edge case specs
│   ├── metrics_collector.py           # Thread-safe high-resolution telemetry accumulator
│   ├── test_runner.py                 # Concurrent workload executor
│   └── results_exporter.py            # Generates results.csv & markdown tables
│
├── evaluation/                        # Quality benchmarking & scoring
│   ├── __init__.py
│   ├── test_questions.py              # 18 curated questions spanning all papers + refusals
│   └── answer_grader.py               # Assigns correct, wrong, refused_correctly/wrongly
│
├── backup_system/                     # Zero-loss state persistence
│   ├── __init__.py
│   └── snapshot_manager.py            # Atomic checkpointing across quota/model changes
│
├── server.py                          # Production FastAPI web API
├── run_pipeline.py                    # CLI for queries and index management
├── run_load_tests.py                  # CLI benchmark harness
├── run_real_measurements.py           # Probing tool for hosted Groq free tier
├── results/                           # Raw telemetry and results.csv output
└── writeup/                           # 2-Page assignment write-up
    └── writeup.md
```

---

## 3. Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment
Copy `.env.example` to `.env` and insert your Groq API key:
```bash
cp .env.example .env
```
*(If no API key is supplied, the system automatically routes to the calibrated stand-in service).*

### Step 3: Download Pinned Research Papers
```bash
python papers/download_papers.py
```

### Step 4: Build Search Index
```bash
python run_pipeline.py --build-index --chunk-setting medium
```

### Step 5: Ask a Question
```bash
python run_pipeline.py "What is the theoretical latency of Moshi's streaming dialogue architecture?"
```

---

## 4. Running the Stand-In Mock Service & Load Tests

1. Start the calibrated stand-in model service in a terminal:
```bash
python -m mock_service.mock_server --port 8001
```

2. Run the comprehensive load testing campaign:
```bash
python run_load_tests.py
```
This executes all 5 required workload scenarios and generates `results/results.csv` and summary tables.

---

## 5. Zero-Loss State Persistence & Model Switch Guarantee
The pipeline includes `backup_system/snapshot_manager.py`. Whenever upstream LLM limits are approached:
1. The `TokenBudgetTracker` proactively switches from primary (`llama-3.3-70b-versatile`) to fallback (`llama-3.1-8b-instant`) before encountering HTTP 429 rejections.
2. If all hosted quotas are exhausted, traffic cascades seamlessly to the stand-in service without in-flight state loss.
3. Mid-pipeline execution checkpoints are committed atomically to `.backup_vault/checkpoints/`.

