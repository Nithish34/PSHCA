---
title: PSHCA Environment Server
emoji: 🀄
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Predictive Self-Healing Cloud Architect (PSHCA) — v2.0

**Built for the Meta Hackathon**

A rigorous real-world simulation of cloud infrastructure designed to train and evaluate AI models acting as Site Reliability Engineers (SRE) or Cloud Architects. Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

## Description & Motivation

Managing cloud infrastructure requires constant vigilance. Systems experience anomalies like CPU spikes, memory leaks, and cascading deployment failures — often with compounding signals across multiple nodes. The **PSHCA** environment simulates a 4-node cluster (2 web servers, 1 primary database, 1 replica) that encounters realistic, escalating anomalies over time.

An AI agent must interpret multi-signal telemetry (CPU, Memory, Latency, Error Rate, Disk I/O, Alerts) and select the correct self-healing action from a predefined action set to prevent a full cluster outage.

### Nine Scenario Variants Across Three Difficulty Tiers

v2.0 introduces **9 scenario variants** to prevent agent memorisation. Each call to `reset()` randomly selects one variant from the active difficulty tier:

| Tier | ID | Title | Correct Action |
|---|---|---|---|
| **Easy** | E1 | CPU Spike — Primary Web Server | `scale_up` / `reboot_server` → `web-server-01` |
| **Easy** | E2 | CPU Spike — Secondary Web Server | `scale_up` / `reboot_server` → `web-server-02` |
| **Easy** | E3 | Replica DB CPU Overload | `reboot_server` / `scale_up` → `db-replica` |
| **Medium** | M1 | Memory Leak — Primary DB | `clear_cache` / `failover_db` → `db-main` |
| **Medium** | M2 | Memory Leak — Replica DB | `clear_cache` / `reboot_server` → `db-replica` |
| **Medium** | M3 | Memory Pressure — Web Server | `reboot_server` / `clear_cache` → `web-server-01` |
| **Hard** | H1 | Cascading Failure — Bad Deployment | `rollback_deployment` → `web-server-01` |
| **Hard** | H2 | Cascading Failure — DB Connection Exhaustion | `failover_db` → `db-main`, then `scale_up` → `web-server-01` |
| **Hard** | H3 | Cascading Failure — Memory Avalanche | `clear_cache` → `db-main`, then `scale_up` → `web-server-01` |

> **H2 and H3 are multi-step**: the agent must execute the correct two-action sequence in order. Partial credit (0.5) is awarded for completing step 1.

---

## v2.0 Key Design Decisions

| Feature | Detail |
|---|---|
| **Random variant selection** | One of 3 variants per tier is chosen randomly on `reset()` — the agent must generalise, not memorise |
| **Negative reward signals** | `−0.05` repeat-action penalty; `−0.10`–`−0.15` for destructive wrong actions (e.g. `rollback` on a memory leak) |
| **Timing-aware rewards** | Early correct action → 1.0; mid-episode → 0.85–0.90; late → 0.70–0.75; last-minute → 0.55 |
| **Multi-step Hard scenarios** | H2/H3 require a correct two-action sequence in order, mirroring real incident runbooks |
| **Realistic degradation curves** | Metrics worsen via non-linear ramp arrays (`[8,12,18,25,30]` CPU; `[8,10,13,17,22]` memory) |
| **Rich telemetry** | `latency_ms`, `error_rate (%)`, `disk_io (%)` alongside CPU/Memory for realistic multi-signal diagnosis |
| **Feedback string** | Every `step()` returns a natural-language `feedback` string explaining the reward, supporting in-context learning |
| **Jittered baseline metrics** | Initial state seeded with ±2–5% jitter — agents cannot hard-code threshold triggers |

---

## Action Space

| Field | Type | Values |
|---|---|---|
| `action_type` | literal string | `reboot_server`, `scale_up`, `rollback_deployment`, `clear_cache`, `failover_db`, `wait` |
| `target_resource` | string | `web-server-01`, `web-server-02`, `db-main`, `db-replica`, or empty for `wait` |

## Observation Space

| Field | Type | Meaning |
|---|---|---|
| `cpu_usage` | `dict[str, float]` | CPU usage percentage per node (0–100) |
| `memory_usage` | `dict[str, float]` | Memory usage percentage per node (0–100) |
| `latency_ms` | `dict[str, float]` | Request latency in milliseconds per node — spikes precede crashes |
| `error_rate` | `dict[str, float]` | Request error rate percentage per node — non-zero signals degradation |
| `disk_io` | `dict[str, float]` | Disk I/O usage percentage per node — high on DB nodes may indicate paging |
| `active_alerts` | `list[str]` | Live `[Warning]`, `[Critical]`, `[Fatal]` alert strings |
| `service_status` | `dict[str, str]` | High-level service health: `Healthy`, `Degraded`, or `Offline` |
| `current_task_info` | `string` | Scenario ID, title, and task description |
| `reward` | `float` | Step reward in range `−1.0` to `1.0` |
| `done` | `bool` | Episode completion flag |
| `metadata.feedback` | `string` | Natural-language explanation of the reward signal |

---

## Baseline Agent

We provide a reproducible baseline script using an OpenAI-compatible client. It implements a **ReAct-style loop** (`thought` + `confidence` + structured action JSON) with:
- Full v2.0 telemetry in every prompt (CPU, Memory, Latency, Error Rate, Disk I/O)
- Episodic memory — the last 6 action outcomes are fed back into each prompt
- Node-aware demo mode — fallback uses the active scenario's correct node, not a hardcoded default
- Structured `[START]` / `[STEP]` / `[END]` log lines for automated grading

```bash
# Set the required environment variables
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export HF_TOKEN="hf_..."
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"

# Run the inference script
python inference.py
```

> Run without env vars for **demo mode** — uses hardcoded optimal strategies to demonstrate the environment.

---

## Live Demo Dashboard

A production-grade live dashboard is served at startup for hackathon demos.

1. Start the server:
   ```bash
   uvicorn server.app:app --reload
   ```

2. Open:
   ```
   http://localhost:8000/dashboard
   ```

**Features:**
- Real-time CPU, memory, latency, and error-rate telemetry cards per node
- Active alert feed and service health badges
- Action stream with per-step reward and feedback trace
- One-click self-healing action buttons for live walkthroughs
- Server-Sent Events (SSE) auto-refresh every 1 second

---

## Benchmark Results

Results from running **Llama-3.3-70B-Instruct** via the Hugging Face Inference API:

| Scenario | Variant | Steps | Reward | Result |
|---|---|---|---|---|
| Easy — CPU Spike web-01 | E1 | 2 | **1.00** | ✅ |
| Easy — CPU Spike web-02 | E2 | 2 | **1.00** | ✅ |
| Easy — Replica DB CPU | E3 | 1 | **1.00** | ✅ |
| Medium — Memory Leak db-main | M1 | 1 | **1.00** | ✅ |
| Medium — Memory Leak db-replica | M2 | 2 | **1.00** | ✅ |
| Medium — Memory Pressure web-01 | M3 | 2 | **1.00** | ✅ |
| Hard — Cascading Deploy Failure | H1 | 2 | **1.00** | ✅ |
| Hard — DB Connection Exhaustion | H2 | 2 | **1.00** | ✅ |
| Hard — Memory Avalanche | H3 | 2 | **1.00** | ✅ |

**Aggregate: 3.00 / 3.00 — all 9 scenarios solved.**

---

## Setup Instructions

### Environment Configuration

```bash
# Copy the example file
cp .env.example .env

# Edit with your values
API_BASE_URL="https://api-inference.huggingface.co/v1"
HF_TOKEN="hf_your_token_here"
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
```

> The `.gitignore` excludes `.env`. Never commit actual credentials.

### Running Locally

```bash
uvicorn server.app:app --reload
```

### Deploying to Hugging Face Spaces

```bash
openenv push
```

---

## Project Structure

```text
PSHCA/
├── __init__.py               # Package exports (PshcaEnv, PshcaAction, PshcaObservation)
├── README.md                 # This file
├── openenv.yaml              # OpenEnv manifest — entrypoint, tasks, model bindings
├── pyproject.toml            # Project metadata and dependencies (v2.0.0)
├── Dockerfile                # Multi-stage container image (root-level for HF Spaces)
├── models.py                 # Pydantic models: PshcaAction, PshcaObservation (v2.0 fields)
├── inference.py              # Hackathon entry point — validates env vars, calls run_evaluation()
├── baseline_inference.py     # ReAct evaluation loop with v2.0 telemetry, episodic memory
├── client.py                 # PshcaEnv client — WebSocket/HTTP wrapper over OpenEnv core
├── preview_dashboard.html    # Live demo dashboard UI (served by app.py)
└── server/
    ├── __init__.py           # Server module exports
    ├── app.py                # FastAPI app — OpenEnv core + custom dashboard endpoints + SSE
    └── PSHCA_environment.py  # Core simulation: 9 scenarios, graders, degradation curves, rewards
```