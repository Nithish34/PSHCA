---
title: PSHCA Environment Server
emoji: 🀄
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Predictive Self-Healing Cloud Architect (PSHCA) Environment

A sophisticated real-world simulation of cloud infrastructure designed to test and train AI models acting as Site Reliability Engineers (SRE) or Cloud Architects.

## Description & Motivation

Managing cloud infrastructure involves constant vigilance. Systems experience anomalies like CPU spikes, memory leaks, and cascading deployment failures. The **PSHCA** environment simulates a 4-node cluster (2 web servers, 1 primary database, 1 replica) that encounters realistic anomalies over time.

An AI agent interacting with this environment must interpret telemetry data (CPU/Memory usage, active alerts) and select the correct self-healing actions from a predefined set of tools to prevent system failure and maintain high availability.

### Three Progressing Tasks

The environment supports 3 difficulty levels with automated graders (0.0 to 1.0 scoring):
1. **Easy:** A single web server experiences a rapid CPU spike. The agent must successfully scale up or reboot it before it crashes.
2. **Medium:** A gradual memory leak is introduced to the primary database. The agent must catch it from the telemetry trend and clear the cache or failover.
3. **Hard:** A cascading failure caused by a bad deployment. Both CPU and memory metrics spike across multiple services. The agent must trace the root cause and execute a deployment rollback.

## Action Space

| Field | Type | Values |
|---|---|---|
| `action_type` | literal string | `reboot_server`, `scale_up`, `rollback_deployment`, `clear_cache`, `failover_db`, `wait` |
| `target_resource` | string | `web-server-01`, `web-server-02`, `db-main`, `db-replica`, or empty for `wait` |

## Observation Space

| Field | Type | Meaning |
|---|---|---|
| `cpu_usage` | dict[str, float] | CPU usage percentage per node |
| `memory_usage` | dict[str, float] | Memory usage percentage per node |
| `active_alerts` | list[str] | Active warnings and critical alerts |
| `service_status` | dict[str, str] | High-level service health such as `Healthy`, `Degraded`, or `Offline` |
| `current_task_info` | string | Scenario description and task context |
| `reward` | float | Step reward in the range `0.0` to `1.0` |
| `done` | bool | Episode completion flag |

## Baseline Agent

We provide a reproducible baseline script that uses an OpenAI-compatible client to evaluate the environment.

The baseline now includes:
- ReAct-style structured decisions (`thought`, `confidence`, action JSON)
- Episodic memory within each scenario episode (recent action outcomes are fed back into prompts)
- Interpretable action logs showing rationale and confidence per step

```bash
# Export the required Hugging Face/OpenAI-compatible variables
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export HF_TOKEN="hf_..."
export MODEL_NAME="gpt-4o-mini"

# Run the inference script
python3 inference.py
```

## Live Demo Dashboard

A built-in live dashboard is available for hackathon demos.

1. Start the server:

```bash
uvicorn server.app:app --reload
```

2. Open:

```text
http://localhost:8000/dashboard
```

Features:
- Real-time CPU and memory telemetry cards per node
- Active alert feed and service health state
- Action stream with reward trace
- One-click self-healing action buttons for live walkthroughs

## Setup Instructions

### Environment Configuration

Before running the inference scripts, you need to set up your environment variables:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and fill in your values:
   ```bash
   API_BASE_URL="https://api-inference.huggingface.co/v1"  # Your API endpoint
   HF_TOKEN="hf_your_token_here"                           # Get from https://huggingface.co/settings/tokens
   MODEL_NAME="gpt-4o-mini"                               # Your chosen model
   ```

3. The `.gitignore` file excludes `.env` to protect your secrets. Never commit actual credentials!

### Running Locally
You can run the environment server locally for development:
```bash
uvicorn server.app:app --reload
```

### Deploying to Hugging Face Spaces
Deploy this OpenEnv environment to Hugging Face Spaces using the `openenv push` command:
```bash
openenv push
```

## Project Structure
```text
PSHCA/
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── models.py              # Action and Observation models
├── inference.py          # LLM Grader and Eval script
├── baseline_inference.py  # Legacy compatibility wrapper
└── server/
    ├── PSHCA_environment.py  # Core environment simulation logic
    ├── app.py             # FastAPI application
    └── Dockerfile         # Container image definition
```