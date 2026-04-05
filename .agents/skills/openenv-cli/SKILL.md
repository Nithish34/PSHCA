---
name: pshca-sre-agent
description: >
  Operate the Predictive Self-Healing Cloud Architect (PSHCA) environment as an
  autonomous Site Reliability Engineer (SRE). Use when asked to diagnose and
  resolve simulated cloud infrastructure incidents — CPU spikes, memory leaks,
  cascading deployment failures — by issuing self-healing actions against a
  4-node cluster via step() / reset() / state(). Do not use for model training
  setup, infrastructure provisioning, or tasks unrelated to incident response.
license: BSD-3-Clause
compatibility: openenv>=0.2.1
metadata:
  domain: cloud-infrastructure
  task_type: incident-response
  difficulty_levels: [easy, medium, hard]
  scenario_variants: 9
  nodes: [web-server-01, web-server-02, db-main, db-replica]
  reward_range: [-1.0, 1.0]
  max_steps: 20
---

# /pshca-sre-agent

Act as an autonomous SRE agent inside the PSHCA OpenEnv environment. Diagnose
live cloud incidents from telemetry observations and issue the correct
self-healing action each turn to stabilise the cluster before it crashes.

---

## 1. Environment Overview

PSHCA simulates a **4-node cloud cluster** running 3 difficulty tiers, each
with multiple randomly-selected scenario variants so the agent must generalise
rather than memorise.

| Difficulty | Variants | Typical Steps | Core Challenge |
|------------|----------|---------------|----------------|
| Easy       | E1 E2 E3 | 1 – 3         | Identify which node is spiking and scale/reboot it |
| Medium     | M1 M2 M3 | 1 – 5         | Find the leaking node and clear memory before OOM |
| Hard       | H1 H2 H3 | 2 – 7         | Trace root cause of cascade; single or multi-step fix |

Episodes cycle automatically: Easy → Medium → Hard → Easy …

---

## 2. Observation Fields

Each call to `step()` or `reset()` returns a `PshcaObservation` with:

```
cpu_usage        dict[node, float]   CPU % per node  (0–100)
memory_usage     dict[node, float]   Memory % per node (0–100)
active_alerts    list[str]           Live alert strings (may include noise)
service_status   dict[str, str]      api / database / cache → Healthy | Degraded | Offline
current_task_info str                Human-readable task description with scenario ID
reward           float | None        Reward for the last action (None on reset)
done             bool                True when episode is over
metadata:
  latency_ms     dict[node, float]   p99 request latency in milliseconds
  error_rate     dict[node, float]   Error rate % per node
  disk_io        dict[node, float]   Disk I/O % per node
  cumulative_reward float            Running total reward this episode
  scenario_state  int                Internal progression counter (1 = early, 4+ = late)
  feedback        str                Plain-English explanation of why this reward was given
  episode_id      str                UUID for the current episode
  step_count      int                Steps elapsed this episode
```

### Alert Severity Prefixes

| Prefix        | Meaning                                      |
|---------------|----------------------------------------------|
| `[Warning]`   | Metric approaching threshold — act soon      |
| `[Critical]`  | Threshold breached — act immediately         |
| `[Fatal]`     | Node crashed / cluster offline — episode over|
| `[Info]`      | Noise / routine event — safe to ignore       |
| `[ESCALATED]` | Severity upgraded automatically by the environment |

> **Alert noise is intentional.** `[Info]` alerts are false positives injected
> to simulate real-world alert storms. Focus on `[Warning]` and above.

---

## 3. Action Space

Submit a `PshcaAction` with two fields: `action_type` and `target_resource`.

### 3.1 Valid Actions

| action_type            | When to Use                                              | Valid Targets                        |
|------------------------|----------------------------------------------------------|--------------------------------------|
| `scale_up`             | CPU rising on a web server; add capacity                 | web-server-01, web-server-02, db-replica |
| `reboot_server`        | CPU critical on any node; hard restart to clear overload | web-server-01, web-server-02, db-main, db-replica |
| `clear_cache`          | Memory leak on a database node; free cache               | db-main, db-replica, web-server-01   |
| `failover_db`          | db-main failing; promote replica to primary              | db-main                              |
| `rollback_deployment`  | Post-deploy cascade; revert to last known-good build     | web-server-01, web-server-02         |
| `wait`                 | Gather one more observation before committing            | (leave empty — `""`)                 |

### 3.2 Action Format

```python
from models import PshcaAction

action = PshcaAction(
    action_type="scale_up",
    target_resource="web-server-01"
)
result = env.step(action)
```

### 3.3 Action Rules

- **Never target a node that does not exist.** Invalid targets incur a −0.5
  penalty and the action has no effect.
- **Never repeat the same action twice in a row.** Exact repeats incur a −0.05
  penalty.
- **Never use destructive actions on unrelated scenarios.** For example,
  `rollback_deployment` during a memory-leak episode returns −0.15.
- **`wait` is almost always wrong.** Metrics worsen every step. Reserve it for
  one observation when the scenario is genuinely ambiguous. Overuse is penalised.

---

## 4. Reward Design

Rewards are continuous and trajectory-aware — not binary end-of-episode scores.

```
+1.00   Perfect: correct action on correct node, early in episode
+0.90   Good: correct action, moderate delay
+0.75   Late: correct action, high metric values at time of fix
+0.50   Partial: correct action family, wrong target
+0.20   Sympathetic: right idea, wrong node
+0.05   Wait: stalling acknowledged but not rewarded
 0.00   No effect: action irrelevant to current scenario
−0.05   Repeat penalty: exact same action as previous step
−0.10   Destructive: wrong action type for this scenario class
−0.15   Highly destructive: e.g. rollback during memory leak
 0.00   Timeout: episode reaches MAX_STEPS=20 without resolution
```

**MTTR bonus:** the environment rewards early intervention. Acting on step 1
when CPU is below 60% scores 1.0. Acting on step 4 when CPU is above 90% scores
0.75. Same action — different timing — different reward.

---

## 5. Decision Rules (SRE Runbook)

Follow this triage logic each turn:

```
1. READ active_alerts — filter out [Info] noise.
2. IDENTIFY the failing metric:
   a. CPU > 70% on a node?       → scale_up or reboot_server on THAT node
   b. Memory > 65% on a node?    → clear_cache or failover_db on THAT node
   c. Post-deploy cascade?        → rollback_deployment on web-server-01
   d. Multi-service failure?      → follow hard scenario two-step sequence
3. CHECK scenario_state in metadata:
   - state 1–2: You are early — act immediately for maximum reward
   - state 3–4: You are mid-episode — still time to recover
   - state 5+:  You are late — cluster may be approaching [Fatal]
4. CHECK cumulative_reward in metadata:
   - Negative cumulative?  You are making mistakes — reconsider your strategy
   - Positive cumulative?  You are on track — continue the same approach
5. ACT with the correct action + correct target.
6. READ feedback in the next observation to understand why you got that reward.
```

---

## 6. Scenario-Specific Guidance

### Easy Scenarios (E1 / E2 / E3)
- One node is experiencing a CPU spike. The others are healthy.
- **Key diagnostic:** compare `cpu_usage` across all four nodes. The outlier is
  the target.
- Correct actions: `scale_up` or `reboot_server` on the spiking node.
- Wrong actions that will be penalised: `failover_db`, `rollback_deployment`.

### Medium Scenarios (M1 / M2 / M3)
- One node's memory is climbing. Latency on that node is also elevated.
- **Key diagnostic:** cross-reference `memory_usage` with `latency_ms`. Both
  will be elevated on the leaking node.
- Correct actions: `clear_cache` or `failover_db` on the leaking node.
- Wrong actions: `rollback_deployment` (−0.15 penalty).

### Hard Scenarios (H1 / H2 / H3)

**H1 — Bad Deployment (single-step fix):**
- Both web servers spike. DB then overloads. Root cause is a bad deployment.
- One correct action: `rollback_deployment` on `web-server-01`.
- Do NOT attempt `scale_up` — symptomatic fix scores only 0.1.

**H2 — DB Connection Exhaustion (two-step fix):**
- Step 1: `failover_db` on `db-main` (scores 0.5, episode continues)
- Step 2: `scale_up` on `web-server-01` (scores 1.0, episode ends)
- Order matters — wrong order scores 0.0.

**H3 — Memory Avalanche (two-step fix):**
- Step 1: `clear_cache` on `db-main` (scores 0.5, episode continues)
- Step 2: `scale_up` on `web-server-01` (scores 1.0, episode ends)

---

## 7. Connection & Usage

### Connect via Python Client

```python
import asyncio
from client import PshcaEnv
from models import PshcaAction

async def run():
    # Connect to live HuggingFace Space
    async with await PshcaEnv.from_env("voldigo/PSHCA") as env:
        obs = await env.reset()
        print(obs.current_task_info)

        while not obs.done:
            # Your agent logic here
            action = PshcaAction(
                action_type="scale_up",
                target_resource="web-server-01"
            )
            obs = await env.step(action)
            print(f"Reward: {obs.reward} | Feedback: {obs.metadata['feedback']}")

asyncio.run(run())
```

### Connect to Local Server

```python
env = PshcaEnv(base_url="http://localhost:8000")
```

### Run Baseline Evaluation

```bash
# Set credentials
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="hf_..."
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"

# Run full 3-scenario evaluation
python inference.py
```

Expected output format:
```
{"type": "START", "task": "easy", ...}
{"type": "STEP",  "task": "easy", "step": 1, "reward": 1.0, ...}
{"type": "END",   "task": "easy", "final_reward": 1.0, "success": true, ...}
```

---

## 8. Validation & Deployment

```bash
# Local structure check
openenv validate --verbose

# Build Docker image
openenv build

# Validate against live server
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
openenv validate --url http://localhost:8000

# Push to HuggingFace Spaces
openenv push --repo-id voldigo/PSHCA
```

Health check endpoint: `GET /health` → must return HTTP 200.
HuggingFace Spaces port: **7860** (configured in Dockerfile CMD).

---

## 9. Environment Contracts

| Contract                  | Value                        |
|---------------------------|------------------------------|
| `spec_version`            | 1                            |
| `SUPPORTS_CONCURRENT_SESSIONS` | True                    |
| `MAX_STEPS`               | 20                           |
| `max_concurrent_envs`     | 1 (default dashboard mode)   |
| Reward range              | [−1.0, 1.0]                  |
| Inference runtime limit   | < 20 minutes on 2 vCPU / 8 GB RAM |
| Required env vars         | `API_BASE_URL`, `HF_TOKEN`, `MODEL_NAME` |
| LLM client                | OpenAI-compatible (`openai` Python SDK) |
| Inference script name     | `inference.py` (root)        |

---

## 10. Guardrails

- Do not target resources outside `[web-server-01, web-server-02, db-main, db-replica]`.
- Do not ignore `[Critical]` or `[Fatal]` alerts — they signal imminent episode failure.
- Do not use `wait` more than once per episode without a clear reason.
- Do not assume the same node is always the problem — scenario variants randomise
  the failing node on every reset.
- Do not skip reading `metadata.feedback` — it is the fastest way to correct
  mistakes mid-episode.
- Do not repeat a failed action — the repeat penalty compounds across steps.