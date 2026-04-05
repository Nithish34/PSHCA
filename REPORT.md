# Predictive Self-Healing Cloud Architect (PSHCA) v2.0 - Environment Report

This report provides a comprehensive breakdown of the inner mechanics, architectural structure, complete file breakdown, and the continuous execution workflow of the PSHCA environment. 

---

## 1. How the Environment Works in Detail

PSHCA simulates a 4-node cloud infrastructure handling real-world site reliability engineering (SRE) anomalies. The environment challenges AI models to diagnose active incidents using multi-signal telemetry and execute precise self-healing protocols to restore system health avoiding cluster failure. 

### Core Mechanics & Components
* **The Cluster Setup**: Consists of 4 targetable servers: `web-server-01`, `web-server-02` (API Layer) and `db-main`, `db-replica` (Database/Cache).
* **Action Space**: The agent can choose from explicit healing actions (`action_type`):
  * `reboot_server`, `scale_up`, `rollback_deployment`, `clear_cache`, `failover_db`, `wait`, `escalate_to_human`.
  * Along with explicitly stating the `target_resource` from the cluster.
* **Observation Space (v2.0 Telemetry)**: Real-time, step-by-step changing diagnostic data is sent back containing:
  * `cpu_usage`, `memory_usage`, `latency_ms`, `error_rate`, `disk_io`
  * `active_alerts` (warnings on SLO breaches or hardware timeouts).
  * `service_status` and contextual variables (`current_task_info`).

### Scenario Evaluation
Scenarios are split into **3 Difficulty Tiers containing 3 Variants each** (9 total). Selecting variants at random per run avoids agent memorization:
* **Easy (`E1`, `E2`, `E3`)**: Point-in-time CPU/traffic spikes requiring `scale_up` or `reboot_server` on a precise node.
* **Medium (`M1`, `M2`, `M3`)**: Gradual accumulating memory leaks exhibiting resource pressure leading to out-of-memory states, requiring `clear_cache` or DB failovers.
* **Hard/Cascading (`H1`, `H2`, `H3`, `C1`, `H4`)**: Compound cascading failures like bad deployments bottlenecking a database which halts API requests. They mandate **ordered multi-step** execution (e.g., Failover DB first, then scale up API).

### Grading & Feedback Loops
The `PSHCA_environment.py` acts as the objective grader. It calculates rewards dynamically:
* Positive Rewards (up to `1.0`): Immediate issue resolve scoring higher depending on quick MTTR (Mean Time to Recovery / step timers).
* Negative Rewards/Penalties: Deductions for blindly repeating previous actions (`-0.05`) or destructive maneuvers (e.g., triggering a DB rollover during a web-server memory leak). 

---

## 2. Complete File Directory & Component Breakdown

The codebase is organized adhering to the OpenEnv framework structure, extending typical endpoints into a FastAPI implementation over docker orchestration.

| File / Directory | Purpose & Detailed Role |
| -- | -- |
| `openenv.yaml` | The absolute source-of-truth metadata manifest for OpenEnv. Maps entry point schemas, data models (`PshcaObservation`, `PshcaAction`), and lists explicit descriptions for the internal Agent Tasks (E1-H3). |
| `models.py` | Built using `pydantic`. Formalizes the structure of the `PshcaAction` schemas for inputs, and extended `PshcaObservation` schemas containing comprehensive cluster telemetry matrices. |
| `server/PSHCA_environment.py` | **The Brain.** Implements core Python logic extending OpenEnv's `Environment` class. Hosts scenario data arrays, randomizers, the `reset()` functions loading jitter metrics, and `step()` functions executing grading rules, mutating node states, emitting natural language feedback, and checking SLO criteria over a 20-step lifetime. |
| `server/app.py` | FastAPI application layer. Wraps `PSHCA_environment.py` for API accessibility via HTTP/WebSockets. Purposefully bypasses standard WebUI limits natively hooking into `preview_dashboard.html` avoiding Gradio overriding. |
| `client.py` | Simplifies remote connections from agent scripts. Handles payload translation between WebSocket byte payloads to the native `PshcaAction` Python models. |
| `baseline_inference.py` | Pre-configured HuggingFace `OpenAI API` Agent. Embeds a heavy SRE system prompt. Parses responses using ReAct structures (Wait -> Observe -> Think -> JSON Action). Automatically steps through the Environment tests sequentially for reporting scores. Include deterministic fallback values overriding HF if network disconnected. |
| `inference.py` | Executable wrapper pointing directly to `baseline_inference.py`. |
| `preview_dashboard.html` | Rich, UI-engineered dynamic dashboard utilizing Server-Sent Events (SSE). Paints raw telemetry outputs into intuitive graph cards simulating DataDog or Grafana live updates. |
| `Dockerfile` | Multi-staged CI container builder utilizing `uv`. Stages base dependencies prior to serving the `uvicorn server.app:app` exposed on port `8000`. |
| `pyproject.toml` | The `uv` managed lock package dependency registry mapping the `openenv-pshca` meta configurations and entry points. |
| `validate-submission.sh` | Shell toolkit ensuring hackathon completion. Checks network pings, local containerization lifecycles, and successful `openenv validate` assertions. |

---

## 3. Workflow of the Environment (Step-by-Step Execution Sequence)

The lifecycle of the PSHCA simulation runs synchronously scaling through isolated incident timelines known as 'Episodes'.

### Phase I: Initialization
1. **Server Boot**: The `server/app.py` begins executing via `uvicorn`, launching the API wrapper and locking `PSHCA_environment` into memory. 
2. **Dashboard Hydration**: The custom front-end polls the `/dashboard/events` stream to visually anchor to internal states.

### Phase II: Starting an Episode
3. **Agent Connected**: `baseline_inference.py` initiates execution invoking the `EnvClient`.
4. **Environment Reset (`.reset()`)**: 
   * Server calculates sequence increments `reset_count`.
   * Randomly selects 1 of the 3 variants within the current difficulty bracket.
   * Calls `_init_cloud_state()` to load nodes with jittered baseline metrics (`latency: ~12ms`, `CPU: ~20%`).
   * Broadcasts the initial `PshcaObservation` contextualized payload to the inference agent.

### Phase III: The Core Step Loop
5. **Agent Inference**:
   * Baseline agent parses the payload appending the last actions (episode memory), the telemetry block, and active alert severity string.
   * Sends the prompt through an LLM layer outputting a formatted JSON response mapping an optimal action and a rationale/thought marker. 
6. **Execution Engine (`.step(action)`)**:
   * Action propagates to `PSHCA_environment.step()`.
   * Server runs scenario-specific metric degradations (e.g., if step=1, add 8% memory; if step=4, add 25% CPU overhead on affected nodes).
   * Server executes the payload over grader parameters (`grader_easy`, `grader_medium`, `grader_hard`):
     * Matches if `(action_type, target_resource)` equals expected resolving vectors.
     * Checks if max 20-step limits hit, or cascading thresholds exceed 100% usage (triggering system crash boolean: `done: True`).
   * Updates state buffers, emits internal string feedback calculating score floats (`1.0` if rapid resolved, penalties otherwise).

### Phase IV: Conclusion
7. **Resolution Loop**: Loop runs bridging Step 5 and 6 until the cluster enters the state `done=False`. 
8. **Logging Event**: A comprehensive `generate_postmortem()` logging event is packaged storing Incident IDs, Severities tracked, Root Causes validated, MTTR counters, and outcomes.
9. **Next Sequence**: Pipeline clears repeating **Phase II** for the next tiered difficulty.
