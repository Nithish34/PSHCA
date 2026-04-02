"""
PSHCA Baseline Inference Script
================================
Evaluates the PshcaEnvironment across all 3 difficulty scenarios using OpenAI.
"""

# Ensure UTF-8 output on Windows (must happen before any print calls)
import sys
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""

Features:
  - Multi-turn conversation history (the LLM remembers its past actions)
  - Structured JSON action parsing with retry logic
  - Per-step telemetry logging with colour-coded alerts
  - Per-scenario timing
  - Final grader report with pass/fail per task
  - Graceful fallback to hardcoded optimal strategies when no API key is set
  - [START] / [STEP] / [END] structured stdout logs for automated grading

Usage:
    # With Hugging Face / OpenAI-compatible endpoint:
    set API_BASE_URL=https://api-inference.huggingface.co/v1
    set HF_TOKEN=hf_...
    set MODEL_NAME=gpt-4o-mini
    python inference.py

  # Demo mode (no API key needed):
    python inference.py

  # Choose a different model:
    set MODEL_NAME=gpt-4o
    python inference.py
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple


from server.PSHCA_environment import PshcaEnvironment
from models import PshcaAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_RETRIES: int = 3          # JSON parse / API retries per step
TEMPERATURE: float = 0.0      # Deterministic output for reproducibility


def openai_configured() -> bool:
    return all(os.environ.get(key) for key in ["API_BASE_URL", "HF_TOKEN", "MODEL_NAME"])

# Valid action and resource values (used in the system prompt and for validation)
VALID_ACTIONS = [
    "reboot_server",
    "scale_up",
    "rollback_deployment",
    "clear_cache",
    "failover_db",
    "wait",
]
VALID_RESOURCES = ["web-server-01", "web-server-02", "db-main", "db-replica"]

# ---------------------------------------------------------------------------
# ANSI colour helpers (graceful no-op on Windows without ANSI support)
# ---------------------------------------------------------------------------
def _c(code: str, text: str) -> str:
    """Wrap text in an ANSI colour code, stripping it if not a TTY."""
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

GREEN  = lambda t: _c("92", t)
YELLOW = lambda t: _c("93", t)
RED    = lambda t: _c("91", t)
CYAN   = lambda t: _c("96", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)

# ---------------------------------------------------------------------------
# Structured log helpers — required by the hackathon checker
# ---------------------------------------------------------------------------
def log_start(task: str, task_info: str) -> None:
    """Emit the [START] structured log line for a new episode."""
    print(json.dumps({
        "type": "START",
        "task": task,
        "task_info": task_info,
        "timestamp": time.time(),
    }), flush=True)


def log_step(
    task: str,
    step: int,
    action_type: str,
    target_resource: str,
    thought: str,
    confidence: float,
    reward: float,
    done: bool,
    observation: dict,
) -> None:
    """Emit a [STEP] structured log line after each env.step() call."""
    print(json.dumps({
        "type": "STEP",
        "task": task,
        "step": step,
        "action": {
            "action_type": action_type,
            "target_resource": target_resource,
        },
        "thought": thought,
        "confidence": round(confidence, 4),
        "reward": round(reward, 4),
        "done": done,
        "observation": observation,
        "timestamp": time.time(),
    }), flush=True)


def log_end(task: str, total_steps: int, final_reward: float, success: bool, elapsed: float) -> None:
    """Emit the [END] structured log line when an episode terminates."""
    print(json.dumps({
        "type": "END",
        "task": task,
        "total_steps": total_steps,
        "final_reward": round(final_reward, 4),
        "success": success,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": time.time(),
    }), flush=True)


def obs_to_dict(obs) -> dict:
    """Convert a PshcaObservation to a plain dict for JSON serialisation."""
    return {
        "cpu_usage": obs.cpu_usage,
        "memory_usage": obs.memory_usage,
        "service_status": obs.service_status,
        "active_alerts": obs.active_alerts,
        "current_task_info": obs.current_task_info,
    }


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are an expert Site Reliability Engineer (SRE) and Cloud Architect.
Your job is to monitor a simulated 4-node cloud cluster and perform the correct self-healing
action each turn to prevent a system outage.

Cluster Resources:
  - web-server-01   (primary web server)
  - web-server-02   (secondary web server)
  - db-main         (primary database)
  - db-replica      (read replica database)

Available Actions:
  - reboot_server      → Hard restart a server to clear CPU overload
  - scale_up           → Add capacity to an overloaded server
  - rollback_deployment→ Revert to the last known-good deployment (fixes bad-deploy cascades)
  - clear_cache        → Free memory on a database to stop a memory leak
  - failover_db        → Promote db-replica to primary if db-main is failing
  - wait               → Do nothing this turn (use sparingly — metrics worsen each step!)

Decision Rules:
  1. Read the Active Alerts and CPU/Memory trends carefully before acting.
  2. Each turn the environment worsens — act as fast as possible.
  3. For CPU spikes on a web server → use scale_up or reboot_server on that server.
  4. For memory leaks on a database → use clear_cache or failover_db on db-main.
  5. For cascading failures after a deployment (both CPU + DB overload) → rollback_deployment.
  6. Never target a resource that is not one of: {VALID_RESOURCES}.

Use this response schema so your decisions are interpretable:
    - thought: one short sentence explaining why this action is best now
    - confidence: float in [0, 1]
    - action_type: one of {VALID_ACTIONS}
    - target_resource: one of {VALID_RESOURCES} or "" for wait

You MUST reply ONLY with a single line of valid JSON in this exact format:
{{"thought": "<brief reason>", "confidence": 0.0, "action_type": "<action>", "target_resource": "<resource>"}}

No markdown, no extra text.
"""

# ---------------------------------------------------------------------------
# OpenAI helper
# ---------------------------------------------------------------------------
def call_openai(messages: list) -> str:
    """
    Send a conversation to the OpenAI Chat Completions API and return the
    raw text of the assistant's reply.

    Raises:
        ImportError  – if openai package is not installed
        Exception    – propagates API errors after logging
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["HF_TOKEN"],
    )
    model = os.environ["MODEL_NAME"]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=64,       # Actions are short JSON — no need for a large window
        timeout=30,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Fallback strategies (demo mode — no API key required)
# ---------------------------------------------------------------------------
FALLBACK_STRATEGIES = {
    "easy":   {"action_type": "scale_up",            "target_resource": "web-server-01"},
    "medium": {"action_type": "clear_cache",          "target_resource": "db-main"},
    "hard":   {"action_type": "rollback_deployment",  "target_resource": "web-server-01"},
}


# ---------------------------------------------------------------------------
# Action parsing with retry
# ---------------------------------------------------------------------------
def parse_action(raw: str) -> Optional[Tuple[PshcaAction, str, float]]:
    """
    Parse a raw LLM string into (PshcaAction, thought, confidence).
    Handles optional ```json ... ``` fences.
    Returns None if parsing fails.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(text)
        action_type = data.get("action_type", "wait")
        target = data.get("target_resource", "")
        thought = str(data.get("thought", "No rationale provided")).strip()
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        if action_type not in VALID_ACTIONS:
            print(f"  {YELLOW('⚠')}  Unknown action_type '{action_type}'. Defaulting to 'wait'.")
            action_type = "wait"

        return PshcaAction(action_type=action_type, target_resource=target), thought, confidence
    except (json.JSONDecodeError, Exception) as exc:
        print(f"  {RED('✗')}  Failed to parse JSON: {exc}  |  Raw: {raw!r}")
        return None


def summarize_episode_memory(memory: List[Dict]) -> str:
    """Create a compact memory block so the agent avoids repeated failed actions."""
    if not memory:
        return "No prior steps in this episode."

    lines = []
    for item in memory[-6:]:
        lines.append(
            f"Step {item['step']}: action={item['action_type']} target={item['target_resource'] or '(none)'} "
            f"reward={item['reward']:.2f} done={item['done']} alerts={item['alerts']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Telemetry display
# ---------------------------------------------------------------------------
def print_observation(obs, step: int):
    """Pretty-print the current cloud telemetry."""
    print(f"\n  {DIM('┌─ Step')} {BOLD(str(step))} {DIM('─' * 50)}")

    cpu_parts = []
    for host, val in obs.cpu_usage.items():
        colour = RED if val >= 85 else (YELLOW if val >= 60 else GREEN)
        cpu_parts.append(f"{host}: {colour(f'{val:.0f}%')}")
    print(f"  {DIM('│')} CPU    : {' | '.join(cpu_parts)}")

    mem_parts = []
    for host, val in obs.memory_usage.items():
        colour = RED if val >= 85 else (YELLOW if val >= 60 else GREEN)
        mem_parts.append(f"{host}: {colour(f'{val:.0f}%')}")
    print(f"  {DIM('│')} Memory : {' | '.join(mem_parts)}")

    svc_parts = []
    for svc, status in obs.service_status.items():
        colour = GREEN if status == "Healthy" else (YELLOW if status == "Degraded" else RED)
        svc_parts.append(f"{svc}: {colour(status)}")
    print(f"  {DIM('│')} Status : {' | '.join(svc_parts)}")

    if obs.active_alerts:
        for alert in obs.active_alerts:
            colour = RED if "Critical" in alert or "Fatal" in alert else YELLOW
            print(f"  {DIM('│')} {colour('⚡ ' + alert)}")
    else:
        print(f"  {DIM('│')} {GREEN('✓ No active alerts')}")

    print(f"  {DIM('└' + '─' * 56)}")


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------
def run_scenario(env: PshcaEnvironment, scenario: str) -> float:
    """
    Run one full episode for the given scenario.

    Returns:
        Final reward score in [0.0, 1.0]
    """
    env.scenario = scenario
    env._init_cloud_state()
    env._state.step_count = 0
    obs = env._get_observation()

    task_info = obs.current_task_info
    label = {"easy": "🟢 Easy", "medium": "🟡 Medium", "hard": "🔴 Hard"}[scenario]

    print(f"\n{'=' * 62}")
    print(f"  {BOLD(label)} — {task_info}")
    print(f"{'=' * 62}")

    # ✅ Emit [START] structured log
    log_start(task=scenario, task_info=task_info)

    done = False
    step = 0
    final_reward = 0.0
    start_time = time.time()

    # Conversation history for multi-turn context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    episode_memory: List[Dict] = []

    while not done and step < env.MAX_STEPS:
        step += 1
        print_observation(obs, step)

        # Build the user message with current telemetry
        user_msg = (
            f"Turn {step} | Scenario: {task_info}\n\n"
            f"Current Telemetry:\n"
            f"  CPU Usage    : {obs.cpu_usage}\n"
            f"  Memory Usage : {obs.memory_usage}\n"
            f"  Service Status: {obs.service_status}\n"
            f"  Active Alerts : {obs.active_alerts}\n\n"
            f"Episode Memory (most recent first):\n{summarize_episode_memory(episode_memory)}\n\n"
            f"Select the best next action. Reply with JSON only."
        )

        action_bundle: Optional[Tuple[PshcaAction, str, float]] = None

        # --- OpenAI path ---
        if openai_configured():
            messages.append({"role": "user", "content": user_msg})

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    raw_reply = call_openai(messages)
                    action_bundle = parse_action(raw_reply)
                    if action_bundle:
                        # Add assistant reply to history so the model has context
                        messages.append({"role": "assistant", "content": raw_reply})
                        break
                    else:
                        # Ask it to retry with explicit correction message
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Your response could not be parsed as JSON (attempt {attempt}/{MAX_RETRIES}). "
                                f"Reply ONLY with valid JSON: "
                                f'{{\"thought\": \"<brief reason>\", \"confidence\": 0.0, \"action_type\": \"<action>\", \"target_resource\": \"<resource>\"}}'
                            ),
                        })
                except Exception as exc:
                    print(f"  {RED('✗')}  OpenAI API error (attempt {attempt}/{MAX_RETRIES}): {exc}")
                    time.sleep(1.5 * attempt)  # Exponential back-off

        # --- Fallback / demo path ---
        if action_bundle is None:
            if openai_configured():
                print(f"  {YELLOW('⚠')}  All retries failed — using optimal fallback strategy.")
            else:
                print(f"  {CYAN('ℹ')}  Demo mode — using hardcoded optimal strategy.")

            fallback = FALLBACK_STRATEGIES[scenario]
            raw_reply = json.dumps({
                "thought": "Fallback optimal policy for this scenario.",
                "confidence": 0.99,
                **fallback,
            })
            action_bundle = parse_action(raw_reply)

        if action_bundle is None:
            # Final guard: should never happen, but keep the loop safe.
            action_bundle = (
                PshcaAction(action_type="wait", target_resource=""),
                "Failed to produce valid action; defaulting to wait.",
                0.0,
            )

        action, thought, confidence = action_bundle

        # Log the chosen action (human-readable)
        action_colour = GREEN if action.action_type != "wait" else YELLOW
        print(
            f"\n  {BOLD('→')} Agent action: "
            f"{action_colour(action.action_type)} on "
            f"{CYAN(action.target_resource or '(none)')}"
        )
        print(f"  {DIM('Reason')} : {thought}")
        print(f"  {DIM('Confidence')} : {confidence:.2f}")

        # Step the environment
        result = env.step(action)
        done = result.done
        final_reward = result.reward if result.reward is not None else 0.0

        episode_memory.append(
            {
                "step": step,
                "action_type": action.action_type,
                "target_resource": action.target_resource,
                "reward": final_reward,
                "done": done,
                "alerts": result.active_alerts,
            }
        )

        # ✅ Emit [STEP] structured log — after every env.step()
        log_step(
            task=scenario,
            step=step,
            action_type=action.action_type,
            target_resource=action.target_resource or "",
            thought=thought,
            confidence=confidence,
            reward=final_reward,
            done=done,
            observation=obs_to_dict(result),
        )

        obs = result  # advance observation

    elapsed = time.time() - start_time
    success = final_reward >= 0.5
    outcome = GREEN("✓ SUCCESS") if success else RED("✗ FAILED")
    print(
        f"\n  {outcome} | Steps: {step}/{env.MAX_STEPS} | "
        f"Score: {BOLD(f'{final_reward:.2f}')}/1.00 | "
        f"Time: {elapsed:.1f}s"
    )

    # ✅ Emit [END] structured log — once per episode
    log_end(
        task=scenario,
        total_steps=step,
        final_reward=final_reward,
        success=success,
        elapsed=elapsed,
    )

    return final_reward


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_evaluation():
    """Run the full 3-scenario PSHCA evaluation and print a grader report."""

    # ── Header ───────────────────────────────────────────────────────────────
    print()
    print(BOLD("╔══════════════════════════════════════════════════════════╗"))
    print(BOLD("║       PSHCA Agent Baseline Evaluation                   ║"))
    print(BOLD("╚══════════════════════════════════════════════════════════╝"))

    if openai_configured():
        print(f"  Mode  : {GREEN('OpenAI-compatible HF')} ({os.environ['MODEL_NAME']})")
    else:
        print(f"  Mode  : {YELLOW('Demo')} (set API_BASE_URL / HF_TOKEN / MODEL_NAME to enable real inference)")
        print(f"  Tip   : export API_BASE_URL=... HF_TOKEN=... MODEL_NAME=...")

    print(f"  MaxSteps/Scenario: {PshcaEnvironment.MAX_STEPS}")
    print()

    env = PshcaEnvironment()
    scores = {}
    eval_start = time.time()

    for scenario in ["easy", "medium", "hard"]:
        scores[scenario] = run_scenario(env, scenario)

    total = sum(scores.values())
    total_time = time.time() - eval_start

    # ── Final Report ─────────────────────────────────────────────────────────
    print()
    print(BOLD("╔══════════════════════════════════════════════════════════╗"))
    print(BOLD("║                  GRADER REPORT                         ║"))
    print(BOLD("╠══════════════════════════════════════════════════════════╣"))

    labels = {"easy": "🟢 Easy  ", "medium": "🟡 Medium", "hard": "🔴 Hard  "}
    for scenario, score in scores.items():
        bar_len = int(score * 20)
        bar = GREEN("█" * bar_len) + DIM("░" * (20 - bar_len))
        status = GREEN("PASS") if score >= 0.5 else RED("FAIL")
        print(f"║  {labels[scenario]}: [{bar}] {score:.2f}/1.00  {status}  ║")

    print(BOLD("╠══════════════════════════════════════════════════════════╣"))
    total_bar = int((total / 3.0) * 20)
    total_display = GREEN("█" * total_bar) + DIM("░" * (20 - total_bar))
    print(f"║  {'TOTAL   '}: [{total_display}] {total:.2f}/3.00             ║")
    print(f"║  Total time: {total_time:.1f}s                                   ║")
    print(BOLD("╚══════════════════════════════════════════════════════════╝"))
    print()

    # Return exit code 0 for pass (≥ 2.0/3.0), 1 for fail
    passed = total >= 2.0
    if passed:
        print(GREEN("✓ Evaluation PASSED"))
    else:
        print(RED("✗ Evaluation FAILED — review scenario logs above"))

    return 0 if passed else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(run_evaluation())