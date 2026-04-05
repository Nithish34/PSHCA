# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Predictive Self-Healing Cloud Architect (PSHCA) Environment — v2.0

Enhancements over v1:
  - Multiple scenario variants per difficulty (agent cannot memorise one answer)
  - Negative reward for repeated / destructive actions
  - Partial-progress reward signals across the full trajectory
  - Richer observation: latency_ms, error_rate, disk_io added
  - Realistic gradual metric degradation curves
  - Feedback string in observation explaining reward signal
"""

from uuid import uuid4
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PshcaAction, PshcaObservation
except ImportError:
    from models import PshcaAction, PshcaObservation


# ---------------------------------------------------------------------------
# Scenario Definitions — multiple variants per difficulty
# ---------------------------------------------------------------------------

EASY_SCENARIOS = [
    {
        "id": "E1",
        "title": "CPU Spike on Primary Web Server",
        "description": "web-server-01 is experiencing a rapid CPU spike due to a traffic burst. Scale or reboot before crash.",
        "spike_node": "web-server-01",
        "correct_actions": [("scale_up", "web-server-01"), ("reboot_server", "web-server-01")],
    },
    {
        "id": "E2",
        "title": "CPU Spike on Secondary Web Server",
        "description": "web-server-02 is overloading. Identify the correct node and mitigate.",
        "spike_node": "web-server-02",
        "correct_actions": [("scale_up", "web-server-02"), ("reboot_server", "web-server-02")],
    },
    {
        "id": "E3",
        "title": "Replica Database CPU Overload",
        "description": "db-replica is taking excess read traffic causing CPU overload. Reboot or scale.",
        "spike_node": "db-replica",
        "correct_actions": [("reboot_server", "db-replica"), ("scale_up", "db-replica")],
    },
]

MEDIUM_SCENARIOS = [
    {
        "id": "M1",
        "title": "Gradual Memory Leak on Primary DB",
        "description": "db-main shows a gradual memory leak. Clear cache or failover before OOM crash.",
        "leak_node": "db-main",
        "correct_actions": [("clear_cache", "db-main"), ("failover_db", "db-main")],
    },
    {
        "id": "M2",
        "title": "Memory Leak on Replica DB",
        "description": "db-replica memory is climbing. Clear its cache before it crashes.",
        "leak_node": "db-replica",
        "correct_actions": [("clear_cache", "db-replica"), ("reboot_server", "db-replica")],
    },
    {
        "id": "M3",
        "title": "Memory Pressure on Web Server",
        "description": "web-server-01 has a memory leak from a runaway worker process. Reboot to clear.",
        "leak_node": "web-server-01",
        "correct_actions": [("reboot_server", "web-server-01"), ("clear_cache", "web-server-01")],
    },
]

HARD_SCENARIOS = [
    {
        "id": "H1",
        "title": "Cascading Failure — Bad Deployment",
        "description": "A faulty deployment caused both web servers to spike, starving the DB. Rollback the deployment.",
        "correct_actions": [("rollback_deployment", "web-server-01")],
        "multi_step": False,
    },
    {
        "id": "H2",
        "title": "Cascading Failure — DB Connection Exhaustion",
        "description": "High web traffic exhausted DB connections causing cascade. Failover DB then scale web servers.",
        "correct_actions": [("failover_db", "db-main"), ("scale_up", "web-server-01")],
        "multi_step": True,
    },
    {
        "id": "H3",
        "title": "Cascading Failure — Memory Avalanche",
        "description": "Memory leak on db-main caused API timeouts, spiking web-server CPU. Clear cache first.",
        "correct_actions": [("clear_cache", "db-main"), ("scale_up", "web-server-01")],
        "multi_step": True,
    },
]


class PshcaEnvironment(Environment):
    """
    PSHCA v2.0 — Predictive Self-Healing Cloud Architect Environment.

    Resources: web-server-01, web-server-02, db-main, db-replica
    Tasks: easy (3 variants) | medium (3 variants) | hard (3 variants)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 20

    SCENARIOS = {
        "easy": EASY_SCENARIOS,
        "medium": MEDIUM_SCENARIOS,
        "hard": HARD_SCENARIOS,
    }

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.scenario = "easy"
        self.active_scenario: Dict[str, Any] = {}

        self.cpu_usage: Dict[str, float] = {}
        self.memory_usage: Dict[str, float] = {}
        self.latency_ms: Dict[str, float] = {}
        self.error_rate: Dict[str, float] = {}
        self.disk_io: Dict[str, float] = {}
        self.service_status: Dict[str, str] = {}
        self.active_alerts: List[str] = []

        self.scenario_state: int = 0
        self.correct_steps_taken: List[str] = []
        self.last_action: Optional[Dict] = None
        self.cumulative_reward: float = 0.0
        self.event_history: List[Dict[str, Any]] = []
        self.max_event_history: int = 200

    def get_task_info(self) -> str:
        if not self.active_scenario:
            return f"Task ({self.scenario}): Awaiting reset."
        d = self.scenario.capitalize()
        sid = self.active_scenario.get("id", "")
        title = self.active_scenario.get("title", "")
        desc = self.active_scenario.get("description", "")
        return f"[{d} | {sid}] {title} — {desc}"

    def _init_cloud_state(self):
        def j(base, spread=2.0):
            return round(base + random.uniform(-spread, spread), 1)

        self.cpu_usage    = {"web-server-01": j(20), "web-server-02": j(22), "db-main": j(15), "db-replica": j(10)}
        self.memory_usage = {"web-server-01": j(40), "web-server-02": j(41), "db-main": j(45), "db-replica": j(43)}
        self.latency_ms   = {"web-server-01": j(45,5), "web-server-02": j(48,5), "db-main": j(12,2), "db-replica": j(14,2)}
        self.error_rate   = {k: 0.0 for k in self.cpu_usage}
        self.disk_io      = {k: j(20, 5) for k in self.cpu_usage}
        self.service_status = {"api": "Healthy", "database": "Healthy", "cache": "Healthy"}
        self.active_alerts = []
        self.scenario_state = 0
        self.correct_steps_taken = []
        self.last_action = None
        self.cumulative_reward = 0.0
        self.event_history = []

    def _repeat_penalty(self, action: PshcaAction) -> float:
        if self.last_action is None:
            return 0.0
        if (self.last_action["action_type"] == action.action_type and
                self.last_action["target_resource"] == action.target_resource):
            return -0.05
        return 0.0

    def _clamp(self, v, lo=0.0, hi=100.0):
        return max(lo, min(hi, v))

    # ── Graders ──────────────────────────────────────────────────────────────

    def grader_easy(self, action: PshcaAction) -> tuple:
        node = self.active_scenario.get("spike_node", "web-server-01")
        correct = self.active_scenario.get("correct_actions", [])
        penalty = self._repeat_penalty(action)

        if action.action_type == "wait":
            return 0.05 + penalty, False, "Stalling: metrics worsen each step. Act now."

        if (action.action_type, action.target_resource) in correct:
            cpu_now = self.cpu_usage.get(node, 0)
            if cpu_now < 60:
                score, msg = 1.0, f"Perfect early intervention! {node} CPU stabilised."
            elif cpu_now < 85:
                score, msg = 0.9, f"Good — caught before crash. CPU was {cpu_now:.0f}%."
            else:
                score, msg = 0.75, f"Late but successful. CPU was critical at {cpu_now:.0f}%."
            self.cpu_usage[node] = max(20.0, self.cpu_usage[node] - 55.0)
            self.latency_ms[node] = 45.0
            self.error_rate[node] = 0.0
            self.active_alerts = []
            self.service_status["api"] = "Healthy"
            return score + penalty, True, msg

        if action.action_type in ("scale_up", "reboot_server"):
            return 0.2 + penalty, False, f"Right action but wrong node. Check which server is spiking — it is {node}."

        if action.action_type in ("failover_db", "rollback_deployment"):
            return -0.1 + penalty, False, "Destructive: DB actions are not needed for a CPU spike."

        return 0.0 + penalty, False, "No effect. Review active alerts and CPU metrics."

    def grader_medium(self, action: PshcaAction, leak_node: str) -> tuple:
        correct = self.active_scenario.get("correct_actions", [])
        penalty = self._repeat_penalty(action)

        if action.action_type == "wait":
            return 0.05 + penalty, False, "Memory leak worsening — every step counts."

        if (action.action_type, action.target_resource) in correct:
            mem_now = self.memory_usage.get(leak_node, 0)
            if mem_now < 65:
                score, msg = 1.0, "Proactive! Memory cleared before service degradation."
            elif mem_now < 85:
                score, msg = 0.85, f"Good response — memory was {mem_now:.0f}%, caught in time."
            else:
                score, msg = 0.7, f"Last-minute fix. Memory was critical at {mem_now:.0f}%."
            self.memory_usage[leak_node] = 45.0
            self.latency_ms[leak_node] = 14.0
            self.disk_io[leak_node] = 20.0
            self.error_rate[leak_node] = 0.0
            self.service_status["database"] = "Healthy"
            self.active_alerts = []
            return score + penalty, True, msg

        if action.action_type in ("clear_cache", "failover_db"):
            return 0.2 + penalty, False, f"Right idea but wrong target. The leak is on {leak_node}."

        if action.action_type == "rollback_deployment":
            return -0.15 + penalty, False, "Destructive: rollback won't fix a memory leak."

        return 0.0 + penalty, False, "No measurable effect. Focus on memory metrics."

    def grader_hard(self, action: PshcaAction) -> tuple:
        penalty = self._repeat_penalty(action)
        correct = self.active_scenario.get("correct_actions", [])
        multi = self.active_scenario.get("multi_step", False)

        if action.action_type == "wait":
            return 0.0 + penalty, False, "Cascading failure is accelerating — do not wait."

        pair = (action.action_type, action.target_resource)

        if not multi:
            if pair in correct:
                if self.scenario_state <= 1:
                    return 1.0 + penalty, True, "Rapid root-cause ID! Full score — cascade stopped."
                elif self.scenario_state == 2:
                    return 0.8 + penalty, True, "Correct — cascade halted."
                else:
                    return 0.55 + penalty, True, "Late but cluster recovered."
            if action.action_type in ("scale_up", "reboot_server"):
                return 0.1 + penalty, False, "Symptomatic fix — addresses overload but not root cause (bad deploy)."
            return -0.1 + penalty, False, "Wrong action worsens the cascade."

        else:
            step_key = f"{action.action_type}:{action.target_resource}"
            if pair == correct[0] and len(self.correct_steps_taken) == 0:
                self.correct_steps_taken.append(step_key)
                return 0.5 + penalty, False, "Step 1 correct! Now address the secondary failure."
            if pair == correct[1] and len(self.correct_steps_taken) == 1:
                self.correct_steps_taken.append(step_key)
                return 1.0 + penalty, True, "Both steps correct — cascade fully resolved!"
            if pair in correct and step_key in self.correct_steps_taken:
                return 0.3 + penalty, False, "Already done. Address the next failure point."
            return 0.0 + penalty, False, "Incorrect step in cascade resolution sequence."

    # ── Record / snapshot ────────────────────────────────────────────────────

    def _record_event(self, action, reward, done, feedback=""):
        self.event_history.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "episode_id": self._state.episode_id,
            "step": self._state.step_count,
            "scenario": self.scenario,
            "scenario_id": self.active_scenario.get("id", ""),
            "action": {"action_type": action.action_type, "target_resource": action.target_resource},
            "reward": reward,
            "cumulative_reward": self.cumulative_reward,
            "done": done,
            "feedback": feedback,
            "alerts": list(self.active_alerts),
            "cpu": self.cpu_usage.copy(),
            "memory": self.memory_usage.copy(),
            "latency": self.latency_ms.copy(),
            "error_rate": self.error_rate.copy(),
            "service_status": self.service_status.copy(),
        })
        if len(self.event_history) > self.max_event_history:
            self.event_history = self.event_history[-self.max_event_history:]

    def get_dashboard_snapshot(self):
        return {
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "max_steps": self.MAX_STEPS,
            "scenario": self.scenario,
            "scenario_id": self.active_scenario.get("id", ""),
            "scenario_title": self.active_scenario.get("title", ""),
            "task_info": self.get_task_info(),
            "cpu_usage": self.cpu_usage.copy(),
            "memory_usage": self.memory_usage.copy(),
            "latency_ms": self.latency_ms.copy(),
            "error_rate": self.error_rate.copy(),
            "disk_io": self.disk_io.copy(),
            "service_status": self.service_status.copy(),
            "active_alerts": list(self.active_alerts),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "recent_events": list(self.event_history[-25:]),
        }

    def _get_observation(self, reward=None, done=False, feedback=""):
        return PshcaObservation(
            cpu_usage=self.cpu_usage.copy(),
            memory_usage=self.memory_usage.copy(),
            latency_ms=self.latency_ms.copy(),
            error_rate=self.error_rate.copy(),
            disk_io=self.disk_io.copy(),
            active_alerts=list(self.active_alerts),
            service_status=self.service_status.copy(),
            current_task_info=self.get_task_info(),
            reward=reward,
            done=done,
            metadata={
                "scenario": self.scenario,
                "scenario_id": self.active_scenario.get("id", ""),
                "scenario_title": self.active_scenario.get("title", ""),
                "latency_ms": self.latency_ms.copy(),
                "error_rate": self.error_rate.copy(),
                "disk_io": self.disk_io.copy(),
                "cumulative_reward": self.cumulative_reward,
                "scenario_state": self.scenario_state,
                "feedback": feedback,
                "step_count": self._state.step_count,
                "episode_id": self._state.episode_id,
            }
        )

    # ── OpenEnv Interface ────────────────────────────────────────────────────

    def reset(self) -> PshcaObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        cycle = self._reset_count % 3
        self.scenario = "easy" if cycle == 1 else ("medium" if cycle == 2 else "hard")
        self.active_scenario = random.choice(self.SCENARIOS[self.scenario])
        self._init_cloud_state()
        self._record_event(PshcaAction(action_type="wait", target_resource=""), 0.0, False, "Episode started.")
        return self._get_observation(feedback="Environment reset. Observe metrics carefully.")

    def step(self, action: PshcaAction) -> PshcaObservation:  # type: ignore[override]
        self._state.step_count += 1
        action_valid = True
        if action.action_type in ("reboot_server", "scale_up", "clear_cache", "failover_db"):
            if action.target_resource not in self.cpu_usage:
                action_valid = False

        if self.scenario == "easy":
            reward, done, feedback = self._step_easy(action, action_valid)
        elif self.scenario == "medium":
            reward, done, feedback = self._step_medium(action, action_valid)
        else:
            reward, done, feedback = self._step_hard(action, action_valid)

        if self._state.step_count >= self.MAX_STEPS:
            done = True
            if reward < 0.5:
                reward, feedback = 0.0, "Timeout — cluster crashed. Episode failed."

        final_score = max(-1.0, min(1.0, reward))
        self.cumulative_reward += final_score
        self.last_action = {"action_type": action.action_type, "target_resource": action.target_resource}
        self._record_event(action, final_score, done, feedback)
        return self._get_observation(reward=final_score, done=done, feedback=feedback)

    # ── Scenario step logic ───────────────────────────────────────────────────

    def _step_easy(self, action, action_valid):
        node = self.active_scenario.get("spike_node", "web-server-01")
        self.scenario_state += 1
        ramp = [8, 12, 18, 25, 30]
        spike = ramp[min(self.scenario_state - 1, len(ramp) - 1)]
        self.cpu_usage[node] = self._clamp(self.cpu_usage[node] + spike)
        self.latency_ms[node] = self._clamp(self.latency_ms[node] + spike * 3, lo=0, hi=5000)
        self.error_rate[node] = self._clamp(self.error_rate[node] + self.scenario_state * 0.5, lo=0, hi=100)

        if self.cpu_usage[node] >= 70 and not any("CPU" in a for a in self.active_alerts):
            self.active_alerts.append(f"[Warning] {node} CPU at {self.cpu_usage[node]:.0f}% — rising")
        if self.cpu_usage[node] >= 90:
            self.active_alerts = [f"[Critical] {node} CPU at {self.cpu_usage[node]:.0f}% — imminent crash"]
            self.service_status["api"] = "Degraded"

        r, d, f = self.grader_easy(action)
        if not d and self.cpu_usage[node] >= 100.0:
            self.cpu_usage[node] = 100.0
            self.service_status["api"] = "Offline"
            self.active_alerts = [f"[Fatal] {node} crashed — CPU overload"]
            return 0.0, True, f"FAILURE: {node} crashed. Respond faster."
        return r, d, f

    def _step_medium(self, action, action_valid):
        node = self.active_scenario.get("leak_node", "db-main")
        self.scenario_state += 1
        leak = [8, 10, 13, 17, 22]
        growth = leak[min(self.scenario_state - 1, len(leak) - 1)]
        self.memory_usage[node] = self._clamp(self.memory_usage[node] + growth)
        self.latency_ms[node] = self._clamp(self.latency_ms[node] + growth * 2, lo=0, hi=5000)
        self.disk_io[node] = self._clamp(self.disk_io[node] + growth * 0.5, lo=0, hi=100)

        if self.memory_usage[node] >= 65 and not any("Memory" in a for a in self.active_alerts):
            self.active_alerts.append(f"[Warning] {node} memory at {self.memory_usage[node]:.0f}% — leak detected")
            self.service_status["database"] = "Degraded"
        if self.memory_usage[node] >= 88:
            self.active_alerts = [f"[Critical] {node} memory at {self.memory_usage[node]:.0f}% — OOM imminent"]
            self.error_rate[node] = min(100, self.error_rate[node] + 20)

        r, d, f = self.grader_medium(action, node)
        if not d and self.memory_usage[node] >= 100.0:
            self.service_status["database"] = "Offline"
            self.active_alerts = [f"[Fatal] {node} OOM crash"]
            return 0.0, True, f"FAILURE: {node} ran out of memory."
        return r, d, f

    def _step_hard(self, action, action_valid):
        self.scenario_state += 1
        sid = self.active_scenario.get("id", "H1")

        if sid == "H1":
            if self.scenario_state == 1:
                self.cpu_usage["web-server-01"] = 88.0
                self.cpu_usage["web-server-02"] = 85.0
                self.latency_ms["web-server-01"] = 820
                self.active_alerts = ["[Warning] API Traffic Spike post-deployment"]
            elif self.scenario_state == 2:
                self.cpu_usage["db-main"] = 92.0
                self.service_status["api"] = "Degraded"
                self.error_rate["web-server-01"] = 18.0
                self.active_alerts.append("[Critical] DB Overload — bad queries from new deployment")
            elif self.scenario_state == 3:
                self.service_status["database"] = "Degraded"
                self.error_rate["db-main"] = 35.0
                self.latency_ms["db-main"] = 1200
        elif sid == "H2":
            if self.scenario_state == 1:
                self.cpu_usage["web-server-01"] = 80.0
                self.memory_usage["db-main"] = 88.0
                self.service_status["database"] = "Degraded"
                self.active_alerts = ["[Critical] DB connection pool exhausted"]
            elif self.scenario_state == 2:
                self.error_rate["db-main"] = 42.0
                self.latency_ms["web-server-01"] = 950
                self.active_alerts.append("[Warning] API response times degraded")
        elif sid == "H3":
            if self.scenario_state == 1:
                self.memory_usage["db-main"] = 85.0
                self.latency_ms["db-main"] = 900
                self.active_alerts = ["[Critical] DB memory leak causing query timeouts"]
            elif self.scenario_state == 2:
                self.cpu_usage["web-server-01"] = 87.0
                self.cpu_usage["web-server-02"] = 82.0
                self.service_status["api"] = "Degraded"
                self.active_alerts.append("[Warning] Web servers overloaded from DB timeouts")

        r, d, f = self.grader_hard(action)
        if d and r > 0:
            self.cpu_usage    = {k: 20.0 for k in self.cpu_usage}
            self.memory_usage = {k: 40.0 for k in self.memory_usage}
            self.latency_ms   = {k: 45.0 for k in self.latency_ms}
            self.error_rate   = {k: 0.0  for k in self.error_rate}
            self.service_status = {"api": "Healthy", "database": "Healthy", "cache": "Healthy"}
            self.active_alerts = []

        if not d and self.scenario_state >= 5:
            self.service_status = {"api": "Offline", "database": "Offline", "cache": "Offline"}
            self.active_alerts = ["[Fatal] Complete cluster outage"]
            return 0.0, True, "FAILURE: Full cluster outage. Cascade uncontrolled."
        return r, d, f

    @property
    def state(self) -> State:
        return self._state