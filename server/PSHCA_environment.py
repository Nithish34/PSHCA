# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Predictive Self-Healing Cloud Architect (PSHCA) Environment Implementation.

A sophisticated simulation of cloud infrastructure experiencing various anomalies
(CPU spikes, Memory Leaks, Cascading failures) that an AI agent must diagnose and resolve.
"""

from uuid import uuid4
import random
from datetime import datetime, timezone
from typing import Any, Dict, List

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PshcaAction, PshcaObservation
except ImportError:
    from models import PshcaAction, PshcaObservation


class PshcaEnvironment(Environment):
    """
    PSHCA Simulation Environment.
    
    This simulates 4 core resources: web-server-01, web-server-02, db-main, db-replica.
    It contains 3 tasks (scenarios):
        - easy: web-server-01 CPU spike
        - medium: db-main memory leak
        - hard: cascading failure across multiple resources
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 20

    def __init__(self):
        """Initialize the PSHCA environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.scenario = "easy"
        
        # Internal Cloud State
        self.cpu_usage = {}
        self.memory_usage = {}
        self.service_status = {}
        self.active_alerts = []
        self.scenario_state = 0 # Track progress of the scenario
        self.event_history: List[Dict[str, Any]] = []
        self.max_event_history = 200

    def get_task_info(self) -> str:
        if self.scenario == "easy":
            return "Task 1 (Easy): High CPU Spike reported on a web server. Diagnose and mitigate before crash."
        elif self.scenario == "medium":
            return "Task 2 (Medium): Gradual DB Memory Leak detected. Preemptively heal the primary database."
        else:
            return "Task 3 (Hard): Cascading Deployment Failure. Multiple services failing. Rollback deployment to stabilize."

    def _init_cloud_state(self):
        """Reset the internal cloud state to healthy."""
        self.cpu_usage = {
            "web-server-01": 20.0,
            "web-server-02": 22.0,
            "db-main": 15.0,
            "db-replica": 10.0,
        }
        self.memory_usage = {
            "web-server-01": 40.0,
            "web-server-02": 41.0,
            "db-main": 45.0,
            "db-replica": 45.0,
        }
        self.service_status = {
            "api": "Healthy",
            "database": "Healthy"
        }
        self.active_alerts = []
        self.scenario_state = 0
        self.event_history = []

    def grader_easy(self, action: PshcaAction) -> tuple[float, bool]:
        """Grade the easy scenario with partial credit for partial mitigation."""
        if action.action_type == "wait":
            return 0.1, False

        if action.action_type == "reboot_server" and action.target_resource == "web-server-01":
            return 1.0, True

        if action.action_type == "scale_up" and action.target_resource == "web-server-01":
            if self.cpu_usage["web-server-01"] >= 95.0:
                self.cpu_usage["web-server-01"] = 70.0
                return 0.4, False
            self.cpu_usage["web-server-01"] = max(20.0, self.cpu_usage["web-server-01"] - 30.0)
            if self.cpu_usage["web-server-01"] < 85.0:
                return 0.7, True
            return 0.4, False

        if action.action_type in ["reboot_server", "scale_up"] and action.target_resource in self.cpu_usage:
            return 0.25, False

        return 0.0, False

    def grader_medium(self, action: PshcaAction) -> tuple[float, bool]:
        """Grade the medium scenario with partial credit based on leak severity."""
        if action.action_type == "wait":
            return 0.1, False

        if action.action_type == "clear_cache" and action.target_resource == "db-main":
            if self.memory_usage["db-main"] >= 90.0:
                self.memory_usage["db-main"] = 75.0
                return 0.4, False
            self.memory_usage["db-main"] = max(45.0, self.memory_usage["db-main"] - 30.0)
            return 0.7 if self.memory_usage["db-main"] < 80.0 else 0.4, self.memory_usage["db-main"] < 80.0

        if action.action_type == "failover_db" and action.target_resource == "db-main":
            if self.memory_usage["db-main"] >= 75.0:
                self.memory_usage["db-main"] = 45.0
                self.service_status["database"] = "Healthy"
                self.active_alerts = []
                return 1.0, True
            return 0.6, True

        if action.action_type in ["clear_cache", "failover_db"] and action.target_resource in self.cpu_usage:
            return 0.25, False

        return 0.0, False

    def grader_hard(self, action: PshcaAction) -> tuple[float, bool]:
        """Grade the hard scenario with timing-aware partial credit."""
        if action.action_type == "wait":
            return 0.0, False

        if action.action_type == "rollback_deployment":
            if self.scenario_state <= 2:
                return 1.0, True
            if self.scenario_state == 3:
                return 0.7, True
            return 0.4, True

        return 0.0, False

    def _record_event(self, action: PshcaAction, reward: float, done: bool):
        """Store a compact, replay-friendly event used by the dashboard and demos."""
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "episode_id": self._state.episode_id,
            "step": self._state.step_count,
            "scenario": self.scenario,
            "action": {
                "action_type": action.action_type,
                "target_resource": action.target_resource,
            },
            "reward": reward,
            "done": done,
            "alerts": list(self.active_alerts),
            "cpu": self.cpu_usage.copy(),
            "memory": self.memory_usage.copy(),
            "service_status": self.service_status.copy(),
        }
        self.event_history.append(event)
        if len(self.event_history) > self.max_event_history:
            self.event_history = self.event_history[-self.max_event_history :]

    def get_dashboard_snapshot(self) -> Dict[str, Any]:
        """Expose a full environment snapshot for dashboard polling/streaming."""
        return {
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "max_steps": self.MAX_STEPS,
            "scenario": self.scenario,
            "task_info": self.get_task_info(),
            "cpu_usage": self.cpu_usage.copy(),
            "memory_usage": self.memory_usage.copy(),
            "service_status": self.service_status.copy(),
            "active_alerts": list(self.active_alerts),
            "recent_events": list(self.event_history[-25:]),
        }

    def _get_observation(self) -> PshcaObservation:
        """Construct the next observation matching the Pydantic schema."""
        return PshcaObservation(
            cpu_usage=self.cpu_usage.copy(),
            memory_usage=self.memory_usage.copy(),
            active_alerts=list(self.active_alerts),
            service_status=self.service_status.copy(),
            current_task_info=self.get_task_info()
        )

    def reset(self) -> PshcaObservation:
        """
        Reset the environment, pick the next scenario (easy -> medium -> hard).
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        
        # Cycle through scenarios based on reset count
        if self._reset_count % 3 == 1:
            self.scenario = "easy"
        elif self._reset_count % 3 == 2:
            self.scenario = "medium"
        else:
            self.scenario = "hard"

        self._init_cloud_state()
        self._record_event(
            PshcaAction(action_type="wait", target_resource=""),
            reward=0.0,
            done=False,
        )

        return self._get_observation()

    def step(self, action: PshcaAction) -> PshcaObservation:  # type: ignore[override]
        """
        Execute a self-healing action and return observation and grader reward.
        """
        self._state.step_count += 1
        reward = 0.0
        done = False
        info = {"task": self.scenario}
        
        # 1. Apply Action
        action_valid = True
        if action.action_type in ["reboot_server", "scale_up", "clear_cache", "failover_db"]:
            if action.target_resource not in self.cpu_usage:
                reward -= 0.5  # Invalid resource
                action_valid = False
                
        # 2. Progress the Simulation (Environment Dynamics) & Calculate Reward
        if self.scenario == "easy":
            reward, done = self._step_easy(action, action_valid)
        elif self.scenario == "medium":
            reward, done = self._step_medium(action, action_valid)
        elif self.scenario == "hard":
            reward, done = self._step_hard(action, action_valid)

        # Truncate if max steps reached
        if self._state.step_count >= self.MAX_STEPS:
            done = True
            if reward < 1.0:
                reward = 0.0 # Time out is a failure

        # Final score scaling: 0.0 to 1.0 needed!
        # Ensure reward stays in [0.0, 1.0] for grader compatibility
        final_score = max(0.0, min(1.0, reward))
        self._record_event(action=action, reward=final_score, done=done)

        return PshcaObservation(
            cpu_usage=self.cpu_usage.copy(),
            memory_usage=self.memory_usage.copy(),
            active_alerts=list(self.active_alerts),
            service_status=self.service_status.copy(),
            current_task_info=self.get_task_info(),
            reward=final_score,
            done=done,
            metadata={
                **info,
                "scenario_state": self.scenario_state,
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "recent_event": self.event_history[-1] if self.event_history else None,
            }
        )

    def _step_easy(self, action: PshcaAction, action_valid: bool):
        reward = 0.1 # Small uptime reward
        done = False
        
        # CPU spikes rapidly
        self.scenario_state += 1
        self.cpu_usage["web-server-01"] += 25.0
        
        if self.cpu_usage["web-server-01"] >= 85.0:
            self.active_alerts = ["[Warning] web-server-01 CPU > 85%"]

        graded_reward, graded_done = self.grader_easy(action)
        if graded_reward > reward:
            reward = graded_reward
        done = done or graded_done

        # Lose Condition
        if not done and self.cpu_usage["web-server-01"] >= 100.0:
            self.service_status["api"] = "Degraded"
            self.cpu_usage["web-server-01"] = 100.0
            self.active_alerts = ["[Critical] web-server-01 Crashed due to high CPU"]
            reward = 0.0
            done = True
            
        return reward, done

    def _step_medium(self, action: PshcaAction, action_valid: bool):
        reward = 0.1
        done = False
        
        self.scenario_state += 1
        self.memory_usage["db-main"] += 15.0
        
        if self.memory_usage["db-main"] >= 75.0:
            self.active_alerts = ["[Warning] db-main Memory Leak Detected"]
            self.service_status["database"] = "Degraded"

        graded_reward, graded_done = self.grader_medium(action)
        if graded_reward > reward:
            reward = graded_reward
        done = done or graded_done

        # Lose Condition
        if not done and self.memory_usage["db-main"] >= 100.0:
            self.memory_usage["db-main"] = 100.0
            self.service_status["database"] = "Offline"
            self.active_alerts = ["[Critical] DB OOM Crash"]
            reward = 0.0
            done = True

        return reward, done

    def _step_hard(self, action: PshcaAction, action_valid: bool):
        reward = 0.1
        done = False
        
        self.scenario_state += 1
        
        # Cascading failure: web servers overload, then DB overloads
        if self.scenario_state == 1:
            self.cpu_usage["web-server-01"] = 90.0
            self.cpu_usage["web-server-02"] = 90.0
            self.active_alerts.append("[Warning] API Traffic Spike post-deployment")
        elif self.scenario_state == 2:
            self.cpu_usage["db-main"] = 95.0
            self.service_status["api"] = "Degraded"
            self.active_alerts.append("[Critical] DB Overload due to bad queries")
            
        graded_reward, graded_done = self.grader_hard(action)
        if graded_reward > reward:
            reward = graded_reward
        if graded_done:
            self.cpu_usage = {k: 20.0 for k in self.cpu_usage}
            self.memory_usage = {k: 40.0 for k in self.memory_usage}
            self.service_status = {"api": "Healthy", "database": "Healthy"}
            self.active_alerts = []
        done = done or graded_done

        # Lose Condition
        if not done and self.scenario_state >= 4:
            self.service_status = {"api": "Offline", "database": "Offline"}
            self.active_alerts = ["[Fatal] Complete Cluster Outage"]
            reward = 0.0
            done = True
            
        return reward, done

    @property
    def state(self) -> State:
        return self._state