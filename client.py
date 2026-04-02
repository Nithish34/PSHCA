# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PSHCA Environment Client — connects to the running FastAPI server via WebSocket."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import PshcaAction, PshcaObservation


class PshcaEnv(EnvClient[PshcaAction, PshcaObservation, State]):
    """
    Client for the PSHCA (Predictive Self-Healing Cloud Architect) Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example — local server:
        >>> with PshcaEnv(base_url="http://localhost:8000") as client:
        ...     obs = client.reset().observation
        ...     print(obs.current_task_info)
        ...     result = client.step(PshcaAction(
        ...         action_type="scale_up", target_resource="web-server-01"
        ...     ))
        ...     print(result.observation.service_status)

    Example — Docker image:
        >>> client = PshcaEnv.from_docker_image("predictive-cloud-architect:latest")
        >>> try:
        ...     client.reset()
        ...     client.step(PshcaAction(action_type="rollback_deployment", target_resource="web-server-01"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: PshcaAction) -> Dict:
        """
        Convert PshcaAction to JSON payload for the step WebSocket message.

        Args:
            action: PshcaAction instance with action_type and target_resource.

        Returns:
            Dictionary suitable for JSON encoding and sending to the server.
        """
        return {
            "action_type": action.action_type,
            "target_resource": action.target_resource,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PshcaObservation]:
        """
        Parse server response into StepResult[PshcaObservation].

        Args:
            payload: JSON response data from the server.

        Returns:
            StepResult containing the cloud telemetry observation.
        """
        obs_data = payload.get("observation", {})
        observation = PshcaObservation(
            cpu_usage=obs_data.get("cpu_usage", {}),
            memory_usage=obs_data.get("memory_usage", {}),
            active_alerts=obs_data.get("active_alerts", []),
            service_status=obs_data.get("service_status", {}),
            current_task_info=obs_data.get("current_task_info", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into a State object.

        Args:
            payload: JSON response from the /state endpoint.

        Returns:
            State object with episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
