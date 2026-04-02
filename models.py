# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the PSHCA Environment.

The Predictive Self-Healing Cloud Architect (PSHCA) environment simulates
cloud infrastructure issues requiring self-healing actions.
"""

from typing import Dict, List, Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class PshcaAction(Action):
    """Action for the Pshca environment - self healing controls."""

    action_type: Literal[
        "reboot_server",
        "scale_up",
        "rollback_deployment",
        "clear_cache",
        "failover_db",
        "wait"
    ] = Field(..., description="The self-healing action to perform")
    
    target_resource: str = Field(
        default="", 
        description="The ID of the server or service to apply the action to (e.g., 'web-server-01', 'db-main')"
    )


class PshcaObservation(Observation):
    """Observation from the Pshca environment - cloud telemetry."""

    cpu_usage: Dict[str, float] = Field(
        default_factory=dict, description="CPU usage percentage per server (0.0 to 100.0)"
    )
    memory_usage: Dict[str, float] = Field(
        default_factory=dict, description="Memory usage percentage per server (0.0 to 100.0)"
    )
    active_alerts: List[str] = Field(
        default_factory=list, description="Active system warnings or critical alerts"
    )
    service_status: Dict[str, str] = Field(
        default_factory=dict, description="Current health status of core services (e.g., 'Healthy', 'Degraded')"
    )
    current_task_info: str = Field(
        default="", description="Information about the current scenario the agent is facing"
    )