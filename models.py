# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the PSHCA Environment — v2.0.

The Predictive Self-Healing Cloud Architect (PSHCA) environment simulates
cloud infrastructure issues requiring self-healing actions. v2.0 exposes
latency, error-rate, and disk I/O as first-class observation fields alongside
the original CPU and memory metrics.
"""

from typing import Dict, List, Literal, Optional

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
        "wait",
        "escalate_to_human"
    ] = Field(..., description="The self-healing action to perform")
    
    target_resource: str = Field(
        default="", 
        description="The ID of the server or service to apply the action to (e.g., 'web-server-01', 'db-main')"
    )


class PshcaObservation(Observation):
    """Observation from the PSHCA environment — full v2.0 cloud telemetry."""

    cpu_usage: Dict[str, float] = Field(
        default_factory=dict, description="CPU usage percentage per server (0.0 to 100.0)"
    )
    memory_usage: Dict[str, float] = Field(
        default_factory=dict, description="Memory usage percentage per server (0.0 to 100.0)"
    )
    latency_ms: Dict[str, float] = Field(
        default_factory=dict,
        description="Request latency in milliseconds per server. Spikes signal imminent failures.",
    )
    error_rate: Dict[str, float] = Field(
        default_factory=dict,
        description="Request error rate percentage per server (0.0 to 100.0). Non-zero = service degradation.",
    )
    disk_io: Dict[str, float] = Field(
        default_factory=dict,
        description="Disk I/O usage percentage per server (0.0 to 100.0). High on DB nodes may indicate paging.",
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