# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import GridEdgeAction, GridEdgeObservation, GridEdgeState

class GridEdgeEnv(EnvClient[GridEdgeAction, GridEdgeObservation, GridEdgeState]):
    """
    Client for the Grid Edge Home Energy Orchestrator environment.
    This client maintains a persistent websockets connection to the environment server.
    """

    def _step_payload(self, action: GridEdgeAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[GridEdgeObservation]:
        obs = payload.get("observation", {})

        observation = GridEdgeObservation(
            # Time
            timestamp_iso=obs.get("timestamp_iso", ""),

            # Economic
            current_grid_tariff=obs.get("current_grid_tariff", 0.0),
            forecast_grid_tariff=obs.get("forecast_grid_tariff", [0.0] * 6),

            # Solar
            current_solar_yield=obs.get("current_solar_yield", 0.0),
            forecast_solar_yield=obs.get("forecast_solar_yield", [0.0] * 6),

            # Battery
            home_battery_soc=obs.get("home_battery_soc", 0.5),

            # EV
            electric_vehicle_soc=obs.get("electric_vehicle_soc", 0.2),
            ev_connection_status=obs.get("ev_connection_status", False),

            # Thermal
            indoor_ambient_temp=obs.get("indoor_ambient_temp", 22.0),
            outdoor_ambient_temp=obs.get("outdoor_ambient_temp", 25.0),

            # Episode signals
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),

            # Diagnostics
            system_diagnostic_msg=obs.get("system_diagnostic_msg", None),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> GridEdgeState:
        return GridEdgeState(
            episode_id=payload.get("episode_id", None),
            step_count=payload.get("step_count", 0),
            cumulative_financial_cost=payload.get("cumulative_financial_cost", 0.0),
            building_thermal_inertia=payload.get("building_thermal_inertia", 22.0),
            true_occupancy_vector=payload.get("true_occupancy_vector", []),
            solar_utilized_kwh=payload.get("solar_utilized_kwh", 0.0),
            solar_available_kwh=payload.get("solar_available_kwh", 0.0),
            rbc_baseline_cost=payload.get("rbc_baseline_cost", 0.0),
        )