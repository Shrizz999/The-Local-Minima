from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field
from typing import List, Optional, Literal

class GridEdgeAction(Action):
    hvac_operational_mode: Literal["off", "cooling", "heating"] = Field(..., description="Operational mode of the HVAC system.")
    hvac_temperature_setpoint: float = Field(..., ge=18.0, le=28.0, description="Target indoor temperature in °C. Must be between 18 and 28.")
    battery_dispatch_command: float = Field(..., ge=-5.0, le=5.0, description="Battery power in kW. Positive=charge, negative=discharge.")
    ev_charging_allocation: float = Field(..., ge=0.0, le=7.2, description="Power allocated to EV charger in kW. 0 means no charging.")
    grid_export_permission: bool = Field(..., description="Whether to allow selling excess solar back to the grid.")

class GridEdgeObservation(Observation):
    timestamp_iso: str = Field(..., description="ISO-8601 timestamp of the current simulation step.")
    current_grid_tariff: float = Field(..., description="Current electricity price in ₹/kWh (MSEDCL ToD rate).")
    forecast_grid_tariff: List[float] = Field(..., min_length=6, max_length=6, description="Predicted ₹/kWh tariff for the next 6 hours.")
    current_solar_yield: float = Field(..., ge=0.0, description="Current rooftop PV output in kW.")
    forecast_solar_yield: List[float] = Field(..., min_length=6, max_length=6, description="Predicted solar generation (kW) for next 6 hours.")
    home_battery_soc: float = Field(..., ge=0.0, le=1.0, description="Home battery state of charge (0.0=empty, 1.0=full).")
    electric_vehicle_soc: float = Field(..., ge=0.0, le=1.0, description="EV battery state of charge (0.0=empty, 1.0=full).")
    ev_connection_status: bool = Field(..., description="True if EV is physically plugged into the home charger.")
    indoor_ambient_temp: float = Field(..., description="Current indoor temperature in °C.")
    outdoor_ambient_temp: float = Field(..., description="Current outdoor temperature in °C.")
    system_diagnostic_msg: Optional[str] = Field(default=None, description="Human-readable warning if an action was invalid or overridden.")

class GridEdgeRewardInfo(Observation):
    aggregate_step_reward: float = Field(..., description="Final scalar reward passed to the RL optimizer.")
    financial_delta_component: float = Field(..., description="Reward portion from energy cost savings or export profits.")
    thermal_penalty_component: float = Field(..., le=0.0, description="Negative penalty for thermal discomfort. Always <= 0.")
    constraint_violation_flag: bool = Field(..., description="True if a physical constraint was violated this step.")
    action_loop_detected: bool = Field(..., description="True if the anti-loop penalty was triggered.")

class GridEdgeState(State):
    cumulative_financial_cost: float = Field(default=0.0, description="Running total of ₹ spent on grid imports minus export revenue.")
    building_thermal_inertia: float = Field(default=22.0, description="True indoor temperature (°C) including thermal mass effects.")
    true_occupancy_vector: List[bool] = Field( default_factory=list, description="Ground-truth occupancy schedule. Determines when comfort is enforced.")
    solar_utilized_kwh: float = Field(default=0.0, description="Cumulative solar energy actually used (for easy task grader).")
    solar_available_kwh: float = Field(default=0.0, description="Cumulative total solar energy available this episode.")
    rbc_baseline_cost: float = Field(default=0.0, description="Naive RBC cost accumulator (for medium task grader).")
