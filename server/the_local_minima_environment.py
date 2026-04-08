from uuid import uuid4
from datetime import datetime, timedelta
from typing import Optional, List
from openenv.core.env_server.interfaces import Environment
from models import GridEdgeAction, GridEdgeObservation, GridEdgeRewardInfo, GridEdgeState
import math
import pvlib

def load_epw(path: str) -> list[dict]:
    weather_df, _ = pvlib.iotools.read_epw(path)
    
    hours = []
    for _, row in weather_df.iterrows():
        hours.append({
            "outdoor_temp": float(row["temp_air"]),
            "ghi": float(row["ghi"]),
        })
    
    return hours

def get_tariff(hour: int) -> float:
    if 6  <= hour < 9:  return 6.50
    if 9  <= hour < 18: return 4.20
    if 18 <= hour < 22: return 8.10
    return 3.80

class HomeConfig:
    # PV system
    panel_area_m2: float = 20.0
    panel_efficiency: float = 0.18

    # Battery
    battery_capacity_kwh: float = 13.5
    battery_max_kw: float = 5.0
    inverter_efficiency: float = 0.95

    # EV
    ev_capacity_kwh: float = 40.0
    ev_max_kw: float = 7.2
    ev_departure_step: int = 28      # step 28 = 7:00 AM
    ev_min_departure_soc: float = 0.80

    # HVAC
    hvac_power_kw: float = 2.0
    hvac_cop: float = 3.0

    # Building thermal
    thermal_mass_kjperdegc: float = 5000.0   # C_bldg
    insulation_resistance: float = 0.05       # R_bldg

    # Comfort
    comfort_deadband_degc: float = 1.5
    thermal_penalty_weight: float = 0.3

    loop_threshold: float = 3.0
    loop_penalty_base: float = 0.2

    # Reward normalization bounds — map raw reward to [0, 1].
    reward_raw_min: float = -25.0
    reward_raw_max: float =   6.0

    load_profile: List[float] = [0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.6, 0.9, 1.0, 0.8, 0.7, 0.7, 0.7, 0.6, 0.6, 0.7, 0.8, 1.2, 1.5, 1.8, 1.6, 1.2, 0.8, 0.5]


class TheLocalMinimaEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state: Optional[GridEdgeState] = None
        self._obs: Optional[GridEdgeObservation] = None
        self._task: str = "solar_self_consumption"
        self._config: HomeConfig = HomeConfig()

        self._weather_data: Optional[List[dict]] = None
        self._episode_start_hour: int = 0

        self._recent_actions: List[dict] = []
        self._consecutive_identical: int = 0

        self.max_steps = 96

    def load_weather(self, weather_data: List[dict], episode_start_hour: int = 0):
        self._weather_data = weather_data
        self._episode_start_hour = episode_start_hour

    def reset(self, task: str = "solar_self_consumption", config: Optional[HomeConfig] = None) -> GridEdgeObservation:
        self._task = task
        self._config = config or HomeConfig()
 
        self._recent_actions = []
        self._consecutive_identical = 0
 
        self._state = GridEdgeState(
            episode_id=str(uuid4()),
            step_count=0,
            cumulative_financial_cost=0.0,
            building_thermal_inertia=22.0,
            true_occupancy_vector=self._build_occupancy_vector(),
            solar_utilized_kwh=0.0,
            solar_available_kwh=0.0,
            rbc_baseline_cost=0.0,
        )
 
        self._obs = GridEdgeObservation(
            timestamp_iso=self._step_to_iso(0),
            current_grid_tariff=get_tariff(0),
            forecast_grid_tariff=[get_tariff(h) for h in range(1, 7)],
            current_solar_yield=self._solar_kw(0),
            forecast_solar_yield=[self._solar_kw(s) for s in range(1, 7)],
            home_battery_soc=0.5,
            electric_vehicle_soc=0.2,
            ev_connection_status=self._ev_connected(0),
            indoor_ambient_temp=22.0,
            outdoor_ambient_temp=self._outdoor_temp(0),
            reward=None,
            done=False,
            system_diagnostic_msg="System initialized.",
        )
 
        return self._obs

    def step(self, action: GridEdgeAction) -> GridEdgeObservation:
        cfg = self._config
        step_idx = self._state.step_count
        hour = self._step_to_hour(step_idx)
        occupied = self._state.true_occupancy_vector[step_idx]
 
        solar_kw = self._solar_kw(step_idx)
        tariff = get_tariff(hour)
        base_load = cfg.load_profile[hour]

        battery_cmd, batt_violated = self._clamp_battery(action.battery_dispatch_command)
        ev_cmd, ev_violated= self._clamp_ev(action.ev_charging_allocation)
        constraint_violated = batt_violated or ev_violated

        hvac_draw = cfg.hvac_power_kw if action.hvac_operational_mode != "off" else 0.0
        total_load = base_load + hvac_draw + ev_cmd + max(0.0, battery_cmd)
        total_gen = solar_kw + max(0.0, -battery_cmd) * cfg.inverter_efficiency
        net_grid = total_load - total_gen
        if net_grid < 0 and not action.grid_export_permission:
            net_grid = 0.0

        new_batt_soc = self._update_battery_soc(battery_cmd)
        new_ev_soc = self._update_ev_soc(ev_cmd)

        new_indoor_temp = self._update_indoor_temp(
            action.hvac_operational_mode,
            action.hvac_temperature_setpoint,
            self._outdoor_temp(step_idx),
            self._ghi(step_idx),
        )

        energy_cost = net_grid * tariff * 0.25
        self._state.cumulative_financial_cost += energy_cost
        self._state.rbc_baseline_cost += base_load * tariff * 0.25

        self._state.solar_available_kwh += solar_kw * 0.25
        self._state.solar_utilized_kwh += min(solar_kw, total_load) * 0.25
 
        financial_delta = -energy_cost
        thermal_penalty = self._thermal_penalty(new_indoor_temp, action.hvac_temperature_setpoint, occupied)
        ev_penalty = self._ev_departure_penalty(step_idx, new_ev_soc)
        violation_penalty = -0.5 if constraint_violated else 0.0
        loop_penalty, loop_detected = self._loop_penalty(action.model_dump(), financial_delta + thermal_penalty)
 
        reward = financial_delta + thermal_penalty + ev_penalty + violation_penalty + loop_penalty

        self._state.step_count += 1
        self._state.building_thermal_inertia = new_indoor_temp
        done = self._state.step_count >= self.max_steps

        diag = "OK"
        if constraint_violated:
            diag = "WARNING: Physical constraint violated - command clamped."
        if ev_penalty < 0:
            diag += "CRITICAL: EV departed below 80% SoC."

        next_step = self._state.step_count
        norm_reward = self._normalize_reward(reward)

        self._obs = GridEdgeObservation(
            timestamp_iso=self._step_to_iso(next_step),
            current_grid_tariff=get_tariff(self._step_to_hour(next_step)),
            forecast_grid_tariff=[get_tariff(self._step_to_hour(next_step + i)) for i in range(1, 7)],
            current_solar_yield=self._solar_kw(next_step),
            forecast_solar_yield=[self._solar_kw(next_step + i) for i in range(1, 7)],
            home_battery_soc=new_batt_soc,
            electric_vehicle_soc=new_ev_soc,
            ev_connection_status=self._ev_connected(next_step),
            indoor_ambient_temp=new_indoor_temp,
            outdoor_ambient_temp=self._outdoor_temp(next_step),
            reward=norm_reward,
            done=done,
            system_diagnostic_msg=diag,
        )
 
        self._last_reward_info = GridEdgeRewardInfo(
            aggregate_step_reward=norm_reward,
            financial_delta_component=round(financial_delta, 4),
            thermal_penalty_component=round(thermal_penalty, 4),
            constraint_violation_flag=constraint_violated,
            action_loop_detected=loop_detected,
        )
 
        return self._obs

    @property
    def state(self) -> GridEdgeState:
        return self._state

    def score(self) -> float:
        if self._task == "solar_self_consumption":
            return self._score_easy()
        elif self._task == "tod_arbitrage":
            return self._score_medium()
        else:
            return self._score_hard()
        
    def _normalize_reward(self, raw: float) -> float:
        cfg = self._config
        span = cfg.reward_raw_max - cfg.reward_raw_min
        return round(max(0.0, min(1.0, (raw - cfg.reward_raw_min) / span)), 4)

    def _clamp_battery(self, cmd: float):
        soc = self._obs.home_battery_soc
        if cmd > 0 and soc >= 0.99:
            return 0.0, True
        if cmd < 0 and soc <= 0.01:
            return 0.0, True
        return cmd, False

    def _clamp_ev(self, cmd: float):
        if not self._obs.ev_connection_status:
            return 0.0, False  # not connected - silently zero, not a violation
        if self._obs.electric_vehicle_soc >= 0.99 and cmd > 0:
            return 0.0, True
        return cmd, False

    def _update_battery_soc(self, cmd_kw: float) -> float:
        cfg = self._config
        soc = self._obs.home_battery_soc
        if cmd_kw >= 0:
            delta = (cmd_kw * 0.25 * cfg.inverter_efficiency) / cfg.battery_capacity_kwh
        else:
            delta = (cmd_kw * 0.25 / cfg.inverter_efficiency) / cfg.battery_capacity_kwh
        return round(max(0.0, min(1.0, soc + delta)), 4)

    def _update_ev_soc(self, cmd_kw: float) -> float:
        cfg = self._config
        if not self._obs.ev_connection_status:
            return self._obs.electric_vehicle_soc
        delta = (cmd_kw * 0.25 * cfg.inverter_efficiency) / cfg.ev_capacity_kwh
        return round(max(0.0, min(1.0, self._obs.electric_vehicle_soc + delta)), 4)

    def _update_indoor_temp(self, mode: str, setpoint: float, outdoor_temp: float, ghi: float) -> float:
        cfg  = self._config
        T_in = self._state.building_thermal_inertia
 
        Q_conduction = (outdoor_temp - T_in) / cfg.insulation_resistance
        Q_solar_gain = (ghi * 5.0 * 0.10) / 1000.0
 
        if mode == "cooling":
            Q_hvac = -cfg.hvac_power_kw * cfg.hvac_cop
        elif mode == "heating":
            Q_hvac = +cfg.hvac_power_kw * cfg.hvac_cop
        else:
            Q_hvac = 0.0
 
        dT = (900 / (cfg.thermal_mass_kjperdegc * 1000)) * (Q_conduction + Q_solar_gain + Q_hvac)
        new_temp = T_in + dT

        if mode != "off" and abs(new_temp - setpoint) < 0.2:
            new_temp = setpoint
 
        return round(new_temp, 3)

    def _thermal_penalty(self, indoor_temp: float, setpoint: float, occupied: bool) -> float:
        if not occupied:
            return 0.0
        cfg = self._config
        deviation = abs(indoor_temp - setpoint) - cfg.comfort_deadband_degc
        if deviation <= 0:
            return 0.0
        return round(-cfg.thermal_penalty_weight * (deviation ** 2), 4)
 
    def _ev_departure_penalty(self, step_idx: int, ev_soc: float) -> float:
        cfg = self._config
        if step_idx == cfg.ev_departure_step - 1 and ev_soc < cfg.ev_min_departure_soc:
            return -5.0
        return 0.0
 
    def _loop_penalty(self, action_dict: dict, step_reward: float):
        cfg = self._config
        if self._recent_actions and action_dict == self._recent_actions[-1]:
            self._consecutive_identical += 1
        else:
            self._consecutive_identical = 0
 
        self._recent_actions.append(action_dict)
        if len(self._recent_actions) > 5:
            self._recent_actions.pop(0)
 
        if self._consecutive_identical >= cfg.loop_threshold and step_reward < 0:
            exponent = min(self._consecutive_identical - cfg.loop_threshold, 4)
            penalty = -cfg.loop_penalty_base * (2 ** exponent)
            return round(penalty, 4), True
 
        return 0.0, False
    
    def _get_weather_row(self, step: int) -> dict:
        if self._weather_data is None:
            return {"outdoor_temp": 25.0, "ghi": 0.0}
        hour_idx = (self._episode_start_hour + step // 4) % len(self._weather_data)
        return self._weather_data[hour_idx]
 
    def _solar_kw(self, step: int) -> float:
        cfg = self._config
        ghi = self._get_weather_row(step)["ghi"]
        return round((ghi * cfg.panel_area_m2 * cfg.panel_efficiency) / 1000.0, 4)
 
    def _ghi(self, step: int) -> float:
        return self._get_weather_row(step)["ghi"]
 
    def _outdoor_temp(self, step: int) -> float:
        return self._get_weather_row(step)["outdoor_temp"]

    def _step_to_hour(self, step: int) -> int:
        return (step // 4) % 24

    def _step_to_iso(self, step: int) -> str:
        base = datetime(2025, 1, 1, 0, 0)
        delta = timedelta(minutes=15 * step)
        return (base + delta).isoformat()

    def _ev_connected(self, step: int) -> bool:
        hour = self._step_to_hour(step)
        return hour >= 22 or hour < 7

    def _build_occupancy_vector(self) -> List[bool]:
        result = []
        for step in range(self.max_steps):
            hour = self._step_to_hour(step)
            result.append(7 <= hour < 9 or 18 <= hour < 23)
        return result

    def _score_easy(self) -> float:
        available = self._state.solar_available_kwh
        if available == 0:
            return 0.0
        return round(max(0.0, min(1.0, self._state.solar_utilized_kwh / available)), 4)

    def _score_medium(self) -> float:
        rbc = self._state.rbc_baseline_cost
        agent = self._state.cumulative_financial_cost
        if rbc == 0:
            return 0.0
        return round(max(0.0, min(1.0, (rbc - agent) / rbc)), 4)

    def _score_hard(self) -> float:
        cfg = self._config
        w1, w3 = 0.4, 0.3
        alpha = 2.0
        cost_ratio = self._state.cumulative_financial_cost / max(self._state.rbc_baseline_cost, 0.001)
        ev_penalty = max(0.0, cfg.ev_min_departure_soc - self._obs.electric_vehicle_soc)
        J = w1 * cost_ratio + w3 * ev_penalty
        return round(max(0.0, min(1.0, math.exp(-alpha * J))), 4)