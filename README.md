---
title: Grid Edge Orchestrator
sdk: docker
emoji: ⚡
colorFrom: blue
colorTo: green
python_version: 3.10+
suggested_hardware: cpu-basic
pinned: false
---

# The "Grid Edge" Home Energy Orchestrator

The "Grid Edge" Home Energy Orchestrator is a production-grade reinforcement learning (RL) environment engineered atop the Meta x PyTorch OpenEnv framework. It is specifically designed to benchmark the long-horizon reasoning, constraint satisfaction, and temporal planning capabilities of frontier Large Language Models (LLMs).



## 1. Description
This project simulates a modern residential microgrid, where an AI agent is responsible for intelligently managing energy flow. Instead of relying on simple, game-like environments, it provides a more realistic and challenging setup that reflects real-world energy systems.

The platform is designed as a high-fidelity testing environment, allowing experimentation with decision-making under dynamic conditions such as changing demand, generation, and storage constraints.

To keep the system modular and scalable, it follows a distributed microservice architecture. The AI agent’s training loop is kept completely separate from the thermodynamic simulation engine, ensuring clear boundaries between learning and physical system modeling.


## 2. Motivation
The global electrical grid is undergoing a major transformation—from a centralized system to a decentralized network powered by intermittent renewable energy sources. In this evolving landscape, residential homes are no longer just passive consumers of electricity; they are prosumers, capable of generating, storing, and even supplying energy back to the grid.

This shift brings new challenges. Energy generation from renewables is inherently unpredictable, while household demand varies throughout the day. Managing this balance requires continuous, real-time decision-making under uncertainty—something traditional rule-based systems struggle to handle effectively.

Our motivation is to explore how intelligent, learning-based agents can address this complexity. By enabling adaptive and data-driven energy management, we aim to improve efficiency, reduce waste, and contribute to a smarter, more resilient energy ecosystem at the household level.


## 3. Action Space Definition
The agent controls the home's energy profile through the `GridEdgeAction` schema, which uses Pydantic field validators to mathematically enforce hardware limitations.

| Field Designation | Data Type | Permissible Range | Operational Description |
| :--- | :--- | :--- | :--- |
| **hvac_operational_mode** | str | ['off', 'cooling', 'heating'] | Dictates the active thermodynamic state. |
| **hvac_temperature_setpoint** | float | 18.0 le x le 28.0 | The target indoor temperature in Celsius. |
| **battery_dispatch_command** | float | -5.0 kW le x le 5.0 kW | Charging (positive) or discharging (negative) in kW. |
| **ev_charging_allocation** | float | 0.0 kW le x le 7.2 kW | Power allocated to Level 2 EV charging in kW. |
| **grid_export_permission** | bool | True / False | Toggle permitting excess solar sales to the grid. |


## 4. Observation Space Definition
The telemetry feed provides a snapshot of the immediate physical state and forward-looking forecasts.

| Field Designation | Data Type | Operational Description |
| :--- | :--- | :--- |
| **current_grid_tariff** | float | Real-time municipal electricity cost per kWh. |
| **forecast_grid_tariff** | list[float] | Predicted electricity prices for the subsequent 6 hours. |
| **current_solar_yield** | float | Instantaneous power output of the rooftop PV array. |
| **home_battery_soc** | float | State of Charge of the stationary battery (0.0 to 1.0). |
| **electric_vehicle_soc** | float | State of Charge of the EV battery. |
| **indoor_ambient_temp** | float | Current interior temperature of the dwelling. |


## 5. Task Descriptions and Difficulty
The environment follows a structured curriculum of three progressively complex tasks. Each task is evaluated using deterministic mathematical graders and is designed not just for technical challenge, but also for real-world impact across stakeholders—consumers, utilities, and the environment.

| Task Name                  | Difficulty | Objectives                                                                | Real-World Impact                                                                                                             | Expected Challenge                                                          |
| :------------------------- | :--------- | :------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- |
| **Solar Self-Consumption** | Easy       | Maximize on-site utilization of solar energy                              | Reduces grid dependency, lowers consumer bills, and minimizes transmission losses—supporting cleaner energy adoption          | Efficiently managing battery charge/discharge during solar peaks            |
| **ToD Economic Arbitrage** | Medium     | Optimize energy usage based on time-of-day pricing (e.g., MSEDCL tariffs) | Helps utilities balance load, reduces peak demand stress on the grid, and enables consumers to save costs                     | Handling non-linear pricing, demand spikes, and weather variability         |
| **Full Orchestration**     | Hard       | Balance comfort, cost, EV charging, and energy flow                       | Aligns consumer convenience with grid stability and sustainability goals—supporting EV adoption and smarter energy ecosystems | Resolving multi-objective conflicts under limited resources and uncertainty |



## 6. Setup and Usage Instructions

### Infrastructure Requirements
**vCPU:** 2 
**RAM:** 8GB 
**Software:** Docker, Python 3.10+, and Hugging Face API credentials.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Shrizz999/The-Local-Minima.git
   cd The-Local-Minima
   ```

2. **Build the Containerized Server:**
   ```bash
   docker build -t grid_edge_env -f src/grid_edge_env/server/Dockerfile .
   ```

3. **Launch the Environment:**
   ```bash
   docker run -p 7860:7860 grid_edge_env
   ```

### Running Inference
Execute the non-negotiable baseline inference script located at the root to interface with the containerized environment:

```bash
python inference.py
```

## 7. Baseline Scores
Performance is evaluated using deterministic physics simulations and linear programming baselines to output reproducible scores.

**Baseline Score Range:** 0.0 to 1.0.
**Rule-Based Controller (Naive):** 0.00 (Standard starting point for evaluation).
**Optimal LP Baseline:** 1.00 (Perfect foresight mathematical optimum).
**Target Success Threshold:** Success is determined when the agent's performance aligns with the requirements specified in the deterministic grader formulations for each task.

---

## 8. Conclusion and Economic Viability

The "Grid Edge" Home Energy Orchestrator demonstrates that the transition to a decentralized grid is not just a technical necessity but a significant financial opportunity for the residential "prosumer." By shifting from a passive consumer to an active orchestrator, a household can transform its energy profile from a monthly liability into a long-term revenue-generating asset.

### Financial Impact: Savings and Earnings
In a typical residential application within the Pune microgrid (utilizing MSEDCL 2024-2025 tariffs), an agent-orchestrated 3 kW system offers substantial returns:

* **Monthly Savings:** By prioritizing solar self-consumption and HVAC optimization, users save between **₹2,800 and ₹4,500 per month** by offsetting expensive grid imports during peak hours.
* **Annual Revenue and Credits:** Total annual benefits range from **₹35,000 to ₹60,000**. This includes net-metering credits earned during high-generation months where excess solar is "sold" back to the municipal grid, effectively zeroing out bills.
* **Lifetime ROI:** Over a standard 25-year operational lifespan, a well-orchestrated 3 kW system in Pune can yield a cumulative return of approximately **₹16.11 lakh**, which is over 11 times the initial net investment.