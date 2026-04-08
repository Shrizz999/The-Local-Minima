import asyncio
import os
import json
import textwrap
import re
from typing import List, Optional, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from client import GridEdgeEnv
from models import GridEdgeAction, GridEdgeObservation

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK", "solar_self_consumption")
BENCHMARK = "grid_edge_v1"
IMAGE_NAME = os.getenv("IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

TASKS = [
    "solar_self_consumption",
    "tod_arbitrage",
    "full_orchestration"
]

MAX_STEPS = 96   
TEMPERATURE = 0.6
MAX_TOKENS = 512 
SUCCESS_SCORE_THRESHOLD = 0.5


SYSTEM_PROMPT = textwrap.dedent("""
        You are an advanced thermodynamic and economic optimization agent managing a 'Grid Edge' Smart Home.
        Your objective is to balance conflicting priorities:
        1. Maximize financial savings via Time-of-Day (ToD) arbitrage.
        2. Maximize solar self-consumption.
        3. Maintain indoor thermal comfort (target: 18.0 - 28.0 C).
        4. Ensure the EV is charged to at least 80% before departure (step 28, 7:00 AM).
        
        You must return a valid JSON object representing your action. Do NOT wrap it in a markdown block.
        The JSON must contain EXACTLY these fields with appropriate mathematical bounds:
        {
            "hvac_operational_mode": "off" | "cooling" | "heating",
            "hvac_temperature_setpoint": float,
            "battery_dispatch_command": float (-5.0 to 5.0, positive=charge from grid/solar, negative=discharge to home/grid),
            "ev_charging_allocation": float (0.0 to 7.2),
            "grid_export_permission": boolean (true or false)
        }
    """).strip()

def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json_action(text: str) -> GridEdgeAction:
    cleaned = strip_think_blocks(text)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return GridEdgeAction(**data)
        except Exception:
            pass
    return GridEdgeAction(
        hvac_operational_mode="off",
        hvac_temperature_setpoint=24.0,
        battery_dispatch_command=0.0,
        ev_charging_allocation=0.0,
        grid_export_permission=False,
    )

def obs_to_dict(obs):
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    elif hasattr(obs, "observation"):
        return obs.observation.model_dump()
    else:
        return {}

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = str(error) if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, obs_dict: Dict[str, Any], last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    obs_json = json.dumps(obs_dict, indent=2, default=str)
    return textwrap.dedent(f"""
        Simulation Step: {step} / {MAX_STEPS}
        Current Observation:
        {obs_json}
        Last Step Reward: {last_reward:.4f}
        Previous 4 Steps (action → reward):
        {history_block}
        Return ONLY valid JSON for your next action.
    """).strip()

def get_model_action(client: OpenAI, step: int, obs_dict: Dict[str, Any], last_reward: float, history: List[str]) -> tuple[GridEdgeAction, Optional[str]]:
    user_prompt = build_user_prompt(step, obs_dict, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False
        )
        raw = (completion.choices[0].message.content or "").strip()
        action = extract_json_action(raw)
        return action, None
    except Exception as exc:
        return extract_json_action(""), str(exc)

async def run_task(client: OpenAI, task_name: str) -> None:
    if IMAGE_NAME:
        env_instance = await GridEdgeEnv.from_docker_image(IMAGE_NAME, task=task_name)
    else:
        env_instance = GridEdgeEnv(base_url=ENV_BASE_URL)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with env_instance as env:
            result = await env.reset(task=task_name)
            obs = result.observation if hasattr(result, "observation") else result
            obs_dict = obs_to_dict(obs)
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                action, error_msg = get_model_action(client, step, obs_dict, last_reward, history)

                action_str = (
                    f"GridEdgeAction(hvac='{action.hvac_operational_mode}',"
                    f"sp={action.hvac_temperature_setpoint},"
                    f"batt={action.battery_dispatch_command},"
                    f"ev={action.ev_charging_allocation},"
                    f"export={action.grid_export_permission})"
                )

                try:
                    result = await env.step(action)
                    obs = result.observation
                    reward = result.reward
                    done = result.done
                    obs_dict = obs_to_dict(obs)

                    diag = obs.system_diagnostic_msg or "OK"
                    if "WARNING" in diag or "CRITICAL" in diag:
                        error_msg = diag

                except Exception as exc:
                    reward = -5.0
                    done = True
                    error_msg = str(exc)

                rewards.append(reward)
                steps_taken = step
                last_reward = reward

                log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
                history.append(f"Step {step}: {action_str} | reward {reward:+.2f}")

                if done:
                    break

            score = env.score()
            success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_name in TASKS:
        await run_task(client, task_name)

if __name__ == "__main__":
    asyncio.run(main())