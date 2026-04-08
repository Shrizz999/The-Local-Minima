# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Grid Edge Home Energy Orchestrator.

This module creates the HTTP/WebSocket server that exposes
TheLocalMinimaEnvironment over endpoints compatible with EnvClient.

On startup, it parses the Pune EPW weather file once and injects the
data into the environment via load_weather(). All subsequent resets()
reuse this pre-loaded data — no file I/O happens during training.

Endpoints (handled automatically by create_app):
    POST /reset       Reset the environment, start a new episode
    POST /step        Execute one action, advance simulation by 15 minutes
    GET  /state       Get current hidden server-side state (debugging only)
    GET  /schema      Get action and observation JSON schemas
    WS   /ws          Persistent WebSocket session for full episode loops

Usage:
    # Development
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production (Docker)
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Direct execution
    python -m server.app
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

try:
    from models import GridEdgeAction, GridEdgeObservation
    from .the_local_minima_environment import TheLocalMinimaEnvironment, load_epw
except ModuleNotFoundError:
    from models import GridEdgeAction, GridEdgeObservation
    from server.the_local_minima_environment import TheLocalMinimaEnvironment, load_epw

EPW_PATH = os.path.join(os.path.dirname(__file__), "data", "pune_weather.epw")

_weather_data = None
_weather_load_error = None

try:
    _weather_data = load_epw(EPW_PATH)
    print(f"[grid-edge] Loaded {len(_weather_data)} hourly weather rows from {EPW_PATH}")
except FileNotFoundError:
    _weather_load_error = (
        f"EPW file not found at {EPW_PATH}. "
        "Place pune_weather.epw in server/data/ before starting the server."
    )
    print(f"[grid-edge] WARNING: {_weather_load_error}")
    print("[grid-edge] Environment will run with fallback weather (25°C, 0 GHI).")
except Exception as e:
    _weather_load_error = f"Failed to parse EPW file: {e}"
    print(f"[grid-edge] WARNING: {_weather_load_error}")

def create_environment() -> TheLocalMinimaEnvironment:
    env = TheLocalMinimaEnvironment()

    if _weather_data is not None:
        env.load_weather(_weather_data)
    else:
        print("[grid-edge] Session started without weather data — using fallback values.")

    return env

app = create_app(
    create_environment,
    GridEdgeAction,
    GridEdgeObservation,
    env_name="grid_edge",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Grid Edge Environment Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)