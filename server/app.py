# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Pshca Environment.

This module creates an HTTP server that exposes the PshcaEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import PshcaAction, PshcaObservation
    from .PSHCA_environment import PshcaEnvironment
except (ModuleNotFoundError, ImportError):
    from models import PshcaAction, PshcaObservation
    from server.PSHCA_environment import PshcaEnvironment

import asyncio
import json
from fastapi import Body
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


# Create the app with web interface and README integration
app = create_app(
    PshcaEnvironment,
    PshcaAction,
    PshcaObservation,
    env_name="PSHCA",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# Dedicated demo environment for live dashboard views.
# This keeps dashboard operations isolated from OpenEnv client sessions.
dashboard_env = PshcaEnvironment()
dashboard_env.reset()
dashboard_lock = asyncio.Lock()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
        """Serve a lightweight real-time dashboard for hackathon demos."""
        return HTMLResponse(
                content="""
<!doctype html>
<html>
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>PSHCA Live Dashboard</title>
    <style>
        :root {
            --bg: #0f172a;
            --panel: #111827;
            --ink: #e5e7eb;
            --accent: #22d3ee;
            --ok: #22c55e;
            --warn: #f59e0b;
            --bad: #ef4444;
        }
        body {
            margin: 0;
            font-family: 'Segoe UI', 'Trebuchet MS', sans-serif;
            background: radial-gradient(circle at 10% 0%, #1f2937, var(--bg));
            color: var(--ink);
        }
        .wrap { max-width: 1100px; margin: 24px auto; padding: 0 16px; }
        .header { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
        .pill { padding: 4px 10px; border-radius: 999px; background: #1f2937; border: 1px solid #334155; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin-top: 16px; }
        .card { background: color-mix(in oklab, var(--panel), black 10%); border: 1px solid #334155; border-radius: 14px; padding: 12px; }
        .k { opacity: 0.8; font-size: 12px; }
        .v { font-size: 22px; font-weight: 700; }
        .bars { display: grid; gap: 8px; margin-top: 8px; }
        .bar { background: #1f2937; border-radius: 8px; overflow: hidden; }
        .fill { height: 10px; background: linear-gradient(90deg, var(--ok), var(--warn), var(--bad)); }
        .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
        button {
            background: #0b1220;
            color: var(--ink);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 8px 10px;
            cursor: pointer;
        }
        button:hover { border-color: var(--accent); }
        .alerts { margin-top: 8px; display: grid; gap: 6px; }
        .alert { font-size: 13px; padding: 6px 8px; border-radius: 8px; background: #1f2937; }
        .stream { margin-top: 16px; max-height: 260px; overflow: auto; font-size: 13px; }
        .ev { padding: 6px 8px; border-bottom: 1px solid #1f2937; }
    </style>
</head>
<body>
    <div class='wrap'>
        <div class='header'>
            <h1>PSHCA Live Incident Recovery Dashboard</h1>
            <div class='pill' id='episode'>episode: -</div>
        </div>

        <div class='grid' id='cpu'></div>
        <div class='grid' id='mem'></div>

        <div class='card'>
            <div class='k'>Scenario</div>
            <div class='v' id='scenario'>-</div>
            <div id='task'></div>
            <div class='actions'>
                <button onclick="doAction('scale_up','web-server-01')">Scale web-01</button>
                <button onclick="doAction('reboot_server','web-server-01')">Reboot web-01</button>
                <button onclick="doAction('clear_cache','db-main')">Clear db-main cache</button>
                <button onclick="doAction('failover_db','db-main')">Failover db-main</button>
                <button onclick="doAction('rollback_deployment','web-server-01')">Rollback</button>
                <button onclick="doAction('wait','')">Wait</button>
                <button onclick="resetEnv()">Reset Episode</button>
            </div>
            <div class='alerts' id='alerts'></div>
        </div>

        <div class='card stream' id='events'></div>
    </div>

    <script>
        function pct(v) { return Math.max(0, Math.min(100, Number(v || 0))); }
        function renderBars(containerId, title, data) {
            const el = document.getElementById(containerId);
            const html = Object.entries(data || {}).map(([k, v]) => `
                <div class='card'>
                    <div class='k'>${title} - ${k}</div>
                    <div class='v'>${pct(v).toFixed(1)}%</div>
                    <div class='bar'><div class='fill' style='width:${pct(v)}%'></div></div>
                </div>
            `).join('');
            el.innerHTML = html;
        }

        function render(state) {
            document.getElementById('episode').textContent = `episode: ${state.episode_id} | step ${state.step_count}/${state.max_steps}`;
            document.getElementById('scenario').textContent = state.scenario;
            document.getElementById('task').textContent = state.task_info;
            renderBars('cpu', 'CPU', state.cpu_usage);
            renderBars('mem', 'Memory', state.memory_usage);
            const alerts = (state.active_alerts || []).map(a => `<div class='alert'>${a}</div>`).join('');
            document.getElementById('alerts').innerHTML = alerts || '<div class="alert">No active alerts</div>';

            const events = (state.recent_events || []).slice().reverse().map(ev => {
                const a = ev.action || {};
                return `<div class='ev'>#${ev.step} [${ev.scenario}] ${a.action_type || '-'} ${a.target_resource || ''} | reward=${(ev.reward ?? 0).toFixed(2)}</div>`;
            }).join('');
            document.getElementById('events').innerHTML = events;
        }

        async function getState() {
            const res = await fetch('/dashboard/state');
            render(await res.json());
        }

        async function doAction(action_type, target_resource) {
            await fetch('/dashboard/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action_type, target_resource })
            });
            await getState();
        }

        async function resetEnv() {
            await fetch('/dashboard/reset', { method: 'POST' });
            await getState();
        }

        const source = new EventSource('/dashboard/events');
        source.onmessage = (e) => render(JSON.parse(e.data));
        getState();
    </script>
</body>
</html>
                """.strip()
        )


@app.post("/dashboard/reset")
async def dashboard_reset():
        async with dashboard_lock:
                obs = dashboard_env.reset()
                return JSONResponse(
                        {
                                "ok": True,
                                "scenario": dashboard_env.scenario,
                                "task_info": obs.current_task_info,
                        }
                )


@app.post("/dashboard/step")
async def dashboard_step(action: PshcaAction = Body(...)):
        async with dashboard_lock:
                obs = dashboard_env.step(action)
                return JSONResponse(
                        {
                                "ok": True,
                                "done": obs.done,
                                "reward": obs.reward,
                                "scenario": dashboard_env.scenario,
                                "snapshot": dashboard_env.get_dashboard_snapshot(),
                        }
                )


@app.get("/dashboard/state")
async def dashboard_state():
        async with dashboard_lock:
                return JSONResponse(dashboard_env.get_dashboard_snapshot())


@app.get("/dashboard/events")
async def dashboard_events():
        async def event_stream():
                while True:
                        async with dashboard_lock:
                                payload = dashboard_env.get_dashboard_snapshot()
                        yield f"data: {json.dumps(payload)}\n\n"
                        await asyncio.sleep(1.0)

        return StreamingResponse(event_stream(), media_type="text/event-stream")


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Start the PSHCA FastAPI server.

    Callable both programmatically (e.g. by the OpenEnv validator) and from
    the command line.  Port defaults to 7860 for HuggingFace Spaces.
    """
    import uvicorn

    # Only parse CLI args when invoked directly — prevents argparse from
    # consuming sys.argv when main() is imported and called by validators.
    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser(description="PSHCA Environment Server")
        parser.add_argument("--host", type=str, default=host)
        parser.add_argument("--port", type=int, default=port)
        args = parser.parse_args()
        host = args.host
        port = args.port

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()