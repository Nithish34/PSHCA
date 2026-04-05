# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# PSHCA Environment v2.0 — FastAPI app with production-grade dashboard

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force-disable Gradio web interface injected by `openenv push`
os.environ["ENABLE_WEB_INTERFACE"] = "false"

try:
    from openenv.core.env_server.http_server import create_fastapi_app
except Exception as e:
    raise ImportError("openenv required. Run: uv sync") from e

from models import PshcaAction, PshcaObservation
from server.PSHCA_environment import PshcaEnvironment

import asyncio, json
from fastapi import Body
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# Use create_fastapi_app — pure FastAPI, no Gradio, ignores ENABLE_WEB_INTERFACE
app = create_fastapi_app(PshcaEnvironment, PshcaAction, PshcaObservation, max_concurrent_envs=1)

dashboard_env = PshcaEnvironment()
dashboard_env.reset()
dashboard_lock = asyncio.Lock()

def get_dashboard_html():
    html_path = os.path.join(os.path.dirname(__file__), '..', 'preview_dashboard.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()

# Strip any conflicting routes openenv may have registered for these paths
app.router.routes = [
    r for r in app.router.routes
    if getattr(r, "path", None) not in ["/", "/dashboard", "/web"]
]

# Serve our custom dashboard at:
#   /          — direct URL access
#   /dashboard — explicit path
#   /web       — HuggingFace Spaces embeds the app in an iframe at /web by default
@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
async def dashboard_page():
    return HTMLResponse(content=get_dashboard_html())

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/dashboard/reset")
async def dashboard_reset():
    async with dashboard_lock:
        obs = dashboard_env.reset()
        return JSONResponse({
            "ok": True,
            "scenario": dashboard_env.scenario,
            "snapshot": dashboard_env.get_dashboard_snapshot()
        })

@app.post("/dashboard/step")
async def dashboard_step(action: PshcaAction = Body(...)):
    async with dashboard_lock:
        obs = dashboard_env.step(action)
        feedback = (obs.metadata or {}).get("feedback", "") if obs.metadata else ""
        return JSONResponse({
            "ok": True,
            "done": obs.done,
            "reward": obs.reward,
            "feedback": feedback,
            "snapshot": dashboard_env.get_dashboard_snapshot()
        })

@app.get("/dashboard/state")
async def dashboard_state():
    async with dashboard_lock:
        return JSONResponse(dashboard_env.get_dashboard_snapshot())

@app.get("/dashboard/events")
async def dashboard_events():
    async def stream():
        while True:
            async with dashboard_lock:
                payload = dashboard_env.get_dashboard_snapshot()
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1.0)
    return StreamingResponse(stream(), media_type="text/event-stream")

def main():
    import uvicorn, argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == '__main__':
    main()