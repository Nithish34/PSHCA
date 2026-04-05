# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# PSHCA Environment v2.0 — FastAPI app with production-grade dashboard + Episode History

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force-disable Gradio web interface injected by `openenv push`
os.environ["ENABLE_WEB_INTERFACE"] = "false"

try:
    from openenv.core.env_server.http_server import create_fastapi_app
except Exception as e:
    raise ImportError("openenv required. Run: uv sync") from e

try:
    from ..models import PshcaAction, PshcaObservation
    from .PSHCA_environment import PshcaEnvironment
except (ModuleNotFoundError, ImportError):
    from models import PshcaAction, PshcaObservation
    from server.PSHCA_environment import PshcaEnvironment

import asyncio, json
from datetime import datetime, timezone
from fastapi import Body
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# Use create_fastapi_app — pure FastAPI, no Gradio, ignores ENABLE_WEB_INTERFACE
app = create_fastapi_app(PshcaEnvironment, PshcaAction, PshcaObservation, env_name="PSHCA", max_concurrent_envs=1)
dashboard_env = PshcaEnvironment()
dashboard_env.reset()
dashboard_lock = asyncio.Lock()

# ── Episode History Store ────────────────────────────────────────────────────
# Persists summaries of ALL completed episodes for the History tab.
# Stored in memory; resets when server restarts.
episode_history: list = []          # list of episode summary dicts
MAX_EPISODE_HISTORY = 100           # keep last 100 episodes

# Snapshot of the episode that is currently in progress
_current_episode_start: dict = {}


def _start_episode_tracking(env: PshcaEnvironment):
    """Called on reset — capture metadata about the new episode."""
    global _current_episode_start
    _current_episode_start = {
        "episode_id":     env._state.episode_id,
        "scenario":       env.scenario,
        "scenario_id":    env.active_scenario.get("id", ""),
        "scenario_title": env.active_scenario.get("title", ""),
        "started_at":     datetime.now(timezone.utc).isoformat(),
    }


def _finish_episode_tracking(env: PshcaEnvironment, final_reward: float, done: bool):
    """Called after every step — if done, archive the completed episode."""
    global episode_history
    if not done or not _current_episode_start:
        return
    summary = {
        **_current_episode_start,
        "finished_at":       datetime.now(timezone.utc).isoformat(),
        "total_steps":       env._state.step_count,
        "final_reward":      round(final_reward, 4),
        "cumulative_reward": round(env.cumulative_reward, 4),
        "outcome":           "SUCCESS" if final_reward >= 0.5 else "FAILED",
        "actions": [
            {
                "step":            ev.get("step"),
                "action_type":     ev.get("action", {}).get("action_type", ""),
                "target_resource": ev.get("action", {}).get("target_resource", ""),
                "reward":          ev.get("reward", 0),
                "feedback":        ev.get("feedback", ""),
            }
            for ev in env.event_history
            if ev.get("step", 0) > 0
        ],
    }
    episode_history.append(summary)
    if len(episode_history) > MAX_EPISODE_HISTORY:
        episode_history = episode_history[-MAX_EPISODE_HISTORY:]


# ── Dashboard HTML ───────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PSHCA — Cloud Incident Command</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#050a14;--surface:#0b1628;--panel:#0f1f36;--border:#1a2e4a;--accent:#00d4ff;--accent2:#ff6b35;--ok:#00e676;--warn:#ffca28;--bad:#ff1744;--ink:#c9d8f0;--ink-dim:#5a7a9a;--mono:'Space Mono',monospace;--sans:'Syne',sans-serif}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,212,255,.015) 2px,rgba(0,212,255,.015) 4px);pointer-events:none;z-index:9999}

/* ── Header ── */
header{display:flex;align-items:center;justify-content:space-between;padding:14px 28px;border-bottom:1px solid var(--border);background:linear-gradient(90deg,#050a14,#0b1628);position:sticky;top:0;z-index:100}
.logo{font-family:var(--sans);font-weight:800;font-size:1.1rem;letter-spacing:.08em;color:var(--accent);display:flex;align-items:center;gap:8px}
.logo span{color:var(--ink-dim);font-weight:400;font-size:.8rem;letter-spacing:.2em}
.status-bar{display:flex;gap:16px;align-items:center;font-family:var(--mono);font-size:.72rem}
.badge{padding:3px 10px;border-radius:4px;border:1px solid;font-weight:700;letter-spacing:.1em;text-transform:uppercase}
.badge.live{color:var(--ok);border-color:var(--ok);background:rgba(0,230,118,.08)}
.badge.warn{color:var(--warn);border-color:var(--warn);background:rgba(255,202,40,.08)}
.badge.crisis{color:var(--bad);border-color:var(--bad);background:rgba(255,23,68,.08);animation:pulse-bad 1s infinite}
@keyframes pulse-bad{0%,100%{opacity:1}50%{opacity:.5}}

/* ── Tab nav ── */
.tab-nav{display:flex;gap:0;border-bottom:1px solid var(--border);background:var(--surface);position:sticky;top:57px;z-index:90}
.tab-btn{padding:11px 24px;font-family:var(--mono);font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;border:none;background:none;color:var(--ink-dim);cursor:pointer;border-bottom:2px solid transparent;transition:all .2s}
.tab-btn:hover{color:var(--ink)}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent);background:rgba(0,212,255,.04)}

/* ── Layout ── */
.tab-panel{display:none}.tab-panel.active{display:block}
.shell{display:grid;grid-template-columns:320px 1fr;min-height:calc(100vh - 100px)}
.sidebar{border-right:1px solid var(--border);background:var(--surface);display:flex;flex-direction:column;overflow-y:auto}
.slab{padding:18px 20px;border-bottom:1px solid var(--border)}
.slab-title{font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;color:var(--ink-dim);margin-bottom:12px;font-family:var(--mono)}

/* ── Scenario badge ── */
.scenario-badge{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:4px;font-family:var(--mono);font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;margin-bottom:10px}
.sc-easy{background:rgba(0,230,118,.12);color:var(--ok);border:1px solid var(--ok)}
.sc-medium{background:rgba(255,202,40,.12);color:var(--warn);border:1px solid var(--warn)}
.sc-hard{background:rgba(255,23,68,.12);color:var(--bad);border:1px solid var(--bad)}
#scenario-title{font-size:1rem;font-weight:600;line-height:1.3;margin-bottom:6px}
#scenario-desc{font-size:.8rem;color:var(--ink-dim);line-height:1.5}

/* ── Progress ── */
.progress-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;font-family:var(--mono);font-size:.72rem}
.progress-track{background:var(--border);border-radius:4px;height:6px;overflow:hidden}
.progress-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width .4s}
.reward-display{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}
.reward-box{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px 12px;text-align:center}
.reward-box .rlabel{font-size:.6rem;color:var(--ink-dim);letter-spacing:.15em;text-transform:uppercase;font-family:var(--mono)}
.reward-box .value{font-size:1.4rem;font-weight:700;font-family:var(--mono);margin-top:2px}
.val-pos{color:var(--ok)}.val-neg{color:var(--bad)}.val-neu{color:var(--accent)}

/* ── Alerts ── */
.alert-item{padding:8px 10px;border-radius:6px;font-size:.78rem;font-family:var(--mono);margin-bottom:6px;border-left:3px solid;animation:slide-in .3s ease}
@keyframes slide-in{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:none}}
.alert-warn{background:rgba(255,202,40,.08);border-color:var(--warn);color:var(--warn)}
.alert-crit{background:rgba(255,23,68,.1);border-color:var(--bad);color:var(--bad)}
.alert-fatal{background:rgba(255,23,68,.15);border-color:var(--bad);color:var(--bad);animation:pulse-bad .8s infinite}
.alert-ok{background:rgba(0,230,118,.08);border-color:var(--ok);color:var(--ok)}
#feedback-box{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px;font-size:.8rem;font-family:var(--mono);line-height:1.5;min-height:48px;color:var(--ink);border-left:3px solid var(--accent)}

/* ── Action controls ── */
.action-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.action-btn{background:var(--panel);border:1px solid var(--border);color:var(--ink);border-radius:8px;padding:10px 8px;font-family:var(--mono);font-size:.7rem;cursor:pointer;text-align:center;transition:all .2s;display:flex;flex-direction:column;align-items:center;gap:4px}
.action-btn:hover{border-color:var(--accent);color:var(--accent);background:rgba(0,212,255,.06)}
.action-btn.active{border-color:var(--accent2);background:rgba(255,107,53,.1);color:var(--accent2)}
.action-btn .icon{font-size:1.2rem}.action-btn .alabel{font-weight:700;letter-spacing:.05em}.action-btn .sub{color:var(--ink-dim);font-size:.62rem}
.target-select{width:100%;background:var(--panel);border:1px solid var(--border);color:var(--ink);padding:8px 10px;border-radius:8px;font-family:var(--mono);font-size:.78rem;margin:8px 0;outline:none;cursor:pointer}
.target-select:focus{border-color:var(--accent)}
.exec-btn{width:100%;padding:12px;border-radius:8px;background:linear-gradient(135deg,var(--accent),#0080aa);color:#000;font-family:var(--sans);font-weight:800;font-size:.85rem;letter-spacing:.1em;text-transform:uppercase;border:none;cursor:pointer;transition:all .2s;margin-top:4px}
.exec-btn:hover{transform:translateY(-1px);box-shadow:0 4px 20px rgba(0,212,255,.3)}
.exec-btn:disabled{opacity:.4;cursor:not-allowed;transform:none}
.reset-btn{width:100%;padding:9px;border-radius:8px;background:transparent;border:1px solid var(--border);color:var(--ink-dim);font-family:var(--mono);font-size:.72rem;cursor:pointer;transition:all .2s;margin-top:6px;letter-spacing:.1em}
.reset-btn:hover{border-color:var(--accent2);color:var(--accent2)}

/* ── Main live panel ── */
.main{padding:20px 24px;overflow-y:auto;background:var(--bg)}
.section-title{font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;color:var(--ink-dim);font-family:var(--mono);margin-bottom:14px;display:flex;align-items:center;gap:8px}
.section-title::after{content:'';flex:1;height:1px;background:var(--border)}
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px;position:relative;overflow:hidden;transition:border-color .3s}
.metric-card.danger{border-color:var(--bad)}.metric-card.warning{border-color:var(--warn)}
.metric-card .node-name{font-family:var(--mono);font-size:.65rem;color:var(--ink-dim);letter-spacing:.1em}
.metric-card .metric-val{font-family:var(--mono);font-size:1.5rem;font-weight:700;margin:4px 0}
.metric-card .metric-sub{font-size:.7rem;color:var(--ink-dim);display:flex;gap:10px;flex-wrap:wrap}
.sparkbar{position:absolute;bottom:0;left:0;right:0;height:3px}
.sparkfill{height:100%;transition:width .5s,background .5s}
.svc-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px}
.svc-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px;text-align:center}
.svc-name{font-family:var(--mono);font-size:.7rem;color:var(--ink-dim);letter-spacing:.1em;margin-bottom:6px}
.svc-status{font-family:var(--mono);font-size:.9rem;font-weight:700}
.svc-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.s-healthy{color:var(--ok)}.s-healthy .svc-dot{background:var(--ok);box-shadow:0 0 6px var(--ok)}
.s-degraded{color:var(--warn)}.s-degraded .svc-dot{background:var(--warn);box-shadow:0 0 6px var(--warn)}
.s-offline{color:var(--bad)}.s-offline .svc-dot{background:var(--bad);box-shadow:0 0 6px var(--bad);animation:pulse-bad .8s infinite}
.task-table{width:100%;border-collapse:collapse;font-size:.78rem;margin-bottom:4px}
.task-table th{font-family:var(--mono);font-size:.65rem;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-dim);padding:8px 12px;border-bottom:1px solid var(--border);text-align:left}
.task-table td{padding:10px 12px;border-bottom:1px solid rgba(26,46,74,.5);font-family:var(--mono)}
.task-table tr:last-child td{border-bottom:none}
.event-log{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden}
.event-log-header{padding:10px 14px;border-bottom:1px solid var(--border);font-family:var(--mono);font-size:.65rem;color:var(--ink-dim);letter-spacing:.15em;text-transform:uppercase;display:flex;justify-content:space-between}
.event-log-body{max-height:220px;overflow-y:auto}
.event-row{display:grid;grid-template-columns:60px 140px 120px 80px 1fr;padding:7px 14px;border-bottom:1px solid rgba(26,46,74,.4);font-family:var(--mono);font-size:.7rem;align-items:center}
.event-row:hover{background:var(--panel)}.event-row:last-child{border-bottom:none}
.ev-step{color:var(--ink-dim)}.ev-act{color:var(--accent);font-weight:700}.ev-tgt{color:var(--ink-dim)}
.ev-rwd{font-weight:700}.ev-rwd.pos{color:var(--ok)}.ev-rwd.neg{color:var(--bad)}.ev-rwd.neu{color:var(--warn)}
.ev-fb{color:var(--ink-dim);font-size:.65rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* ── History tab ── */
.history-panel{padding:24px;background:var(--bg);min-height:calc(100vh - 100px)}
.history-toolbar{display:flex;align-items:center;gap:12px;margin-bottom:20px;flex-wrap:wrap}
.history-toolbar h2{font-size:1.1rem;font-weight:800;color:var(--ink);flex:1}
.filter-btn{padding:6px 14px;border-radius:6px;border:1px solid var(--border);background:var(--surface);color:var(--ink-dim);font-family:var(--mono);font-size:.68rem;cursor:pointer;letter-spacing:.08em;transition:all .2s}
.filter-btn:hover,.filter-btn.active{border-color:var(--accent);color:var(--accent);background:rgba(0,212,255,.06)}
.filter-btn.f-easy.active{border-color:var(--ok);color:var(--ok);background:rgba(0,230,118,.08)}
.filter-btn.f-medium.active{border-color:var(--warn);color:var(--warn);background:rgba(255,202,40,.08)}
.filter-btn.f-hard.active{border-color:var(--bad);color:var(--bad);background:rgba(255,23,68,.08)}
.filter-btn.f-success.active{border-color:var(--ok);color:var(--ok)}
.filter-btn.f-failed.active{border-color:var(--bad);color:var(--bad)}
.clear-btn{padding:6px 14px;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--bad);font-family:var(--mono);font-size:.68rem;cursor:pointer;transition:all .2s}
.clear-btn:hover{border-color:var(--bad);background:rgba(255,23,68,.08)}

/* ── Stats row ── */
.stats-row{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:24px}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px;text-align:center}
.stat-card .slabel{font-size:.6rem;letter-spacing:.15em;text-transform:uppercase;color:var(--ink-dim);font-family:var(--mono);margin-bottom:4px}
.stat-card .svalue{font-size:1.6rem;font-weight:800;font-family:var(--mono)}
.stat-card .ssub{font-size:.65rem;color:var(--ink-dim);margin-top:2px;font-family:var(--mono)}

/* ── Episode cards grid ── */
.ep-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.ep-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;transition:border-color .2s,transform .2s;cursor:pointer}
.ep-card:hover{border-color:var(--accent);transform:translateY(-2px)}
.ep-card.success{border-left:4px solid var(--ok)}
.ep-card.failed{border-left:4px solid var(--bad)}
.ep-header{padding:14px 16px;display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid var(--border)}
.ep-id{font-family:var(--mono);font-size:.65rem;color:var(--ink-dim)}
.ep-outcome{font-family:var(--mono);font-size:.72rem;font-weight:700;padding:3px 10px;border-radius:4px}
.ep-outcome.success{color:var(--ok);background:rgba(0,230,118,.1);border:1px solid var(--ok)}
.ep-outcome.failed{color:var(--bad);background:rgba(255,23,68,.1);border:1px solid var(--bad)}
.ep-body{padding:14px 16px}
.ep-title{font-size:.9rem;font-weight:700;margin-bottom:4px;line-height:1.3}
.ep-meta{display:flex;gap:14px;font-family:var(--mono);font-size:.68rem;color:var(--ink-dim);margin-bottom:12px;flex-wrap:wrap}
.ep-meta span{display:flex;align-items:center;gap:4px}
.ep-reward-bar{height:6px;background:var(--border);border-radius:4px;overflow:hidden;margin-bottom:10px}
.ep-reward-fill{height:100%;border-radius:4px;transition:width .4s}
.ep-actions{display:flex;flex-direction:column;gap:4px}
.ep-action-row{display:grid;grid-template-columns:28px 110px 110px 60px 1fr;align-items:center;font-family:var(--mono);font-size:.65rem;padding:4px 6px;border-radius:4px;background:var(--panel)}
.ep-action-row .ea-step{color:var(--ink-dim)}
.ep-action-row .ea-act{color:var(--accent);font-weight:700}
.ep-action-row .ea-tgt{color:var(--ink-dim)}
.ep-action-row .ea-rwd{font-weight:700}
.ep-action-row .ea-rwd.pos{color:var(--ok)}.ep-action-row .ea-rwd.neg{color:var(--bad)}.ep-action-row .ea-rwd.neu{color:var(--warn)}
.ep-action-row .ea-fb{color:var(--ink-dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ep-ts{font-family:var(--mono);font-size:.62rem;color:var(--ink-dim);margin-top:8px;text-align:right}

/* ── Empty state ── */
.empty-state{text-align:center;padding:80px 20px;color:var(--ink-dim)}
.empty-state .big-icon{font-size:3rem;margin-bottom:16px}
.empty-state h3{font-size:1.1rem;font-weight:700;margin-bottom:8px;color:var(--ink)}
.empty-state p{font-size:.85rem;line-height:1.6;max-width:360px;margin:0 auto}

/* ── Done overlay ── */
#done-overlay{display:none;position:fixed;inset:0;background:rgba(5,10,20,.85);z-index:200;align-items:center;justify-content:center;flex-direction:column}
#done-overlay.show{display:flex}
.done-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:40px 48px;text-align:center;max-width:440px}
.done-title{font-size:2rem;font-weight:800;margin-bottom:8px}
.done-sub{color:var(--ink-dim);font-family:var(--mono);font-size:.85rem}
.done-score{font-family:var(--mono);font-size:3rem;font-weight:700;margin:20px 0}
.done-btn{padding:12px 32px;border-radius:8px;border:none;background:var(--accent);color:#000;font-family:var(--sans);font-weight:800;font-size:.9rem;cursor:pointer;margin-top:12px}
.hist-btn{padding:12px 32px;border-radius:8px;border:1px solid var(--border);background:transparent;color:var(--ink);font-family:var(--sans);font-weight:600;font-size:.85rem;cursor:pointer;margin-top:8px;margin-left:8px}

::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:var(--surface)}::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
</style>
</head>
<body>

<!-- Done overlay -->
<div id="done-overlay">
  <div class="done-card">
    <div class="done-title" id="done-title">Episode Complete</div>
    <div class="done-sub" id="done-sub"></div>
    <div class="done-score" id="done-score">-</div>
    <div class="done-sub" id="done-feedback" style="margin-top:8px;font-size:.78rem;max-width:340px;margin-inline:auto;line-height:1.5"></div>
    <div style="margin-top:16px">
      <button class="done-btn" onclick="resetEnv()">Next Episode &rarr;</button>
      <button class="hist-btn" onclick="switchTab('history');document.getElementById('done-overlay').classList.remove('show')">View History</button>
    </div>
  </div>
</div>

<header>
  <div class="logo">&#11041; PSHCA <span>CLOUD INCIDENT COMMAND v2.0</span></div>
  <div class="status-bar">
    <span id="hdr-episode" style="color:var(--ink-dim)">ep: -</span>
    <span id="hdr-step" style="color:var(--ink-dim)">step: 0/20</span>
    <div class="badge live" id="hdr-status">LIVE</div>
  </div>
</header>

<!-- Tab navigation -->
<div class="tab-nav">
  <button class="tab-btn active" id="tab-live" onclick="switchTab('live')">&#9679; Live Incident</button>
  <button class="tab-btn" id="tab-history" onclick="switchTab('history')">&#128196; Episode History</button>
</div>

<!-- ═══════════════════════════════════════════════════
     LIVE TAB
═══════════════════════════════════════════════════ -->
<div class="tab-panel active" id="panel-live">
<div class="shell">
<aside class="sidebar">
  <div class="slab">
    <div class="slab-title">Active Incident</div>
    <div id="scenario-badge" class="scenario-badge sc-easy">&#9679; Easy &middot; E1</div>
    <div id="scenario-title">Awaiting reset&hellip;</div>
    <div id="scenario-desc" style="margin-top:6px;"></div>
  </div>
  <div class="slab">
    <div class="slab-title">Episode Progress</div>
    <div class="progress-row">
      <span id="prog-label" style="font-family:var(--mono);font-size:.72rem;">Step 0 / 20</span>
      <span id="prog-pct" style="font-family:var(--mono);font-size:.72rem;color:var(--ink-dim);">0%</span>
    </div>
    <div class="progress-track"><div class="progress-fill" id="prog-fill" style="width:0%"></div></div>
    <div class="reward-display">
      <div class="reward-box"><div class="rlabel">Last Reward</div><div class="value val-neu" id="last-reward">-</div></div>
      <div class="reward-box"><div class="rlabel">Cumulative</div><div class="value val-neu" id="cum-reward">0.00</div></div>
    </div>
  </div>
  <div class="slab">
    <div class="slab-title">Active Alerts</div>
    <div id="alerts-container"><div class="alert-item alert-ok">&#10003; All systems nominal</div></div>
  </div>
  <div class="slab">
    <div class="slab-title">Agent Feedback</div>
    <div id="feedback-box">Reset the environment to begin incident response.</div>
  </div>
  <div class="slab">
    <div class="slab-title">Action Controls</div>
    <div class="action-grid">
      <div class="action-btn" onclick="selectAction('scale_up')" id="btn-scale_up"><span class="icon">&#9889;</span><span class="alabel">Scale Up</span><span class="sub">add capacity</span></div>
      <div class="action-btn" onclick="selectAction('reboot_server')" id="btn-reboot_server"><span class="icon">&#128260;</span><span class="alabel">Reboot</span><span class="sub">hard restart</span></div>
      <div class="action-btn" onclick="selectAction('clear_cache')" id="btn-clear_cache"><span class="icon">&#129529;</span><span class="alabel">Clear Cache</span><span class="sub">free memory</span></div>
      <div class="action-btn" onclick="selectAction('failover_db')" id="btn-failover_db"><span class="icon">&#128256;</span><span class="alabel">Failover DB</span><span class="sub">promote replica</span></div>
      <div class="action-btn" onclick="selectAction('rollback_deployment')" id="btn-rollback_deployment"><span class="icon">&#9194;</span><span class="alabel">Rollback</span><span class="sub">revert deploy</span></div>
      <div class="action-btn" onclick="selectAction('wait')" id="btn-wait"><span class="icon">&#9208;</span><span class="alabel">Wait</span><span class="sub">observe only</span></div>
    </div>
    <select id="target-select" class="target-select">
      <option value="">- Select Target Resource -</option>
      <option value="web-server-01">web-server-01 (primary web)</option>
      <option value="web-server-02">web-server-02 (secondary web)</option>
      <option value="db-main">db-main (primary database)</option>
      <option value="db-replica">db-replica (read replica)</option>
    </select>
    <button class="exec-btn" id="exec-btn" onclick="executeAction()" disabled>Execute Action</button>
    <button class="reset-btn" onclick="resetEnv()">&#8634; Reset / Next Scenario</button>
  </div>
  <div class="slab">
    <div class="slab-title">Task Reference</div>
    <table class="task-table">
      <thead><tr><th>Difficulty</th><th>Variants</th><th>Steps</th></tr></thead>
      <tbody>
        <tr><td style="color:var(--ok)">&#9679; Easy</td><td>E1 E2 E3</td><td>1-3</td></tr>
        <tr><td style="color:var(--warn)">&#9679; Medium</td><td>M1 M2 M3</td><td>1-5</td></tr>
        <tr><td style="color:var(--bad)">&#9679; Hard</td><td>H1 H2 H3</td><td>2-7</td></tr>
      </tbody>
    </table>
  </div>
</aside>
<main class="main">
  <div class="section-title">CPU Usage</div>
  <div class="metric-grid" id="cpu-grid"></div>
  <div class="section-title">Memory Usage</div>
  <div class="metric-grid" id="mem-grid"></div>
  <div class="section-title">Latency &amp; Error Rate</div>
  <div class="metric-grid" id="lat-grid"></div>
  <div class="section-title">Service Health</div>
  <div class="svc-grid" id="svc-grid"></div>
  <div class="section-title">Action History (Current Episode)</div>
  <div class="event-log">
    <div class="event-log-header">
      <span>Step &middot; Action &middot; Target &middot; Reward &middot; Feedback</span>
      <span id="log-count">0 events</span>
    </div>
    <div class="event-log-body" id="event-log-body">
      <div style="padding:16px;text-align:center;color:var(--ink-dim);font-family:var(--mono);font-size:.75rem;">No actions yet - reset and begin.</div>
    </div>
  </div>
</main>
</div>
</div>

<!-- ═══════════════════════════════════════════════════
     HISTORY TAB
═══════════════════════════════════════════════════ -->
<div class="tab-panel" id="panel-history">
<div class="history-panel">

  <!-- Toolbar -->
  <div class="history-toolbar">
    <h2>&#128196; Episode History</h2>
    <button class="filter-btn f-all active" id="f-all" onclick="setFilter('all')">All</button>
    <button class="filter-btn f-easy" id="f-easy" onclick="setFilter('easy')">Easy</button>
    <button class="filter-btn f-medium" id="f-medium" onclick="setFilter('medium')">Medium</button>
    <button class="filter-btn f-hard" id="f-hard" onclick="setFilter('hard')">Hard</button>
    <button class="filter-btn f-success" id="f-success" onclick="setFilter('success')">&#10003; Success</button>
    <button class="filter-btn f-failed" id="f-failed" onclick="setFilter('failed')">&#10007; Failed</button>
    <button class="clear-btn" onclick="clearHistory()">&#128465; Clear</button>
  </div>

  <!-- Stats row -->
  <div class="stats-row" id="stats-row">
    <div class="stat-card"><div class="slabel">Total Episodes</div><div class="svalue val-neu" id="stat-total">0</div></div>
    <div class="stat-card"><div class="slabel">Success Rate</div><div class="svalue val-pos" id="stat-sr">-</div><div class="ssub" id="stat-sr-sub"></div></div>
    <div class="stat-card"><div class="slabel">Avg Reward</div><div class="svalue val-neu" id="stat-avg">-</div></div>
    <div class="stat-card"><div class="slabel">Best Score</div><div class="svalue val-pos" id="stat-best">-</div></div>
    <div class="stat-card"><div class="slabel">Avg Steps</div><div class="svalue val-neu" id="stat-steps">-</div></div>
  </div>

  <!-- Episode cards -->
  <div class="ep-grid" id="ep-grid">
    <div class="empty-state">
      <div class="big-icon">&#128203;</div>
      <h3>No episodes yet</h3>
      <p>Complete an episode on the Live Incident tab to see it appear here. Each resolved or failed incident is automatically recorded.</p>
    </div>
  </div>

</div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
var selectedAction=null, lastDone=false;
var allEpisodes=[], activeFilter='all';

// ── Tab switching ──────────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active');});
  document.querySelectorAll('.tab-panel').forEach(function(p){p.classList.remove('active');});
  document.getElementById('tab-'+tab).classList.add('active');
  document.getElementById('panel-'+tab).classList.add('active');
  if(tab==='history') loadHistory();
}

// ── Action selection ───────────────────────────────────────────────────────
function selectAction(a){
  selectedAction=a;
  document.querySelectorAll('.action-btn').forEach(function(b){b.classList.remove('active');});
  document.getElementById('btn-'+a).classList.add('active');
  document.getElementById('exec-btn').disabled=false;
  var ts=document.getElementById('target-select');
  if(a==='wait'){ts.value='';ts.disabled=true;}else{ts.disabled=false;}
}

// ── Execute action ─────────────────────────────────────────────────────────
async function executeAction(){
  if(!selectedAction)return;
  var target=document.getElementById('target-select').value;
  if(selectedAction!=='wait'&&!target){document.getElementById('feedback-box').textContent='Select a target resource first.';return;}
  document.getElementById('exec-btn').disabled=true;
  try{
    var res=await fetch('/dashboard/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action_type:selectedAction,target_resource:target||''})});
    var data=await res.json();
    render(data.snapshot,data.reward,data.done,data.feedback);
    if(data.done){lastDone=true;showDone(data.reward,data.feedback);}
  }finally{document.getElementById('exec-btn').disabled=false;}
}

// ── Reset ──────────────────────────────────────────────────────────────────
async function resetEnv(){
  document.getElementById('done-overlay').classList.remove('show');
  lastDone=false;selectedAction=null;
  document.querySelectorAll('.action-btn').forEach(function(b){b.classList.remove('active');});
  document.getElementById('exec-btn').disabled=true;
  var ts=document.getElementById('target-select');ts.disabled=false;ts.value='';
  await fetch('/dashboard/reset',{method:'POST'});
  await getState();
}

async function getState(){var res=await fetch('/dashboard/state');render(await res.json());}

// ── Live render ────────────────────────────────────────────────────────────
function render(state,lastReward,done,feedback){
  if(!state)return;
  var epShort=(state.episode_id||'').slice(0,8);
  document.getElementById('hdr-episode').textContent='ep: '+epShort;
  document.getElementById('hdr-step').textContent='step: '+state.step_count+'/'+state.max_steps;
  var alerts=state.active_alerts||[];
  var hasCrit=alerts.some(function(a){return a.indexOf('[Critical]')>-1||a.indexOf('[Fatal]')>-1;});
  var hasWarn=alerts.some(function(a){return a.indexOf('[Warning]')>-1;});
  var badge=document.getElementById('hdr-status');
  badge.textContent=hasCrit?'CRISIS':hasWarn?'WARNING':'LIVE';
  badge.className='badge '+(hasCrit?'crisis':hasWarn?'warn':'live');
  var sc=state.scenario||'easy';var sid=state.scenario_id||'';
  var sbadge=document.getElementById('scenario-badge');
  sbadge.textContent='\u25CF '+sc.charAt(0).toUpperCase()+sc.slice(1)+' \u00B7 '+sid;
  sbadge.className='scenario-badge sc-'+sc;
  document.getElementById('scenario-title').textContent=state.scenario_title||'No active scenario';
  var desc=(state.task_info||'').split('\u2014').slice(1).join('\u2014').trim();
  document.getElementById('scenario-desc').textContent=desc;
  var pct=Math.round((state.step_count/state.max_steps)*100);
  document.getElementById('prog-label').textContent='Step '+state.step_count+' / '+state.max_steps;
  document.getElementById('prog-pct').textContent=pct+'%';
  document.getElementById('prog-fill').style.width=pct+'%';
  if(lastReward!==undefined&&lastReward!==null){
    var lrEl=document.getElementById('last-reward');
    lrEl.textContent=(lastReward>=0?'+':'')+lastReward.toFixed(2);
    lrEl.className='value '+(lastReward>0?'val-pos':lastReward<0?'val-neg':'val-neu');
  }
  var cum=state.cumulative_reward||0;
  var cumEl=document.getElementById('cum-reward');
  cumEl.textContent=(cum>=0?'+':'')+cum.toFixed(2);
  cumEl.className='value '+(cum>0?'val-pos':cum<0?'val-neg':'val-neu');
  var ac=document.getElementById('alerts-container');
  if(alerts.length===0){ac.innerHTML='<div class="alert-item alert-ok">\u2713 All systems nominal</div>';}
  else{ac.innerHTML=alerts.map(function(a){var cls=a.indexOf('[Fatal]')>-1?'alert-fatal':a.indexOf('[Critical]')>-1?'alert-crit':'alert-warn';return '<div class="alert-item '+cls+'">'+a+'</div>';}).join('');}
  if(feedback)document.getElementById('feedback-box').textContent=feedback;
  renderMetrics('cpu-grid',state.cpu_usage||{},state.latency_ms||{},state.error_rate||{},'cpu');
  renderMetrics('mem-grid',state.memory_usage||{},state.latency_ms||{},state.disk_io||{},'mem');
  renderLatency('lat-grid',state.latency_ms||{},state.error_rate||{});
  var sg=document.getElementById('svc-grid');
  sg.innerHTML=Object.entries(state.service_status||{}).map(function(e){var svc=e[0],st=e[1];var cls=st==='Healthy'?'s-healthy':st==='Degraded'?'s-degraded':'s-offline';return '<div class="svc-card"><div class="svc-name">'+svc.toUpperCase()+'</div><div class="svc-status '+cls+'"><span class="svc-dot"></span>'+st+'</div></div>';}).join('');
  var events=(state.recent_events||[]).filter(function(e){return e.step>0;});
  var lb=document.getElementById('event-log-body');
  document.getElementById('log-count').textContent=events.length+' events';
  if(events.length===0){lb.innerHTML='<div style="padding:16px;text-align:center;color:var(--ink-dim);font-family:var(--mono);font-size:.75rem;">No actions yet.</div>';}
  else{lb.innerHTML=events.slice().reverse().map(function(ev){var a=ev.action||{};var rwd=ev.reward!=null?ev.reward:0;var rwdCls=rwd>0?'pos':rwd<0?'neg':'neu';return '<div class="event-row"><span class="ev-step">#'+ev.step+'</span><span class="ev-act">'+(a.action_type||'-')+'</span><span class="ev-tgt">'+(a.target_resource||'(none)')+'</span><span class="ev-rwd '+rwdCls+'">'+(rwd>=0?'+':'')+rwd.toFixed(2)+'</span><span class="ev-fb">'+(ev.feedback||'')+'</span></div>';}).join('');}
}

function renderMetrics(id,primary,secondary,tertiary,type){
  var el=document.getElementById(id);
  el.innerHTML=Object.entries(primary).map(function(e){var node=e[0],val=e[1];var pct=Math.max(0,Math.min(100,val));var cls=pct>=90?'danger':pct>=70?'warning':'';var col=pct>=90?'var(--bad)':pct>=70?'var(--warn)':'var(--ok)';var lat=secondary[node]!==undefined?secondary[node].toFixed(0)+'ms':'-';var tert=tertiary[node]!==undefined?tertiary[node].toFixed(1)+'%':'-';var tl=type==='cpu'?'err':'disk';return '<div class="metric-card '+cls+'"><div class="node-name">'+node+'</div><div class="metric-val" style="color:'+col+'">'+pct.toFixed(1)+'%</div><div class="metric-sub"><span>lat '+lat+'</span><span>'+tl+' '+tert+'</span></div><div class="sparkbar"><div class="sparkfill" style="width:'+pct+'%;background:'+col+'"></div></div></div>';}).join('');
}

function renderLatency(id,latency,errorRate){
  var el=document.getElementById(id);
  el.innerHTML=Object.entries(latency).map(function(e){var node=e[0],val=e[1];var lat=Math.max(0,val);var pct=Math.min(100,(lat/2000)*100);var cls=lat>=1000?'danger':lat>=400?'warning':'';var col=lat>=1000?'var(--bad)':lat>=400?'var(--warn)':'var(--ok)';var err=errorRate[node]||0;return '<div class="metric-card '+cls+'"><div class="node-name">'+node+'</div><div class="metric-val" style="color:'+col+'">'+lat.toFixed(0)+'<span style="font-size:.7rem">ms</span></div><div class="metric-sub"><span>err '+err.toFixed(1)+'%</span></div><div class="sparkbar"><div class="sparkfill" style="width:'+pct+'%;background:'+col+'"></div></div></div>';}).join('');
}

function showDone(reward,feedback){
  var success=reward>=0.5;
  document.getElementById('done-title').textContent=success?'\u2713 Incident Resolved':'\u2717 Cluster Failed';
  document.getElementById('done-title').style.color=success?'var(--ok)':'var(--bad)';
  document.getElementById('done-sub').textContent=success?'Cluster successfully stabilised.':'The cluster crashed.';
  document.getElementById('done-score').textContent=(reward>=0?'+':'')+(reward||0).toFixed(2);
  document.getElementById('done-score').style.color=success?'var(--ok)':'var(--bad)';
  document.getElementById('done-feedback').textContent=feedback||'';
  document.getElementById('done-overlay').classList.add('show');
}

// ── History tab ────────────────────────────────────────────────────────────
function setFilter(f){
  activeFilter=f;
  document.querySelectorAll('.filter-btn').forEach(function(b){b.classList.remove('active');});
  document.getElementById('f-'+f).classList.add('active');
  renderHistory();
}

async function loadHistory(){
  try{
    var res=await fetch('/dashboard/history');
    allEpisodes=await res.json();
    renderHistory();
  }catch(e){console.error('History load failed',e);}
}

async function clearHistory(){
  if(!confirm('Clear all episode history?'))return;
  await fetch('/dashboard/history',{method:'DELETE'});
  allEpisodes=[];
  renderHistory();
}

function renderHistory(){
  // Filter
  var filtered=allEpisodes.filter(function(ep){
    if(activeFilter==='all')return true;
    if(activeFilter==='easy')return ep.scenario==='easy';
    if(activeFilter==='medium')return ep.scenario==='medium';
    if(activeFilter==='hard')return ep.scenario==='hard';
    if(activeFilter==='success')return ep.outcome==='SUCCESS';
    if(activeFilter==='failed')return ep.outcome==='FAILED';
    return true;
  });

  // Stats (always computed on full set)
  var total=allEpisodes.length;
  var successes=allEpisodes.filter(function(e){return e.outcome==='SUCCESS';}).length;
  var sr=total>0?Math.round((successes/total)*100):0;
  var avgRwd=total>0?(allEpisodes.reduce(function(s,e){return s+e.final_reward;},0)/total):0;
  var bestRwd=total>0?Math.max.apply(null,allEpisodes.map(function(e){return e.final_reward;})):0;
  var avgSteps=total>0?Math.round(allEpisodes.reduce(function(s,e){return s+e.total_steps;},0)/total):0;

  document.getElementById('stat-total').textContent=total;
  document.getElementById('stat-sr').textContent=total>0?sr+'%':'-';
  document.getElementById('stat-sr-sub').textContent=total>0?(successes+'/'+total+' resolved'):'';
  document.getElementById('stat-avg').textContent=total>0?(avgRwd>=0?'+':'')+avgRwd.toFixed(2):'-';
  document.getElementById('stat-best').textContent=total>0?(bestRwd>=0?'+':'')+bestRwd.toFixed(2):'-';
  document.getElementById('stat-steps').textContent=total>0?avgSteps:'-';

  var grid=document.getElementById('ep-grid');
  if(filtered.length===0){
    grid.innerHTML='<div class="empty-state"><div class="big-icon">&#128203;</div><h3>'+(total===0?'No episodes yet':'No matching episodes')+'</h3><p>'+(total===0?'Complete an episode on the Live Incident tab to see it appear here.':'Try a different filter to see more episodes.')+'</p></div>';
    return;
  }

  // Render cards — newest first
  grid.innerHTML=filtered.slice().reverse().map(function(ep,idx){
    var success=ep.outcome==='SUCCESS';
    var outcomeClass=success?'success':'failed';
    var outcomeText=success?'\u2713 RESOLVED':'\u2717 FAILED';
    var scCls=ep.scenario==='easy'?'var(--ok)':ep.scenario==='medium'?'var(--warn)':'var(--bad)';
    var rewardPct=Math.max(0,Math.min(100,(ep.final_reward+1)/2*100));
    var rewardCol=ep.final_reward>=0.5?'var(--ok)':ep.final_reward>=0?'var(--warn)':'var(--bad)';
    var epShort=(ep.episode_id||'').slice(0,8);
    var ts=ep.finished_at?new Date(ep.finished_at).toLocaleTimeString():'';

    var actionsHtml=(ep.actions||[]).map(function(a){
      var rwd=a.reward!=null?a.reward:0;
      var rwdCls=rwd>0?'pos':rwd<0?'neg':'neu';
      return '<div class="ep-action-row">'
        +'<span class="ea-step">#'+a.step+'</span>'
        +'<span class="ea-act">'+(a.action_type||'-')+'</span>'
        +'<span class="ea-tgt">'+(a.target_resource||'(none)')+'</span>'
        +'<span class="ea-rwd '+rwdCls+'">'+(rwd>=0?'+':'')+rwd.toFixed(2)+'</span>'
        +'<span class="ea-fb">'+(a.feedback||'')+'</span>'
        +'</div>';
    }).join('');

    return '<div class="ep-card '+outcomeClass+'">'
      +'<div class="ep-header">'
        +'<div>'
          +'<div class="ep-id">EP-'+epShort+'</div>'
          +'<div style="font-family:var(--mono);font-size:.62rem;color:'+scCls+';margin-top:2px;">'
            +(ep.scenario||'').toUpperCase()+' &middot; '+(ep.scenario_id||'')
          +'</div>'
        +'</div>'
        +'<span class="ep-outcome '+outcomeClass+'">'+outcomeText+'</span>'
      +'</div>'
      +'<div class="ep-body">'
        +'<div class="ep-title">'+(ep.scenario_title||'Unknown Scenario')+'</div>'
        +'<div class="ep-meta">'
          +'<span>&#9201; '+ep.total_steps+' steps</span>'
          +'<span>&#127942; '+(ep.final_reward>=0?'+':'')+ep.final_reward.toFixed(2)+'</span>'
          +'<span>&#931; '+(ep.cumulative_reward>=0?'+':'')+ep.cumulative_reward.toFixed(2)+'</span>'
        +'</div>'
        +'<div class="ep-reward-bar"><div class="ep-reward-fill" style="width:'+rewardPct+'%;background:'+rewardCol+'"></div></div>'
        +'<div class="ep-actions">'+actionsHtml+'</div>'
        +'<div class="ep-ts">'+ts+'</div>'
      +'</div>'
    +'</div>';
  }).join('');
}

// ── SSE live updates ────────────────────────────────────────────────────────
var sse=new EventSource('/dashboard/events');
sse.onmessage=function(e){if(!lastDone)render(JSON.parse(e.data));};

// ── Initial load ────────────────────────────────────────────────────────────
getState();
</script>
</body>
</html>"""


# Strip any conflicting routes openenv may have registered for these paths
app.router.routes = [
    r for r in app.router.routes
    if getattr(r, "path", None) not in ["/", "/dashboard", "/web"]
]

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
async def dashboard_page():
    return HTMLResponse(content=DASHBOARD_HTML)


@app.post("/dashboard/reset")
async def dashboard_reset():
    async with dashboard_lock:
        obs = dashboard_env.reset()
        _start_episode_tracking(dashboard_env)
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
        _finish_episode_tracking(dashboard_env, obs.reward or 0.0, obs.done or False)
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


@app.get("/dashboard/history")
async def dashboard_history():
    """Return all completed episode summaries for the History tab."""
    return JSONResponse(episode_history)


@app.delete("/dashboard/history")
async def dashboard_clear_history():
    """Clear all episode history."""
    global episode_history
    episode_history = []
    return JSONResponse({"ok": True, "cleared": True})


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