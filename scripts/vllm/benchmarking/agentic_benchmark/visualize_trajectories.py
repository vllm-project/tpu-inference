#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Interactive trajectory and turn waterfall visualizer for agentic RL benchmarks.

Similar to Google Trellis (https://github.com/google/trellis/pull/67), this tool
renders an interactive waterfall timeline displaying multi-turn trajectories,
turn-by-turn latency segmentation (model generation vs. environment/tool idle time),
batch launch boundaries, and per-turn metrics (prompt/gen tokens, TTFT, TPOT, tok/s).

Input formats supported:
1. Trajectory JSONL from benchmark_agentic.py (--save-trajectory-file).
2. Responses JSONL from benchmark_agentic.py (--save-responses-file).
3. Trellis inference_metrics.jsonl files.
4. Raw log files with [DEBUG_INFERENCE][Turn] and [DEBUG_INFERENCE][Trajectory] lines.
5. Built-in --demo mode generating realistic sample data.
"""

import argparse
import http.server
import json
import os
import re
import socketserver
import sys
import webbrowser
from typing import Any, Dict, List, Optional, Tuple


def generate_demo_data() -> Dict[str, Any]:
    """Generates realistic synthetic multi-batch agentic benchmark trajectories."""
    import random
    rng = random.Random(42)

    num_batches = 3
    concurrency = 4
    group_size = 4
    time_between_batches = 20.0
    batch_launch_times = [round(b * time_between_batches, 3) for b in range(num_batches)]

    trajectories = []
    turns_all = []
    traj_idx = 0

    for b in range(num_batches):
        b_launch = batch_launch_times[b]
        for g in range(1, concurrency + 1):
            g_idx = b * concurrency + g
            prompt_len = rng.randint(4000, 8000)
            for s in range(group_size):
                traj_idx += 1
                traj_id = f"b{b}_g{g_idx}_s{s}"
                num_turns = rng.randint(6, 12)
                t_cur = b_launch + rng.uniform(0.05, 0.4)
                traj_turns = []
                tot_out = 0
                tot_model = 0.0
                tot_tool = 0.0
                t_start = t_cur

                for turn in range(1, num_turns + 1):
                    # Turn 1 prefill is longer on cache miss
                    is_miss = (s == 0 and turn == 1)
                    ttft_s = rng.uniform(0.4, 0.8) if is_miss else rng.uniform(0.05, 0.15)
                    gen_toks = rng.randint(150, 450)
                    tpot_ms = rng.uniform(4.5, 7.5)
                    decode_s = (gen_toks - 1) * (tpot_ms / 1000.0)
                    model_time_s = round(ttft_s + decode_s, 3)
                    tool_time_s = round(rng.uniform(1.0, 3.5), 3) if turn < num_turns else 0.0

                    turn_start = round(t_cur, 3)
                    turn_end = round(turn_start + model_time_s, 3)
                    tot_out += gen_toks
                    tot_model += model_time_s
                    tot_tool += tool_time_s

                    t_rec = {
                        "type": "turn",
                        "batch_idx": b,
                        "group_idx": g_idx,
                        "stream_idx": s,
                        "traj_id": traj_id,
                        "turn": turn,
                        "num_turns": num_turns,
                        "start_time_s": turn_start,
                        "end_time_s": turn_end,
                        "model_time_s": model_time_s,
                        "tool_time_s": tool_time_s,
                        "prompt_tokens": prompt_len + (turn - 1) * 200,
                        "output_tokens": gen_toks,
                        "ttft_ms": round(ttft_s * 1000.0, 1),
                        "tpot_ms": round(tpot_ms, 2),
                        "total_time_ms": round(model_time_s * 1000.0, 1),
                        "success": True,
                    }
                    traj_turns.append(t_rec)
                    turns_all.append(t_rec)
                    t_cur = turn_end + tool_time_s

                traj_rec = {
                    "type": "trajectory",
                    "traj_id": traj_id,
                    "batch_idx": b,
                    "group_idx": g_idx,
                    "stream_idx": s,
                    "num_turns": num_turns,
                    "start_time_s": round(t_start, 3),
                    "end_time_s": round(t_cur, 3),
                    "duration_s": round(t_cur - t_start, 3),
                    "model_time_s": round(tot_model, 3),
                    "tool_time_s": round(tot_tool, 3),
                    "prompt_tokens": prompt_len,
                    "output_tokens": tot_out,
                    "status": "COMPLETED",
                }
                trajectories.append(traj_rec)

    total_duration = max(t["end_time_s"] for t in trajectories)
    meta = {
        "type": "benchmark_meta",
        "num_batches": num_batches,
        "time_between_batches": time_between_batches,
        "concurrency": concurrency,
        "total_groups": concurrency * num_batches,
        "total_trajectories": len(trajectories),
        "total_turns": len(turns_all),
        "total_duration_sec": round(total_duration, 2),
        "batch_launch_times": batch_launch_times,
    }
    return {
        "meta": meta,
        "trajectories": trajectories,
        "turns": turns_all,
    }


def parse_metrics_file(path: str) -> Dict[str, Any]:
    """Parses various metrics and log formats into unified trajectory and turn records."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    meta = {
        "num_batches": 1,
        "time_between_batches": 0.0,
        "concurrency": 1,
        "batch_launch_times": [0.0],
    }
    trajectories: Dict[str, Dict[str, Any]] = {}
    turns: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Format 1: JSONL file
    is_jsonl = False
    for line in lines[:10]:
        if line.startswith("{") and line.endswith("}"):
            is_jsonl = True
            break

    if is_jsonl:
        for line in lines:
            try:
                rec = json.loads(line)
            except Exception:
                continue

            r_type = rec.get("type")
            if r_type == "benchmark_meta":
                meta.update(rec)
            elif r_type == "trajectory":
                traj_id = rec.get("traj_id")
                if traj_id:
                    trajectories[traj_id] = rec
            elif r_type == "turn" or ("turn" in rec and "group_idx" in rec):
                # Turn record
                b_idx = rec.get("batch_idx", 0)
                g_idx = rec.get("group_idx", 0)
                s_idx = rec.get("stream_idx", 0)
                traj_id = rec.get("traj_id") or f"b{b_idx}_g{g_idx}_s{s_idx}"
                t_num = rec.get("turn", 1)
                num_turns = rec.get("num_turns", 1)
                model_time = rec.get("model_time_s")
                if model_time is None and "total_time_ms" in rec:
                    model_time = round(rec["total_time_ms"] / 1000.0, 4)

                turn_rec = {
                    "type": "turn",
                    "batch_idx": b_idx,
                    "group_idx": g_idx,
                    "stream_idx": s_idx,
                    "traj_id": traj_id,
                    "turn": t_num,
                    "num_turns": num_turns,
                    "start_time_s": rec.get("start_time_s", 0.0),
                    "end_time_s": rec.get("end_time_s", model_time or 0.0),
                    "model_time_s": model_time or 0.0,
                    "tool_time_s": rec.get("tool_time_s", 0.0),
                    "prompt_tokens": rec.get("input_history_tokens") or rec.get("prompt_tokens", 0),
                    "output_tokens": rec.get("output_tokens") or rec.get("completion_tokens", 0),
                    "ttft_ms": rec.get("ttft_ms"),
                    "tpot_ms": rec.get("tpot_ms"),
                    "total_time_ms": rec.get("total_time_ms"),
                    "success": rec.get("success", True),
                    "error": rec.get("error"),
                }
                turns.append(turn_rec)

    # Format 2: Fallback parser for Tunix / Trellis [DEBUG_INFERENCE] text logs
    if not turns:
        # Regex for [DEBUG_INFERENCE][Turn] and [DEBUG_INFERENCE][Trajectory]
        turn_re = re.compile(
            r"\[DEBUG_INFERENCE\]\[Turn\]\s+(?:prompt_id=(?P<pid>[^\s,]+)|traj_id=(?P<tid>[^\s,]+)).*?"
            r"step_index=(?P<step>\d+).*?"
            r"(?:model_time_sec=(?P<mtime>[\d\.]+)|latency=(?P<lat>[\d\.]+)).*?"
            r"(?:prompt_tokens=(?P<ptok>\d+))?.*?"
            r"(?:completion_tokens=(?P<ctok>\d+))?",
            re.IGNORECASE,
        )
        for line in lines:
            m = turn_re.search(line)
            if m:
                tid = m.group("tid") or m.group("pid") or "traj_0"
                step = int(m.group("step"))
                mtime = float(m.group("mtime") or m.group("lat") or 0.0)
                ptok = int(m.group("ptok") or 0)
                ctok = int(m.group("ctok") or 0)
                turns.append({
                    "type": "turn",
                    "batch_idx": 0,
                    "group_idx": 0,
                    "stream_idx": 0,
                    "traj_id": tid,
                    "turn": step + 1,
                    "num_turns": step + 1,
                    "start_time_s": 0.0,
                    "end_time_s": mtime,
                    "model_time_s": mtime,
                    "tool_time_s": 0.0,
                    "prompt_tokens": ptok,
                    "output_tokens": ctok,
                    "success": True,
                })

    # Synthesize trajectory summary records if missing
    traj_turns_map: Dict[str, List[Dict[str, Any]]] = {}
    for t in turns:
        traj_turns_map.setdefault(t["traj_id"], []).append(t)

    for traj_id, t_list in traj_turns_map.items():
        if traj_id not in trajectories:
            t_list.sort(key=lambda x: x.get("turn", 0))
            first = t_list[0]
            last = t_list[-1]
            tot_out = sum(x.get("output_tokens", 0) for x in t_list)
            tot_model = sum(x.get("model_time_s", 0.0) for x in t_list)
            tot_tool = sum(x.get("tool_time_s", 0.0) for x in t_list)
            t_start = first.get("start_time_s", 0.0)
            t_end = last.get("end_time_s", 0.0) + last.get("tool_time_s", 0.0)
            b_idx = first.get("batch_idx", 0)
            trajectories[traj_id] = {
                "type": "trajectory",
                "traj_id": traj_id,
                "batch_idx": b_idx,
                "group_idx": first.get("group_idx", 0),
                "stream_idx": first.get("stream_idx", 0),
                "num_turns": len(t_list),
                "start_time_s": round(t_start, 4),
                "end_time_s": round(t_end, 4),
                "duration_s": round(max(0.0, t_end - t_start), 4),
                "model_time_s": round(tot_model, 4),
                "tool_time_s": round(tot_tool, 4),
                "prompt_tokens": first.get("prompt_tokens", 0),
                "output_tokens": tot_out,
                "status": "COMPLETED" if all(x.get("success", False) for x in t_list) else "FAILED",
            }

    # Discover batch launch times if not explicitly recorded
    batches_seen = sorted(set(t.get("batch_idx", 0) for t in trajectories.values()))
    if len(batches_seen) > 1 and len(meta.get("batch_launch_times", [])) <= 1:
        launch_times = []
        for b in batches_seen:
            b_trajs = [tr for tr in trajectories.values() if tr.get("batch_idx") == b]
            launch_times.append(min((tr.get("start_time_s", 0.0) for tr in b_trajs), default=0.0))
        meta["batch_launch_times"] = launch_times
        meta["num_batches"] = len(batches_seen)

    total_duration = max((tr.get("end_time_s", 0.0) for tr in trajectories.values()), default=1.0)
    meta["total_duration_sec"] = round(total_duration, 3)
    meta["total_trajectories"] = len(trajectories)
    meta["total_turns"] = len(turns)

    return {
        "meta": meta,
        "trajectories": list(trajectories.values()),
        "turns": turns,
    }


def generate_html(data: Dict[str, Any], title: str = "Agentic RL Trajectory Turn Waterfall") -> str:
    """Generates a standalone, dependency-free interactive HTML waterfall visualization."""
    data_json = json.dumps(data)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #1e293b;
      --panel-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --grid: #334155;
      --accent: #38bdf8;
      --success: #22c55e;
      --danger: #ef4444;
      --warning: #f59e0b;
      --b0: #3b82f6;
      --b1: #10b981;
      --b2: #f59e0b;
      --b3: #ec4899;
      --b4: #8b5cf6;
      --b5: #06b6d4;
      --b6: #f97316;
      --b7: #6366f1;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.4;
      padding: 16px 20px 40px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--panel-border);
    }}
    h1 {{
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #fff;
    }}
    .subtitle {{
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 2px;
    }}
    .stats-bar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .stat-card {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 6px;
      padding: 10px 12px;
    }}
    .stat-card .label {{
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
    }}
    .stat-card .val {{
      font-size: 18px;
      font-weight: 700;
      color: #fff;
    }}
    .stat-card .unit {{
      font-size: 12px;
      font-weight: 400;
      color: var(--text-muted);
      margin-left: 2px;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 6px;
      padding: 8px 14px;
      margin-bottom: 14px;
      font-size: 13px;
    }}
    .control-group {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    label {{
      font-weight: 600;
      color: var(--text-muted);
      font-size: 12px;
    }}
    select, input[type="text"] {{
      background: #0b1120;
      color: #fff;
      border: 1px solid var(--panel-border);
      padding: 5px 8px;
      border-radius: 4px;
      font-size: 12px;
      outline: none;
    }}
    select:focus, input:focus {{
      border-color: var(--accent);
    }}
    .toggle {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }}
    .toggle input {{ cursor: pointer; }}
    .chart-container {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 6px;
      padding: 14px 14px 6px;
      position: relative;
    }}
    .chart-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      font-size: 12px;
    }}
    .chart-scroll {{
      overflow-x: auto;
      max-height: 580px;
      overflow-y: auto;
      border: 1px solid var(--panel-border);
      border-radius: 4px;
      background: #090e1a;
      position: relative;
    }}
    svg {{
      display: block;
      user-select: none;
    }}
    .legend {{
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 10px;
      padding-top: 8px;
      border-top: 1px solid var(--panel-border);
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .tag {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 11px;
      font-weight: 600;
      background: #334155;
      color: #fff;
    }}
    #tooltip {{
      position: fixed;
      display: none;
      background: rgba(15, 23, 42, 0.96);
      border: 1px solid var(--accent);
      border-radius: 6px;
      padding: 10px 14px;
      font-size: 12px;
      color: #fff;
      pointer-events: none;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.7);
      z-index: 1000;
      white-space: pre-line;
      max-width: 380px;
      backdrop-filter: blur(4px);
    }}
    .profile-card {{
      margin-top: 16px;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 6px;
      padding: 12px 16px;
    }}
    .profile-card h3 {{
      font-size: 14px;
      margin-bottom: 8px;
      color: #fff;
    }}
  </style>
</head>
<body>
  <div id="tooltip"></div>

  <header>
    <div>
      <h1>{title}</h1>
      <div class="subtitle" id="subtitle-info">Continuous multi-batch trajectory turn waterfall & latency profiling</div>
    </div>
    <div style="display:flex; gap:8px;">
      <span class="tag" style="background:var(--accent); color:#000;">Interactive Explorer</span>
    </div>
  </header>

  <div class="stats-bar" id="stats-bar">
    <!-- Populated dynamically -->
  </div>

  <div class="controls">
    <div class="control-group">
      <label for="batch-filter">Batch:</label>
      <select id="batch-filter">
        <option value="all">All Batches</option>
      </select>
    </div>

    <div class="control-group">
      <label for="search-input">Search / Group:</label>
      <input type="text" id="search-input" placeholder="e.g. g1, s0, b0">
    </div>

    <div class="control-group">
      <label for="sort-select">Sort by:</label>
      <select id="sort-select">
        <option value="start">Start Time</option>
        <option value="duration">Trajectory Duration</option>
        <option value="turns">Turn Count</option>
        <option value="id">Trajectory ID</option>
      </select>
    </div>

    <div class="control-group">
      <label class="toggle">
        <input type="checkbox" id="toggle-tool-time" checked>
        <span>Show Tool / Env Idle Time</span>
      </label>
    </div>

    <div class="control-group" style="margin-left: auto;">
      <label for="zoom-scale">Zoom:</label>
      <input type="range" id="zoom-scale" min="0.5" max="3.0" step="0.1" value="1.0" style="width: 100px;">
      <span id="zoom-val" style="font-size: 11px; color: var(--text-muted); width: 32px;">1.0x</span>
    </div>
  </div>

  <div class="chart-container">
    <div class="chart-header">
      <div id="chart-count-info" style="font-weight:600; color:var(--text);">Loading trajectories...</div>
      <div style="display:flex; gap:8px;">
        <span class="tag" style="background:#000; color:#fff; border:1px solid #475569;">Black line: Batch Dispatch Boundary</span>
      </div>
    </div>

    <div class="chart-scroll" id="scroll-box">
      <div id="svg-host"></div>
    </div>

    <div class="legend">
      <span style="font-weight:600; color:#fff;">Legend:</span>
      <div class="legend-item">
        <span style="display:inline-block;width:18px;height:13px;background:var(--b0);border-radius:2px;text-align:center;line-height:13px;color:#fff;font-size:9px;font-weight:700;">1</span>
        <span>Model Generation (Turn #)</span>
      </div>
      <div class="legend-item">
        <span style="display:inline-block;width:18px;height:13px;background:var(--b0);opacity:0.4;border-radius:2px;border:1px solid var(--b0);"></span>
        <span>Environment / Tool Idle Time</span>
      </div>
      <div class="legend-item">
        <span style="display:inline-block;width:16px;height:13px;background:#000;border:1px solid #fff;border-radius:2px;color:#fff;text-align:center;line-height:11px;font-size:8px;font-weight:700;">b0</span>
        <span>Solid black line: Batch Launch Marker</span>
      </div>
      <div class="legend-item" style="margin-left:auto;">
        <span>Colors: Batch C[batch % 8]</span>
      </div>
    </div>
  </div>

  <div class="profile-card" id="profile-container">
    <h3>Turn-by-Turn Latency Profile (Average Model vs Tool Duration)</h3>
    <div id="turn-profile-host"></div>
  </div>

  <script>
    const RAW_DATA = {data_json};
    const COLORS = [
      "#3b82f6", "#10b981", "#f59e0b", "#ec4899",
      "#8b5cf6", "#06b6d4", "#f97316", "#6366f1"
    ];

    const tooltip = document.getElementById("tooltip");
    function showTip(e, text) {{
      tooltip.textContent = text;
      tooltip.style.display = "block";
      const x = Math.min(e.clientX + 14, window.innerWidth - 380);
      const y = Math.min(e.clientY + 14, window.innerHeight - 180);
      tooltip.style.left = x + "px";
      tooltip.style.top = y + "px";
    }}
    function hideTip() {{
      tooltip.style.display = "none";
    }}

    // Greedy swimlane row packing
    function packSwimlanes(items) {{
      items.sort((a, b) => a.start_time_s - b.start_time_s || b.duration_s - a.duration_s);
      const rowEnds = [];
      const rowAssigned = [];
      for (let i = 0; i < items.length; i++) {{
        const it = items[i];
        let placed = false;
        for (let r = 0; r < rowEnds.length; r++) {{
          if (rowEnds[r] <= it.start_time_s) {{
            rowAssigned[i] = r;
            rowEnds[r] = it.end_time_s;
            placed = true;
            break;
          }}
        }}
        if (!placed) {{
          rowAssigned[i] = rowEnds.length;
          rowEnds.push(it.end_time_s);
        }}
      }}
      return {{ nrows: Math.max(rowEnds.length, 1), rows: rowAssigned }};
    }}

    function renderStats(trajs, turns, meta) {{
      const sb = document.getElementById("stats-bar");
      sb.innerHTML = "";
      const totalDur = meta.total_duration_sec || 0;
      const numTrajs = trajs.length;
      const numTurns = turns.length;
      const avgDur = numTrajs ? (trajs.reduce((a, b) => a + (b.duration_s || 0), 0) / numTrajs).toFixed(2) : "0.00";
      const avgModel = numTurns ? (turns.reduce((a, b) => a + (b.model_time_s || 0), 0) / numTurns).toFixed(2) : "0.00";
      const avgTool = numTurns ? (turns.reduce((a, b) => a + (b.tool_time_s || 0), 0) / numTurns).toFixed(2) : "0.00";
      const totOutToks = turns.reduce((a, b) => a + (b.output_tokens || 0), 0);
      const tps = totalDur > 0 ? (totOutToks / totalDur).toFixed(1) : "0.0";

      const cards = [
        {{ label: "Simulated Batches", val: meta.num_batches || 1, unit: "" }},
        {{ label: "Total Trajectories", val: numTrajs, unit: "" }},
        {{ label: "Total Turns", val: numTurns, unit: "" }},
        {{ label: "Avg Trajectory", val: avgDur, unit: "s" }},
        {{ label: "Avg Model Turn", val: avgModel, unit: "s" }},
        {{ label: "Avg Tool Idle", val: avgTool, unit: "s" }},
        {{ label: "Output Tokens", val: totOutToks.toLocaleString(), unit: "" }},
        {{ label: "Throughput", val: tps, unit: "tok/s" }}
      ];

      cards.forEach(c => {{
        const el = document.createElement("div");
        el.className = "stat-card";
        el.innerHTML = `<div class="label">${{c.label}}</div><div class="val">${{c.val}}<span class="unit">${{c.unit}}</span></div>`;
        sb.appendChild(el);
      }});
    }}

    function renderChart() {{
      const host = document.getElementById("svg-host");
      host.innerHTML = "";

      const batchFilter = document.getElementById("batch-filter").value;
      const searchQuery = document.getElementById("search-input").value.trim().toLowerCase();
      const sortBy = document.getElementById("sort-select").value;
      const showToolTime = document.getElementById("toggle-tool-time").checked;
      const zoom = parseFloat(document.getElementById("zoom-scale").value);

      let trajs = [...RAW_DATA.trajectories];
      if (batchFilter !== "all") {{
        const bTarget = parseInt(batchFilter, 10);
        trajs = trajs.filter(t => t.batch_idx === bTarget);
      }}
      if (searchQuery) {{
        trajs = trajs.filter(t => (
          t.traj_id.toLowerCase().includes(searchQuery) ||
          String(t.group_idx).includes(searchQuery) ||
          String(t.stream_idx).includes(searchQuery)
        ));
      }}

      // Sorting
      if (sortBy === "start") {{
        trajs.sort((a, b) => a.start_time_s - b.start_time_s || a.traj_id.localeCompare(b.traj_id));
      }} else if (sortBy === "duration") {{
        trajs.sort((a, b) => b.duration_s - a.duration_s);
      }} else if (sortBy === "turns") {{
        trajs.sort((a, b) => b.num_turns - a.num_turns);
      }} else if (sortBy === "id") {{
        trajs.sort((a, b) => a.traj_id.localeCompare(b.traj_id));
      }}

      document.getElementById("chart-count-info").textContent =
        `Showing ${{trajs.length}} trajectories (${{RAW_DATA.meta.total_turns}} total turns across ${{RAW_DATA.meta.num_batches}} batches)`;

      if (!trajs.length) {{
        host.innerHTML = '<div style="padding:40px; text-align:center; color:var(--text-muted);">No trajectories match filter.</div>';
        return;
      }}

      const packing = packSwimlanes(trajs);
      trajs.forEach((t, i) => {{ t.row = packing.rows[i]; }});

      const tStart = 0.0;
      const tEnd = Math.max(...trajs.map(t => t.end_time_s), RAW_DATA.meta.total_duration_sec || 1.0);
      const span = Math.max(tEnd - tStart, 1.0);

      const left = 80, right = 40, top = 28, rowH = 22, barH = 16, bottom = 28;
      const baseW = Math.max(document.getElementById("scroll-box").clientWidth - 20, 800);
      const plotW = Math.round((baseW - left - right) * zoom);
      const svgW = left + plotW + right;
      const plotH = packing.nrows * rowH;
      const svgH = top + plotH + bottom;

      const tx = (t) => left + ((t - tStart) / span) * plotW;

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("width", svgW);
      svg.setAttribute("height", svgH);
      svg.setAttribute("viewBox", `0 0 ${{svgW}} ${{svgH}}`);

      // Time grid
      const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
      const stp = steps.find(s => span / s <= 14) || 60;
      for (let t = 0; t <= tEnd; t += stp) {{
        const gx = tx(t);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", gx); line.setAttribute("x2", gx);
        line.setAttribute("y1", top - 4); line.setAttribute("y2", top + plotH);
        line.setAttribute("stroke", "var(--grid)");
        line.setAttribute("opacity", "0.4");
        svg.appendChild(line);

        const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
        txt.setAttribute("x", gx);
        txt.setAttribute("y", svgH - 10);
        txt.setAttribute("text-anchor", "middle");
        txt.setAttribute("fill", "var(--text-muted)");
        txt.setAttribute("font-size", "10");
        txt.textContent = t >= 60 ? (t / 60).toFixed(1) + "m" : t + "s";
        svg.appendChild(txt);
      }}

      // Swimlane row labels & guidelines
      for (let r = 0; r < packing.nrows; r++) {{
        const ry = top + r * rowH;
        const rlab = document.createElementNS("http://www.w3.org/2000/svg", "text");
        rlab.setAttribute("x", left - 8);
        rlab.setAttribute("y", ry + barH - 3);
        rlab.setAttribute("text-anchor", "end");
        rlab.setAttribute("fill", "var(--text-muted)");
        rlab.setAttribute("font-size", "10");
        rlab.textContent = "lane " + (r + 1);
        svg.appendChild(rlab);

        const gline = document.createElementNS("http://www.w3.org/2000/svg", "line");
        gline.setAttribute("x1", left); gline.setAttribute("x2", left + plotW);
        gline.setAttribute("y1", ry + barH + 3); gline.setAttribute("y2", ry + barH + 3);
        gline.setAttribute("stroke", "var(--grid)");
        gline.setAttribute("stroke-dasharray", "2 4");
        gline.setAttribute("opacity", "0.25");
        svg.appendChild(gline);
      }}

      // Trajectories & turns map
      const turnsByTraj = {{}};
      RAW_DATA.turns.forEach(t => {{
        turnsByTraj[t.traj_id] = turnsByTraj[t.traj_id] || [];
        turnsByTraj[t.traj_id].push(t);
      }});

      trajs.forEach(it => {{
        const ry = top + it.row * rowH;
        const bCol = COLORS[it.batch_idx % COLORS.length];
        const tTurns = (turnsByTraj[it.traj_id] || []).sort((a, b) => a.turn - b.turn);

        tTurns.forEach(turn => {{
          const mX0 = tx(turn.start_time_s);
          const mX1 = tx(turn.end_time_s);
          const mW = Math.max(mX1 - mX0, 1.5);

          // 1. Model generation bar
          const mRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          mRect.setAttribute("x", mX0);
          mRect.setAttribute("y", ry);
          mRect.setAttribute("width", mW);
          mRect.setAttribute("height", barH);
          mRect.setAttribute("fill", bCol);
          mRect.setAttribute("rx", "2");
          mRect.setAttribute("stroke", "rgba(0,0,0,0.35)");
          mRect.setAttribute("stroke-width", "0.5");
          mRect.style.cursor = "pointer";

          const tokRate = turn.model_time_s > 0 ? (turn.output_tokens / turn.model_time_s).toFixed(1) : "0.0";
          const mTip = `Trajectory: ${{it.traj_id}} (Batch ${{it.batch_idx}}, Group ${{it.group_idx}}, Stream ${{it.stream_idx}})
Turn: ${{turn.turn}} / ${{it.num_turns}}
Phase: Model Generation
Duration: ${{turn.model_time_s.toFixed(2)}}s
Tokens: ${{turn.prompt_tokens.toLocaleString()}} prompt / ${{turn.output_tokens.toLocaleString()}} gen (${{tokRate}} tok/s)
TTFT: ${{turn.ttft_ms != null ? turn.ttft_ms.toFixed(1) + 'ms' : 'N/A'}} | TPOT: ${{turn.tpot_ms != null ? turn.tpot_ms.toFixed(2) + 'ms' : 'N/A'}}
Span: +${{turn.start_time_s.toFixed(2)}}s → +${{turn.end_time_s.toFixed(2)}}s
Trajectory Total: ${{it.duration_s.toFixed(1)}}s (${{it.output_tokens.toLocaleString()}} tokens)`;

          mRect.addEventListener("mousemove", e => showTip(e, mTip));
          mRect.addEventListener("mouseleave", hideTip);
          svg.appendChild(mRect);

          // Turn number label inside generation block if wide enough
          if (mW >= 13) {{
            const numTxt = document.createElementNS("http://www.w3.org/2000/svg", "text");
            numTxt.setAttribute("x", mX0 + mW / 2);
            numTxt.setAttribute("y", ry + barH / 2 + 3.5);
            numTxt.setAttribute("text-anchor", "middle");
            numTxt.setAttribute("fill", "#ffffff");
            numTxt.setAttribute("font-size", mW >= 20 ? "9.5" : "8");
            numTxt.setAttribute("font-weight", "700");
            numTxt.setAttribute("pointer-events", "none");
            numTxt.textContent = String(turn.turn);
            svg.appendChild(numTxt);
          }}

          // 2. Tool / Environment Idle Time bar
          if (showToolTime && turn.tool_time_s > 0) {{
            const eX0 = mX1;
            const eX1 = tx(turn.end_time_s + turn.tool_time_s);
            const eW = Math.max(eX1 - eX0, 1.0);

            const eRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            eRect.setAttribute("x", eX0);
            eRect.setAttribute("y", ry);
            eRect.setAttribute("width", eW);
            eRect.setAttribute("height", barH);
            eRect.setAttribute("fill", bCol);
            eRect.setAttribute("opacity", "0.4");
            eRect.setAttribute("rx", "1");
            eRect.setAttribute("stroke", bCol);
            eRect.setAttribute("stroke-width", "0.5");
            eRect.style.cursor = "pointer";

            const eTip = `Trajectory: ${{it.traj_id}} (Batch ${{it.batch_idx}}, Group ${{it.group_idx}}, Stream ${{it.stream_idx}})
Turn: ${{turn.turn}} / ${{it.num_turns}}
Phase: Tool / Environment Idle Time
Duration: ${{turn.tool_time_s.toFixed(2)}}s
Span: +${{turn.end_time_s.toFixed(2)}}s → +${{(turn.end_time_s + turn.tool_time_s).toFixed(2)}}s`;

            eRect.addEventListener("mousemove", e => showTip(e, eTip));
            eRect.addEventListener("mouseleave", hideTip);
            svg.appendChild(eRect);
          }}
        }});
      }});

      // Batch launch vertical boundary lines
      const launchTimes = RAW_DATA.meta.batch_launch_times || [0.0];
      launchTimes.forEach((bTime, bIdx) => {{
        const bx = tx(bTime);
        const bline = document.createElementNS("http://www.w3.org/2000/svg", "line");
        bline.setAttribute("x1", bx); blline = bline; bline.setAttribute("x2", bx);
        bline.setAttribute("y1", top - 6); bline.setAttribute("y2", top + plotH + 4);
        bline.setAttribute("stroke", "#000");
        bline.setAttribute("stroke-width", "2");
        bline.setAttribute("opacity", "0.95");
        svg.appendChild(bline);

        const badge = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        badge.setAttribute("x", bx - 14);
        badge.setAttribute("y", top - 22);
        badge.setAttribute("width", "28");
        badge.setAttribute("height", "15");
        badge.setAttribute("rx", "3");
        badge.setAttribute("fill", "#000");
        badge.setAttribute("stroke", "#475569");
        badge.setAttribute("stroke-width", "1");
        svg.appendChild(badge);

        const blbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
        blbl.setAttribute("x", bx);
        blbl.setAttribute("y", top - 11);
        blbl.setAttribute("text-anchor", "middle");
        blbl.setAttribute("fill", "#fff");
        blbl.setAttribute("font-size", "9.5");
        blbl.setAttribute("font-weight", "700");
        blbl.textContent = "b" + bIdx;
        svg.appendChild(blbl);

        const bTip = (e) => showTip(e, `Batch ${{bIdx}} launch boundary dispatched at +${{bTime.toFixed(2)}}s`);
        [bline, badge, blbl].forEach(el => {{
          el.addEventListener("mousemove", bTip);
          el.addEventListener("mouseleave", hideTip);
        }});
      }});

      host.appendChild(svg);
    }}

    function renderTurnProfile() {{
      const host = document.getElementById("turn-profile-host");
      host.innerHTML = "";
      const turns = RAW_DATA.turns;
      if (!turns.length) return;

      const maxTurn = Math.max(...turns.map(t => t.turn));
      const modelSum = new Array(maxTurn + 1).fill(0);
      const toolSum = new Array(maxTurn + 1).fill(0);
      const count = new Array(maxTurn + 1).fill(0);

      turns.forEach(t => {{
        modelSum[t.turn] += (t.model_time_s || 0);
        toolSum[t.turn] += (t.tool_time_s || 0);
        count[t.turn] += 1;
      }});

      const rows = [];
      for (let i = 1; i <= maxTurn; i++) {{
        if (count[i] > 0) {{
          rows.push({{
            turn: i,
            avgModel: modelSum[i] / count[i],
            avgTool: toolSum[i] / count[i],
            n: count[i],
          }});
        }}
      }}

      const maxVal = Math.max(...rows.map(r => r.avgModel + r.avgTool), 0.1);
      const table = document.createElement("div");
      table.style.cssText = "display:grid; grid-template-columns: 60px 1fr 100px; gap:8px; align-items:center; font-size:12px;";

      rows.slice(0, 20).forEach(r => {{
        const rowDiv = document.createElement("div");
        rowDiv.textContent = `Turn ${{r.turn}}:`;
        rowDiv.style.color = "var(--text-muted)";

        const barContainer = document.createElement("div");
        barContainer.style.cssText = "display:flex; height:14px; background:#090e1a; border-radius:3px; overflow:hidden;";

        const mPct = (r.avgModel / maxVal * 100).toFixed(1);
        const tPct = (r.avgTool / maxVal * 100).toFixed(1);

        barContainer.innerHTML = `
          <div style="width:${{mPct}}%; background:var(--b0);" title="Model: ${{r.avgModel.toFixed(2)}}s"></div>
          <div style="width:${{tPct}}%; background:var(--b0); opacity:0.4;" title="Tool: ${{r.avgTool.toFixed(2)}}s"></div>
        `;

        const valTxt = document.createElement("div");
        valTxt.style.color = "var(--text-muted)";
        valTxt.textContent = `${{r.avgModel.toFixed(2)}}s + ${{r.avgTool.toFixed(2)}}s`;

        table.appendChild(rowDiv);
        table.appendChild(barContainer);
        table.appendChild(valTxt);
      }});

      host.appendChild(table);
    }}

    // Initialization
    function init() {{
      const bSel = document.getElementById("batch-filter");
      const numBatches = RAW_DATA.meta.num_batches || 1;
      for (let b = 0; b < numBatches; b++) {{
        const opt = document.createElement("option");
        opt.value = String(b);
        opt.textContent = `Batch ${{b}}`;
        bSel.appendChild(opt);
      }}

      renderStats(RAW_DATA.trajectories, RAW_DATA.turns, RAW_DATA.meta);
      renderChart();
      renderTurnProfile();

      document.getElementById("batch-filter").addEventListener("change", renderChart);
      document.getElementById("search-input").addEventListener("input", renderChart);
      document.getElementById("sort-select").addEventListener("change", renderChart);
      document.getElementById("toggle-tool-time").addEventListener("change", renderChart);
      document.getElementById("zoom-scale").addEventListener("input", (e) => {{
        document.getElementById("zoom-val").textContent = parseFloat(e.target.value).toFixed(1) + "x";
        renderChart();
      }});
      window.addEventListener("resize", renderChart);
    }}

    window.addEventListener("DOMContentLoaded", init);
  </script>
</body>
</html>
"""
    return html_template


def main():
    parser = argparse.ArgumentParser(
        description="Visualize multi-turn agentic RL trajectories and turn latencies."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input trajectory metrics JSONL file (e.g. from benchmark_agentic.py --save-trajectory-file).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="trajectory_waterfall.html",
        help="Output HTML file path (default: trajectory_waterfall.html).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Agentic RL Trajectory Turn Waterfall",
        help="Title displayed in the visualization report.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic demo data to preview the visualization.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a local HTTP server to view the generated visualization.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for local HTTP server (default: 8080).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML report in the default web browser.",
    )

    args = parser.parse_args()

    if args.demo or not args.input:
        if not args.input and not args.demo:
            print("No input file provided. Generating interactive demo data (--demo)...")
        data = generate_demo_data()
    else:
        print(f"Parsing metrics from: {args.input}")
        data = parse_metrics_file(args.input)

    html_content = generate_html(data, title=args.title)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated trajectory waterfall visualization: {os.path.abspath(args.output)}")
    print(f"Total trajectories: {len(data['trajectories'])}, Total turns: {len(data['turns'])}")

    if args.open:
        try:
            webbrowser.open(f"file://{os.path.abspath(args.output)}")
        except Exception:
            pass

    if args.serve:
        out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
        out_file = os.path.basename(args.output)
        os.chdir(out_dir)
        handler = http.server.SimpleHTTPRequestHandler
        print(f"Serving visualization at http://localhost:{args.port}/{out_file}")
        with socketserver.TCPServer(("", args.port), handler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")


if __name__ == "__main__":
    main()
