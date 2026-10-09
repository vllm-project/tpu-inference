#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Interactive trajectory and turn waterfall visualizer for agentic RL benchmarks.

This tool renders an interactive waterfall timeline displaying multi-turn trajectories,
turn-by-turn latency segmentation (model generation vs. environment/tool idle time),
batch launch boundaries, per-turn metrics (prompt/gen tokens, TTFT, TPOT, tok/s),
and a multi-run comparison dropdown to switch seamlessly between benchmark runs
(e.g., comparing different trace files or engine flags such as DP_SCHED_BATCH_PREFILL).

Input formats supported:
1. One or more trajectory JSONL files from benchmark_agentic.py (--save-trajectory-file).
2. Responses JSONL from benchmark_agentic.py (--save-responses-file).
3. Distributed RL inference_metrics.jsonl files.
4. Raw log files with [DEBUG_INFERENCE][Turn] and [DEBUG_INFERENCE][Trajectory] lines.
5. Built-in --demo mode generating realistic sample data across multiple runs.
"""

import argparse
import http.server
import json
import os
import re
import socketserver
import sys
import webbrowser
from typing import Any, Dict, List, Optional, Tuple, Union


def generate_single_demo_dataset(
    seed: int = 42,
    num_batches: int = 3,
    concurrency: int = 4,
    group_size: int = 4,
    time_between_batches: float = 20.0,
    has_tool_time: bool = True,
    model_factor: float = 1.0,
    run_name: str = "Demo Run",
    dp_sched_prefill: bool = False,
) -> Dict[str, Any]:
    """Generates a realistic synthetic multi-batch agentic benchmark trajectory dataset."""
    import random
    rng = random.Random(seed)

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
                    base_ttft = rng.uniform(0.4, 0.8) if is_miss else rng.uniform(0.05, 0.15)
                    if dp_sched_prefill and turn > 1:
                        # DP batch prefill batches prefills together, reducing tail TTFT under load
                        base_ttft *= 0.82
                    ttft_s = base_ttft * model_factor

                    gen_toks = rng.randint(150, 450)
                    tpot_ms = rng.uniform(4.5, 7.5) * model_factor
                    decode_s = (gen_toks - 1) * (tpot_ms / 1000.0)
                    model_time_s = round(ttft_s + decode_s, 3)

                    if has_tool_time and turn < num_turns:
                        tool_time_s = round(rng.uniform(0.8, 3.2), 3)
                    else:
                        tool_time_s = 0.0

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

    total_duration = max((t["end_time_s"] for t in trajectories), default=1.0)
    meta = {
        "run_name": run_name,
        "model": "Qwen/Qwen3.5-397B-A17B-FP8",
        "num_batches": num_batches,
        "concurrency": concurrency,
        "group_size": group_size,
        "time_between_batches_sec": time_between_batches,
        "batch_launch_times": batch_launch_times,
        "total_trajectories": len(trajectories),
        "total_turns": len(turns_all),
        "total_duration_sec": total_duration,
        "dp_sched_batch_prefill": dp_sched_prefill,
        "trace_file": "gs://wenxindong-vm/rl/mlperf2026/agentic_benchmark/gbs1024_trace_file_tool_time.jsonl" if has_tool_time else "gbs1024_trace_file.jsonl",
    }
    return {
        "meta": meta,
        "trajectories": trajectories,
        "turns": turns_all,
    }


def generate_demo_runs() -> Dict[str, Dict[str, Any]]:
    """Generates synthetic runs comparing different benchmark configurations."""
    return {
        "Demo: Tool Time (DP_SCHED=false)": generate_single_demo_dataset(
            seed=42, has_tool_time=True, dp_sched_prefill=False, model_factor=1.0,
            run_name="Demo: Tool Time (DP_SCHED=false)"
        ),
        "Demo: Tool Time (DP_SCHED=true)": generate_single_demo_dataset(
            seed=43, has_tool_time=True, dp_sched_prefill=True, model_factor=0.92,
            run_name="Demo: Tool Time (DP_SCHED=true)"
        ),
        "Demo: Baseline (No Tool Time)": generate_single_demo_dataset(
            seed=44, has_tool_time=False, dp_sched_prefill=False, model_factor=1.0,
            run_name="Demo: Baseline (No Tool Time)"
        ),
    }


def parse_metrics_file(path: str, run_name: Optional[str] = None) -> Dict[str, Any]:
    """Parses a trajectory metrics JSONL or text log file into structured records."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input metrics file not found: {path}")

    trajectories: Dict[str, Dict[str, Any]] = {}
    turns: List[Dict[str, Any]] = []
    base_name = run_name or os.path.splitext(os.path.basename(path))[0]
    meta: Dict[str, Any] = {
        "source_file": os.path.abspath(path),
        "run_name": base_name,
        "batch_launch_times": [],
    }

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

    # Format 2: Fallback parser for [DEBUG_INFERENCE] text logs
    if not turns:
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
            first = t_list[0]
            t_start = min((x["start_time_s"] for x in t_list), default=0.0)
            t_end = max((x["end_time_s"] for x in t_list), default=0.0)
            tot_model = sum((x["model_time_s"] for x in t_list))
            tot_tool = sum((x.get("tool_time_s", 0.0) for x in t_list))
            tot_out = sum((x.get("output_tokens", 0) for x in t_list))
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


def generate_html(
    runs_data: Union[Dict[str, Any], Dict[str, Dict[str, Any]]],
    title: str = "Agentic RL Trajectory Turn Waterfall",
) -> str:
    """Generates a standalone, dependency-free interactive HTML waterfall visualization."""
    if "trajectories" in runs_data and "turns" in runs_data:
        single_name = runs_data.get("meta", {}).get("run_name") or "Default Run"
        all_runs: Dict[str, Dict[str, Any]] = {single_name: runs_data}
    else:
        all_runs = runs_data  # type: ignore

    runs_json = json.dumps(all_runs)

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>__TITLE__</title>
  <style>
    :root {
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
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.4;
      padding: 16px 20px 40px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--panel-border);
    }
    h1 {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #fff;
    }
    .subtitle {
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 2px;
    }
    .stats-bar {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .stat-card {
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 6px;
      padding: 10px 12px;
    }
    .stat-card .label {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
    }
    .stat-card .val {
      font-size: 18px;
      font-weight: 700;
      color: #fff;
    }
    .stat-card .unit {
      font-size: 12px;
      font-weight: 400;
      color: var(--text-muted);
      margin-left: 2px;
    }
    .controls {
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
    }
    .control-group {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    label {
      font-weight: 600;
      color: var(--text-muted);
      font-size: 12px;
    }
    select, input[type="text"] {
      background: #0b1120;
      color: #fff;
      border: 1px solid var(--panel-border);
      padding: 5px 8px;
      border-radius: 4px;
      font-size: 12px;
      outline: none;
    }
    select:focus, input:focus {
      border-color: var(--accent);
    }
    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    .toggle input { cursor: pointer; }
    .chart-container {
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 6px;
      padding: 14px 14px 6px;
      position: relative;
    }
    .chart-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      font-size: 12px;
    }
    .chart-scroll {
      overflow-x: auto;
      max-height: 580px;
      overflow-y: auto;
      border: 1px solid var(--panel-border);
      border-radius: 4px;
      background: #0b1120;
      position: relative;
    }
    #tooltip {
      position: fixed;
      display: none;
      background: #1e293b;
      border: 1px solid #475569;
      color: #f8fafc;
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 12px;
      line-height: 1.4;
      pointer-events: none;
      z-index: 1000;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
      white-space: pre-line;
      max-width: 360px;
    }
    .legend {
      display: flex;
      gap: 16px;
      align-items: center;
      margin-top: 10px;
      font-size: 12px;
      color: var(--text-muted);
      flex-wrap: wrap;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .hatched-legend {
      width: 18px;
      height: 12px;
      background: repeating-linear-gradient(
        45deg,
        #475569,
        #475569 3px,
        #1e293b 3px,
        #1e293b 6px
      );
      border: 1px solid #64748b;
      border-radius: 2px;
    }
    .tag {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .profile-card {
      margin-top: 16px;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 6px;
      padding: 12px 16px;
    }
    .profile-card h3 {
      font-size: 14px;
      margin-bottom: 8px;
      color: #fff;
    }
    .btn-action {
      background: #1e293b;
      color: #38bdf8;
      border: 1px solid #0284c7;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s, border-color 0.15s;
    }
    .btn-action:hover {
      background: #0284c7;
      color: #ffffff;
    }
    .btn-neutral {
      background: #334155;
      color: #f8fafc;
      border: 1px solid #475569;
      padding: 3px 7px;
      border-radius: 4px;
      font-size: 11px;
      cursor: pointer;
      transition: background 0.15s;
    }
    .btn-neutral:hover {
      background: #475569;
    }
    .measure-hint {
      font-size: 11px;
      color: #38bdf8;
      background: rgba(56, 189, 248, 0.08);
      border: 1px solid rgba(56, 189, 248, 0.28);
      padding: 3px 8px;
      border-radius: 4px;
      display: flex;
      align-items: center;
      gap: 5px;
      user-select: none;
    }
    .measure-hint kbd {
      background: #0f172a;
      border: 1px solid #38bdf8;
      border-radius: 3px;
      padding: 1px 5px;
      font-family: monospace;
      font-size: 10px;
      color: #38bdf8;
      font-weight: 700;
    }
    body.shift-measuring,
    body.shift-measuring #scroll-box,
    body.shift-measuring #throughput-scroll-box,
    body.shift-measuring svg {
      cursor: crosshair !important;
    }
  </style>
</head>
<body>
  <div id="tooltip"></div>

  <header>
    <div>
      <h1>__TITLE__</h1>
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
    <div class="control-group" id="run-select-group">
      <label for="run-select" style="color:var(--accent); font-weight:700;">Run / Benchmark:</label>
      <select id="run-select" style="font-weight:600; min-width: 220px;"></select>
    </div>

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

    <div class="control-group" style="margin-left: auto; display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
      <div class="measure-hint" id="measure-hint" title="Hold Shift and drag horizontally on either chart to measure elapsed time duration & turns">
        <span>Hold <kbd>Shift</kbd> + Drag to measure &Delta;t</span>
      </div>
      <label for="zoom-scale">Zoom:</label>
      <input type="range" id="zoom-scale" min="0.02" max="3.0" step="0.02" value="1.0" style="width: 100px;">
      <span id="zoom-val" style="font-size: 11px; color: var(--text-muted); min-width: 34px;">1.0x</span>
      <button id="zoom-fit-btn" type="button" class="btn-action" title="Fit entire benchmark timeline horizontally to screen">Fit All</button>
      <button id="zoom-reset-btn" type="button" class="btn-neutral" title="Reset zoom to 1.0x">1.0x</button>
      <input type="file" id="run-file-input" accept=".jsonl,.json" style="display:none;">
      <button id="load-file-btn" type="button" class="btn-neutral" title="Load local trajectory JSONL into visualizer">+ Add Run File</button>
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
        <div class="hatched-legend"></div>
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

  <div class="profile-card" id="throughput-card">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">
      <div>
        <h3>Trajectory Generation Throughput Over Time</h3>
        <div style="font-size:12px; color:var(--text-muted); margin-top:2px;">Generation throughput (tok/s) across turns for each trajectory over benchmark timeline. Color-coded by batch with turn markers.</div>
      </div>
      <div style="display:flex; gap:14px; align-items:center; font-size:12px;">
        <label class="toggle">
          <input type="checkbox" id="toggle-agg-tps" checked>
          <span>Show Aggregate Throughput</span>
        </label>
        <label class="toggle">
          <input type="checkbox" id="toggle-turn-labels" checked>
          <span>Turn # Labels</span>
        </label>
      </div>
    </div>
    <div class="chart-scroll" id="throughput-scroll-box" style="max-height:360px;">
      <div id="throughput-svg-host"></div>
    </div>
    <div class="legend" style="margin-top:8px;">
      <span style="font-weight:600; color:#fff;">Throughput Legend:</span>
      <div class="legend-item">
        <span style="display:inline-block;width:18px;height:3px;background:var(--b0);border-radius:2px;"></span>
        <span>Trajectory Throughput Line</span>
      </div>
      <div class="legend-item">
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--b0);border:1.5px solid #fff;"></span>
        <span>Turn Marker (Turn #)</span>
      </div>
      <div class="legend-item">
        <span style="display:inline-block;width:18px;height:2px;background:#38bdf8;border-top:2px dashed #38bdf8;"></span>
        <span>Aggregate Throughput (tok/s)</span>
      </div>
      <div class="legend-item">
        <span style="display:inline-block;width:16px;height:12px;background:#000;border:1px solid #fff;border-radius:2px;color:#fff;text-align:center;line-height:10px;font-size:8px;font-weight:700;">b0</span>
        <span>Batch Launch Boundary</span>
      </div>
    </div>
  </div>

  <script>
    const ALL_RUNS = __RUNS_JSON__;
    let currentRunKey = Object.keys(ALL_RUNS)[0] || "Default Run";
    let RAW_DATA = ALL_RUNS[currentRunKey] || { meta: {}, trajectories: [], turns: [] };

    const COLORS = [
      "#3b82f6", "#10b981", "#f59e0b", "#ec4899",
      "#8b5cf6", "#06b6d4", "#f97316", "#6366f1"
    ];

    const tooltip = document.getElementById("tooltip");
    let measurement = null; // { startSec: number, endSec: number, active: boolean }
    let isShiftMeasuring = false;

    function showTip(e, text) {
      if (e.shiftKey || (measurement && measurement.active) || isShiftMeasuring) {
        hideTip();
        return;
      }
      tooltip.textContent = text;
      tooltip.style.display = "block";
      const x = Math.min(e.clientX + 14, window.innerWidth - 380);
      const y = Math.min(e.clientY + 14, window.innerHeight - 180);
      tooltip.style.left = x + "px";
      tooltip.style.top = y + "px";
    }
    function hideTip() {
      tooltip.style.display = "none";
    }

    // Greedy swimlane row packing
    function packSwimlanes(items) {
      items.sort((a, b) => a.start_time_s - b.start_time_s || b.duration_s - a.duration_s);
      const rowEnds = [];
      const rowAssigned = [];
      for (let i = 0; i < items.length; i++) {
        const it = items[i];
        let placed = false;
        for (let r = 0; r < rowEnds.length; r++) {
          if (rowEnds[r] <= it.start_time_s) {
            rowAssigned[i] = r;
            rowEnds[r] = it.end_time_s;
            placed = true;
            break;
          }
        }
        if (!placed) {
          rowAssigned[i] = rowEnds.length;
          rowEnds.push(it.end_time_s);
        }
      }
      return { nrows: Math.max(rowEnds.length, 1), rows: rowAssigned };
    }

    function renderStats(trajs, turns, meta) {
      const sb = document.getElementById("stats-bar");
      sb.innerHTML = "";

      const totalTrajs = trajs.length;
      const totalTurns = turns.length;
      const totalGenToks = turns.reduce((acc, t) => acc + (t.output_tokens || 0), 0);
      const totalModelSec = turns.reduce((acc, t) => acc + (t.model_time_s || 0), 0);
      const totalToolSec = turns.reduce((acc, t) => acc + (t.tool_time_s || 0), 0);
      const wallSec = meta.total_duration_sec || Math.max(...trajs.map(t => t.end_time_s), 1.0);

      const genThroughput = totalDurationSec => totalDurationSec > 0 ? (totalGenToks / totalDurationSec).toFixed(1) : "0.0";
      const avgTpot = turns.length > 0 ? (turns.filter(t => t.tpot_ms).reduce((acc, t) => acc + t.tpot_ms, 0) / (turns.filter(t => t.tpot_ms).length || 1)).toFixed(2) : "--";
      const avgTtft = turns.length > 0 ? (turns.filter(t => t.ttft_ms).reduce((acc, t) => acc + t.ttft_ms, 0) / (turns.filter(t => t.ttft_ms).length || 1)).toFixed(1) : "--";

      const cards = [
        { label: "Trajectories", val: totalTrajs, unit: "" },
        { label: "Total Turns", val: totalTurns, unit: "" },
        { label: "Wall Time", val: wallSec.toFixed(1), unit: "s" },
        { label: "Throughput", val: genThroughput(wallSec), unit: "tok/s" },
        { label: "Avg TTFT", val: avgTtft, unit: "ms" },
        { label: "Avg TPOT", val: avgTpot, unit: "ms" },
        { label: "Model Gen Time", val: totalModelSec.toFixed(1), unit: "s" },
        { label: "Tool Idle Time", val: totalToolSec.toFixed(1), unit: "s" },
      ];

      cards.forEach(c => {
        const el = document.createElement("div");
        el.className = "stat-card";
        el.innerHTML = `<div class="label">${c.label}</div><div class="val">${c.val}<span class="unit">${c.unit}</span></div>`;
        sb.appendChild(el);
      });
    }

    function populateRunSelector() {
      const rSel = document.getElementById("run-select");
      rSel.innerHTML = "";
      const runKeys = Object.keys(ALL_RUNS);
      runKeys.forEach(k => {
        const opt = document.createElement("option");
        opt.value = k;
        opt.textContent = k;
        if (k === currentRunKey) opt.selected = true;
        rSel.appendChild(opt);
      });
      const group = document.getElementById("run-select-group");
      if (group) {
        group.style.display = runKeys.length > 0 ? "flex" : "none";
      }
    }

    function populateBatchFilter() {
      const bSel = document.getElementById("batch-filter");
      const prevVal = bSel.value;
      bSel.innerHTML = '<option value="all">All Batches</option>';
      const numBatches = (RAW_DATA.meta && RAW_DATA.meta.num_batches) || 1;
      for (let b = 0; b < numBatches; b++) {
        const opt = document.createElement("option");
        opt.value = String(b);
        opt.textContent = `Batch ${b}`;
        bSel.appendChild(opt);
      }
      if (prevVal === "all" || parseInt(prevVal) < numBatches) {
        bSel.value = prevVal;
      } else {
        bSel.value = "all";
      }
    }

    function updateSubtitle() {
      const sub = document.getElementById("subtitle-info");
      if (!sub) return;
      const m = (RAW_DATA && RAW_DATA.meta) ? RAW_DATA.meta : {};
      const model = m.model || m.model_name || "";
      const trace = m.trace_file ? m.trace_file.split("/").pop() : "";
      const dp = m.dp_sched_batch_prefill !== undefined ? `DP_SCHED=${m.dp_sched_batch_prefill}` : "";
      const parts = ["Continuous multi-batch trajectory turn waterfall & latency profiling"];
      if (model) parts.push(`Model: ${model}`);
      if (trace) parts.push(`Trace: ${trace}`);
      if (dp) parts.push(dp);
      sub.textContent = parts.join(" | ");
    }

    function switchRun(runKey) {
      if (!ALL_RUNS[runKey]) return;
      currentRunKey = runKey;
      RAW_DATA = ALL_RUNS[runKey];
      populateBatchFilter();
      renderStats(RAW_DATA.trajectories, RAW_DATA.turns, RAW_DATA.meta);
      renderChart();
      renderThroughputChart();
      updateSubtitle();
    }

    function parseClientJSONL(text, filename) {
      const lines = text.split("\\n").map(l => l.trim()).filter(l => l);
      const meta = { source_file: filename, run_name: filename.replace(/\\.[^/.]+$/, ""), batch_launch_times: [] };
      const trajectories = {};
      const turns = [];

      for (const line of lines) {
        if (!line.startsWith("{")) continue;
        let rec;
        try { rec = JSON.parse(line); } catch (e) { continue; }
        const rType = rec.type;
        if (rType === "benchmark_meta") {
          Object.assign(meta, rec);
        } else if (rType === "trajectory") {
          if (rec.traj_id) trajectories[rec.traj_id] = rec;
        } else if (rType === "turn" || (rec.turn && rec.group_idx !== undefined)) {
          const bIdx = rec.batch_idx || 0;
          const gIdx = rec.group_idx || 0;
          const sIdx = rec.stream_idx || 0;
          const trajId = rec.traj_id || `b${bIdx}_g${gIdx}_s${sIdx}`;
          const tNum = rec.turn || 1;
          const numTurns = rec.num_turns || 1;
          let mTime = rec.model_time_s;
          if (mTime === undefined && rec.total_time_ms) mTime = rec.total_time_ms / 1000.0;
          turns.push({
            type: "turn",
            batch_idx: bIdx,
            group_idx: gIdx,
            stream_idx: sIdx,
            traj_id: trajId,
            turn: tNum,
            num_turns: numTurns,
            start_time_s: rec.start_time_s || 0.0,
            end_time_s: rec.end_time_s || (mTime || 0.0),
            model_time_s: mTime || 0.0,
            tool_time_s: rec.tool_time_s || 0.0,
            prompt_tokens: rec.input_history_tokens || rec.prompt_tokens || 0,
            output_tokens: rec.output_tokens || rec.completion_tokens || 0,
            ttft_ms: rec.ttft_ms,
            tpot_ms: rec.tpot_ms,
            total_time_ms: rec.total_time_ms,
            success: rec.success !== undefined ? rec.success : true,
            error: rec.error
          });
        }
      }

      // Synthesize trajectories if needed
      const turnsByTraj = {};
      turns.forEach(t => {
        turnsByTraj[t.traj_id] = turnsByTraj[t.traj_id] || [];
        turnsByTraj[t.traj_id].push(t);
      });
      for (const [tid, tList] of Object.entries(turnsByTraj)) {
        if (!trajectories[tid]) {
          const first = tList[0];
          const tStart = Math.min(...tList.map(t => t.start_time_s));
          const tEnd = Math.max(...tList.map(t => t.end_time_s));
          const totModel = tList.reduce((acc, t) => acc + (t.model_time_s || 0), 0);
          const totTool = tList.reduce((acc, t) => acc + (t.tool_time_s || 0), 0);
          const totOut = tList.reduce((acc, t) => acc + (t.output_tokens || 0), 0);
          trajectories[tid] = {
            type: "trajectory",
            traj_id: tid,
            batch_idx: first.batch_idx || 0,
            group_idx: first.group_idx || 0,
            stream_idx: first.stream_idx || 0,
            num_turns: tList.length,
            start_time_s: tStart,
            end_time_s: tEnd,
            duration_s: Math.max(0, tEnd - tStart),
            model_time_s: totModel,
            tool_time_s: totTool,
            prompt_tokens: first.prompt_tokens || 0,
            output_tokens: totOut,
            status: tList.every(t => t.success) ? "COMPLETED" : "FAILED"
          };
        }
      }

      const batches = [...new Set(Object.values(trajectories).map(t => t.batch_idx))].sort((a,b)=>a-b);
      meta.num_batches = meta.num_batches || batches.length || 1;
      meta.total_trajectories = Object.keys(trajectories).length;
      meta.total_turns = turns.length;
      meta.total_duration_sec = Math.max(...Object.values(trajectories).map(t => t.end_time_s), 1.0);
      return { meta, trajectories: Object.values(trajectories), turns };
    }

    function renderChart() {
      const bFilter = document.getElementById("batch-filter").value;
      const searchVal = document.getElementById("search-input").value.trim().toLowerCase();
      const sortVal = document.getElementById("sort-select").value;
      const showTool = document.getElementById("toggle-tool-time").checked;
      const zoom = parseFloat(document.getElementById("zoom-scale").value);

      let trajs = [...RAW_DATA.trajectories];

      if (bFilter !== "all") {
        const b = parseInt(bFilter);
        trajs = trajs.filter(t => t.batch_idx === b);
      }

      if (searchVal) {
        trajs = trajs.filter(t => {
          return t.traj_id.toLowerCase().includes(searchVal) ||
                 `g${t.group_idx}`.includes(searchVal) ||
                 `s${t.stream_idx}`.includes(searchVal) ||
                 `b${t.batch_idx}`.includes(searchVal);
        });
      }

      if (sortVal === "start") {
        trajs.sort((a, b) => a.start_time_s - b.start_time_s || a.group_idx - b.group_idx);
      } else if (sortVal === "duration") {
        trajs.sort((a, b) => b.duration_s - a.duration_s);
      } else if (sortVal === "turns") {
        trajs.sort((a, b) => b.num_turns - a.num_turns);
      } else if (sortVal === "id") {
        trajs.sort((a, b) => a.traj_id.localeCompare(b.traj_id));
      }

      document.getElementById("chart-count-info").textContent =
        `Showing ${trajs.length} trajectories (${RAW_DATA.meta.total_turns} total turns across ${RAW_DATA.meta.num_batches} batches)`;

      const host = document.getElementById("svg-host");
      host.innerHTML = "";

      if (trajs.length === 0) {
        host.innerHTML = '<div style="padding: 40px; text-align: center; color: var(--text-muted);">No trajectories match filter.</div>';
        return;
      }

      const tStart = 0.0;
      const tEnd = Math.max(...trajs.map(t => t.end_time_s), RAW_DATA.meta.total_duration_sec || 1.0);
      const span = Math.max(tEnd - tStart, 1.0);

      const pxPerSec = Math.max(30 * zoom, 0.1);
      const svgWidth = Math.max(Math.round(span * pxPerSec) + labelW + 40, 600);
      const labelW = 120;
      const rowHeight = 22;
      const rowGap = 6;
      const headerH = 28;

      const numRows = trajs.length;
      const svgHeight = headerH + numRows * (rowHeight + rowGap) + 30;

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("width", svgWidth);
      svg.setAttribute("height", svgHeight);
      svg.style.display = "block";

      const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
      const pattern = document.createElementNS("http://www.w3.org/2000/svg", "pattern");
      pattern.setAttribute("id", "tool-hatch");
      pattern.setAttribute("width", "6");
      pattern.setAttribute("height", "6");
      pattern.setAttribute("patternTransform", "rotate(45 0 0)");
      pattern.setAttribute("patternUnits", "userSpaceOnUse");
      pattern.innerHTML = '<line x1="0" y1="0" x2="0" y2="6" stroke="#64748b" stroke-width="2.5" opacity="0.7"/>';
      defs.appendChild(pattern);
      svg.appendChild(defs);

      // Time axis grid
      const axisG = document.createElementNS("http://www.w3.org/2000/svg", "g");
      let stepSec = 30;
      if (pxPerSec >= 25) stepSec = 5;
      else if (pxPerSec >= 12) stepSec = 10;
      else if (pxPerSec >= 5) stepSec = 20;
      else if (pxPerSec >= 2) stepSec = 30;
      else if (pxPerSec >= 0.8) stepSec = 60;
      else stepSec = 120;

      for (let sec = 0; sec <= span; sec += stepSec) {
        const x = labelW + Math.round(sec * pxPerSec);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", x);
        line.setAttribute("y1", headerH - 6);
        line.setAttribute("x2", x);
        line.setAttribute("y2", svgHeight - 10);
        line.setAttribute("stroke", "#334155");
        line.setAttribute("stroke-dasharray", "2,3");
        axisG.appendChild(line);

        const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
        txt.setAttribute("x", x);
        txt.setAttribute("y", headerH - 10);
        txt.setAttribute("fill", "#94a3b8");
        txt.setAttribute("font-size", "11");
        txt.setAttribute("text-anchor", "middle");
        txt.textContent = `${sec}s`;
        axisG.appendChild(txt);
      }
      svg.appendChild(axisG);

      // Map turns by trajectory
      const trajTurns = {};
      RAW_DATA.turns.forEach(t => {
        trajTurns[t.traj_id] = trajTurns[t.traj_id] || [];
        trajTurns[t.traj_id].push(t);
      });

      // Render Trajectory Rows
      trajs.forEach((traj, idx) => {
        const y = headerH + idx * (rowHeight + rowGap);
        const rowG = document.createElementNS("http://www.w3.org/2000/svg", "g");

        // Row background on hover
        const rowBg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rowBg.setAttribute("x", 0);
        rowBg.setAttribute("y", y - 2);
        rowBg.setAttribute("width", svgWidth);
        rowBg.setAttribute("height", rowHeight + 4);
        rowBg.setAttribute("fill", "transparent");
        rowBg.style.cursor = "pointer";
        rowG.appendChild(rowBg);

        // Label
        const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
        lbl.setAttribute("x", 6);
        lbl.setAttribute("y", y + rowHeight / 2 + 4);
        lbl.setAttribute("fill", "#e2e8f0");
        lbl.setAttribute("font-size", "11");
        lbl.setAttribute("font-family", "monospace");
        lbl.textContent = traj.traj_id;
        rowG.appendChild(lbl);

        // Turn blocks
        const tList = trajTurns[traj.traj_id] || [];
        tList.sort((a, b) => a.turn - b.turn);

        tList.forEach(turn => {
          const mX = labelW + Math.round((turn.start_time_s - tStart) * pxPerSec);
          const mW = Math.max(Math.round(turn.model_time_s * pxPerSec), 1.5);
          const color = COLORS[turn.batch_idx % COLORS.length];

          // Model computation block
          const mRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          mRect.setAttribute("x", mX);
          mRect.setAttribute("y", y);
          mRect.setAttribute("width", mW);
          mRect.setAttribute("height", rowHeight);
          mRect.setAttribute("rx", 3);
          mRect.setAttribute("fill", color);
          mRect.style.cursor = "pointer";

          const tipText = [
            `Trajectory: ${turn.traj_id} (Turn ${turn.turn}/${turn.num_turns})`,
            `Batch: ${turn.batch_idx} | Group: ${turn.group_idx} | Stream: ${turn.stream_idx}`,
            `Model Time: ${turn.model_time_s.toFixed(3)} s`,
            `Tool/Env Idle: ${turn.tool_time_s.toFixed(3)} s`,
            `Prompt Tokens: ${turn.prompt_tokens}`,
            `Generated Tokens: ${turn.output_tokens}`,
            turn.ttft_ms ? `TTFT: ${turn.ttft_ms} ms` : null,
            turn.tpot_ms ? `TPOT: ${turn.tpot_ms} ms` : null,
            `Start: ${turn.start_time_s.toFixed(3)} s -> End: ${turn.end_time_s.toFixed(3)} s`,
          ].filter(Boolean).join("\n");

          mRect.addEventListener("mousemove", (e) => showTip(e, tipText));
          mRect.addEventListener("mouseleave", hideTip);
          rowG.appendChild(mRect);

          // Turn number inside model block if width permits
          if (mW >= 12) {
            const tTxt = document.createElementNS("http://www.w3.org/2000/svg", "text");
            tTxt.setAttribute("x", mX + mW / 2);
            tTxt.setAttribute("y", y + rowHeight / 2 + 4);
            tTxt.setAttribute("fill", "#ffffff");
            tTxt.setAttribute("font-size", mW >= 18 ? "10" : "8");
            tTxt.setAttribute("font-weight", "700");
            tTxt.setAttribute("text-anchor", "middle");
            tTxt.style.pointerEvents = "none";
            tTxt.textContent = String(turn.turn);
            rowG.appendChild(tTxt);
          }

          // Tool / environment idle block
          if (showTool && turn.tool_time_s > 0) {
            const toolX = mX + mW;
            const toolW = Math.max(Math.round(turn.tool_time_s * pxPerSec), 1.5);
            const toolRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            toolRect.setAttribute("x", toolX);
            toolRect.setAttribute("y", y + 2);
            toolRect.setAttribute("width", toolW);
            toolRect.setAttribute("height", rowHeight - 4);
            toolRect.setAttribute("rx", 2);
            toolRect.setAttribute("fill", "url(#tool-hatch)");
            toolRect.setAttribute("stroke", "#475569");
            toolRect.setAttribute("stroke-width", "0.5");
            toolRect.style.cursor = "pointer";

            const toolTip = `Tool Call Idle Gap (Turn ${turn.turn} -> ${turn.turn + 1})\n` +
                            `Duration: ${turn.tool_time_s.toFixed(3)} s\n` +
                            `Time: ${(turn.end_time_s).toFixed(3)} s -> ${(turn.end_time_s + turn.tool_time_s).toFixed(3)} s`;
            toolRect.addEventListener("mousemove", (e) => showTip(e, toolTip));
            toolRect.addEventListener("mouseleave", hideTip);
            rowG.appendChild(toolRect);
          }
        });

        rowBg.addEventListener("mouseenter", () => { rowBg.setAttribute("fill", "rgba(255,255,255,0.04)"); });
        rowBg.addEventListener("mouseleave", () => { rowBg.setAttribute("fill", "transparent"); });
        svg.appendChild(rowG);
      });

      // Batch launch vertical boundary markers
      const launchTimes = RAW_DATA.meta.batch_launch_times || [0.0];
      launchTimes.forEach((bTime, bIdx) => {
        const bX = labelW + Math.round(bTime * pxPerSec);
        const bG = document.createElementNS("http://www.w3.org/2000/svg", "g");

        const vLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        vLine.setAttribute("x1", bX);
        vLine.setAttribute("y1", headerH - 8);
        vLine.setAttribute("x2", bX);
        vLine.setAttribute("y2", svgHeight - 6);
        vLine.setAttribute("stroke", "#000000");
        vLine.setAttribute("stroke-width", "2.5");
        bG.appendChild(vLine);

        const badge = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        badge.setAttribute("x", bX - 12);
        badge.setAttribute("y", 2);
        badge.setAttribute("width", 24);
        badge.setAttribute("height", 16);
        badge.setAttribute("rx", 3);
        badge.setAttribute("fill", "#000000");
        badge.setAttribute("stroke", "#ffffff");
        badge.setAttribute("stroke-width", "1");
        bG.appendChild(badge);

        const badgeTxt = document.createElementNS("http://www.w3.org/2000/svg", "text");
        badgeTxt.setAttribute("x", bX);
        badgeTxt.setAttribute("y", 14);
        badgeTxt.setAttribute("fill", "#ffffff");
        badgeTxt.setAttribute("font-size", "10");
        badgeTxt.setAttribute("font-weight", "700");
        badgeTxt.setAttribute("text-anchor", "middle");
        badgeTxt.textContent = `b${bIdx}`;
        bG.appendChild(badgeTxt);

        bG.addEventListener("mousemove", (e) => showTip(e, `Batch ${bIdx} Launch Boundary\nTime: ${bTime.toFixed(2)}s`));
        bG.addEventListener("mouseleave", hideTip);
        svg.appendChild(bG);
      });

      const measureLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      measureLayer.setAttribute("id", "waterfall-measure-layer");
      svg.appendChild(measureLayer);

      host.appendChild(svg);
    }

    function renderThroughputChart() {
      const host = document.getElementById("throughput-svg-host");
      if (!host) return;
      host.innerHTML = "";

      const bFilter = document.getElementById("batch-filter").value;
      const searchVal = document.getElementById("search-input").value.trim().toLowerCase();
      const zoom = parseFloat(document.getElementById("zoom-scale").value);
      const showAgg = document.getElementById("toggle-agg-tps") ? document.getElementById("toggle-agg-tps").checked : true;
      const showLabels = document.getElementById("toggle-turn-labels") ? document.getElementById("toggle-turn-labels").checked : true;

      let trajs = [...RAW_DATA.trajectories];
      if (bFilter !== "all") {
        const b = parseInt(bFilter);
        trajs = trajs.filter(t => t.batch_idx === b);
      }
      if (searchVal) {
        trajs = trajs.filter(t => {
          return t.traj_id.toLowerCase().includes(searchVal) ||
                 `g${t.group_idx}`.includes(searchVal) ||
                 `s${t.stream_idx}`.includes(searchVal) ||
                 `b${t.batch_idx}`.includes(searchVal);
        });
      }

      if (trajs.length === 0) {
        host.innerHTML = '<div style="padding: 30px; text-align: center; color: var(--text-muted);">No trajectories match filter.</div>';
        return;
      }

      // Map turns by trajectory
      const trajTurns = {};
      RAW_DATA.turns.forEach(t => {
        trajTurns[t.traj_id] = trajTurns[t.traj_id] || [];
        trajTurns[t.traj_id].push(t);
      });

      const tStart = 0.0;
      const tEnd = Math.max(...trajs.map(t => t.end_time_s), RAW_DATA.meta.total_duration_sec || 1.0);
      const span = Math.max(tEnd - tStart, 1.0);

      const pxPerSec = Math.max(30 * zoom, 0.1);
      const svgWidth = Math.max(Math.round(span * pxPerSec) + padL + padR, 600);
      const svgHeight = 280;
      const padL = 70;
      const padR = 40;
      const padT = 24;
      const padB = 36;
      const plotW = svgWidth - padL - padR;
      const plotH = svgHeight - padT - padB;

      // Extract points per trajectory: (midpoint_time, throughput_tps)
      const trajPoints = {};
      let maxTps = 50.0;

      trajs.forEach(tr => {
        const tList = trajTurns[tr.traj_id] || [];
        tList.sort((a, b) => a.turn - b.turn);
        const pts = [];
        tList.forEach(t => {
          const mTime = Math.max(t.model_time_s || 0.0, 0.001);
          const tps = (t.output_tokens || 0) / mTime;
          const tMid = t.start_time_s + mTime / 2.0;
          if (tps > maxTps) maxTps = tps;
          pts.push({
            time: tMid,
            tps: tps,
            turn: t.turn,
            num_turns: t.num_turns,
            traj_id: tr.traj_id,
            batch_idx: tr.batch_idx,
            group_idx: tr.group_idx,
            stream_idx: tr.stream_idx,
            model_time_s: t.model_time_s,
            tool_time_s: t.tool_time_s,
            prompt_tokens: t.prompt_tokens,
            output_tokens: t.output_tokens,
            ttft_ms: t.ttft_ms,
            tpot_ms: t.tpot_ms,
          });
        });
        trajPoints[tr.traj_id] = pts;
      });

      // Nice ceiling for y-axis
      const yMax = Math.ceil((maxTps * 1.15) / 25) * 25;

      const scaleX = t => padL + ((t - tStart) / span) * (span * pxPerSec);
      const scaleY = tps => padT + plotH - (Math.max(0, tps) / yMax) * plotH;

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("width", svgWidth);
      svg.setAttribute("height", svgHeight);
      svg.style.display = "block";

      // Gridlines & Y-Axis ticks
      const yAxisG = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const ySteps = 4;
      for (let i = 0; i <= ySteps; i++) {
        const val = Math.round((yMax / ySteps) * i);
        const y = scaleY(val);

        const gridLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        gridLine.setAttribute("x1", padL);
        gridLine.setAttribute("y1", y);
        gridLine.setAttribute("x2", padL + Math.round(span * pxPerSec));
        gridLine.setAttribute("y2", y);
        gridLine.setAttribute("stroke", "#334155");
        gridLine.setAttribute("stroke-dasharray", i === 0 ? "none" : "2,3");
        gridLine.setAttribute("stroke-width", i === 0 ? "1.5" : "1");
        yAxisG.appendChild(gridLine);

        const yTxt = document.createElementNS("http://www.w3.org/2000/svg", "text");
        yTxt.setAttribute("x", padL - 10);
        yTxt.setAttribute("y", y + 4);
        yTxt.setAttribute("fill", "#94a3b8");
        yTxt.setAttribute("font-size", "11");
        yTxt.setAttribute("text-anchor", "end");
        yTxt.textContent = `${val} tok/s`;
        yAxisG.appendChild(yTxt);
      }
      svg.appendChild(yAxisG);

      // Time X-Axis
      const xAxisG = document.createElementNS("http://www.w3.org/2000/svg", "g");
      let stepSec = 30;
      if (pxPerSec >= 25) stepSec = 5;
      else if (pxPerSec >= 12) stepSec = 10;
      else if (pxPerSec >= 5) stepSec = 20;
      else if (pxPerSec >= 2) stepSec = 30;
      else if (pxPerSec >= 0.8) stepSec = 60;
      else stepSec = 120;

      for (let sec = 0; sec <= span; sec += stepSec) {
        const x = scaleX(sec);
        const xLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        xLine.setAttribute("x1", x);
        xLine.setAttribute("y1", padT);
        xLine.setAttribute("x2", x);
        xLine.setAttribute("y2", padT + plotH);
        xLine.setAttribute("stroke", "#1e293b");
        xLine.setAttribute("stroke-dasharray", "2,3");
        xAxisG.appendChild(xLine);

        const xTxt = document.createElementNS("http://www.w3.org/2000/svg", "text");
        xTxt.setAttribute("x", x);
        xTxt.setAttribute("y", padT + plotH + 18);
        xTxt.setAttribute("fill", "#94a3b8");
        xTxt.setAttribute("font-size", "11");
        xTxt.setAttribute("text-anchor", "middle");
        xTxt.textContent = `${sec}s`;
        xAxisG.appendChild(xTxt);
      }
      svg.appendChild(xAxisG);

      // Batch Launch Boundary markers
      const launchTimes = RAW_DATA.meta.batch_launch_times || [0.0];
      launchTimes.forEach((bTime, bIdx) => {
        const bX = scaleX(bTime);
        const bG = document.createElementNS("http://www.w3.org/2000/svg", "g");

        const vLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        vLine.setAttribute("x1", bX);
        vLine.setAttribute("y1", padT - 6);
        vLine.setAttribute("x2", bX);
        vLine.setAttribute("y2", padT + plotH);
        vLine.setAttribute("stroke", "#000000");
        vLine.setAttribute("stroke-width", "2");
        bG.appendChild(vLine);

        const badge = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        badge.setAttribute("x", bX - 12);
        badge.setAttribute("y", 2);
        badge.setAttribute("width", 24);
        badge.setAttribute("height", 16);
        badge.setAttribute("rx", 3);
        badge.setAttribute("fill", "#000000");
        badge.setAttribute("stroke", "#ffffff");
        badge.setAttribute("stroke-width", "1");
        bG.appendChild(badge);

        const badgeTxt = document.createElementNS("http://www.w3.org/2000/svg", "text");
        badgeTxt.setAttribute("x", bX);
        badgeTxt.setAttribute("y", 14);
        badgeTxt.setAttribute("fill", "#ffffff");
        badgeTxt.setAttribute("font-size", "10");
        badgeTxt.setAttribute("font-weight", "700");
        badgeTxt.setAttribute("text-anchor", "middle");
        badgeTxt.textContent = `b${bIdx}`;
        bG.appendChild(badgeTxt);

        bG.addEventListener("mousemove", (e) => showTip(e, `Batch ${bIdx} Launch Boundary
Time: ${bTime.toFixed(2)}s`));
        bG.addEventListener("mouseleave", hideTip);
        svg.appendChild(bG);
      });

      // Optional Aggregate Throughput Line (Rolling active tok/s)
      if (showAgg && span > 0) {
        const numBuckets = Math.min(Math.round(span * 2), 600);
        const dtBucket = span / numBuckets;
        const aggCurve = [];

        for (let k = 0; k <= numBuckets; k++) {
          const tCur = tStart + k * dtBucket;
          let activeToksPerSec = 0.0;
          RAW_DATA.turns.forEach(t => {
            if (tCur >= t.start_time_s && tCur <= t.end_time_s) {
              const dur = Math.max(t.model_time_s, 0.001);
              activeToksPerSec += (t.output_tokens || 0) / dur;
            }
          });
          aggCurve.push({ t: tCur, val: activeToksPerSec });
        }

        const maxAgg = Math.max(...aggCurve.map(p => p.val), 1.0);
        // Normalize agg to fit comfortably in top area
        const aggPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
        let dStr = "";
        aggCurve.forEach((pt, idx) => {
          const x = scaleX(pt.t);
          const y = scaleY((pt.val / maxAgg) * (yMax * 0.9));
          dStr += (idx === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
        });
        aggPath.setAttribute("d", dStr);
        aggPath.setAttribute("fill", "none");
        aggPath.setAttribute("stroke", "#38bdf8");
        aggPath.setAttribute("stroke-width", "2.5");
        aggPath.setAttribute("stroke-dasharray", "4,3");
        aggPath.setAttribute("opacity", "0.85");
        aggPath.style.cursor = "pointer";

        aggPath.addEventListener("mousemove", (e) => {
          showTip(e, `Aggregate Active Token Generation Rate
Peak: ${maxAgg.toFixed(1)} tok/s`);
        });
        aggPath.addEventListener("mouseleave", hideTip);
        svg.appendChild(aggPath);
      }

      // Draw Trajectory Curves and Turn Markers
      const linesG = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const markersG = document.createElementNS("http://www.w3.org/2000/svg", "g");

      trajs.forEach(tr => {
        const pts = trajPoints[tr.traj_id] || [];
        if (pts.length === 0) return;
        const color = COLORS[tr.batch_idx % COLORS.length];

        // Polyline connecting turns of trajectory
        let pathD = "";
        pts.forEach((pt, idx) => {
          const x = scaleX(pt.time);
          const y = scaleY(pt.tps);
          pathD += (idx === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
        });

        const trajPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
        trajPath.setAttribute("d", pathD);
        trajPath.setAttribute("fill", "none");
        trajPath.setAttribute("stroke", color);
        trajPath.setAttribute("stroke-width", "1.8");
        trajPath.setAttribute("opacity", "0.7");
        trajPath.setAttribute("class", `traj-tps-line traj-${tr.traj_id}`);
        trajPath.style.cursor = "pointer";

        const highlightTraj = () => {
          svg.querySelectorAll(".traj-tps-line").forEach(l => l.setAttribute("opacity", "0.15"));
          trajPath.setAttribute("opacity", "1.0");
          trajPath.setAttribute("stroke-width", "3.2");
        };
        const resetTraj = () => {
          svg.querySelectorAll(".traj-tps-line").forEach(l => l.setAttribute("opacity", "0.7"));
          trajPath.setAttribute("stroke-width", "1.8");
        };

        trajPath.addEventListener("mouseenter", highlightTraj);
        trajPath.addEventListener("mouseleave", resetTraj);
        linesG.appendChild(trajPath);

        // Turn Markers along trajectory line
        pts.forEach(pt => {
          const x = scaleX(pt.time);
          const y = scaleY(pt.tps);

          const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          circle.setAttribute("cx", x);
          circle.setAttribute("cy", y);
          circle.setAttribute("r", "5");
          circle.setAttribute("fill", color);
          circle.setAttribute("stroke", "#ffffff");
          circle.setAttribute("stroke-width", "1.5");
          circle.style.cursor = "pointer";

          const tipMsg = [
            `Trajectory: ${pt.traj_id} (Turn ${pt.turn}/${pt.num_turns})`,
            `Batch: ${pt.batch_idx} | Group: ${pt.group_idx} | Stream: ${pt.stream_idx}`,
            `Generation Throughput: ${pt.tps.toFixed(1)} tok/s`,
            `Tokens Generated: ${pt.output_tokens}`,
            `Turn Model Time: ${pt.model_time_s.toFixed(3)} s`,
            pt.ttft_ms ? `TTFT: ${pt.ttft_ms} ms` : null,
            pt.tpot_ms ? `TPOT: ${pt.tpot_ms} ms` : null,
            `Time: ${pt.time.toFixed(2)} s`,
          ].filter(Boolean).join("\\n");

          circle.addEventListener("mousemove", (e) => {
            highlightTraj();
            showTip(e, tipMsg);
          });
          circle.addEventListener("mouseleave", () => {
            resetTraj();
            hideTip();
          });
          markersG.appendChild(circle);

          // Turn number inside/above marker if enabled
          if (showLabels) {
            const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
            lbl.setAttribute("x", x);
            lbl.setAttribute("y", y - 8);
            lbl.setAttribute("fill", "#e2e8f0");
            lbl.setAttribute("font-size", "9");
            lbl.setAttribute("font-weight", "700");
            lbl.setAttribute("text-anchor", "middle");
            lbl.style.pointerEvents = "none";
            lbl.textContent = `t${pt.turn}`;
            markersG.appendChild(lbl);
          }
        });
      });

      svg.appendChild(linesG);
      svg.appendChild(markersG);

      const tMeasureLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      tMeasureLayer.setAttribute("id", "throughput-measure-layer");
      svg.appendChild(tMeasureLayer);

      host.appendChild(svg);
      renderMeasurementOverlays();
    }

    function updateZoomDisplay(val) {
      const zVal = parseFloat(val);
      const zv = document.getElementById("zoom-val");
      if (!zv) return;
      if (zVal < 0.2) {
        zv.textContent = zVal.toFixed(2) + "x";
      } else {
        zv.textContent = zVal.toFixed(1) + "x";
      }
    }

    function fitAllHorizontal() {
      const scrollBox = document.getElementById("scroll-box");
      const containerW = (scrollBox && scrollBox.clientWidth > 100) ? scrollBox.clientWidth : (window.innerWidth - 60);
      const trajs = RAW_DATA.trajectories || [];
      const tEnd = Math.max(...trajs.map(t => t.end_time_s), RAW_DATA.meta.total_duration_sec || 1.0);
      const span = Math.max(tEnd, 1.0);
      const labelW = 120;
      const availW = Math.max(containerW - labelW - 50, 400);
      const targetPxPerSec = Math.max(availW / span, 0.1);
      const targetZoom = Math.min(Math.max(targetPxPerSec / 30.0, 0.02), 3.0);
      const zInput = document.getElementById("zoom-scale");
      if (zInput) {
        zInput.value = targetZoom.toFixed(3);
        updateZoomDisplay(targetZoom);
        renderChart();
        renderThroughputChart();
      }
    }

    function clearMeasurement() {
      measurement = null;
      renderMeasurementOverlays();
    }

    function renderMeasurementOverlays() {
      renderMeasurementForChart("waterfall-measure-layer", 120, () => {
        const zoom = parseFloat(document.getElementById("zoom-scale").value);
        return Math.max(30 * zoom, 0.1);
      });
      renderMeasurementForChart("throughput-measure-layer", 70, () => {
        const zoom = parseFloat(document.getElementById("zoom-scale").value);
        return Math.max(30 * zoom, 0.1);
      });
    }

    function renderMeasurementForChart(layerId, padLeft, getPxPerSec) {
      const layer = document.getElementById(layerId);
      if (!layer) return;
      layer.innerHTML = "";
      if (!measurement) return;

      const svg = layer.ownerSVGElement;
      if (!svg) return;
      const svgHeight = parseFloat(svg.getAttribute("height")) || 300;
      const svgWidth = parseFloat(svg.getAttribute("width")) || 900;
      const pxPerSec = getPxPerSec();

      const tMin = Math.max(0, Math.min(measurement.startSec, measurement.endSec));
      const tMax = Math.max(0, Math.max(measurement.startSec, measurement.endSec));
      const dt = tMax - tMin;

      const x1 = padLeft + Math.round(tMin * pxPerSec);
      const x2 = padLeft + Math.round(tMax * pxPerSec);
      const width = Math.max(x2 - x1, 1);

      // Shaded selection box
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", x1);
      rect.setAttribute("y", 0);
      rect.setAttribute("width", width);
      rect.setAttribute("height", svgHeight);
      rect.setAttribute("fill", "rgba(56, 189, 248, 0.18)");
      rect.setAttribute("stroke", "#38bdf8");
      rect.setAttribute("stroke-width", "1.5");
      rect.setAttribute("stroke-dasharray", "4,3");
      rect.style.pointerEvents = "none";
      layer.appendChild(rect);

      // Boundary vertical lines
      [x1, x2].forEach(bx => {
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", bx);
        line.setAttribute("y1", 0);
        line.setAttribute("x2", bx);
        line.setAttribute("y2", svgHeight);
        line.setAttribute("stroke", "#0284c7");
        line.setAttribute("stroke-width", "2");
        line.style.pointerEvents = "none";
        layer.appendChild(line);
      });

      // Active turns / tokens overlapping this time window
      const activeTurns = (RAW_DATA.turns || []).filter(t => t.start_time_s <= tMax && t.end_time_s >= tMin);
      const activeTokens = activeTurns.reduce((acc, t) => acc + (t.output_tokens || 0), 0);

      // Header measurement badge
      const dtSecStr = dt >= 1.0 ? `${dt.toFixed(3)}s` : `${(dt * 1000).toFixed(1)}ms`;
      const dtMsStr = dt >= 1.0 ? `(${(dt * 1000).toFixed(0)} ms)` : `(${dt.toFixed(4)}s)`;
      const rangeStr = `${tMin.toFixed(2)}s → ${tMax.toFixed(2)}s`;
      const turnStr = `${activeTurns.length} turns, ${activeTokens.toLocaleString()} tokens`;
      const pillText = `Δt: ${dtSecStr} ${dtMsStr}  |  ${rangeStr}  |  ${turnStr}  [✕]`;

      const pillG = document.createElementNS("http://www.w3.org/2000/svg", "g");
      pillG.style.cursor = "pointer";
      pillG.setAttribute("title", "Click to clear measurement (or press Esc)");
      pillG.onclick = (e) => {
        e.stopPropagation();
        clearMeasurement();
      };

      const approxW = Math.min(Math.max(pillText.length * 7 + 24, 250), svgWidth - 20);
      const centerX = Math.max(Math.min((x1 + x2) / 2, svgWidth - approxW / 2 - 10), approxW / 2 + 10);
      const pillY = 5;

      const pillBg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      pillBg.setAttribute("x", centerX - approxW / 2);
      pillBg.setAttribute("y", pillY);
      pillBg.setAttribute("width", approxW);
      pillBg.setAttribute("height", 22);
      pillBg.setAttribute("rx", 4);
      pillBg.setAttribute("fill", "#0f172a");
      pillBg.setAttribute("stroke", "#38bdf8");
      pillBg.setAttribute("stroke-width", "1.5");
      pillG.appendChild(pillBg);

      const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
      txt.setAttribute("x", centerX);
      txt.setAttribute("y", pillY + 15);
      txt.setAttribute("fill", "#38bdf8");
      txt.setAttribute("font-size", "11");
      txt.setAttribute("font-weight", "700");
      txt.setAttribute("font-family", "monospace");
      txt.setAttribute("text-anchor", "middle");
      txt.textContent = pillText;
      pillG.appendChild(txt);

      layer.appendChild(pillG);
    }

    function setupMeasuring() {
      window.addEventListener("keydown", (e) => {
        if (e.key === "Shift") {
          isShiftMeasuring = true;
          document.body.classList.add("shift-measuring");
        }
        if (e.key === "Escape") {
          clearMeasurement();
        }
      });

      window.addEventListener("keyup", (e) => {
        if (e.key === "Shift") {
          isShiftMeasuring = false;
          document.body.classList.remove("shift-measuring");
        }
      });

      attachChartMeasureListener("scroll-box", 120, () => {
        const zoom = parseFloat(document.getElementById("zoom-scale").value);
        return Math.max(30 * zoom, 0.1);
      });

      attachChartMeasureListener("throughput-scroll-box", 70, () => {
        const zoom = parseFloat(document.getElementById("zoom-scale").value);
        return Math.max(30 * zoom, 0.1);
      });
    }

    function attachChartMeasureListener(containerId, padLeft, getPxPerSec) {
      const container = document.getElementById(containerId);
      if (!container) return;

      let isDragging = false;

      container.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        if (e.shiftKey) {
          e.preventDefault();
          isDragging = true;
          const svg = container.querySelector("svg");
          if (!svg) return;
          const rect = svg.getBoundingClientRect();
          const svgX = e.clientX - rect.left;
          const pxPerSec = getPxPerSec();
          const trajs = RAW_DATA.trajectories || [];
          const tEnd = Math.max(...trajs.map(t => t.end_time_s), RAW_DATA.meta.total_duration_sec || 1.0);
          const span = Math.max(tEnd, 1.0);
          const t = Math.max(0, Math.min(span, (svgX - padLeft) / pxPerSec));
          measurement = { startSec: t, endSec: t, active: true };
          renderMeasurementOverlays();
          container.setPointerCapture(e.pointerId);
        } else {
          if (measurement && !measurement.active) {
            measurement = null;
            renderMeasurementOverlays();
          }
        }
      });

      container.addEventListener("pointermove", (e) => {
        if (isDragging && measurement && measurement.active) {
          e.preventDefault();
          const svg = container.querySelector("svg");
          if (!svg) return;
          const rect = svg.getBoundingClientRect();
          const svgX = e.clientX - rect.left;
          const pxPerSec = getPxPerSec();
          const trajs = RAW_DATA.trajectories || [];
          const tEnd = Math.max(...trajs.map(t => t.end_time_s), RAW_DATA.meta.total_duration_sec || 1.0);
          const span = Math.max(tEnd, 1.0);
          const t = Math.max(0, Math.min(span, (svgX - padLeft) / pxPerSec));
          measurement.endSec = t;
          renderMeasurementOverlays();
        }
      });

      const stopMeasure = (e) => {
        if (isDragging) {
          isDragging = false;
          try { container.releasePointerCapture(e.pointerId); } catch (_) {}
          if (measurement) {
            measurement.active = false;
            if (Math.abs(measurement.endSec - measurement.startSec) < 0.005) {
              measurement = null;
            }
            renderMeasurementOverlays();
          }
        }
      };

      container.addEventListener("pointerup", stopMeasure);
      container.addEventListener("pointercancel", stopMeasure);
    }

    // Initialization
    function init() {
      populateRunSelector();
      populateBatchFilter();
      updateSubtitle();

      renderStats(RAW_DATA.trajectories, RAW_DATA.turns, RAW_DATA.meta);
      renderChart();
      renderThroughputChart();
      setupMeasuring();

      document.getElementById("run-select").addEventListener("change", (e) => switchRun(e.target.value));
      document.getElementById("batch-filter").addEventListener("change", () => { renderChart(); renderThroughputChart(); });
      document.getElementById("search-input").addEventListener("input", () => { renderChart(); renderThroughputChart(); });
      document.getElementById("sort-select").addEventListener("change", renderChart);
      document.getElementById("toggle-tool-time").addEventListener("change", renderChart);
      if (document.getElementById("toggle-agg-tps")) {
        document.getElementById("toggle-agg-tps").addEventListener("change", renderThroughputChart);
      }
      if (document.getElementById("toggle-turn-labels")) {
        document.getElementById("toggle-turn-labels").addEventListener("change", renderThroughputChart);
      }
      document.getElementById("zoom-scale").addEventListener("input", (e) => {
        updateZoomDisplay(e.target.value);
        renderThroughputChart();
        renderChart();
      });
      const fitBtn = document.getElementById("zoom-fit-btn");
      if (fitBtn) fitBtn.addEventListener("click", fitAllHorizontal);
      const resetBtn = document.getElementById("zoom-reset-btn");
      if (resetBtn) resetBtn.addEventListener("click", () => {
        const zInput = document.getElementById("zoom-scale");
        if (zInput) {
          zInput.value = "1.0";
          updateZoomDisplay(1.0);
          renderChart();
          renderThroughputChart();
        }
      });

      // Add Run file upload listener
      const fileInput = document.getElementById("run-file-input");
      const loadBtn = document.getElementById("load-file-btn");
      if (loadBtn && fileInput) {
        loadBtn.addEventListener("click", () => fileInput.click());
        fileInput.addEventListener("change", (e) => {
          const file = e.target.files && e.target.files[0];
          if (!file) return;
          const reader = new FileReader();
          reader.onload = (evt) => {
            try {
              const text = evt.target.result;
              const parsed = parseClientJSONL(text, file.name);
              const runName = file.name.replace(/\\.[^/.]+$/, "");
              ALL_RUNS[runName] = parsed;
              populateRunSelector();
              document.getElementById("run-select").value = runName;
              switchRun(runName);
            } catch (err) {
              alert("Failed to parse run metrics JSONL file: " + err);
            }
          };
          reader.readAsText(file);
        });
      }

      window.addEventListener("resize", () => { renderChart(); renderThroughputChart(); });
    }

    try {
      init();
    } catch (err) {
      console.error("Initialization error:", err);
      const errBox = document.getElementById("chart-count-info");
      if (errBox) {
        errBox.innerHTML = '<span style="color:#ef4444; font-weight:700;">Error: ' + err.message + '</span>';
      }
    }
  </script>
</body>
</html>
"""
    return html_template.replace("__TITLE__", title).replace("__RUNS_JSON__", runs_json)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize multi-turn agentic RL trajectories and turn latencies across one or more benchmark runs."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[],
        help="One or more input trajectory metrics JSONL files, or 'Run Label=path/to/file.jsonl'.",
    )
    parser.add_argument(
        "--run",
        "-r",
        action="append",
        dest="explicit_runs",
        default=[],
        help="Explicitly named run in format 'Run Label=path/to/file.jsonl' (can be specified multiple times).",
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
        help="Generate synthetic multi-run demo data to preview the comparison visualization.",
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

    runs_data: Dict[str, Dict[str, Any]] = {}

    # 1. Parse explicitly named runs via -r / --run
    for item in args.explicit_runs:
        if "=" in item:
            name, path = item.rsplit("=", 1)
        else:
            name, path = os.path.splitext(os.path.basename(item))[0], item
        print(f"Loading named run '{name}' from: {path}")
        runs_data[name] = parse_metrics_file(path, run_name=name)

    # 2. Parse positional inputs
    for item in args.inputs:
        if "=" in item:
            name, path = item.rsplit("=", 1)
        else:
            name, path = os.path.splitext(os.path.basename(item))[0], item
        print(f"Loading run '{name}' from: {path}")
        runs_data[name] = parse_metrics_file(path, run_name=name)

    # 3. Fallback to demo mode if no inputs provided or --demo requested
    if args.demo or not runs_data:
        if not runs_data and not args.demo:
            print("No input files provided. Generating interactive multi-run demo data (--demo)...")
        demo_runs = generate_demo_runs()
        runs_data.update(demo_runs)

    html_content = generate_html(runs_data, title=args.title)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated trajectory waterfall visualization: {os.path.abspath(args.output)}")
    print(f"Total runs embedded: {len(runs_data)} -> {list(runs_data.keys())}")
    for name, rdata in runs_data.items():
        print(f"  - [{name}]: {len(rdata['trajectories'])} trajectories, {len(rdata['turns'])} turns")

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
