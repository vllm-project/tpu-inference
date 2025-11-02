import gzip
import json
def extract_op_times_from_json(json_gz_path, name):
    """Parses a trace.json.gz file and extracts operation durations."""
    with gzip.open(json_gz_path, 'rt', encoding='utf-8') as f:
        trace_data = json.load(f)
    t = 0
    cnt = 0
    trace_events = trace_data.get('traceEvents', [])
    for event in trace_events:
        if event.get('ph') == 'X':  # Complete event
            if "ragged" in event["name"]:
              print(event["name"])
            op_name = event["name"]
            duration_us = event["dur"]
            if name in op_name:
                t += event['dur']
                cnt += 1
                continue

    return t, cnt


print(extract_op_times_from_json('/tmp/jax-trace/plugins/profile/2025_10_31_18_43_16/t1v-n-1479201e-w-0.trace.json.gz', 'jit_ragged_paged_attention'))