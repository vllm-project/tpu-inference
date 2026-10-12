#!/usr/bin/env python3
"""Turn two /metrics scrapes into a server-side report for one benchmark leg.

Why this exists: TTFT is not a prefill measurement. vLLM splits a request's
pre-first-token life into two phases and exports them separately --
`vllm:request_queue_time_seconds` is `scheduled_ts - queued_ts` and
`vllm:request_prefill_time_seconds` is `first_token_ts - scheduled_ts`
(v1/metrics/stats.py). TTFT is their sum, so a TTFT number cannot distinguish
"prefill is slow" from "the request sat in the admission queue". On build #12
median TTFT moved 2.8% between the conc-8 and conc-160 legs while actual prompt
throughput moved 3.9x, which is what a metric that is blind to the distinction
looks like.

Counters are cumulative from server start, so everything here is a delta
between two scrapes taken around one leg.

This is instrumentation. It must never fail a build: every metric is optional
and a missing one prints as "n/a" rather than raising.

  qwen38_2p4t_metrics.py --pre A.prom --post B.prom [--series S.prom]
                         [--bench bench_cN.json] [--label cN] [--json out.json]
"""
import argparse
import json
import re
import sys

# Prometheus client appends _total to Counter names and _bucket/_sum/_count to
# Histogram names, so the exposed name is not always the name in loggers.py.
LE_RE = re.compile(r'le="([^"]+)"')


def parse(path):
    """Scalar samples summed over label sets, plus histogram buckets by `le`.

    Summing over label sets is what we want: the labels are (model_name,
    engine) and we always want the whole server. It is also why gauges are
    handled the same way -- `vllm:num_requests_running` summed over engines is
    the global batch.
    """
    scalars, buckets = {}, {}
    try:
        fh = open(path, errors="replace")
    except OSError as e:
        print(f"[metrics] cannot read {path}: {e}", file=sys.stderr)
        return scalars, buckets
    with fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "{" in line:
                name, _, rest = line.partition("{")
                labels, _, val = rest.rpartition("}")
            else:
                name, _, val = line.partition(" ")
                labels = ""
            try:
                v = float(val.strip().split()[0])
            except (ValueError, IndexError):
                continue
            if name.endswith("_bucket"):
                m = LE_RE.search(labels)
                if m:
                    buckets.setdefault(name, {})
                    buckets[name][m.group(1)] = buckets[name].get(m.group(1), 0.0) + v
            else:
                scalars[name] = scalars.get(name, 0.0) + v
    return scalars, buckets


def get(scalars, name):
    """Look a metric up under both the bare and the _total-suffixed name."""
    for k in (name, name + "_total"):
        if k in scalars:
            return scalars[k]
    return None


def delta(pre, post, name):
    a, b = get(pre, name), get(post, name)
    if a is None or b is None:
        return None
    return b - a


def hist_mean_ms(pre, post, base):
    """Mean of a seconds-valued histogram over the window, in ms.

    _sum/_count rather than buckets, so there is no bucket-edge error.
    """
    s = delta(pre, post, base + "_sum")
    n = delta(pre, post, base + "_count")
    if s is None or n is None or n <= 0:
        return None, (n or 0)
    return s * 1000.0 / n, n


def fmt(v, unit="", width=10, prec=1):
    if v is None:
        return f"{'n/a':>{width}}"
    return f"{v:>{width},.{prec}f}{unit}"


def bucket_hist(pre_b, post_b, name):
    """Per-bin counts from cumulative `le` buckets, as [(upper, count)]."""
    a, b = pre_b.get(name, {}), post_b.get(name, {})
    if not b:
        return []
    def key(le):
        return float("inf") if le == "+Inf" else float(le)
    edges = sorted(b, key=key)
    out, prev = [], 0.0
    for le in edges:
        cum = b[le] - a.get(le, 0.0)
        out.append((le, max(0.0, cum - prev)))
        prev = cum
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--series")
    ap.add_argument("--bench", help="bench_cN.json, for the timed-window cross-check")
    ap.add_argument("--label", default="leg")
    ap.add_argument("--window-s", type=float,
                    help="wall-clock between the two scrapes")
    ap.add_argument("--json", help="write the report here as well")
    args = ap.parse_args()

    pre, pre_b = parse(args.pre)
    post, post_b = parse(args.post)
    if not post:
        print(f"[metrics] {args.label}: no samples in {args.post}; skipping report")
        return 0

    w = args.window_s if args.window_s and args.window_s > 0 else None
    rep = {"label": args.label, "window_s": w}

    prompt_tok = delta(pre, post, "vllm:prompt_tokens")
    gen_tok = delta(pre, post, "vllm:generation_tokens")
    finished = delta(pre, post, "vllm:request_success")
    preempt = delta(pre, post, "vllm:num_preemptions")
    pc_hits = delta(pre, post, "vllm:prefix_cache_hits")
    pc_q = delta(pre, post, "vllm:prefix_cache_queries")

    print(f"--- [metrics] {args.label}: server-side, delta over the scrape window"
          + (f" ({w:.1f} s)" if w else ""))
    print("    The window spans the whole `vllm bench serve` invocation, so it")
    print("    includes the warmup requests as well as the timed ones.")
    print()
    print(f"  prefill (prompt) tokens   {fmt(prompt_tok, prec=0)}"
          + (f"   {fmt(prompt_tok / w, ' tok/s', 9)}" if w and prompt_tok else ""))
    print(f"  decode (generation) tok   {fmt(gen_tok, prec=0)}"
          + (f"   {fmt(gen_tok / w, ' tok/s', 9)}" if w and gen_tok else ""))
    print(f"  requests finished         {fmt(finished, prec=0)}")
    print(f"  preemptions               {fmt(preempt, prec=0)}")
    if pc_q:
        print(f"  prefix cache hit rate     {fmt(100.0 * pc_hits / pc_q, ' %', 10)}"
              "   (must be ~0: a hit means prefill was skipped, not fast)")
    rep.update(prompt_tokens=prompt_tok, generation_tokens=gen_tok,
               requests_finished=finished, preemptions=preempt,
               prompt_tokens_per_s=(prompt_tok / w if w and prompt_tok else None))

    # The whole point of the file. queue + prefill should reconstruct TTFT; the
    # residual is a check that the three histograms are talking about the same
    # requests.
    q_ms, q_n = hist_mean_ms(pre, post, "vllm:request_queue_time_seconds")
    p_ms, _ = hist_mean_ms(pre, post, "vllm:request_prefill_time_seconds")
    d_ms, _ = hist_mean_ms(pre, post, "vllm:request_decode_time_seconds")
    t_ms, _ = hist_mean_ms(pre, post, "vllm:time_to_first_token_seconds")
    print()
    print(f"  TTFT decomposition, mean per request over {int(q_n):,} requests:")
    tot = (q_ms or 0) + (p_ms or 0)
    for nm, v in (("queue   (WAITING)", q_ms), ("prefill (PREFILL)", p_ms)):
        share = f"   {100.0 * v / tot:5.1f}%" if v is not None and tot else ""
        print(f"    {nm:<22}{fmt(v, ' ms')}{share}")
    print(f"    {'-' * 40}")
    print(f"    {'queue + prefill':<22}{fmt(tot if tot else None, ' ms')}")
    print(f"    {'TTFT (reported)':<22}{fmt(t_ms, ' ms')}")
    if t_ms and tot:
        print(f"    {'residual':<22}{fmt(t_ms - tot, ' ms')}"
              f"   {100.0 * abs(t_ms - tot) / t_ms:.1f}% of TTFT")
    print(f"    {'decode  (DECODE)':<22}{fmt(d_ms, ' ms')}")
    rep.update(queue_time_ms=q_ms, prefill_time_ms=p_ms,
               decode_time_ms=d_ms, ttft_ms=t_ms)

    # Is `max_num_batched_tokens` actually the binding constraint? If the mass
    # sits in the top bin the budget is saturated and raising it should buy
    # prefill throughput; if it sits below, the budget is not what is limiting
    # us and raising it buys nothing.
    bins = bucket_hist(pre_b, post_b, "vllm:iteration_tokens_total_bucket")
    steps = sum(c for _, c in bins)
    tok_sum = delta(pre, post, "vllm:iteration_tokens_total_sum")
    print()
    if steps > 0:
        print(f"  engine steps              {fmt(steps, prec=0)}"
              + (f"   mean {tok_sum / steps:,.0f} tok/step" if tok_sum else ""))
        print("  tokens per step (share of steps -- the max_num_batched_tokens test):")
        for le, c in bins:
            if c <= 0:
                continue
            print(f"    <= {le:>7}   {c:>10,.0f}   {100.0 * c / steps:5.1f}%")
        rep["iteration_token_bins"] = {le: c for le, c in bins}
        rep["engine_steps"] = steps
    else:
        print("  engine steps              n/a  (vllm:iteration_tokens_total absent)")

    # Cross-check the counter against the client's own volume accounting. These
    # measure different windows -- the counter includes warmups, the bench JSON
    # does not -- so they should differ by roughly the warmup share, and a much
    # larger gap means one of the two is not measuring what we think.
    if args.bench:
        try:
            b = json.load(open(args.bench))
        except (OSError, ValueError) as e:
            print(f"\n  [metrics] could not read {args.bench}: {e}")
        else:
            dur, tin = b.get("duration"), b.get("total_input_tokens")
            if dur and tin:
                print()
                print(f"  timed window only (from {args.bench.split('/')[-1]}):")
                print(f"    prompt tok/s            {fmt(tin / dur, '', 10)}"
                      f"   ({tin:,} tok / {dur:.1f} s, warmups excluded)")
                print(f"    median TTFT             {fmt(b.get('median_ttft_ms'), ' ms')}"
                      "   <- the metric this file exists to replace")
                rep["timed_prompt_tokens_per_s"] = tin / dur

    if args.series:
        n = 0
        try:
            with open(args.series, errors="replace") as fh:
                n = sum(1 for line in fh if line.startswith("# SNAPSHOT"))
        except OSError:
            pass
        if n:
            print(f"\n  timeseries: {n} scrapes in "
                  f"{args.series.split('/')[-1]} (slice any sub-window offline)")

    if args.json:
        try:
            with open(args.json, "w") as fh:
                json.dump(rep, fh, indent=2, sort_keys=True)
        except OSError as e:
            print(f"[metrics] could not write {args.json}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # instrumentation must not fail the build
        print(f"[metrics] report failed, continuing: {e!r}", file=sys.stderr)
        sys.exit(0)
