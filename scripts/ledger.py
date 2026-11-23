#!/usr/bin/env python3
"""Experiment ledger for the SOL optimization loop.

Every Codex iteration becomes one JSON line in sre/ledger.jsonl. The ledger computes the
fraction of SOL, the gain over the best accepted result, and raises flags:

  IMPOSSIBLE       measured value beats the speed-of-light  -> measurement bug
  NOISE            gain at or below 3x the noise floor       -> not a real win
  MODEL-MISMATCH   measured gain < 30% of predicted gain     -> gap model is wrong
  REGRESSION       worse than the best accepted result

Usage:
  ledger.py init --sol sre/sol.json [--sol-value X] --metric decode_tok_s --target-frac 0.85 [--noise 0.01]
  ledger.py add  --label "fuse rmsnorm+qkv" --value 1520 --commit abc123 --verdict accept \
                 [--target fusion] [--predicted-gain 0.06] [--runs 1510,1525,1519] [--notes "..."]
  ledger.py show [--stall]

Acceptance requires a valid baseline and a known noise floor. An accept with IMPOSSIBLE,
NOISE or REGRESSION exits 2 without appending a row. Use a non-accept verdict to record
the observation for future reference.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

DEFAULT = os.environ.get("SRE_LEDGER", "sre/ledger.jsonl")
VERDICTS = ["baseline", "accept", "reject", "revert", "drift", "probe"]
BLOCKING_FLAGS = {"IMPOSSIBLE", "NOISE", "REGRESSION"}


def read_number(value, label, *, allow_zero=False):
    requirement = "non-negative" if allow_zero else "positive"
    message = f"{label} must be a finite {requirement} number"
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if isinstance(value, bool) or not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ValueError(message)
    return result


def load(path):
    if not os.path.exists(path):
        sys.exit(f"no ledger at {path}; run `ledger.py init` first")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("each ledger row must be a JSON object")
    if not rows or rows[0].get("type") != "init":
        sys.exit("ledger is missing its init row")
    init = rows[0]
    init["sol_value"] = read_number(init.get("sol_value"), "ledger SOL")
    if init.get("noise") is not None:
        init["noise"] = read_number(init["noise"], "ledger noise", allow_zero=True)
    return init, rows[1:]


def append(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")


def better(init, a, b):
    return a < b if init["lower_is_better"] else a > b


def valid_best_row(init, row):
    verdict = row.get("verdict")
    if verdict not in ("baseline", "accept"):
        return False
    flags = set(row.get("flags") or ())
    if "IMPOSSIBLE" in flags or (verdict == "accept" and BLOCKING_FLAGS & flags):
        return False
    try:
        value = read_number(row.get("value"), "historical value")
    except ValueError:
        return False
    return not better(init, value, init["sol_value"])


def best_accepted(init, rows):
    vals = [float(r["value"]) for r in rows if valid_best_row(init, r)]
    if not vals:
        return None
    return min(vals) if init["lower_is_better"] else max(vals)


def frac(init, v):
    s = init["sol_value"]
    return s / v if init["lower_is_better"] else v / s


def cmd_init(a):
    if os.path.exists(a.ledger) and not a.force:
        sys.exit(f"{a.ledger} exists; use --force to overwrite")
    sol = a.sol_value
    if sol is None:
        with open(a.sol, encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise ValueError("SOL file must contain a JSON object")
        src = d.get("achievable") or d.get("theoretical") or {}
        op = src.get("operating_point") if isinstance(src, dict) else None
        if not isinstance(op, dict) or not op:
            sys.exit("sol.json has no operating_point; pass --sol-value explicitly")
        sol = op.get("tpot_ms" if a.lower_is_better else "tok_s")
    sol = read_number(sol, "SOL value")
    target = read_number(a.target_frac, "--target-frac")
    if target > 1:
        raise ValueError("--target-frac must be at most 1")
    noise = None if a.noise is None else read_number(a.noise, "--noise", allow_zero=True)
    row = {"type": "init", "metric": a.metric, "sol_value": sol, "target_frac": target,
           "noise": noise, "lower_is_better": a.lower_is_better, "sol_source": a.sol, "ts": time.time()}
    os.makedirs(os.path.dirname(a.ledger) or ".", exist_ok=True)
    with open(a.ledger, "w", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
    print(f"ledger initialized: {a.metric} SOL={sol:.4g}, target {target:.0%} of SOL")


def cmd_add(a):
    init, rows = load(a.ledger)
    runs = [read_number(x, "--runs measurement") for x in a.runs.split(",")] if a.runs is not None else []
    value = a.value if a.value is not None else (statistics.median(runs) if runs else None)
    if value is None:
        sys.exit("need --value or --runs")
    value = read_number(value, "--value or median of --runs")
    predicted = None if a.predicted_gain is None else read_number(a.predicted_gain, "--predicted-gain", allow_zero=True)
    spread = None
    if len(runs) >= 2:
        spread = read_number((max(runs) - min(runs)) / statistics.median(runs), "measurement spread", allow_zero=True)
    baselines = [row for row in rows if row["verdict"] == "baseline" and valid_best_row(init, row)]
    noise = init.get("noise")
    if noise is None:
        noise = next((row["spread"] for row in baselines if row.get("spread") is not None), None)
    if noise is not None:
        noise = read_number(noise, "noise floor", allow_zero=True)
    if a.verdict == "accept" and (not baselines or noise is None):
        print("cannot record accept without a valid baseline and measured noise floor. "
              "Record baseline repetitions or initialize with an observed --noise value.", file=sys.stderr)
        return 2
    best = best_accepted(init, rows)
    flags = []
    if better(init, value, init["sol_value"]):
        flags.append("IMPOSSIBLE")
    gain = None
    if best is not None:
        gain = (best - value) / best if init["lower_is_better"] else (value - best) / best
        if not math.isfinite(gain):
            raise ValueError("gain must be finite")
        if gain < 0:
            flags.append("REGRESSION")
        elif gain <= 3 * (noise or 0):
            flags.append("NOISE")
        if predicted and gain >= 0 and gain < 0.3 * predicted:
            flags.append("MODEL-MISMATCH")
    if a.verdict == "accept" and BLOCKING_FLAGS.intersection(flags):
        print(f"cannot record accept for {a.label!r}: {', '.join(flags)}", file=sys.stderr)
        return 2
    row = {"type": "iter", "n": len(rows) + 1, "label": a.label, "target": a.target, "value": value,
           "runs": runs, "spread": spread, "commit": a.commit, "verdict": a.verdict,
           "predicted_gain": predicted, "gain_vs_best": gain,
           "sol_frac": read_number(frac(init, value), "fraction of SOL"),
           "flags": flags, "codex_run": a.codex_run, "notes": a.notes, "ts": time.time()}
    append(a.ledger, row)
    g = f"{gain:+.2%}" if gain is not None else "n/a"
    print(f"#{row['n']} {a.label}: {value:.4g} ({row['sol_frac']:.1%} of SOL, gain {g}) verdict={a.verdict}"
          + (f"  FLAGS: {', '.join(flags)}" if flags else ""))
    if "IMPOSSIBLE" in flags:
        print("!! Above speed-of-light: treat as a measurement bug (sync, token count, cached work).")
    return 2 if "IMPOSSIBLE" in flags else 0


def cmd_show(a):
    init, rows = load(a.ledger)
    best = best_accepted(init, rows)
    print(f"metric {init['metric']}  SOL {init['sol_value']:.4g}  target {init['target_frac']:.0%}")
    print(f"{'#':>3} {'verdict':>8} {'value':>10} {'%SOL':>6} {'gain':>8}  {'target':<12} label / flags")
    for r in rows:
        g = f"{r['gain_vs_best']:+.1%}" if r.get("gain_vs_best") is not None else ""
        fl = (" [" + ",".join(r["flags"]) + "]") if r["flags"] else ""
        print(f"{r['n']:>3} {r['verdict']:>8} {r['value']:>10.4g} {r['sol_frac']:>6.1%} {g:>8}  "
              f"{(r.get('target') or ''):<12} {r['label']}{fl}")
    if best is not None:
        f = frac(init, best)
        status = "TARGET REACHED" if f >= init["target_frac"] else f"gap {init['target_frac'] - f:.1%} to target"
        print(f"\nbest accepted {best:.4g} = {f:.1%} of SOL; {status}")
    if a.stall:
        acc = [r for r in rows if r["verdict"] == "accept" and valid_best_row(init, r)]
        last = acc[-3:]
        if len(last) == 3 and all((r.get("gain_vs_best") or 0) < 0.01 for r in last):
            print("STALL: last 3 accepted iterations each gained < 1%. Re-derive the waterfall.")
        attempts = [r for r in rows if r["verdict"] != "accept" or valid_best_row(init, r)]
        tgt = attempts[-1].get("target") if attempts else None
        if tgt:
            same = [r for r in attempts if r.get("target") == tgt][-3:]
            cum = sum(max(0, r.get("gain_vs_best") or 0) for r in same if r["verdict"] == "accept")
            if len(same) == 3 and cum < 0.02:
                print(f"RABBIT-HOLE: 3 iterations on target '{tgt}' with {cum:.1%} cumulative gain. Redirect.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=DEFAULT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--sol", default="sre/sol.json")
    i.add_argument("--sol-value", type=float)
    i.add_argument("--metric", required=True)
    i.add_argument("--target-frac", type=float, default=0.85)
    i.add_argument("--noise", type=float, help="relative noise floor, e.g. 0.01")
    i.add_argument("--lower-is-better", action="store_true")
    i.add_argument("--force", action="store_true")
    d = sub.add_parser("add")
    d.add_argument("--label", required=True)
    d.add_argument("--value", type=float)
    d.add_argument("--runs", help="comma-separated repeat measurements")
    d.add_argument("--commit")
    d.add_argument("--verdict", choices=VERDICTS, required=True)
    d.add_argument("--target", help="gap term this iteration attacks (for stall detection)")
    d.add_argument("--predicted-gain", type=float)
    d.add_argument("--codex-run")
    d.add_argument("--notes")
    s = sub.add_parser("show")
    s.add_argument("--stall", action="store_true")
    a = ap.parse_args()
    try:
        return {"init": cmd_init, "add": cmd_add, "show": cmd_show}[a.cmd](a)
    except (OSError, ValueError) as exc:
        sys.exit(f"ledger: {exc}")


if __name__ == "__main__":
    sys.exit(main())
