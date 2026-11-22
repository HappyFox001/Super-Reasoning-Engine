#!/usr/bin/env python3
"""Experiment ledger for the SOL optimization loop.

Every Codex iteration becomes one JSON line in sre/ledger.jsonl. The ledger computes the
fraction of SOL, the gain over the best accepted result, and raises flags:

  IMPOSSIBLE       measured value beats the speed-of-light  -> measurement bug
  NOISE            gain smaller than 3x the noise floor      -> not a real win
  MODEL-MISMATCH   measured gain < 30% of predicted gain     -> gap model is wrong
  REGRESSION       worse than the best accepted result

Usage:
  ledger.py init --sol sre/sol.json [--sol-value X] --metric decode_tok_s --target-frac 0.85 [--noise 0.01]
  ledger.py add  --label "fuse rmsnorm+qkv" --value 1520 --commit abc123 --verdict accept \
                 [--target fusion] [--predicted-gain 0.06] [--runs 1510,1525,1519] [--notes "..."]
  ledger.py show [--stall]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

DEFAULT = os.environ.get("SRE_LEDGER", "sre/ledger.jsonl")
VERDICTS = ["baseline", "accept", "reject", "revert", "drift", "probe"]


def load(path):
    if not os.path.exists(path):
        sys.exit(f"no ledger at {path}; run `ledger.py init` first")
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if not rows or rows[0].get("type") != "init":
        sys.exit("ledger is missing its init row")
    return rows[0], rows[1:]


def append(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def better(init, a, b):
    return a < b if init["lower_is_better"] else a > b


def best_accepted(init, rows):
    vals = [r["value"] for r in rows if r["verdict"] in ("baseline", "accept")]
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
        with open(a.sol) as f:
            d = json.load(f)
        src = d.get("achievable") or d["theoretical"]
        op = src.get("operating_point")
        if not op:
            sys.exit("sol.json has no operating_point; pass --sol-value explicitly")
        sol = op["tpot_ms"] if a.lower_is_better else op["tok_s"]
    row = {"type": "init", "metric": a.metric, "sol_value": float(sol), "target_frac": a.target_frac,
           "noise": a.noise, "lower_is_better": a.lower_is_better, "sol_source": a.sol, "ts": time.time()}
    os.makedirs(os.path.dirname(a.ledger) or ".", exist_ok=True)
    with open(a.ledger, "w") as f:
        f.write(json.dumps(row) + "\n")
    print(f"ledger initialized: {a.metric} SOL={sol:.4g}, target {a.target_frac:.0%} of SOL")


def cmd_add(a):
    init, rows = load(a.ledger)
    runs = [float(x) for x in a.runs.split(",")] if a.runs else []
    value = a.value if a.value is not None else (statistics.median(runs) if runs else None)
    if value is None:
        sys.exit("need --value or --runs")
    spread = (max(runs) - min(runs)) / statistics.median(runs) if len(runs) >= 2 else None
    base_spread = next((r["spread"] for r in rows if r["verdict"] == "baseline" and r.get("spread")), None)
    noise = init.get("noise") or base_spread or spread
    best = best_accepted(init, rows)
    flags = []
    if better(init, value, init["sol_value"]):
        flags.append("IMPOSSIBLE")
    gain = None
    if best is not None:
        gain = (best - value) / best if init["lower_is_better"] else (value - best) / best
        if gain < 0:
            flags.append("REGRESSION")
        elif noise and gain < 3 * noise:
            flags.append("NOISE")
        if a.predicted_gain and gain >= 0 and gain < 0.3 * a.predicted_gain:
            flags.append("MODEL-MISMATCH")
    row = {"type": "iter", "n": len(rows) + 1, "label": a.label, "target": a.target, "value": value,
           "runs": runs, "spread": spread, "commit": a.commit, "verdict": a.verdict,
           "predicted_gain": a.predicted_gain, "gain_vs_best": gain, "sol_frac": frac(init, value),
           "flags": flags, "codex_run": a.codex_run, "notes": a.notes, "ts": time.time()}
    append(a.ledger, row)
    g = f"{gain:+.2%}" if gain is not None else "n/a"
    print(f"#{row['n']} {a.label}: {value:.4g} ({row['sol_frac']:.1%} of SOL, gain {g}) verdict={a.verdict}"
          + (f"  FLAGS: {', '.join(flags)}" if flags else ""))
    if "IMPOSSIBLE" in flags:
        print("!! Above speed-of-light: treat as a measurement bug (sync, token count, cached work).")
    if a.verdict == "accept" and any(f in flags for f in ("IMPOSSIBLE", "NOISE", "REGRESSION")):
        print("!! Accept verdict conflicts with flags; reconsider.")
    sys.exit(2 if "IMPOSSIBLE" in flags else 0)


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
        acc = [r for r in rows if r["verdict"] == "accept"]
        last = acc[-3:]
        if len(last) == 3 and all((r.get("gain_vs_best") or 0) < 0.01 for r in last):
            print("STALL: last 3 accepted iterations each gained < 1%. Re-derive the waterfall.")
        tgt = rows[-1].get("target") if rows else None
        if tgt:
            same = [r for r in rows if r.get("target") == tgt][-3:]
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
    {"init": cmd_init, "add": cmd_add, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
