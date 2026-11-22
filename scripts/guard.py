#!/usr/bin/env python3
"""Post-Codex diff guard.

Checks the working tree of a repository against a base commit for:
  - changes to protected paths (benchmark, correctness check, contract, AGENTS.md, ...)
  - changes outside the allowed scope of the current task
  - oversized diffs
  - red-flag patterns in added lines (benchmark gaming, contract violations, env changes)

Exit code 0 = clean, 1 = violations (details on stdout, JSON with --json).

  guard.py --repo . --base HEAD --protect-file sre/protected.txt \
           --allow 'csrc/kernels/*' --allow 'engine/layers/*.py' --max-lines 600
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys

RED_FLAGS = [
    (r"\btime\.sleep\(|\busleep\(|\bstd::this_thread::sleep", "sleep added"),
    (r"\b(atol|rtol|tolerance|max_abs_diff|kl_threshold)\s*=", "tolerance changed"),
    (r"pytest\.mark\.skip|unittest\.skip|@skip\b|xfail", "test skipped"),
    (r"\bpip install|\bconda install|\bapt(-get)? install|\buv (pip|add)\b", "dependency install"),
    (r"TODO:?\s*(restore|re-?enable|revert)|HACK|FIXME: remove", "temporary hack"),
    (r"num_layers\s*=\s*\d|layers\[:\s*\d|skip_layer|skip_attention", "layers skipped"),
    (r"(warmup|n_iters|num_iters|iterations|repeat)\s*=\s*\d", "benchmark iteration/warmup change"),
    (r"lru_cache|functools\.cache|_cached_output|memoize", "caching that may elide work"),
    (r"CUDA_LAUNCH_BLOCKING|TORCH_CUDNN|NCCL_[A-Z_]+\s*=|os\.environ\[", "environment variable change"),
    (r"float8|fp8|int4|nf4|e4m3|e5m2|bitsandbytes", "precision keyword (check contract)"),
]


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


def changed_files(repo, base):
    files = set(l for l in git(repo, "diff", "--name-only", base).splitlines() if l)
    for l in git(repo, "status", "--porcelain").splitlines():
        files.add(l[3:].split(" -> ")[-1])
    return sorted(files)


def numstat(repo, base):
    total = 0
    for l in git(repo, "diff", "--numstat", base).splitlines():
        a, d, _ = l.split("\t", 2)
        if a != "-":
            total += int(a) + int(d)
    return total


def added_lines(repo, base):
    out, cur = [], None
    for l in git(repo, "diff", "-U0", base).splitlines():
        if l.startswith("+++ "):
            cur = l[6:] if l.startswith("+++ b/") else l[4:]
        elif l.startswith("+") and not l.startswith("+++"):
            out.append((cur, l[1:]))
    return out


def match_any(path, globs):
    for g in globs:
        if fnmatch.fnmatch(path, g):
            return True
        if g.endswith(("/**", "/")) and path.startswith(g.rstrip("*").rstrip("/") + "/"):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--base", default="HEAD")
    ap.add_argument("--protect", action="append", default=[], help="glob of protected paths (repeatable)")
    ap.add_argument("--protect-file", help="file with one protected glob per line")
    ap.add_argument("--allow", action="append", default=[], help="glob of paths the task may edit (repeatable)")
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--ignore-flag", action="append", default=[], help="red-flag label to ignore (e.g. when the contract allows fp8)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    protect = list(a.protect)
    if a.protect_file:
        with open(a.protect_file) as f:
            protect += [l.strip() for l in f if l.strip() and not l.startswith("#")]
    protect += ["AGENTS.md", "sre/contract.yaml", "sre/protected.txt"]

    files = changed_files(a.repo, a.base)
    v = {"protected": [p for p in files if match_any(p, protect)],
         "out_of_scope": [p for p in files if a.allow and not match_any(p, a.allow) and not p.startswith("sre/runs/")],
         "diff_lines": numstat(a.repo, a.base), "red_flags": []}
    for path, line in added_lines(a.repo, a.base):
        for pat, label in RED_FLAGS:
            if label in a.ignore_flag:
                continue
            if re.search(pat, line, re.I):
                v["red_flags"].append({"file": path, "flag": label, "line": line.strip()[:160]})
    too_big = a.max_lines and v["diff_lines"] > a.max_lines
    hard = bool(v["protected"] or v["out_of_scope"] or too_big)
    v["verdict"] = "VIOLATION" if hard else ("REVIEW" if v["red_flags"] else "CLEAN")
    v["changed_files"] = files

    if a.json:
        print(json.dumps(v, indent=2))
    else:
        print(f"guard: {v['verdict']}  ({len(files)} files, {v['diff_lines']} changed lines)")
        for p in v["protected"]:
            print(f"  PROTECTED  {p}")
        for p in v["out_of_scope"]:
            print(f"  OUT-SCOPE  {p}")
        if too_big:
            print(f"  DIFF-SIZE  {v['diff_lines']} > {a.max_lines}")
        for r in v["red_flags"]:
            print(f"  FLAG       {r['flag']:<36} {r['file']}: {r['line']}")
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
