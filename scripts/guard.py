#!/usr/bin/env python3
"""Post-Codex diff guard.

Checks the working tree of a repository against a base commit for:
  - changes to protected paths (benchmark, correctness check, contract, AGENTS.md, ...)
  - changes outside the allowed scope of the current task
  - oversized diffs
  - red-flag patterns in added lines (benchmark gaming, contract violations, env changes)

Exit code 0 = clean or needs review, 1 = hard violation.
Inspect the verdict in stdout or use --json for a machine-readable result.

  guard.py --repo . --base HEAD --protect-file sre/protected.txt \
           --allow 'csrc/kernels/*' --allow 'engine/layers/*.py' --max-lines 600
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
from pathlib import Path
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
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", repo, *args], capture_output=True, check=True,
    ).stdout.decode("utf-8", errors="surrogateescape")


def diff(repo, base, *args):
    return git(repo, "-c", "core.quotePath=true", "diff", "--no-ext-diff", "--no-textconv",
               "--no-color", "--find-renames", *args, base, "--")


def diff_stats(repo, base, *args):
    """Read both endpoints of tracked renames without relying on display quoting."""
    entries = iter(diff(repo, base, "--numstat", "-z", *args).split("\0")[:-1])
    files = set()
    total = 0
    for entry in entries:
        added, deleted, path = entry.split("\t", 2)
        if path:
            files.add(path)
        else:
            files.update((next(entries), next(entries)))
        if added != "-":
            total += int(added) + int(deleted)
    return files, total


def tracked_additions(repo, base):
    path = None
    in_hunk = False
    for line in diff(repo, base, "-U0", "--src-prefix=a/", "--dst-prefix=b/",
                     "--output-indicator-new=+", "--output-indicator-old=-",
                     "--output-indicator-context= ").split("\n"):
        if line.startswith("diff --git "):
            path = None
            in_hunk = False
        elif line.startswith("@@ "):
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            yield path, line[1:]
        elif not in_hunk and line.startswith("+++ "):
            name = line[4:].removesuffix("\t")
            if name.startswith('"'):
                name = ast.literal_eval("b" + name).decode("utf-8", errors="surrogateescape")
            path = name.removeprefix("b/")


def untracked_lines(repo, path):
    file = Path(repo) / path
    if file.is_symlink():
        yield os.readlink(file)
    elif file.is_file():
        with file.open("rb") as contents:
            # Git's ordinary diff also treats files with an early NUL as binary.
            if b"\0" in contents.read(8000):
                return
            contents.seek(0)
            for line in contents:
                yield line.decode("utf-8", errors="surrogateescape").removesuffix("\n")


def match_any(path, globs):
    for g in globs:
        if fnmatch.fnmatchcase(path, g):
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
        with open(a.protect_file, encoding="utf-8") as f:
            patterns = map(str.strip, f)
            protect += [pattern for pattern in patterns if pattern and not pattern.startswith("#")]
    protect += ["AGENTS.md", "sre/contract.yaml", "sre/protected.txt"]

    # Validate the base as a commit before passing it to diff as an argument.
    base = git(a.repo, "rev-parse", "--verify", "--end-of-options", a.base + "^{commit}").strip()
    tracked, total = diff_stats(a.repo, base)
    staged, _ = diff_stats(a.repo, base, "--cached")
    untracked = git(a.repo, "ls-files", "-z", "--others", "--exclude-standard").split("\0")[:-1]
    log_paths = {p for p in untracked if p.startswith("sre/runs/")}
    files = sorted(tracked | staged | set(untracked))
    v = {"protected": [p for p in files if match_any(p, protect)],
         "out_of_scope": [p for p in files if a.allow and p not in log_paths and not match_any(p, a.allow)],
         "diff_lines": total, "red_flags": []}
    patterns = [(re.compile(pat, re.I), label) for pat, label in RED_FLAGS if label not in a.ignore_flag]

    def check_line(path, line):
        for pattern, label in patterns:
            if pattern.search(line):
                v["red_flags"].append({"file": path, "flag": label, "line": line.strip()[:160]})

    for path, line in tracked_additions(a.repo, base):
        check_line(path, line)
    for path in sorted(set(untracked) - log_paths):
        for line in untracked_lines(a.repo, path):
            v["diff_lines"] += 1
            check_line(path, line)
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
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.stderr.decode("utf-8", errors="replace").strip() or str(exc))
    except OSError as exc:
        sys.exit(str(exc))
