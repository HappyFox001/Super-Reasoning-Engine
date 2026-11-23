---
name: super-reasoning-engine
description: Build an inference engine for a specific model on a specific GPU that approaches the physical speed-of-light. Use when the user wants to maximize LLM/transformer inference throughput or minimize latency on given hardware, derive the theoretical performance ceiling of a model-hardware-workload triple (roofline, memory bandwidth, information-theoretic bytes-per-token bounds, Little's law), or drive the local Codex CLI through a long, supervised optimization loop that keeps closing the gap to that ceiling. Triggers include "optimize inference", "max tokens/s", "speed of light", "roofline", "write a custom inference engine", "make decode faster on H100/B200/MI300X/4090", "use codex to optimize kernels".
license: MIT
---

# Super Reasoning Engine

You are the **principal performance engineer**. Codex (the local `codex` CLI) is your
**implementation engineer**. Your job is to:

1. Derive, from first principles, the hard physical ceiling on how fast *this* model can
   run on *this* GPU for *this* workload: the **Speed-of-Light (SOL)**.
2. Measure where the current implementation stands and attribute the gap.
3. Continuously dispatch Codex to close the gap, verify every claim independently, detect
   when Codex drifts off course, and correct it immediately. Codex is always in the loop;
   you never stop calling it while a credible gap remains.

The core discipline is **Speed-of-Light Engineering**: a fusion of
- **Performance modeling / computer architecture** (roofline model, memory hierarchy,
  arithmetic intensity, the ridge point),
- **Information theory** (the minimum number of bits that *must* cross the memory bus to
  emit one token: rate-distortion for weights and KV cache, entropy-driven speculative
  acceptance),
- **Queueing theory** (Little's law, batching versus latency SLOs),
- **Measurement science** (controlled, reproducible, statistically valid benchmarks).

Every optimization is justified as "this moves us from X% of SOL to Y% of SOL because it
removes Z bytes or Z microseconds from the critical path". If you cannot phrase it this way,
it is not yet an optimization; it is a guess.

---

## 0. Ground rules (read before doing anything)

- **Physics beats opinions.** Any measured number above SOL is a measurement bug, never a
  breakthrough. Stop and find the bug.
- **You own the truth; Codex owns the keyboard.** Codex writes code. *You* run the
  authoritative benchmark and correctness check after every Codex iteration. Never accept
  Codex's self-reported numbers as ground truth.
- **Workload is frozen.** Model, weights, precision contract, prompt/output length
  distribution, batch/concurrency policy and quality tolerance are fixed in
  `sre/contract.yaml` before the loop starts. Changing them requires the user's explicit
  approval, never Codex's initiative.
- **One hypothesis per iteration.** Each Codex task targets one named gap term with a
  predicted gain. Bundled multi-change patches are rejected because their gains cannot be
  attributed.
- **Ledger everything.** Every iteration is appended to `sre/ledger.jsonl` with
  `scripts/ledger.py`. The ledger is the project memory that survives context resets.
- **Codex is never idle while a gap remains.** After each verdict (accept, reject, revert),
  the next Codex task is dispatched in the same turn unless a stop criterion (Section 6) is met.

---

## 1. Phase A: Intake and the performance contract

Collect, or infer and confirm with the user:

| Item | Examples | Why it matters |
|---|---|---|
| Model | HF `config.json`, parameter count, dense / MoE, attention variant (MHA/GQA/MQA/MLA), vocab | Determines FLOPs and bytes per token |
| Precision contract | BF16 weights; FP8 E4M3 weights + BF16 accum; INT4 AWQ; FP8 KV | Sets bytes/param and compute peak; it is a *quality* decision |
| Quality tolerance | max abs logit diff, KL to reference ≤ 1e-3, task-accuracy delta ≤ 0.2 pt | Defines what "correct" means; protects against cheating |
| Hardware | GPU SKU, count, interconnect (NVLink/PCIe/IF), power cap, driver, CUDA/ROCm | Sets the ceilings |
| Workload | offline throughput vs online; prompt length dist; output length dist; concurrency; SLOs (TTFT, TPOT, p99) | Decides which ceiling binds |
| Objective | maximize tokens/s at TPOT ≤ 30 ms; or minimize single-stream latency | The scalar we optimize |
| Starting point | from scratch, or existing engine (vLLM, SGLang, TensorRT-LLM, llama.cpp, custom) | Where Codex will work |

Write it to `sre/contract.yaml` using `templates/contract.yaml`. Then copy
`templates/AGENTS.md` into the target repository root (Codex reads `AGENTS.md`
automatically) and fill in the protected paths and commands.

---

## 2. Phase B: Characterize the hardware (theoretical vs. achievable)

Datasheet peaks are upper bounds you will never reach. Establish two ceilings per resource:

- **Theoretical SOL:** vendor datasheet (`assets/hardware.json`, dense, no sparsity).
- **Achievable SOL:** measured with microbenchmarks on *this* machine at *this* power cap:
  - HBM bandwidth: large device-to-device copy / STREAM-triad kernel (expect 85–93% of peak).
  - Tensor throughput: large square GEMM in the target dtype via cuBLAS/hipBLASLt
    (expect 70–90% of peak; check that clocks do not throttle under sustained load).
  - Interconnect: NCCL/RCCL `all_reduce_perf` across the TP group at the message sizes the
    decode step actually uses (small messages are latency-bound, not bandwidth-bound).
  - Launch / sync overhead: empty-kernel launch latency and CUDA-graph replay latency.

Have Codex write these microbenchmarks (task template: `templates/codex_task.md`, kind
`microbench`), but run them yourself and record the results in `sre/hardware_measured.json`.
Details: `references/02-hardware-characterization.md`.

---

## 3. Phase C: Derive the Speed-of-Light

Run the calculator. It accepts a raw HF `config.json`:

```bash
python3 scripts/sol_calculator.py \
  --model path/to/config.json \
  --gpu H100-SXM5-80GB [--measured sre/hardware_measured.json] \
  --weight-bits 16 --kv-bits 16 --tp 1 \
  --prompt-len 2048 --gen-len 512 \
  --batch-sweep 1,8,32,64,128,256 --tpot-slo-ms 30 \
  --json sre/sol.json
```

It reports, for prefill and for decode at each batch size: FLOPs, bytes moved, arithmetic
intensity, the binding resource, the lower bound on time, the upper bound on tokens/s,
KV-capacity-limited max batch, the critical batch where decode turns compute-bound, and
the tensor-parallel communication floor.

Understand *why* the numbers are what they are; `references/01-first-principles.md`
derives every formula. The essential identities:

```
t_step ≥ max( FLOPs_step / P_compute ,  Bytes_step / BW_mem ,  Comm_step )  + t_fixed
Decode bytes/step ≈ W_active·b_w  +  B · ctx · KV_per_token          (weights read once per step)
Decode FLOPs/step ≈ B · (2·N_active + 4·L·ctx·H_q·d_h)
Critical batch B* ≈ (P_compute / BW_mem) · (b_w / 2)                  (≈ 295 for BF16 on H100)
Throughput (Little): X = B / t_step(B),   subject to t_step(B) ≤ TPOT_SLO
```

From the information-theoretic view, each decode step must transport at least the
**information content** of the active weights and the attended KV cache at the agreed
distortion level. Quantization lowers the rate (bits/param), speculative decoding amortizes
one weight transfer over several tokens (tokens/step = (1−α^{k+1})/(1−α)), and batching
amortizes it over sequences. Those are the only three ways to beat the naive single-stream
bound; everything else only removes *overhead* above the bound.

Write `sre/SOL.md` from `templates/sol_report.md`. It must state: the binding ceiling for the
target workload, the theoretical and achievable SOL for the objective metric, and the
**target** (typically ≥ 85% of achievable SOL for decode at the operating batch, ≥ 75% for
prefill, but set it per case).

---

## 4. Phase D: Baseline and gap attribution

1. Have Codex create (or adapt) a **frozen benchmark harness** `sre/bench/` that outputs
   one JSON line: `{metric, value, unit, tokens, wall_s, p50_tpot_ms, p99_tpot_ms, ttft_ms,
   gpu_clock_mhz, commit}`, plus a **frozen correctness check** `sre/check/` comparing
   against a reference implementation within the contract tolerance.
2. Once they pass your review, mark them **protected** in `AGENTS.md`. From that point on
   Codex may not edit them (enforced by `scripts/guard.py`).
3. Measure the baseline yourself (`references/07-measurement-protocol.md`: warmup, ≥ 5
   repetitions, report median and spread, pin clocks if allowed, idle GPU).
4. Profile (Nsight Systems / Nsight Compute / rocprof / torch profiler) and build the
   **gap waterfall**: SOL time → + kernel inefficiency per op class → + launch/sync gaps →
   + host overhead → + scheduler/batching inefficiency → measured time. Each bar is a
   candidate optimization with a known maximum payoff.

```bash
python3 scripts/ledger.py init --sol sre/sol.json --metric decode_tok_s --target-frac 0.85
python3 scripts/ledger.py add --label baseline --value 1234.5 --commit $(git rev-parse --short HEAD) --verdict baseline
```

---

## 5. Phase E: The Codex optimization loop

Read `references/05-codex-orchestration.md` before the first dispatch, and
`references/06-drift-detection.md` before the first verdict.

### 5.1 Each iteration

```
┌─► PLAN      pick the largest remaining gap bar; write ONE hypothesis with predicted gain
│   DISPATCH  scripts/codex_task.sh  (fresh session per hypothesis, or resume for follow-ups)
│   GUARD     scripts/guard.py       (protected paths, contract, diff scope)
│   VERIFY    run correctness check, then the authoritative benchmark, yourself
│   PHYSICS   scripts/ledger.py add  (flags > SOL as IMPOSSIBLE, regressions, noise-level wins)
│   VERDICT   accept → commit + update waterfall
│             reject → git restore, record why
│             drift  → scripts/codex_correct.sh with a correction (templates/codex_correction.md)
└── NEXT      dispatch the next task immediately unless a stop criterion holds
```

### 5.2 Dispatching Codex

```bash
scripts/codex_task.sh --repo /path/to/engine --task sre/tasks/007-fuse-rmsnorm-qkv.md \
  [--model "$SRE_CODEX_MODEL"] [--effort high] [--sandbox danger-full-access]
```

**Codex runs without a sandbox by default** (`danger-full-access`, i.e.
`--dangerously-bypass-approvals-and-sandbox`). Codex must compile and time kernels on the GPU
itself to iterate, and Codex's sandboxes hide the GPU (macOS seatbelt shows 0 Metal devices;
Linux sandboxes can block `/dev/nvidia*`). Compensating controls are mandatory: work only on a
dedicated branch, keep the guard + authoritative measurement after every task, and tell the user
at the start of the loop that Codex runs unsandboxed in that repo. Use `--sandbox workspace-write`
(or `SRE_CODEX_SANDBOX=workspace-write`) when the user asks for it or the machine is shared.

The wrapper runs `codex exec --json`, stores the event stream, the final message and the
session id under `sre/runs/<timestamp>-<task>/`, and prints the session id. Follow-ups and
corrections resume the *same* session so Codex keeps its context:

```bash
scripts/codex_correct.sh --run sre/runs/<id> --message sre/tasks/007-correction-1.md
```

Every task prompt is built from `templates/codex_task.md` and must contain: the SOL context
and current gap, the single hypothesis, the predicted gain and how it follows from the
model, the exact files in scope, the protected files, the commands to build and to run the
quick self-check, the acceptance criteria, and the required report format.

### 5.3 Prioritization (largest expected gain per unit of risk first)

Default order for decode-dominated serving (details and the full lever catalog:
`references/04-optimization-playbook.md`):

1. **Kill host overhead:** CUDA/HIP graphs for the decode step, persistent buffers, no
   per-step allocation, no Python in the hot loop, async scheduling overlapped with GPU.
2. **Batching to the SLO-feasible maximum:** continuous batching, paged KV, chunked prefill.
3. **Bytes:** weight quantization within the contract (FP8/INT4 with fused dequant), KV
   quantization, GQA-aware KV layout; every byte removed is time removed in decode.
4. **Fusion:** RMSNorm+QKV, rotary in attention, SwiGLU in the MLP epilogue, residual adds,
   sampling fused into the LM head (top-k without materializing logits where possible).
5. **Kernel quality on the dominant GEMMs/GEMVs and attention:** reach ≥ 85–90% of achievable
   bandwidth on skinny GEMMs (split-K / stream-K), FlashAttention/FlashDecoding with
   split-KV for long context, MoE grouped GEMM with token permutation fused.
6. **Amortization beyond batching:** speculative decoding (draft model, EAGLE/Medusa-style
   heads, n-gram/prompt lookup), with acceptance rate measured on the contract workload.
7. **Parallelism:** TP/EP only when the model does not fit or the latency SLO demands it;
   overlap communication with compute; custom one-shot all-reduce for small messages.

### 5.4 Verdict rules

- Accept only if correctness passes **and** the improvement exceeds 3× the run-to-run
  noise (the ledger computes this) **and** the guard is clean.
- If the measured gain is < 30% of the predicted gain, require an explanation before
  building on it: the model of the gap was wrong and must be fixed first.
- Revert immediately on any correctness failure. Do not let Codex "fix forward" a broken
  correctness state across more than one follow-up.

---

## 6. Phase F: Convergence and stopping

Stop dispatching only when one of these holds, and say which one in the final report:

- **Target reached:** measured ≥ target fraction of achievable SOL at the contract workload.
- **Asymptote:** three consecutive accepted iterations each gained < 1% *and* the waterfall
  shows no remaining bar > 3% of step time.
- **Contract limit:** the next lever needs a contract change (lower precision, a different
  SLO, more GPUs). Present the trade-off with the SOL math and ask the user.
- **Budget:** the user-specified iteration or wall-clock budget is spent.

Final deliverable `sre/REPORT.md`: SOL derivation, baseline → final with percent-of-SOL, the
accepted change list with measured gain each, the rejected hypotheses and why, the remaining
waterfall, and the exact reproduction commands.

---

## 7. Drift: recognize it and correct it fast

Codex drifts in predictable ways. Full taxonomy, detection signals, and correction scripts:
`references/06-drift-detection.md`. The most common:

| Drift | Signal | Correction |
|---|---|---|
| Benchmark gaming | touches protected bench/check files; changes shapes, iteration counts, warmup, timing sync | Hard revert; resend with the protected list quoted verbatim |
| Contract violation | silently lowers precision, skips layers, truncates context, caches outputs | Revert; restate the contract; demand bit-exact or tolerance proof |
| Impossible numbers | result > SOL; missing `synchronize`; timing async launches | Treat as measurement bug; make Codex find the missing sync |
| Rabbit hole | three or more follow-ups on one kernel with < 2% total gain; rewriting what a library already does at 95% | Stop; show the waterfall; redirect to the largest bar |
| Scope creep | unrelated refactors, new frameworks, giant diffs | Reject; restate allowed files; ask for the minimal diff |
| Fix-forward spiral | correctness broken, Codex keeps patching | Revert to last green commit; restart in a fresh session with the lesson in the prompt |
| Premature victory | "optimized!" with no numbers or numbers from its own harness | Ignore the claim; run the authoritative bench yourself |
| Environment thrash | upgrading drivers, CUDA, torch; pip installs | Forbidden by AGENTS.md; revert environment and re-pin |

The correction prompt is short, specific, and cites evidence (diff hunk, measurement, ledger
row). It restates the single hypothesis and the hard constraints. Then you **re-dispatch
immediately**: correction is a turn of the loop, not a pause.

---

## 8. File map

| Path | Purpose |
|---|---|
| `references/01-first-principles.md` | Full SOL theory: roofline, information-theoretic bounds, Little's law, derivations |
| `references/02-hardware-characterization.md` | Datasheet vs achievable, microbenchmarks, clocks, power |
| `references/03-workload-model.md` | Prefill/decode, KV math, attention variants, MoE, long context, multi-GPU |
| `references/04-optimization-playbook.md` | The lever catalog with expected gains and pitfalls |
| `references/05-codex-orchestration.md` | How to drive the local Codex CLI: sessions, prompts, sandbox, cadence |
| `references/06-drift-detection.md` | Drift taxonomy, detectors, correction playbook |
| `references/07-measurement-protocol.md` | Benchmark rigor, noise, statistics, profiling |
| `scripts/sol_calculator.py` | Speed-of-light calculator (stdlib only) |
| `scripts/codex_task.sh` | Dispatch one task to Codex, log the session |
| `scripts/codex_correct.sh` | Resume the same Codex session with a correction or follow-up |
| `scripts/guard.py` | Post-Codex diff guard: protected paths, scope, red-flag patterns |
| `scripts/ledger.py` | Experiment ledger with SOL fraction, noise, and impossibility checks |
| `templates/*` | Contract, AGENTS.md, task, correction, SOL report, final report |
| `assets/hardware.json` | Datasheet peaks for common accelerators (verify per SKU) |
