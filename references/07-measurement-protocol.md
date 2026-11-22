# 07: Measurement Protocol

A number you cannot reproduce is not a result. This protocol governs every authoritative
measurement in the loop.

## 1. The frozen harness

`sre/bench/` is written once (by Codex, reviewed by you), then protected. It must:

- Load the model exactly as production does (same dtype, same kernels).
- Use the contract workload: fixed seeds, fixed prompt set or length distribution, fixed
  output lengths (`ignore_eos` with fixed `max_tokens` for throughput tests).
- Warm up (≥ 3 full iterations, including graph capture and autotuning) before timing.
- Time with device events or wall clock **after** a device synchronize, around the whole
  measured region.
- Count tokens from the outputs, never from the request parameters.
- Emit one JSON line per run:
  `{"metric":"decode_tok_s","value":..., "unit":"tok/s","tokens":..., "wall_s":...,
   "p50_tpot_ms":..., "p99_tpot_ms":..., "ttft_ms":..., "batch":..., "ctx":...,
   "gpu_clock_mhz":..., "commit":"..."}`

## 2. The frozen correctness check

`sre/check/` compares against a reference (HF transformers in FP32/BF16, or the baseline
engine at the last verified commit):

- Teacher-forced logits on a fixed prompt set: max abs diff, mean KL divergence, top-1
  agreement.
- Greedy generation token-match rate over N tokens (expect divergence after near-ties in
  low precision; use KL as the primary criterion).
- Optional task-level eval (a few hundred items) for precision-changing tasks.

Thresholds live in `sre/contract.yaml`, not in the check code.

## 3. Repetition and noise

- ≥ 5 repetitions; report median, min, and relative spread `(max−min)/median`.
- Establish the noise floor at baseline: run the unchanged engine 10 times. `ledger.py init`
  stores it; accept a change only if gain > 3 × noise.
- Interleave A/B when the difference is small: A, B, A, B, … to cancel thermal drift.

## 4. Environment hygiene

- No other GPU processes; record `nvidia-smi` before and after.
- Note clocks and power state; lock clocks for A/B where allowed.
- Same driver/toolkit across the whole ledger; a change invalidates comparisons, so
  re-baseline.

## 5. Profiling for attribution

| Tool | Use |
|---|---|
| Nsight Systems (`nsys profile -t cuda,nvtx,osrt`) | timeline: gaps, launches, syncs, overlap |
| Nsight Compute (`ncu --set full -k regex:<kernel>`) | per-kernel: achieved BW, FLOP/s, occupancy, stall reasons |
| `rocprofv3` / Omniperf | AMD equivalents |
| PyTorch profiler | operator-level view, quick |

From one decode step's trace, build the **gap waterfall**:

```
SOL time                                 (from sol_calculator, achievable)
+ Σ per-kernel (actual − kernel SOL)     grouped: GEMM, attention, norm/act, sampling, comm
+ idle gaps between kernels              (launch / dependency latency)
+ host-side stalls                       (GPU idle waiting on CPU)
= measured step time
```

Store it as `sre/waterfall.md` and update it after each accepted change. The largest bar is
the next task.

## 6. Sanity rules

- A result above achievable SOL → measurement bug (check sync, token count, cached work).
- A big gain with no profile change → measurement bug.
- A gain that disappears under interleaved A/B → noise.
- A gain only at one batch size → report it as such; check the others for regressions.
