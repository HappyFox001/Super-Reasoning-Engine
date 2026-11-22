# 01: First Principles of the Speed-of-Light

This reference derives every bound the skill uses. Read it once fully; afterwards use it to
justify each hypothesis you send to Codex.

## 1. The disciplines involved

"How fast can this model possibly run on this chip" is a question in four overlapping fields:

| Field | Question it answers | Tool |
|---|---|---|
| Computer architecture / performance modeling | Which resource saturates first? | Roofline model, arithmetic intensity, ridge point |
| Information theory | What is the minimum data that must move per token? | Rate (bits/element), rate-distortion, entropy, speculative acceptance |
| Queueing theory | How do batching, concurrency and latency trade off? | Little's law, utilization vs latency |
| Measurement science | Is a number real? | Controlled experiments, noise floors, physical plausibility |

Together they form **Speed-of-Light (SOL) analysis**: compute the bound, measure the
distance, attribute the distance, remove it.

## 2. Notation

| Symbol | Meaning |
|---|---|
| `N` | parameters touched per token (for MoE: active params, `N_active`) |
| `W` | parameters resident in memory (MoE: all experts) |
| `b_w`, `b_kv`, `b_a` | bytes per weight, KV element, activation element |
| `L` | layers; `H_q`, `H_kv` query / KV heads; `d_h` head dim; `D` hidden size |
| `S` | prompt (prefill) length; `ctx` current context length during decode |
| `B` | decode batch (concurrent sequences in one step) |
| `P` | peak tensor throughput in the compute dtype (FLOP/s) |
| `BW` | memory bandwidth (B/s); `BW_link` interconnect bandwidth |
| `p` | tensor-parallel degree |

## 3. The roofline model

For a kernel performing `F` FLOPs while moving `Q` bytes to/from DRAM:

```
t ≥ max(F / P, Q / BW)          arithmetic intensity I = F / Q   (FLOP/byte)
attainable FLOP/s = min(P, I · BW)
ridge point I* = P / BW
```

Kernels with `I < I*` are memory-bound; their SOL is `Q / BW`. Kernels with `I > I*` are
compute-bound; their SOL is `F / P`. For an H100 SXM in BF16, `I* ≈ 989e12 / 3.35e12 ≈ 295`
FLOP/byte. For FP8 compute the ridge doubles (~590).

A whole decode step is a sequence of kernels; its SOL is the **sum** of per-kernel SOLs
(kernels serialize), which is ≥ the max of aggregate F/P and Q/BW. Overlap (e.g. streams,
persistent kernels, PDL/programmatic dependent launch) lets you approach the aggregate bound.

## 4. Bytes and FLOPs per decode step

Per decode step with batch `B` and per-sequence context `ctx`:

```
Weights read:      Q_w   = N_active_unique(B) · b_w
KV read:           Q_kv  = B · ctx · KV_tok
KV write:          B · KV_tok                       (negligible)
KV per token:      KV_tok = 2 · L · H_kv · d_h · b_kv        (MHA/GQA/MQA)
                   KV_tok = L · (d_c + d_rope) · b_kv          (MLA latent cache)
Linear FLOPs:      F_lin = 2 · N_active · B
Attention FLOPs:   F_att = 4 · L · H_q · d_h · ctx · B
```

`N_active_unique(B)` matters for MoE: with `E` experts, top-`k` routing and `B` tokens, the
expected number of distinct experts touched per layer is `E · (1 − (1 − k/E)^B)`. Small
batches read few experts; large batches read nearly all of them. Dense models read all of `N`.

Decode step SOL:

```
t_step(B) ≥ max( (F_lin + F_att) / P ,  (Q_w + Q_kv) / BW )
```

### Critical batch

Ignoring attention, decode turns compute-bound when `2·N·B / P = N·b_w / BW`:

```
B* = (P / BW) · (b_w / 2)
```

BF16 on H100: `295 · 1 = 295`. With FP8 weights *and* FP8 compute, `B* = 590 · 0.5 = 295`
(both sides scale). With INT4 weights dequantized into BF16 compute, `B* = 295 · 0.25 ≈ 74`:
weight-only quantization makes decode compute-bound sooner, so INT4 shines at low batch and
loses its edge at high batch. Always compute `B*` for the *actual* (storage dtype, compute
dtype) pair.

Attention KV traffic does **not** amortize over the batch: each sequence reads its own cache.
At long context, `Q_kv` dominates and decode stays memory-bound at any batch size. The
remedies there are KV bytes (GQA/MLA, KV quantization) and sparsity, not batching.

## 5. Prefill

Prefill processes `S` tokens at once; weights are read once, so it is compute-bound for
`S` beyond a few hundred:

```
F_prefill = 2 · N_active · S  +  2 · L · H_q · d_h · S²      (causal: half of 4·S²)
t_prefill ≥ F_prefill / P
TTFT ≥ t_prefill + queueing + first decode step
```

FlashAttention-class kernels keep attention compute-bound and IO-optimal; the SOL target for
prefill attention is ~70–80% of `P` on Hopper/Blackwell class hardware with good kernels.

## 6. Information-theoretic lower bounds

The roofline says *time ≥ bytes / bandwidth*. Information theory asks *what is the minimum
number of bytes?*

1. **Weights as a source.** At a given quality (distortion `D`), the weights need at least
   `R(D)` bits per parameter, where `R` is the rate-distortion function of the weight
   distribution under the loss the model actually cares about (output KL, task accuracy),
   not MSE on weights. Practical quantizers (FP8, INT4 group-wise, vector/lattice codebooks)
   are points on an empirical rate-distortion curve. The contract fixes the allowed `D`; your
   job is to reach the lowest rate at that `D` *and* decode it at bandwidth speed (fused
   dequant inside the GEMM/GEMV, never a separate pass).
2. **Lossless headroom.** Entropy coding of BF16 weights (the exponent field has low entropy)
   yields ~1.3–1.5× compression losslessly. It only pays off if decompression runs at > BW
   on-chip; treat it as an advanced lever.
3. **KV cache as a source.** Same logic per element; plus *structural* rate reduction: GQA
   shares K/V across `H_q/H_kv` heads, MLA stores a low-rank latent, and sliding-window or
   sparse attention reads only part of the cache.
4. **Tokens per transfer.** One weight transfer can serve several tokens:
   - across sequences (batching, `B`),
   - across time (speculative decoding). With per-token acceptance probability `α` and draft
     length `k`, expected tokens per verification step:
     `E[tokens] = (1 − α^{k+1}) / (1 − α)`.
     For exact (lossless) speculative sampling, `α = 1 − TV(p_target, p_draft)` per position,
     so acceptance is bounded by the distributional distance between draft and target; low-
     entropy (predictable) outputs such as code and structured text accept much better.
     Effective speedup ≈ `E[tokens] / (1 + c·k)` where `c` is draft cost relative to a target
     step; it requires the verify step of `k+1` tokens to still be memory-bound (true for
     `B·(k+1) < B*`).

The **information floor per emitted token** therefore is:

```
Q_token ≥ [ R_w(D) · N_active_unique / (B · E[tokens]) ]  +  [ ctx · KV_tok(R_kv) ]
```

Weights amortize over `B · E[tokens]`; KV does not. This single line explains why
long-context serving is KV-bound and short-context serving is weight-bound.

## 7. Queueing: throughput vs latency

Little's law: `concurrency = throughput × latency`. For decode at batch `B`:

```
throughput X(B) = B / t_step(B)            tokens/s
per-token latency (TPOT) = t_step(B)
```

Memory-bound regime: `t_step` is nearly flat in `B`, so throughput grows almost linearly
with batch for free. Compute-bound regime: `t_step ∝ B`, throughput saturates at `P / (2N)`.
The optimal operating point is **the largest `B` such that `t_step(B) ≤ TPOT_SLO` and the KV
cache fits**:

```
B_max_mem = (HBM · u − W·b_w − workspace) / (ctx_avg · KV_tok)     (u ≈ 0.9)
B_op      = min(B_max_mem, max{B : t_step(B) ≤ TPOT_SLO})
```

Online serving adds queueing delay: at utilization `ρ`, M/M/1-style waiting grows as
`ρ/(1−ρ)`; p99 latency explodes past ~80–85% utilization. Report offline SOL and the
SLO-constrained SOL separately.

## 8. Multi-GPU

Tensor parallelism splits weights so each GPU reads `N·b_w / p`, but adds two all-reduces
per layer (after attention output projection and after the MLP), each of size
`B · D · b_a`:

```
t_allreduce ≈ 2·(p−1)/p · (B·D·b_a) / BW_link  +  2·(p−1)·α_hop
```

Decode messages are tiny (e.g. `B=8, D=8192, BF16` = 128 KB), so the latency term `α`
dominates; one-shot / two-shot custom all-reduce kernels over NVLink and fusing the
all-reduce into the GEMM epilogue are the levers. Pipeline parallelism adds bubbles and helps
throughput, not latency. Expert parallelism adds all-to-all with load-imbalance tail latency.

## 9. Fixed costs (`t_fixed`)

Not bounded by physics but by engineering; they are the first things to eliminate:

- Kernel launch: ~2–5 µs each from the host; a 64-layer model with ~12 kernels/layer = ~770
  launches = 2–4 ms per step, which can exceed the entire memory SOL. CUDA Graphs cut this to
  a single replay (~10–20 µs).
- Host work: Python scheduling, tokenization, sampling on CPU, `.item()` syncs.
- Kernel tails: wave quantization (grid not a multiple of SM count), small-kernel ramp-up.

A credible engine target is `t_fixed < 3%` of `t_step`.

## 10. The SOL statement

Every SOL report ends with a single sentence of this form:

> "For <model> on <GPU> under <contract>, decode at B=<B_op>, ctx=<ctx> is <memory|compute>-
> bound; theoretical SOL is <X> tok/s, achievable SOL (measured BW/P) is <Y> tok/s; the
> current engine reaches <Z> tok/s = <Z/Y>% of achievable SOL; the largest gap term is <term>."
