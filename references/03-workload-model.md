# 03: Workload Model

The SOL depends on the workload as much as on the hardware. This file maps model and
workload features to the terms of the bound.

## 1. Reading a model config

From HF `config.json`:

| Key(s) | Symbol |
|---|---|
| `hidden_size` | `D` |
| `num_hidden_layers` | `L` |
| `num_attention_heads` | `H_q` |
| `num_key_value_heads` (absent → = `H_q`) | `H_kv` |
| `head_dim` (absent → `D / H_q`) | `d_h` |
| `intermediate_size` | dense MLP width |
| `vocab_size`, `tie_word_embeddings` | LM head size |
| `num_local_experts` / `num_experts` / `n_routed_experts` | `E` |
| `num_experts_per_tok` | `k` |
| `moe_intermediate_size`, `n_shared_experts`, `first_k_dense_replace` | MoE geometry |
| `kv_lora_rank`, `qk_rope_head_dim` | MLA latent cache |
| `sliding_window`, `layer_types` | windowed attention |

Parameter count per layer (gated MLP, e.g. SwiGLU):

```
attn  = D·(H_q·d_h) + 2·D·(H_kv·d_h) + (H_q·d_h)·D
mlp   = 3 · D · I                                   (gate, up, down)
moe   = E · 3·D·I_moe  (resident),  k · 3·D·I_moe  (active)  + router D·E
embed = V · D;  lm_head = V · D (unless tied; it is always *read* each decode step)
```

The embedding lookup reads only `B` rows; the LM head reads the full `V·D` matrix every step.
Count it in `N_active` for bytes.

## 2. Prefill vs decode mix

Total time for a request ≈ `t_prefill(S) + Σ_{i=1..G} t_step(B, S+i)`. For chat workloads
with `S ≈ 2k`, `G ≈ 500`, decode dominates GPU-time; for RAG or summarization with `S ≈ 32k`,
`G ≈ 200`, prefill can dominate. Weight the optimization plan by each phase's share of
end-to-end GPU time **on the contract workload**, not on a toy prompt.

Chunked prefill mixes prefill chunks into decode batches: it raises arithmetic intensity of
decode steps (good) but lengthens them (TPOT impact). Model the chunk size explicitly.

## 3. Attention variants

| Variant | KV_tok | Decode behaviour |
|---|---|---|
| MHA | `2·L·H_q·d_h·b_kv` | KV-bound early |
| GQA (group g = H_q/H_kv) | `2·L·H_kv·d_h·b_kv` | KV reduced g×; kernel must load each KV head once per group |
| MQA | `2·L·d_h·b_kv` | minimal KV; parallelism over heads is low, split-KV needed |
| MLA | `L·(r_kv + d_rope)·b_kv` | tiny cache; decode attention becomes compute-heavier (absorbed up-projection) |
| Sliding window W | min(ctx, W) instead of ctx | KV read bounded |

Decode attention SOL is `B · ctx · KV_tok / BW` for the KV read. FlashDecoding (split the
KV sequence across CTAs, reduce partials) is required when `B · H_kv` is too small to fill the
GPU's SMs.

## 4. MoE

- Bytes: distinct experts touched per layer `E·(1−(1−k/E)^B)` × expert bytes. At `B=1` you
  read `k` experts; at large `B` you read all `E`.
- Compute: `2·N_active·B` as usual.
- Kernels: grouped GEMM over experts with token permutation/unpermutation fused; load
  imbalance creates tail effects. Track tokens-per-expert histograms.
- Multi-GPU: expert parallelism replaces weight reads with all-to-all traffic.

## 5. Sampling and the LM head

`V·D` bytes per step (for V=152k, D=5120, BF16: 1.56 GB, i.e. ~0.5 ms on H100). Logits of
size `B·V` in FP32 are written and read for sampling. Fuse top-k/top-p and argmax into the LM
head epilogue where the contract's sampling semantics allow; never copy logits to host.

## 6. Context length distribution

Use the contract's actual distribution (p50/p90/p99 of `S` and `G`). The SOL calculator
evaluates a representative `ctx`; for heavy-tailed distributions run it at p50 and p90 and
report both. Paged KV with block size 16–64 keeps memory waste low and enables high `B`.

## 7. Multi-GPU decision

Use the smallest `p` such that:

1. weights + KV for `B_op` fit, and
2. `t_step` at `B_op` meets the TPOT SLO.

Beyond that, prefer data-parallel replicas for throughput. Compute the all-reduce floor
(01 §8) before choosing `p`: at `p=8` on PCIe, communication alone can exceed the memory SOL.
