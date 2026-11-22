# 04: Optimization Playbook

Each lever lists: **which SOL term it attacks**, the **max payoff** (from the gap waterfall),
how to **verify** it, and the **traps** Codex tends to fall into. Pick levers by largest
measured gap, never by fashion.

Two classes of work, always label which one a task is:

- **Floor-lowering** (changes the physics term): quantization, KV compression, batching,
  speculative decoding, attention variant at the contract level.
- **Overhead-removing** (approaches the floor): graphs, fusion, better kernels, scheduling.

## A. Host and launch overhead (overhead-removing)

| Lever | Detail | Verify | Traps |
|---|---|---|---|
| CUDA/HIP Graphs for decode | capture per batch-size bucket; static input/output buffers; update with `cudaGraphExecUpdate` or bucket padding | nsys timeline: one graph launch per step, no gaps | capturing allocations; graph per exact B (explosion); hidden host syncs breaking capture |
| No host syncs in loop | remove `.item()`, `.cpu()`, `torch.nonzero` on device data, implicit syncs | nsys: no `cudaStreamSynchronize` per step | moving sync elsewhere without removing it |
| Async scheduler | CPU prepares step n+1 while GPU runs step n; pinned ring buffers | GPU idle gaps < 3% | race conditions on block tables |
| Persistent kernels / megakernel | one kernel runs many layers | step time vs sum of kernels | huge engineering cost; only after everything else |
| Programmatic dependent launch (Hopper+) | overlap next kernel prologue with tail | tail gaps shrink | arch-specific |

## B. Batching and scheduling (floor-lowering via amortization)

| Lever | Detail | Traps |
|---|---|---|
| Continuous batching | admit/evict per step | starving prefill or decode |
| Paged KV cache | block tables, block size 16–64 | fragmentation, gather cost in attention |
| Chunked prefill | cap tokens per step | chunk too large → TPOT SLO break |
| Prefix caching | reuse KV of shared prefixes | correctness with eviction |
| Batch-size buckets | pad B to graph buckets | too coarse → wasted compute |

## C. Bytes (floor-lowering)

| Lever | Detail | Verify | Traps |
|---|---|---|---|
| FP8 weights (E4M3, per-channel/per-block scales) | Hopper/Blackwell/MI300 native | contract tolerance; GEMM BW at decode shapes | per-tensor scales hurting quality; dequant as separate kernel |
| INT4/FP4 weights (AWQ/GPTQ/NVFP4/MXFP4), fused dequant | best at low B | quality eval on contract tasks | separate dequant pass doubles traffic; slow at high B |
| KV FP8 / INT8 / INT4 | per-head or per-token scales | long-context quality | quantizing in prefill but not decode, or vice versa |
| Layout | KV contiguous per head-group; weights pre-swizzled for the kernel | achieved BW | runtime transposes |

Changing precision is a **contract** decision: Codex may implement it only when the contract
allows that precision.

## D. Fusion (overhead-removing)

Typical decode fusions, in order of payoff:

1. RMSNorm + QKV projection (or norm into previous residual add)
2. RoPE inside attention (or into the QKV epilogue) + KV-cache write
3. SwiGLU/GeGLU activation into gate/up GEMM epilogue
4. Residual add into down/out projection epilogue
5. All-reduce + residual + norm (TP)
6. LM head + softmax/top-k/argmax sampling

Each fusion saves one activation round-trip (`2·B·D·b_a` bytes) and one launch. At small B the
launch matters more than the bytes; under graphs the bytes and kernel tails matter.

## E. Kernel quality (overhead-removing)

- **Skinny GEMM / GEMV:** target ≥ 85–90% of achievable BW for the decode shapes. Split-K or
  stream-K to fill all SMs; vectorized loads; avoid wave quantization; for B ≤ 16 a custom
  GEMV often beats library GEMM.
- **Decode attention:** FlashDecoding with split-KV, GQA-aware (load each KV head once per
  group), paged-KV gather fused, FP8 KV dequant fused.
- **Prefill attention:** FlashAttention-3 class (warp specialization, TMA, async WGMMA on
  Hopper; tcgen05 on Blackwell); target 70%+ of tensor peak.
- **MoE:** grouped GEMM, fused permute/unpermute, balanced tiling for skewed expert loads.
- Always compare against the best library implementation first (cuBLASLt, CUTLASS,
  FlashInfer, FlashAttention, Triton autotuned). Writing a custom kernel that is slower than
  the library is the most common rabbit hole.

## F. Speculative decoding (floor-lowering via temporal amortization)

- Options: small draft model, EAGLE-style feature-level draft heads, Medusa heads, n-gram /
  prompt-lookup (free for code and RAG).
- Measure `α` and `E[tokens]` on the **contract workload**; acceptance on toy prompts lies.
- Must remain lossless (exact rejection sampling) unless the contract allows otherwise.
- Interacts with batching: at large B the verify step becomes compute-bound and speculation
  stops paying. Model it: enable only below a batch threshold.

## G. Parallelism

- TP all-reduce: custom one-shot/two-shot all-reduce for messages < 1 MB over NVLink; fuse
  with residual/norm; overlap with next GEMM.
- Avoid TP across PCIe for latency-critical decode unless unavoidable.
- EP for large MoE; balance with redundant experts for hot experts.

## H. What not to do

- Do not chase prefill when decode is 90% of GPU time (or vice versa).
- Do not optimize a kernel that is already at ≥ 90% of its own roofline.
- Do not trade correctness for speed outside the contract.
- Do not replace a tuned library kernel without a measured gap on the real shapes.
