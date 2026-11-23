# Estimate inference ceilings without a GPU

This example uses a synthetic 7.24-billion-parameter decoder with grouped-query attention and the bundled H100 SXM hardware entry. The calculator needs the model dimensions and hardware values. It runs locally with Python's standard library and uses no model weights.

Run from the repository root:

```bash
python3 scripts/sol_calculator.py \
  --model examples/sol-quickstart/config.json \
  --gpu H100-SXM5-80GB \
  --weight-bits 16 --kv-bits 16 --compute-dtype bf16 --tp 1 \
  --prompt-len 2048 --gen-len 512 --batch-sweep 1,32,64,128 \
  --tpot-slo-ms 12 --json /tmp/sre-example-sol.json
```

| Workload | Binding resource | Theoretical latency floor | Theoretical throughput ceiling |
| --- | --- | --- | --- |
| Prefill, 2,048 tokens | Compute | 30.55 ms per prompt | 67.04k tokens/s |
| Decode, batch 1 | Memory | 4.335 ms/token | 231 tokens/s |
| Decode, batch 64 | Memory | 10.014 ms/token | 6,391 tokens/s |
| Decode, max batch under 12 ms TPOT | Memory | 12.00 ms/token | 7,168 tokens/s at batch 86 |

Decode uses an average context of 2,304 tokens. The calculator estimates 14.22 GB of weights and 301.99 MB of KV reads per step at batch 1. Increasing the batch amortizes the weight reads, but grows the KV traffic. The capacity estimate allows batch 165 at maximum context, while the 12 ms TPOT constraint lowers the modeled operating point to batch 86.

These are calculated bounds. Estimate the achievable ceiling for a real machine by passing its measured bandwidth and compute throughput with `--measured`. Use the frozen workload and benchmark to measure the engine. [Hardware characterization](../../references/02-hardware-characterization.md)
