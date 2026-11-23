# 02: Hardware Characterization

Datasheets give the **theoretical** ceiling. The loop targets the **achievable** ceiling,
measured on the exact machine. Both go in the SOL report.

## 1. Collect the facts

```bash
nvidia-smi -q | egrep -i "product name|max clocks|power limit|memory total|bus|link"
nvidia-smi topo -m                  # NVLink / PCIe topology
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE
nvcc --version; python -c "import torch;print(torch.__version__, torch.version.cuda)"
# AMD
rocm-smi --showproductname --showmeminfo vram --showpower --showclocks
rocminfo | grep -i gfx
```

Record: exact SKU (SXM vs PCIe vs NVL matters: different TDP, clocks and bandwidth), memory
size, power cap, application clocks, driver, toolkit, ECC state, MIG/MPS state, and the
interconnect topology.

## 2. Datasheet peaks

`assets/hardware.json` lists dense (non-sparse) peaks for common parts. Always:

- Use **dense** numbers. "With sparsity" figures assume 2:4 structured sparsity you almost
  certainly do not have.
- Use the peak for the **compute dtype and accumulate dtype** the kernels actually use.
  Consumer GeForce parts halve tensor throughput with FP32 accumulation.
- Check the SKU's power envelope: power-capped parts do not sustain boost clocks under GEMM.

## 3. Microbenchmarks (have Codex write them; run them yourself)

Each microbenchmark must: warm up, run ≥ 20 iterations timed with device events, synchronize
correctly, report median and min, and print the clock observed during the run.

| Ceiling | Method | Typical achievable |
|---|---|---|
| HBM read BW | read-only reduction over ≥ 4 GB buffer, vectorized 128-bit loads | 88–93% of peak |
| HBM copy BW | device-to-device copy (count read+write bytes) | 85–92% |
| Tensor FLOP/s | square GEMM M=N=K=8192 (and the model's real shapes) via cuBLASLt / hipBLASLt | 70–90% |
| Skinny GEMM BW | the actual decode shapes `[B, D] × [D, D_out]` for B ∈ {1..256} | report as achieved BW |
| L2 BW / size effects | sweep buffer sizes across L2 capacity | find the cliff |
| Launch latency | empty kernel, back-to-back; graph replay of N empty kernels | µs per launch |
| All-reduce | `nccl-tests all_reduce_perf -b 8K -e 256M -f 2 -g <p>` | latency at decode sizes |
| Host↔device | pinned H2D/D2H | relevant for offload only |

The **skinny GEMM sweep** is the most important one for decode: it tells you how much of HBM
bandwidth the GEMV/skinny-GEMM path can actually reach at each batch size. That curve, not the
datasheet, is the practical decode ceiling.

Record results in `sre/hardware_measured.json`:

```json
{
  "gpu": "H100-SXM5-80GB",
  "hbm_bw_Bps": 3.05e12,
  "tensor_flops": {"bf16": 7.6e14, "fp8": 1.45e15},
  "launch_us": 2.6,
  "graph_replay_us_per_kernel": 0.9,
  "allreduce_us": {"131072": 9.5, "1048576": 14.0},
  "clock_mhz_under_gemm": 1755,
  "power_limit_w": 700
}
```

`--measured sre/hardware_measured.json` supplies `hbm_bw_Bps`, `tensor_flops`, and optional
`link_bw_Bps` overrides for the achievable SOL. The other fields preserve measurement context.
To model measured fixed overhead or collective latency, pass `--fixed-us` and
`--allreduce-latency-us` explicitly for the chosen operating point.

## 4. Stability controls

- Lock clocks if permitted (`nvidia-smi -lgc <min>,<max>`, `-lmc`) for low-noise A/B tests,
  but report final results at default clocks too.
- Check thermals and power throttling (`nvidia-smi dmon -s pucvmet`) during long runs.
- Ensure no other process is on the GPU (`nvidia-smi --query-compute-apps=pid --format=csv`).
- Disable MPS/MIG unless they are part of the deployment.

## 5. Apple / other accelerators

The same method applies to any accelerator: find the memory bandwidth and matrix throughput
achievable from your kernel language (Metal, Triton, Pallas, …), then run the same SOL
analysis. Unified-memory systems have one bandwidth number shared by CPU and GPU; account for
contention.
