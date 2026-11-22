#!/usr/bin/env python3
"""Speed-of-Light (SOL) calculator for transformer inference.

Reads a Hugging Face config.json (dense, GQA/MQA, MoE, MLA) and a hardware spec, then
reports lower bounds on prefill and decode time, upper bounds on throughput, the critical
batch, KV-capacity limits, tensor-parallel communication floors and the SLO-feasible
operating point. Standard library only.

Derivations: references/01-first-principles.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HW = os.path.join(HERE, "..", "assets", "hardware.json")


# ---------------------------------------------------------------------------- model
def _get(cfg, *keys, default=None):
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


class Model:
    def __init__(self, cfg: dict):
        if "text_config" in cfg and "hidden_size" not in cfg:
            cfg = {**cfg, **cfg["text_config"]}
        self.D = int(_get(cfg, "hidden_size", "d_model", "n_embd"))
        self.L = int(_get(cfg, "num_hidden_layers", "n_layer", "num_layers"))
        self.Hq = int(_get(cfg, "num_attention_heads", "n_head"))
        self.Hkv = int(_get(cfg, "num_key_value_heads", "num_kv_heads", default=self.Hq))
        self.dh = int(_get(cfg, "head_dim", default=self.D // self.Hq))
        self.I = int(_get(cfg, "intermediate_size", "ffn_dim", "n_inner", default=4 * self.D))
        self.V = int(_get(cfg, "vocab_size"))
        self.tied = bool(_get(cfg, "tie_word_embeddings", default=False))
        self.window = _get(cfg, "sliding_window", default=None)

        # MoE
        self.E = int(_get(cfg, "num_local_experts", "num_experts", "n_routed_experts", default=0) or 0)
        self.k = int(_get(cfg, "num_experts_per_tok", "moe_top_k", default=0) or 0)
        self.I_moe = int(_get(cfg, "moe_intermediate_size", default=self.I))
        self.n_shared = int(_get(cfg, "n_shared_experts", default=0) or 0)
        self.shared_I = int(_get(cfg, "shared_expert_intermediate_size", default=self.n_shared * self.I_moe) or 0)
        self.first_dense = int(_get(cfg, "first_k_dense_replace", default=0) or 0)
        self.moe = self.E > 0 and self.k > 0
        self.moe_layers = (self.L - self.first_dense) if self.moe else 0
        self.dense_layers = self.L - self.moe_layers

        # MLA (DeepSeek-style latent attention)
        self.kv_lora = _get(cfg, "kv_lora_rank", default=None)
        self.mla = self.kv_lora is not None
        if self.mla:
            self.kv_lora = int(self.kv_lora)
            self.q_lora = _get(cfg, "q_lora_rank", default=None)
            self.rope = int(_get(cfg, "qk_rope_head_dim", default=64))
            self.nope = int(_get(cfg, "qk_nope_head_dim", default=128))
            self.vdim = int(_get(cfg, "v_head_dim", default=128))

    # parameters -----------------------------------------------------------
    def attn_params(self) -> float:
        D, Hq, Hkv, dh = self.D, self.Hq, self.Hkv, self.dh
        if self.mla:
            qd = Hq * (self.nope + self.rope)
            q = D * qd if not self.q_lora else D * self.q_lora + self.q_lora * qd
            kv = D * (self.kv_lora + self.rope) + self.kv_lora * Hq * (self.nope + self.vdim)
            o = Hq * self.vdim * D
            return q + kv + o
        return D * Hq * dh + 2 * D * Hkv * dh + Hq * dh * D

    def dense_mlp_params(self) -> float:
        return 3 * self.D * self.I

    def expert_params(self) -> float:
        return 3 * self.D * self.I_moe

    def shared_params(self) -> float:
        return 3 * self.D * self.shared_I if self.shared_I else 0

    def embed_params(self) -> float:
        return self.V * self.D

    def lm_head_params(self) -> float:
        return self.V * self.D

    def total_params(self) -> float:
        p = self.L * self.attn_params() + self.dense_layers * self.dense_mlp_params()
        if self.moe:
            p += self.moe_layers * (self.E * self.expert_params() + self.shared_params() + self.D * self.E)
        p += self.embed_params() + (0 if self.tied else self.lm_head_params())
        return p

    def active_params(self) -> float:
        """Parameters that take part in the FLOPs of one token (excludes embedding lookup)."""
        p = self.L * self.attn_params() + self.dense_layers * self.dense_mlp_params()
        if self.moe:
            p += self.moe_layers * (self.k * self.expert_params() + self.shared_params() + self.D * self.E)
        return p + self.lm_head_params()

    def weights_read_per_step(self, B: int) -> float:
        """Distinct parameters read from HBM in one decode step with B sequences."""
        p = self.L * self.attn_params() + self.dense_layers * self.dense_mlp_params()
        if self.moe:
            touched = self.E * (1 - (1 - self.k / self.E) ** B)
            p += self.moe_layers * (touched * self.expert_params() + self.shared_params() + self.D * self.E)
        return p + self.lm_head_params()

    # attention --------------------------------------------------------------
    def kv_elems_per_token(self) -> float:
        if self.mla:
            return self.L * (self.kv_lora + self.rope)
        return 2 * self.L * self.Hkv * self.dh

    def eff_ctx(self, ctx: float) -> float:
        return min(ctx, self.window) if self.window else ctx

    def attn_flops_decode_per_token(self, ctx: float) -> float:
        c = self.eff_ctx(ctx)
        if self.mla:  # absorbed: scores over (kv_lora+rope), values over kv_lora
            return self.L * self.Hq * c * (2 * (self.kv_lora + self.rope) + 2 * self.kv_lora)
        return 4 * self.L * self.Hq * self.dh * c

    def attn_flops_prefill(self, S: float) -> float:
        d = (self.nope + self.rope + self.vdim) / 2 if self.mla else self.dh
        if self.window and S > self.window:
            return 4 * self.L * self.Hq * d * S * self.window
        return 2 * self.L * self.Hq * d * S * S  # causal: half of 4*S^2


# ---------------------------------------------------------------------------- hardware
def load_hw(name: str, path: str, measured: str | None):
    with open(path) as f:
        db = json.load(f)
    if name not in db:
        keys = ", ".join(k for k in db if not k.startswith("_"))
        sys.exit(f"unknown --gpu {name!r}; known: {keys} (or add it to {path})")
    hw = json.loads(json.dumps(db[name]))
    theo = json.loads(json.dumps(hw))
    if measured:
        with open(measured) as f:
            m = json.load(f)
        if "hbm_bw_Bps" in m:
            hw["hbm_bw_Bps"] = float(m["hbm_bw_Bps"])
        for k, v in (m.get("tensor_flops") or {}).items():
            hw["tensor_flops"][k] = float(v)
        if "link_bw_Bps" in m:
            hw["link_bw_Bps"] = float(m["link_bw_Bps"])
    return theo, hw


# ---------------------------------------------------------------------------- bounds
def step_bound(m: Model, hw: dict, a, B: int, ctx: float):
    P = hw["tensor_flops"][a.compute_dtype] * a.tp
    BW = hw["hbm_bw_Bps"] * a.tp
    bw_b, kv_b = a.weight_bits / 8, a.kv_bits / 8
    q_w = m.weights_read_per_step(B) * bw_b
    q_kv = B * m.eff_ctx(ctx) * m.kv_elems_per_token() * kv_b
    flops = B * (2 * m.active_params() + m.attn_flops_decode_per_token(ctx))
    t_mem, t_cmp = (q_w + q_kv) / BW, flops / P
    t_comm = comm_time(m, hw, a, B)
    t = max(t_mem, t_cmp) + t_comm + a.fixed_us * 1e-6
    return {
        "B": B, "ctx": ctx, "bytes_weights": q_w, "bytes_kv": q_kv, "flops": flops,
        "intensity": flops / (q_w + q_kv), "t_mem_s": t_mem, "t_compute_s": t_cmp,
        "t_comm_s": t_comm, "t_step_s": t, "tok_s": B / t, "tpot_ms": t * 1e3,
        "bound": "compute" if t_cmp > t_mem else "memory",
    }


def comm_time(m: Model, hw: dict, a, B: int) -> float:
    if a.tp <= 1:
        return 0.0
    msg = B * m.D * a.act_bits / 8
    per = 2 * (a.tp - 1) / a.tp * msg / hw["link_bw_Bps"] + a.allreduce_latency_us * 1e-6
    return 2 * m.L * per  # two all-reduces per layer


def prefill_bound(m: Model, hw: dict, a, S: int):
    P = hw["tensor_flops"][a.compute_dtype] * a.tp
    BW = hw["hbm_bw_Bps"] * a.tp
    flops = 2 * m.active_params() * S + m.attn_flops_prefill(S)
    q = m.weights_read_per_step(S) * a.weight_bits / 8 + S * m.kv_elems_per_token() * a.kv_bits / 8
    t = max(flops / P, q / BW)
    return {"S": S, "flops": flops, "bytes": q, "t_s": t, "tok_s": S / t,
            "bound": "compute" if flops / P > q / BW else "memory"}


def kv_capacity_batch(m: Model, hw: dict, a, ctx_max: float) -> int:
    usable = hw["hbm_bytes"] * a.tp * a.mem_util - m.total_params() * a.weight_bits / 8 - a.workspace_gb * 1e9 * a.tp
    per_seq = m.eff_ctx(ctx_max) * m.kv_elems_per_token() * a.kv_bits / 8
    return max(0, int(usable // per_seq)) if per_seq > 0 else 0


def critical_batch(m: Model, hw: dict, a, ctx: float) -> int | None:
    for B in range(1, 8193):
        r = step_bound(m, hw, a, B, ctx)
        if r["t_compute_s"] >= r["t_mem_s"]:
            return B
    return None


def slo_batch(m: Model, hw: dict, a, ctx: float, cap: int) -> int:
    best = 0
    for B in range(1, max(1, cap) + 1):
        if step_bound(m, hw, a, B, ctx)["tpot_ms"] <= a.tpot_slo_ms:
            best = B
        else:
            break
    return best


# ---------------------------------------------------------------------------- report
def fmt(x, unit=""):
    for div, suf in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            return f"{x / div:.2f}{suf}{unit}"
    return f"{x:.2f}{unit}"


def analyze(m, hw, a):
    ctx_avg = a.prompt_len + a.gen_len / 2
    ctx_max = a.prompt_len + a.gen_len
    batches = [int(b) for b in a.batch_sweep.split(",") if b]
    b_mem = kv_capacity_batch(m, hw, a, ctx_max)
    out = {
        "decode": [step_bound(m, hw, a, B, ctx_avg) for B in batches],
        "prefill": prefill_bound(m, hw, a, a.prompt_len),
        "critical_batch": critical_batch(m, hw, a, ctx_avg),
        "kv_capacity_batch": b_mem,
    }
    if a.tpot_slo_ms:
        b_slo = slo_batch(m, hw, a, ctx_avg, b_mem)
        out["slo_batch"] = b_slo
        if b_slo:
            op = step_bound(m, hw, a, b_slo, ctx_avg)
            out["operating_point"] = op
    else:
        b_op = max(1, b_mem)
        out["operating_point"] = step_bound(m, hw, a, b_op, ctx_avg)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF config.json path")
    ap.add_argument("--gpu", required=True, help="key in hardware.json")
    ap.add_argument("--hw-file", default=DEFAULT_HW)
    ap.add_argument("--measured", help="hardware_measured.json with achievable ceilings")
    ap.add_argument("--weight-bits", type=float, default=16, help="storage bits/weight incl. scale overhead, e.g. 4.25")
    ap.add_argument("--kv-bits", type=float, default=16)
    ap.add_argument("--act-bits", type=float, default=16)
    ap.add_argument("--compute-dtype", default="bf16", help="key of tensor_flops: bf16|fp16|fp8|int8|fp4")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--allreduce-latency-us", type=float, default=8.0)
    ap.add_argument("--fixed-us", type=float, default=0.0, help="per-step fixed overhead to include (0 = pure SOL)")
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--gen-len", type=int, default=512)
    ap.add_argument("--batch-sweep", default="1,8,32,64,128,256")
    ap.add_argument("--tpot-slo-ms", type=float, default=0.0)
    ap.add_argument("--mem-util", type=float, default=0.9)
    ap.add_argument("--workspace-gb", type=float, default=2.0, help="per-GPU activation/workspace reserve")
    ap.add_argument("--json", help="write full results here")
    a = ap.parse_args()

    with open(a.model) as f:
        m = Model(json.load(f))
    theo_hw, ach_hw = load_hw(a.gpu, a.hw_file, a.measured)
    if a.compute_dtype not in theo_hw["tensor_flops"]:
        sys.exit(f"{a.gpu} has no tensor_flops entry for {a.compute_dtype}")

    theo = analyze(m, theo_hw, a)
    ach = analyze(m, ach_hw, a) if a.measured else None

    print(f"== Model: D={m.D} L={m.L} Hq={m.Hq} Hkv={m.Hkv} dh={m.dh} V={m.V}"
          f"{' MoE E=%d k=%d' % (m.E, m.k) if m.moe else ''}{' MLA' if m.mla else ''}")
    print(f"   total params {fmt(m.total_params())}, active/token {fmt(m.active_params())}, "
          f"KV/token {fmt(m.kv_elems_per_token() * a.kv_bits / 8, 'B')}")
    print(f"== Hardware: {a.gpu} x{a.tp}  BW {fmt(theo_hw['hbm_bw_Bps'], 'B/s')}  "
          f"P[{a.compute_dtype}] {fmt(theo_hw['tensor_flops'][a.compute_dtype], 'FLOP/s')}  "
          f"ridge {theo_hw['tensor_flops'][a.compute_dtype] / theo_hw['hbm_bw_Bps']:.0f} FLOP/B")
    if ach:
        print(f"   achievable: BW {fmt(ach_hw['hbm_bw_Bps'], 'B/s')}  P {fmt(ach_hw['tensor_flops'][a.compute_dtype], 'FLOP/s')}")
    print(f"== Precision: weights {a.weight_bits} b, KV {a.kv_bits} b, compute {a.compute_dtype}")
    pf = theo["prefill"]
    print(f"\nPrefill S={a.prompt_len}: {fmt(pf['flops'], 'FLOP')}, bound {pf['bound']}, "
          f"t >= {pf['t_s'] * 1e3:.2f} ms ({fmt(pf['tok_s'])} tok/s)"
          + (f"  | achievable {ach['prefill']['t_s'] * 1e3:.2f} ms" if ach else ""))
    print(f"\nDecode (ctx avg {a.prompt_len + a.gen_len / 2:.0f}), theoretical SOL:")
    hdr = f"{'B':>6} {'bound':>8} {'W bytes':>10} {'KV bytes':>10} {'AI':>7} {'TPOT ms':>9} {'tok/s':>10}"
    if ach:
        hdr += f" {'ach tok/s':>10}"
    print(hdr)
    for i, r in enumerate(theo["decode"]):
        line = (f"{r['B']:>6} {r['bound']:>8} {fmt(r['bytes_weights'], 'B'):>10} {fmt(r['bytes_kv'], 'B'):>10} "
                f"{r['intensity']:>7.1f} {r['tpot_ms']:>9.3f} {r['tok_s']:>10.0f}")
        if ach:
            line += f" {ach['decode'][i]['tok_s']:>10.0f}"
        print(line)
    b_star_w = theo_hw["tensor_flops"][a.compute_dtype] / theo_hw["hbm_bw_Bps"] * (a.weight_bits / 8) / 2
    print(f"\nWeight-only critical batch B* = (P/BW)*(b_w/2) = {b_star_w:.0f} (ignores KV traffic)")
    cb = theo["critical_batch"]
    print(f"Critical batch incl. KV at this ctx: "
          + (str(cb) if cb else "never: KV traffic keeps decode memory-bound at this context length"))
    print(f"KV-capacity max batch at ctx {a.prompt_len + a.gen_len}: {theo['kv_capacity_batch']}")
    if a.tp > 1:
        print(f"TP all-reduce floor per step at B=1: {comm_time(m, theo_hw, a, 1) * 1e3:.3f} ms")
    if "slo_batch" in theo:
        print(f"Max batch meeting TPOT <= {a.tpot_slo_ms} ms: {theo['slo_batch']}")
    op = theo.get("operating_point")
    if op:
        msg = f"Operating point B={op['B']}: {op['bound']}-bound, TPOT {op['tpot_ms']:.2f} ms, {op['tok_s']:.0f} tok/s (theoretical)"
        if ach and ach.get("operating_point"):
            msg += f", {ach['operating_point']['tok_s']:.0f} tok/s (achievable)"
        print(msg)

    if a.json:
        res = {"args": vars(a), "model": {"total_params": m.total_params(), "active_params": m.active_params(),
                                          "kv_bytes_per_token": m.kv_elems_per_token() * a.kv_bits / 8},
               "theoretical": theo, "achievable": ach}
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
