<p align="center">
  <img src="assets/logo.svg" width="112" alt="Super Reasoning Engine logo: a roofline meeting its ceiling">
</p>

<h1 align="center">Super Reasoning Engine</h1>

<p align="center">
  <b>Speed-of-light inference engineering as an agent skill.</b><br>
  Derive the physical ceiling of <i>your</i> model on <i>your</i> GPU, then let a supervised Codex loop close the gap.
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#the-theory-in-one-page">Theory</a> ·
  <a href="#the-codex-loop">Codex loop</a> ·
  <a href="#repository-layout">Layout</a>
</p>

<p align="center">
  <a href="promo/super-reasoning-engine-promo.mp4">
    <img src="assets/social-preview.png" width="720" alt="Watch the 30-second Super Reasoning Engine promo">
  </a>
  <br>
  <sub>▶ <a href="promo/super-reasoning-engine-promo.mp4">Watch the 30-second promo</a> (with sound): a live run on laya-mlx, Apple M2</sub>
</p>

---

Most inference optimization is folklore: try a flag, try a kernel, see what happens.
**Super Reasoning Engine** (SRE) replaces folklore with a contract and a ceiling:

1. **A ceiling.** Before writing any code, SRE derives the *Speed-of-Light* (SOL) of the
   model × hardware × workload triple from the roofline model, information-theoretic
   bytes-per-token bounds and Little's law, using datasheet peaks *and* measured ceilings.
2. **A gap.** It measures the current engine against that ceiling and decomposes the
   difference into a waterfall: kernel inefficiency, launch gaps, host overhead, batching.
3. **A relentless loop.** It drives the local [Codex CLI](https://github.com/openai/codex)
   as an implementation engineer, one hypothesis per iteration, while the orchestrating
   agent (Claude) owns the benchmark, the correctness check and the verdict. When Codex drifts
   (gaming the benchmark, breaking the precision contract, falling into rabbit holes), SRE
   detects it from evidence and corrects it in the same session, then keeps going.

The output is an inference engine whose performance is stated as **a percentage of physics**,
not as a speedup over some arbitrary baseline.

## Install

SRE follows the open [Agent Skills](https://agentskills.io) layout: a `SKILL.md` with
frontmatter, plus `references/`, `scripts/`, `templates/` and `assets/` loaded on demand.

```bash
# Claude Code: personal skill
git clone https://github.com/HappyFox001/Super-Reasoning-Engine.git ~/.claude/skills/super-reasoning-engine

# or project-scoped
git clone https://github.com/HappyFox001/Super-Reasoning-Engine.git .claude/skills/super-reasoning-engine
```

**Requirements**

| Component | Needed for |
|---|---|
| Python ≥ 3.9 (stdlib only) | `sol_calculator.py`, `ledger.py`, `guard.py` |
| [Codex CLI](https://github.com/openai/codex), logged in | the optimization loop (`codex exec`) |
| git | guard, revert and attribution |
| CUDA/ROCm toolchain + profiler on the GPU host | building and measuring the engine |

Codex runs **without a sandbox by default** so it can compile and time GPU kernels itself (Codex's
sandboxes hide the GPU). SRE compensates with a dedicated branch, a diff guard and independent
measurement every iteration. Set `SRE_CODEX_SANDBOX=workspace-write` to keep the sandbox.

Pick the Codex model with `SRE_CODEX_MODEL` (for example an Astra-class model such as
`gpt-6.2-astra`). If it isn't set, SRE uses the model in your `~/.codex/config.toml`.

Then just ask your agent:

> *"Use super-reasoning-engine to build the fastest possible decode path for Qwen3-32B on one H100 SXM, FP8 weights allowed, TPOT ≤ 30 ms."*

## How it works

```
 ┌──────────────┐   ┌───────────────────┐   ┌─────────────────┐   ┌────────────────────────┐
 │ A. Contract  │──►│ B. Hardware truth │──►│ C. Speed-of-    │──►│ D. Baseline + gap      │
 │ model, dtype,│   │ datasheet peaks + │   │    Light (SOL)  │   │    waterfall (profile) │
 │ SLO, quality │   │ measured ceilings │   │ sol_calculator  │   │ ledger init            │
 └──────────────┘   └───────────────────┘   └─────────────────┘   └───────────┬────────────┘
                                                                              │
          ┌───────────────────────────────────────────────────────────────────▼──────────┐
          │ E. Codex loop:  PLAN → DISPATCH (codex exec) → GUARD → VERIFY → PHYSICS CHECK │
          │                 → VERDICT (accept / reject / correct) → NEXT, until F         │
          └───────────────────────────────────────────────────────────────────┬──────────┘
                                                                              │
                                          ┌───────────────────────────────────▼──────────┐
                                          │ F. Stop: target % of SOL, asymptote,          │
                                          │    contract limit, or budget → REPORT.md      │
                                          └──────────────────────────────────────────────┘
```

## The theory in one page

The binding question: *what is the minimum time to produce one token?*

```
t_step ≥ max( FLOPs / P_compute , Bytes / BW_mem , Comm ) + t_fixed
```

| Regime | Dominant term | Bound |
|---|---|---|
| Prefill (long prompts) | compute | `t ≥ (2·N·S + 2·L·S²·H·d) / P` |
| Decode, small batch | memory | `t ≥ (W_active·b_w + B·ctx·KV_tok) / BW` |
| Decode, large batch | compute | above `B* ≈ (P/BW)·(b_w/2)`: about 295 for BF16 on H100 |
| Tensor parallel | interconnect | `2·(p−1)/p · msg / BW_link + hops·α` per all-reduce |

**The information-theoretic view.** A decode step must move at least the information content
of the active weights and the attended KV cache, at the distortion the quality contract
allows. Only three things change that floor:

- **Rate:** fewer bits per weight or KV element (quantization; rate-distortion trade-off).
- **Reuse across sequences:** batching, i.e. the same bytes serve `B` tokens.
- **Reuse across time:** speculative decoding, i.e. the same bytes serve
  `(1−α^{k+1})/(1−α)` tokens in expectation, where `α` is driven by how predictable (low
  entropy) the target distribution is relative to the draft.

Everything else (fusion, graphs, better kernels, scheduling) removes *overhead above* the
floor. SRE always tells you which of the two kinds of work you are doing.

Full derivations are in [`references/01-first-principles.md`](references/01-first-principles.md).

```bash
python3 scripts/sol_calculator.py --model config.json --gpu H100-SXM5-80GB \
  --weight-bits 8 --kv-bits 8 --prompt-len 2048 --gen-len 512 \
  --batch-sweep 1,16,64,128,256 --tpot-slo-ms 30
```

## The Codex loop

The orchestrator never lets Codex grade its own work.

| Step | Who | Tool |
|---|---|---|
| Plan one hypothesis with a predicted gain | Claude | gap waterfall, `ledger.py show` |
| Implement | **Codex** | `scripts/codex_task.sh` → `codex exec --json` |
| Guard the diff | Claude | `scripts/guard.py` (protected bench/check, scope, red flags) |
| Correctness + authoritative benchmark | Claude | frozen `sre/check`, `sre/bench` |
| Physics check | Claude | `ledger.py add` (flags > SOL, noise-level wins, regressions) |
| Correct drift in the same session | Claude → Codex | `scripts/codex_correct.sh` → `codex exec resume` |

Drift patterns SRE detects and corrects: benchmark gaming, precision-contract violations,
impossible (> SOL) numbers, rabbit holes, scope creep, fix-forward spirals, premature victory
claims and environment thrash. See [`references/06-drift-detection.md`](references/06-drift-detection.md).

## Repository layout

```
SKILL.md                         entry point the agent loads
references/
  01-first-principles.md         roofline, information-theoretic bounds, queueing
  02-hardware-characterization.md datasheet vs achievable, microbenchmarks
  03-workload-model.md           prefill/decode, KV, GQA/MLA, MoE, long context, multi-GPU
  04-optimization-playbook.md    lever catalog, expected gains, pitfalls
  05-codex-orchestration.md      driving the Codex CLI: sessions, prompts, sandbox
  06-drift-detection.md          drift taxonomy and correction playbook
  07-measurement-protocol.md     benchmark rigor and statistics
scripts/
  sol_calculator.py              speed-of-light calculator (reads HF config.json)
  codex_task.sh                  dispatch a task to Codex, log session
  codex_correct.sh               resume the same Codex session with a correction
  guard.py                       post-Codex diff guard
  ledger.py                      experiment ledger with SOL-fraction and noise checks
templates/                       contract, AGENTS.md, task, correction, SOL report, final report
assets/hardware.json             datasheet peaks for common accelerators
```

## Design principles

- **State performance as a fraction of physics.** "2× faster" is meaningless; "88% of
  achievable HBM SOL at B=64" is a fact.
- **Separate the author from the judge.** Codex writes; the orchestrator measures.
- **One hypothesis, one diff, one number.** Attribution beats velocity.
- **Freeze the workload.** The contract changes only with a human decision.
- **Never stop at the first plateau.** Re-derive the waterfall; the largest bar is the next task.

## Contributing

Contributions are welcome: hardware entries (with datasheet sources), new drift detectors,
model-family workload notes (MLA, hybrid SSM/attention, diffusion LMs), and case studies
with reproducible ledgers. Please keep `scripts/` dependency-free.

## License

MIT, see [LICENSE](LICENSE).
