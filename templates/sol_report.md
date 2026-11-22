# Speed-of-Light Report: <model> on <GPU xN>

_Date: <YYYY-MM-DD> · Contract: `sre/contract.yaml` @ <hash>_

## 1. Inputs
| | |
|---|---|
| Model | <name>, total <X>B params, active <Y>B/token, <dense/MoE>, <MHA/GQA/MLA> |
| Precision | weights <..>, KV <..>, compute <..> |
| Hardware (theoretical) | BW <..> TB/s, P <..> TFLOP/s (<dtype>), ridge <..> FLOP/B |
| Hardware (achievable, measured) | BW <..> TB/s (<..>%), P <..> TFLOP/s (<..>%), launch <..> µs |
| Workload | prompt p50/p90 <..>, gen p50/p90 <..>, TPOT SLO <..> ms |

## 2. Bounds
| Phase / batch | Bound | Bytes | FLOPs | Theoretical | Achievable |
|---|---|---|---|---|---|
| Prefill S=<S> | compute | | | <ms> | <ms> |
| Decode B=1 | memory | | | <tok/s> | <tok/s> |
| Decode B=<B_op> | <..> | | | <tok/s> | <tok/s> |

- Critical batch B*: <..>
- KV-capacity max batch: <..>
- SLO-feasible max batch: <..>
- TP communication floor: <..> ms/step (if p > 1)

## 3. Information-theoretic floor per token
- Weight bytes per token at B_op: <..> (rate <..> bits/param)
- KV bytes per token at ctx <..>: <..>
- Which term dominates: <..>. Floor-lowering options within contract: <..>

## 4. SOL statement
> For <model> on <GPU> under <contract>, decode at B=<..>, ctx=<..> is <..>-bound; theoretical
> SOL is <..> tok/s, achievable SOL is <..> tok/s; the current engine reaches <..> tok/s =
> <..>% of achievable SOL; the largest gap term is <..>.

## 5. Target
<target>% of achievable SOL = <value>. Rationale: <..>
