# sre/STATE.md: loop state (survives context resets; paste relevant parts into every task)

## Current
- Best accepted: <value> = <X>% of achievable SOL @ commit <sha>
- Target: <Y>% · Iteration <n> / budget <N>
- Active Codex run: <sre/runs/...> (session <id>) | none

## Gap waterfall (latest)
| Term | ms/step | % of step | Max payoff |
|---|---|---|---|
| SOL | | | |
| GEMM inefficiency | | | |
| Attention inefficiency | | | |
| Norm/act/elementwise | | | |
| Sampling / LM head | | | |
| Launch / idle gaps | | | |
| Host stalls | | | |

## Hypothesis queue (largest expected gain first)
1. <hypothesis> (term, predicted gain, risk)

## Banned approaches
- <approach>: <evidence/reason> (iteration <n>)

## Open questions for the user
- <contract changes that would unlock more SOL, with the math>
