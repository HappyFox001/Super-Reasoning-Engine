# Skill selection and behavior checks

Use fresh sessions with the skill installed and no skill name in the prompt. For a positive
case, confirm the host loads this skill in its activity trace or transcript and verify the
expected action. For a negative case, confirm the skill stays unloaded. Repeat in Codex and
Claude Code when testing a new client version. Record the client version, invocation mode, and
observed result.

| Prompt | Expectation |
|---|---|
| Estimate the theoretical decode throughput of my HF model on an H100 SXM at BF16, with batch 32 and a 2,048-token prompt. | Load SRE and use the supplied model config or request it. Produce an estimate with the operating point. |
| Profile the decode bottleneck in our inference engine and supervise one kernel experiment with a 30 ms TPOT bound. | Load SRE. Establish the workload and baseline, dispatch one scoped hypothesis, and require independent correctness and benchmark evidence. |
| The worker reports 101% of its modeled SOL and changed the frozen timing harness. Decide whether to accept the patch. | Load SRE in the existing optimization context. Block acceptance and record the guard and measurement findings. |
| Help me write a faster solution to this reasoning puzzle. | Use ordinary reasoning without activating SRE. |
| Profile a slow training loop and reduce time to convergence. | Use the training workflow without activating SRE. |

To isolate explicit discovery from automatic selection, repeat the first prompt through the
skill picker or named invocation. The automated suite checks scripts without a GPU or model
call. The prompts above check host behavior with live agents.
