# Changelog

## v0.1.0

- Added project and personal installation instructions for Codex and Claude Code.
  Documented native Codex orchestration and the local CLI worker workflow.
- Added a no-GPU calculator example and a measured Apple M2 and MLX inference case.
- Hardened acceptance and recovery. The guard checks new files without staging them,
  and the ledger rejects invalid acceptances before writing them.
- Fixed CLI wrapper path handling, retained the initial Git base across follow-ups,
  and added portable timeouts. The CLI worker now defaults to `workspace-write`.
  Use `--sandbox danger-full-access` when explicitly authorized on an isolated runner.
- Added Linux and macOS regression testing and live-client skill selection prompts.
