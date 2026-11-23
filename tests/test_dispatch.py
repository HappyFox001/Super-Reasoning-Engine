"""Exercise the Codex wrappers without making an authenticated model call."""

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SESSION = "123e4567-e89b-12d3-a456-426614174000"
BYPASS = "--dangerously-bypass-approvals-and-sandbox"
FAKE_CODEX = '''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

args = sys.argv[1:]
session = "123e4567-e89b-12d3-a456-426614174000"
print("diagnostic preceding the JSON event", flush=True)
print(json.dumps({"type": "item.completed", "thread_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"}), flush=True)
if not os.environ.get("FAKE_NO_SESSION"):
    print(json.dumps({"type": "thread.started", "thread_id": session}), flush=True)
if os.environ.get("FAKE_SIGNAL_PROBE"):
    child_code = """import signal, time, sys
from pathlib import Path
def stop(signum, _frame):
    Path("child-stopped").write_text(str(signum))
    sys.exit(128 + signum)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
Path("child-ready").touch()
time.sleep(30)
"""
    child = subprocess.Popen([sys.executable, "-c", child_code])

    def stop(signum, _frame):
        child.wait(timeout=1)
        Path("worker-stopped").write_text(str(signum))
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not Path("child-ready").exists():
        time.sleep(0.01)
    Path("worker-ready").write_text(str(os.getpid()))
    time.sleep(30)
if os.environ.get("FAKE_SLOW"):
    time.sleep(30)
prompt = sys.stdin.read()
assert "--json" in args and args[-1] == "-"
if args[:2] == ["exec", "resume"]:
    assert args[2] == session
else:
    assert Path(args[args.index("-C") + 1]) == Path.cwd()

if not os.environ.get("FAKE_NO_WRITES"):
    with Path("worker.txt").open("a") as target:
        target.write(prompt)
    Path("new 'file'.txt").write_text(prompt)

out = Path(args[args.index("-o") + 1])
assert out.is_absolute()
out.write_text(json.dumps({"argv": args, "cwd": str(Path.cwd()), "prompt": prompt}))
print("worker stderr", file=sys.stderr)
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
'''


class DispatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        self.repo = root / """engine "double" 'single'"""
        self.caller = root / "other 'working' directory"
        self.repo.mkdir()
        self.caller.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "SRE Test")
        self.git("config", "user.email", "sre-test@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        (self.repo / "worker.txt").write_text("baseline\n", encoding="utf-8")
        (self.repo / "user.txt").write_text("user baseline\n", encoding="utf-8")
        self.git("add", "worker.txt", "user.txt")
        self.git("commit", "-qm", "baseline")
        self.base = self.git("rev-parse", "HEAD")
        self.index = self.repo / ".git" / "index"

        self.task = self.caller / """task "quoted".md"""
        self.task.write_text("run one measurable hypothesis\n", encoding="utf-8")
        self.message = self.caller / "correction 'quoted'.md"
        self.message.write_text("correct the original task\n", encoding="utf-8")
        self.fake = self.caller / """fake "codex" """
        self.fake.write_text(FAKE_CODEX, encoding="utf-8")
        self.fake.chmod(0o755)
        self.env = {
            key: value for key, value in os.environ.items() if not key.startswith("SRE_CODEX_")
        }
        self.env["SRE_CODEX_BIN"] = f"./{self.fake.name}"

    def git(self, *args):
        result = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()

    def call(self, script, *args, env=None, code=0):
        result = subprocess.run(
            [str(SCRIPTS / script), *map(str, args)],
            cwd=self.caller,
            env={**self.env, **(env or {})},
            text=True,
            capture_output=True,
            timeout=12,
        )
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return result

    def dispatch(self, *args, env=None, code=0, runs_dir="run records"):
        return self.call(
            "codex_task.sh",
            "--repo", os.path.relpath(self.repo, self.caller),
            "--task", self.task.name,
            "--runs-dir", runs_dir,
            *args,
            env=env,
            code=code,
        )

    def correct(self, run, *args, env=None, code=0):
        return self.call(
            "codex_correct.sh",
            "--run", os.path.relpath(run, self.caller),
            "--message", self.message.name,
            *args,
            env=env,
            code=code,
        )

    def run_dir(self, result):
        key, directory = result.stdout.splitlines()[-1].split("=", 1)
        self.assertEqual(key, "RUN_DIR")
        path = Path(directory)
        self.assertTrue(path.is_absolute())
        return path

    def read_json(self, run, name):
        return json.loads((run / name).read_text(encoding="utf-8"))

    def test_relative_dispatch_and_followup_preserve_git_index(self):
        user_file = self.repo / "user.txt"
        user_file.write_text("user staged change\n", encoding="utf-8")
        self.git("add", "user.txt")
        original_index = self.index.read_bytes()

        custom_dir = """logs "double" 'single'"""
        result = self.dispatch("--model", """fake-"model" """, "--effort", "high", runs_dir=custom_dir)
        run = self.run_dir(result)

        self.assertEqual(run.parent, self.caller / custom_dir)
        self.assertEqual((run / "task.md").read_bytes(), self.task.read_bytes())
        self.assertEqual((run / "session_id").read_text(encoding="utf-8").strip(), SESSION)
        self.assertEqual((run / "stderr.log").read_text(encoding="utf-8"), "worker stderr\n")
        meta = self.read_json(run, "meta.json")
        report = self.read_json(run, "last_message.md")
        self.assertEqual(meta["repo"], str(self.repo))
        self.assertEqual(meta["model"], """fake-"model" """)
        self.assertEqual(meta["sandbox"], "workspace-write")
        self.assertEqual(meta["base_commit"], self.base)
        self.assertEqual(report["cwd"], str(self.repo))
        self.assertEqual(report["prompt"], self.task.read_text(encoding="utf-8"))
        self.assertIn("-s", report["argv"])
        self.assertIn("workspace-write", report["argv"])
        self.assertNotIn(BYPASS, report["argv"])
        self.assertEqual(self.index.read_bytes(), original_index)
        self.assertEqual(self.read_json(run, "untracked.json"), ["new 'file'.txt"])
        self.assertIn("+run one measurable hypothesis", (run / "diff.patch").read_text(encoding="utf-8"))

        self.git("commit", "--only", "-qm", "worker checkpoint", "--", "worker.txt")
        self.assertNotEqual(self.git("rev-parse", "HEAD"), self.base)
        original_index = self.index.read_bytes()

        followup = self.run_dir(self.correct(run))
        self.assertEqual(followup, run / "followup-01")
        self.assertEqual((followup / "message.md").read_bytes(), self.message.read_bytes())
        report = self.read_json(followup, "last_message.md")
        meta = self.read_json(followup, "meta.json")
        self.assertEqual(report["cwd"], str(self.repo))
        self.assertEqual(report["prompt"], self.message.read_text(encoding="utf-8"))
        self.assertIn('sandbox_mode="workspace-write"', report["argv"])
        self.assertEqual(meta["base_commit"], self.base)
        self.assertEqual(meta["session_id"], SESSION)
        self.assertEqual(self.index.read_bytes(), original_index)
        self.assertEqual(user_file.read_text(encoding="utf-8"), "user staged change\n")
        patch = (followup / "diff.patch").read_text(encoding="utf-8")
        self.assertIn("+run one measurable hypothesis", patch)
        self.assertIn("+correct the original task", patch)

    def test_explicit_sandbox_modes_and_resume_mismatch(self):
        for mode in ("read-only", "danger-full-access"):
            with self.subTest(mode=mode):
                env = {"FAKE_NO_WRITES": "1"}
                run = self.run_dir(self.dispatch("--sandbox", mode, env=env))
                report = self.read_json(run, "last_message.md")
                self.assertEqual(self.read_json(run, "meta.json")["sandbox"], mode)
                self.assertIn(BYPASS if mode == "danger-full-access" else mode, report["argv"])
                result = self.correct(run, env={"SRE_CODEX_SANDBOX": "workspace-write"}, code=78)
                self.assertIn("start a new task", result.stderr)
                self.assertFalse((run / "followup-01").exists())

                followup = self.run_dir(self.correct(run, env=env))
                self.assertEqual(self.read_json(followup, "meta.json")["sandbox"], mode)
                report = self.read_json(followup, "last_message.md")
                expected = BYPASS if mode == "danger-full-access" else 'sandbox_mode="read-only"'
                self.assertIn(expected, report["argv"])

    def test_missing_and_invalid_session_is_actionable(self):
        result = self.dispatch(env={"FAKE_NO_SESSION": "1", "FAKE_NO_WRITES": "1"}, code=65)
        run = self.run_dir(result)
        self.assertIn("no valid Codex session ID", result.stderr)
        self.assertEqual(self.read_json(run, "meta.json")["session_id"], "")
        session = run / "session_id"
        session.unlink()
        self.assertIn("missing Codex session ID", self.correct(run, code=65).stderr)
        session.write_text("--last\n", encoding="utf-8")
        self.assertIn("invalid Codex session ID", self.correct(run, code=65).stderr)
        self.assertFalse((run / "followup-01").exists())

    def test_worker_error_preserves_logs_and_exit(self):
        run = self.run_dir(self.dispatch(env={"FAKE_EXIT": "23"}, code=23))
        self.assertEqual(self.read_json(run, "meta.json")["exit_code"], 23)
        self.assertIn("worker stderr", (run / "stderr.log").read_text(encoding="utf-8"))
        self.assertIn("thread.started", (run / "events.jsonl").read_text(encoding="utf-8"))

    def test_timeout_does_not_require_coreutils(self):
        result = self.dispatch("--timeout-min", "0.02", env={"FAKE_SLOW": "1"}, code=124)
        run = self.run_dir(result)
        self.assertIn("TIMEOUT", result.stderr)
        self.assertEqual(self.read_json(run, "meta.json")["exit_code"], 124)
        self.assertEqual(self.read_json(run, "meta.json")["session_id"], SESSION)

    def test_wrapper_signals_stop_timed_worker_and_child(self):
        run = self.run_dir(self.dispatch(env={"FAKE_NO_WRITES": "1"}))
        commands = {
            signal.SIGTERM: [
                str(SCRIPTS / "codex_task.sh"),
                "--repo", str(self.repo), "--task", str(self.task),
                "--runs-dir", "signal records",
            ],
            signal.SIGINT: [
                str(SCRIPTS / "codex_correct.sh"),
                "--run", str(run), "--message", str(self.message),
            ],
        }
        ready = self.repo / "worker-ready"
        for signum, command in commands.items():
            with self.subTest(signal=signum):
                for name in ("child-ready", "worker-ready", "child-stopped", "worker-stopped"):
                    (self.repo / name).unlink(missing_ok=True)
                process = subprocess.Popen(
                    [*command, "--timeout-min", "1"],
                    cwd=self.caller,
                    env={**self.env, "FAKE_SIGNAL_PROBE": "1"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                try:
                    deadline = time.monotonic() + 5
                    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(ready.exists(), "fake worker never became ready")
                    process.send_signal(signum)
                    stdout, stderr = process.communicate(timeout=6)
                    self.assertEqual(process.returncode, 128 + signum, stdout + stderr)
                    for name in ("worker-stopped", "child-stopped"):
                        self.assertEqual((self.repo / name).read_text(), str(int(signum)))
                finally:
                    if process.poll() is None:
                        process.kill()
                        if ready.exists():
                            try:
                                os.killpg(int(ready.read_text()), signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        process.communicate(timeout=5)
        self.assertFalse((run / ".followup.lock").exists())


if __name__ == "__main__":
    unittest.main()
