"""Measure two prefill paths on the same fixed SmolLM workload."""

import argparse
import hashlib
import json
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step, generation_stream
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

MODEL_ID = "mlx-community/SmolLM-135M-Instruct-4bit"
MODEL_REVISION = "642e06afe3fab57fd6cc518637c471af0a569e1e"
ENGINE_REVISION = "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd"
PREFIX_TOKENS = 1024
OUTPUT_TOKENS = 32
WARMUP_REPEATS = 3
REPEATS = 7
PREFIX_TEXT = (
    "The station log records blue wind vanes, silver rain gauges, and a red lantern. "
    "Each reading includes the place, an hour, and the observer on duty. "
)
SUFFIXES = (
    "\nQuestion: What color is the lantern?\nAnswer:",
    "\nQuestion: Name two instruments in the log.\nAnswer:",
    "\nQuestion: What information goes with each reading?\nAnswer:",
)


def run_requests(model, prefix, suffixes, cache):
    requests = []
    for suffix in suffixes:
        mx.synchronize(generation_stream)
        start = time.perf_counter()
        if cache is None:
            prompt = mx.array(prefix + suffix)
        else:
            extra = cache[0].offset - len(prefix)
            if trim_prompt_cache(cache, extra) != extra:
                raise RuntimeError("Could not restore the shared prefix")
            if any(layer.offset != len(prefix) for layer in cache):
                raise RuntimeError("Layer caches disagree on shared-prefix length")
            prompt = mx.array(suffix)

        generated = generate_step(prompt, model, max_tokens=OUTPUT_TOKENS, prompt_cache=cache)
        first_token, _ = next(generated)
        first_token_time = time.perf_counter()
        tokens = [first_token, *(token for token, _ in generated)]
        mx.synchronize(generation_stream)
        end = time.perf_counter()
        if len(tokens) != OUTPUT_TOKENS:
            raise RuntimeError("Unexpected generated token count")
        requests.append(
            {
                "ttft_ms": round((first_token_time - start) * 1_000, 6),
                "request_ms": round((end - start) * 1_000, 6),
                "tokens": tokens,
            }
        )
    return requests


def summarize(repetitions, mode):
    samples = [request for repetition in repetitions for request in repetition[mode]]
    return {
        f"{metric}_p50": round(statistics.median(sample[metric] for sample in samples), 4)
        for metric in ("ttft_ms", "request_ms")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise SystemExit("An Apple Silicon Metal GPU is required")
    manifest = Path(__file__).with_name("model.sha256").read_text(encoding="utf-8")
    for line in manifest.splitlines():
        expected, filename = line.split()
        with (args.model_dir / filename).open("rb") as model_file:
            actual = hashlib.file_digest(model_file, "sha256").hexdigest()
        if actual != expected:
            raise SystemExit(f"Model checksum mismatch: {filename}")

    mx.set_default_device(mx.gpu)
    model, tokenizer = load(args.model_dir, tokenizer_config={"trust_remote_code": False})
    prefix = tokenizer.encode(
        PREFIX_TEXT * 64,
        add_special_tokens=False,
        truncation=True,
        max_length=PREFIX_TOKENS,
    )
    suffixes = [tokenizer.encode(suffix, add_special_tokens=False) for suffix in SUFFIXES]
    if len(prefix) != PREFIX_TOKENS or not all(suffixes):
        raise RuntimeError("The pinned tokenizer did not produce the expected context")

    mx.synchronize(generation_stream)
    cache_started = time.perf_counter()
    cache = make_prompt_cache(model)
    with mx.stream(generation_stream):
        model(mx.array(prefix)[None], cache=cache)
        mx.eval([layer.state for layer in cache])
    mx.synchronize(generation_stream)
    cache_prime_ms = round((time.perf_counter() - cache_started) * 1_000, 6)

    reference = None
    for _ in range(WARMUP_REPEATS):
        baseline_tokens = [sample["tokens"] for sample in run_requests(model, prefix, suffixes, None)]
        cached_tokens = [sample["tokens"] for sample in run_requests(model, prefix, suffixes, cache)]
        if cached_tokens != baseline_tokens or (reference is not None and baseline_tokens != reference):
            raise RuntimeError("Generated tokens changed during warm-up")
        reference = baseline_tokens

    raw = []
    for repeat in range(REPEATS):
        order = ["baseline", "prefix_cache"] if repeat % 2 == 0 else ["prefix_cache", "baseline"]
        repetition = {"order": order}
        for mode in order:
            active_cache = None if mode == "baseline" else cache
            requests = run_requests(model, prefix, suffixes, active_cache)
            if [request["tokens"] for request in requests] != reference:
                raise RuntimeError(f"{mode} changed tokens in repeat {repeat + 1}")
            for request in requests:
                del request["tokens"]
            repetition[mode] = requests
        raw.append(repetition)

    device = mx.device_info()
    results = {
        "hardware": {
            "architecture": platform.machine(),
            "macos": platform.mac_ver()[0],
            "gpu": device.get("device_name"),
            "gpu_architecture": device.get("architecture"),
            "unified_memory_bytes": device.get("memory_size"),
        },
        "software": {
            "python": platform.python_version(),
            "mlx": version("mlx"),
            "mlx_lm": version("mlx-lm"),
            "mlx_lm_source_revision": ENGINE_REVISION,
            "transformers": version("transformers"),
        },
        "workload": {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "weight_bits": 4,
            "sampling": "greedy, fixed length",
            "batch_size": 1,
            "prefix_tokens": len(prefix),
            "suffix_token_counts": [len(suffix) for suffix in suffixes],
            "output_tokens_per_request": OUTPUT_TOKENS,
            "cache_prime_ms": cache_prime_ms,
            "warmup_repetitions": WARMUP_REPEATS,
            "repetitions": REPEATS,
        },
        "quality": {
            "exact_candidate_token_match": True,
            "candidate_tokens_compared": REPEATS * len(suffixes) * OUTPUT_TOKENS,
            "reference_token_sequences": reference,
            "reference_sha256": hashlib.sha256(json.dumps(reference).encode()).hexdigest(),
        },
        "summary": {mode: summarize(raw, mode) for mode in ("baseline", "prefix_cache")},
        "raw": raw,
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
