# Prefix-cache reuse on Apple M2

The baseline runs full prefill on every request. The candidate builds a KV cache for a shared 1,024-token prefix once, then prefills only each request's 12–13 remaining prompt tokens. Both paths use the same weights, full prompt token sequence, and greedy generation of 32 tokens at batch one. The candidate resets cache offsets to the prefix boundary between requests.

The measured host was an Apple M2 with a 10-core GPU, 24 GiB unified memory and macOS 15.6. The environment used native Python 3.12.11, MLX 0.31.2, and MLX LM 0.31.3. The 4-bit SmolLM-135M-Instruct weights and the engine release are pinned. [Model revision](https://huggingface.co/mlx-community/SmolLM-135M-Instruct-4bit/tree/642e06afe3fab57fd6cc518637c471af0a569e1e) · [MLX LM source revision](https://github.com/ml-explore/mlx-lm/tree/ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd)

| Metric | Baseline median | Prefix cache median | Reduction |
| --- | ---: | ---: | ---: |
| Time to first token | 128.63 ms | 22.43 ms | 82.6% |
| Request time | 250.85 ms | 131.58 ms | 47.5% |

After three warm-up passes per path, seven timed repetitions alternated the order in which the paths ran. Each repetition contains all three prompts once per path. [Raw timing samples and reference token IDs](results.json)

- **Quality.** All 672 measured candidate token IDs matched the frozen uncached baseline. This check covers the three fixed prompts.
- **Timing.** The reported median uses 21 request samples per path. Wall time starts when each tokenized request is passed to the engine and ends after synchronizing the GPU. TTFT ends at delivery of the first token ID. Timed requests include cache reset, prompt tensor creation and generation. Model loading, tokenization and the one-time cache prime of 121.41 ms are measured separately or outside the request window.
- **Variation.** The seven baseline repeat medians ranged from 126.28–168.71 ms for TTFT and 237.00–322.62 ms for request time. Candidate repeat medians ranged from 21.58–23.09 ms and 128.22–162.00 ms, respectively. This is an observed warm-prefix result on one Mac. The experiment did not establish an independent ten-run baseline noise floor or a validated SOL for cached request latency, so it does not assign an SRE acceptance verdict or report a percentage of SOL.

## Reproduce

From the repository root, with Apple Silicon, native Python 3.12 and `uv`, create a temporary environment and download the pinned model. The weights are about 76 MB.

```bash
run_dir="$(mktemp -d)"
case_dir="examples/apple-m2-mlx"
python3.12 -c 'import platform; assert platform.machine() == "arm64"'
uv venv --python "$(command -v python3.12)" "$run_dir/venv"
UV_CACHE_DIR="$run_dir/uv-cache" uv pip sync \
  --python "$run_dir/venv/bin/python" "$case_dir/requirements.txt"

model_url="https://huggingface.co/mlx-community/SmolLM-135M-Instruct-4bit/resolve/642e06afe3fab57fd6cc518637c471af0a569e1e"
mkdir -p "$run_dir/model"
while read -r checksum filename; do
  curl -fLsS --retry 2 "$model_url/$filename?download=true" \
    -o "$run_dir/model/$filename" || break
done < "$case_dir/model.sha256"
```

Run the benchmark locally. It verifies model file checksums before loading and requires an available Metal GPU.

```bash
HF_HOME="$run_dir/hf" HF_HUB_OFFLINE=1 "$run_dir/venv/bin/python" \
  "$case_dir/benchmark.py" --model-dir "$run_dir/model" \
  --output "$run_dir/results.json"
```

The command prints the measured medians. The output JSON records every request sample and the exact-match result.
