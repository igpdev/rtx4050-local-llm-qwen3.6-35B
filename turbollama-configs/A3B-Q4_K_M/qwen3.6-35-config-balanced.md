## Overview
**Turbo Precision** – Balances context length and KV cache fidelity for high‑accuracy coding tasks.

- **TurboQuant enabled** (`TURBO_LAYER_ADAPTIVE=1`) – still leverages adaptive optimization for MoE scheduling.
- **Higher precision KV cache** (`q8_0`) – preserves detailed attention information, ideal for tracking variable names, function signatures, and complex code structures.
- **Fixed GPU/CPU split** (`--n-gpu-layers 64`, `--n-cpu-moe 64`) – predictable resource allocation, runs comfortably on 6 GB VRAM.
- **32k context window** (`-c 32768`) – long enough for most agentic coding sessions without the fidelity loss of heavier compression.

**Best for**:
Code generation, debugging, refactoring, and tool‑calling workflows where output quality and token‑level precision matter more than absolute maximum context length.

**Trade‑offs**:
Slightly shorter effective context than the `turbo-adaptive-fit` variant, but maintains `q8_0` KV cache for better accuracy in long reasoning chains. Generation speed ~16–19 tok/s on RTX 4050.

## Command
```bash
TURBO_LAYER_ADAPTIVE=1 ./build/bin/llama-server \
  -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8084 \
  --n-gpu-layers 64 \
  --n-cpu-moe 64 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn on \
  -c 32768 \
  -n 16384 \
  --threads 10 \
  --threads-batch 12 \
  -b 512 \
  -ub 512 \
  --prio 2 \
  --poll 0 \
  --temp 0.1 \
  --top-p 1 \
  --top-k 0 \
  --repeat-penalty 1.0 \
  --cont-batching \
  --cache-prompt \
  --ctx-checkpoints 10 \
  --jinja \
  --reasoning auto
```

## Flags explanation

- `TURBO_LAYER_ADAPTIVE=1`
  TurboQuant-specific environment variable that enables adaptive layer optimization for MoE models.

- `-m`
  Path to the model file.

- `--host` / `--port`
  Defines where the server listens.
  `0.0.0.0` means the server accepts connections from any network interface.

- `--n-gpu-layers 64`
  Number of transformer layers loaded onto the GPU.
  `64` places all layers for this model on GPU. More GPU layers generally improve generation speed.

- `--n-cpu-moe 64`
  Forces all MoE expert layers to run on CPU.
  Although the model contains 35B total parameters, only about 3B are active per token. Keeping experts on CPU frees VRAM for attention layers, which have a larger impact on inference speed.

- `--cache-type-k q8_0` / `--cache-type-v q8_0`
  Controls KV-cache compression format.
  `q8_0` preserves enough precision for long coding sessions, helping the model reliably track:
  - variable names
  - function signatures
  - file structure
  - tool outputs

- `--flash-attn on`
  Enables memory-efficient Flash Attention.
  Important for long-context agentic workloads where the context fills with file contents, search results, and tool outputs on limited VRAM.

- `-c 32768`
  Total context window size.
  Large enough to prevent truncation during extended coding sessions with accumulated prompts and tool interactions.

- `-n 16384`
  Maximum output tokens per response.
  Allows the model to generate:
  - complete functions
  - entire files
  - long code reviews
  - multi-step refactors

  in a single response.

- `--threads 10`
  CPU generation threads.
  Since MoE experts run on CPU, this directly impacts token generation speed. Typically set to the number of physical CPU cores.

- `--threads-batch 12`
  CPU threads used during prompt prefill.
  Affects how quickly large prompts and file contents are processed before generation begins.

- `-b 512`
  Prefill batch size.
  Controls how many tokens are processed in parallel during prompt ingestion. Balanced for single-user local workloads.

- `-ub 512`
  Micro-batch size per GPU call.
  Tuned for RTX 4050 6GB-class GPUs.

- `--prio 2`
  Runs the server with above-normal OS scheduling priority.
  Helps maintain responsiveness when other applications are competing for CPU time.

- `--poll 0`
  Disables busy-loop polling while idle.
  Reduces unnecessary CPU usage, heat, and power draw during long agentic sessions.

- `--temp 0.1`
  Low temperature for deterministic output.
  Better for coding tasks where correctness and reproducibility matter more than creativity.

- `--top-p 1.0`
  Disables nucleus sampling.
  Allows the model to consider the full probability distribution instead of truncating candidate tokens.

- `--top-k 0`
  Disables top-k sampling limits.
  Prevents artificial token restrictions that can hurt syntax accuracy or identifier selection in code generation.

- `--repeat-penalty 1.0`
  Disables repetition penalties.
  Code naturally repeats tokens frequently (identifiers, brackets, keywords). Penalizing repetition can degrade output quality.

- `--cont-batching`
  Enables continuous batching.
  Allows the server to accept and process new requests while generation is still running, reducing latency during rapid tool-call workflows.

- `--cache-prompt`
  Caches the KV state of the system prompt.
  Since agent frameworks often resend the same system prompt repeatedly, this avoids reprocessing it every request and significantly improves performance.

- `--ctx-checkpoints 7`
  Saves periodic KV-cache checkpoints.
  Long sessions can restore state quickly without rebuilding the entire accumulated context from scratch.
  Tuned to 7 (vs 10) to reduce VRAM pressure on 6 GB mobile GPU while still covering append-forward agentic context patterns.


- `--jinja`
  Enables Jinja templating support required for Qwen3 tool-call formatting.
  Necessary for proper compatibility with qwen-code CLI tool calling.

- `--reasoning auto`
  Lets the model dynamically choose when to use deeper reasoning.
  Simple edits stay fast, while complex debugging or architectural tasks receive additional reasoning time automatically.
