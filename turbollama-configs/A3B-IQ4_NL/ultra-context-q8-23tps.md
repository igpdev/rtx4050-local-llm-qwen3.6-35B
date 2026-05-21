# Qwen3.6-35B A3B — Ultra Context Q8

## Model Information
- **Model**: Qwen3.6 35B A3B
- **Quantization**: IQ4_NL GGUF
- **llama.cpp**: TurboQuant (`TURBO_LAYER_ADAPTIVE=1`)
- **Suitable for**: Ultra-long-context agentic coding, repository-scale tool use

## Runtime Configuration
Optimized for:
- RTX 4050 Laptop GPU (6 GB VRAM)
- TurboQuant adaptive layer offloading (`TURBO_LAYER_ADAPTIVE=1 -ngl 999`)
- CUDA Flash Attention
- 160K token context window
- High-fidelity `q8_0` KV cache
- Continuous batching with large batch sizes (`-b 2048 -ub 2048`)
- MoE experts fully offloaded to RAM
- Extended reasoning budget (8192 tokens)

## Expected Performance
- Expect ~23 tokens/sec during generation
- Performance depends on:
  - Context size (speed decreases as context fills toward 160K)
  - Prompt complexity and prefill length
  - TurboQuant adaptive layer shifting between GPU/CPU
  - `q8_0` KV cache prioritises quality over memory savings
  - Concurrent requests
- `TURBO_LAYER_ADAPTIVE=1` dynamically shifts layers to fit the model into available VRAM
- `--no-mmap` combined with `--mlock` trades RAM usage for deterministic latency

## Use Cases
Agentic coding, code generation, debugging, code review, architecture decisions.
Best suited for extremely long sessions where context accumulates heavily — large codebases, multi-file edits, long tool-use chains.

Not ideal for: Creative writing, chat, brainstorming. For those use `--temp 0.7`, `--top-p 0.95`, `--top-k 20`.

## Download Model
> **Requires the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)** — install it with:
> ```bash
> pip install huggingface_hub
> ```

```bash
hf download Qwen/Qwen3.6-35B-A3B-GGUF Qwen_Qwen3.6-35B-A3B-IQ4_NL.gguf --local-dir ~/models
```

---

## Overview
**Ultra Context Mode** – Pushes context to 160K tokens using TurboQuant adaptive layer offloading with high-fidelity `q8_0` KV cache on a standard TurboQuant llama.cpp build.

- **Adaptive layer offloading** (`TURBO_LAYER_ADAPTIVE=1 -ngl 999`) – dynamically shifts layers between GPU and CPU to maximise the layers that fit in 6 GB VRAM.
- **All MoE experts on CPU** (`--n-cpu-moe 999`) – offloads the sparsely activated 3B experts to RAM, reserving GPU bandwidth for attention layers.
- **High-fidelity KV cache** (`q8_0`) – prioritises generation quality over memory savings across the full 160K context.
- **Large batch sizes** (`-b 2048 -ub 2048`) – maximises prefill throughput for long system prompts and document ingestion.
- **Extended reasoning budget** (`--reasoning-budget 8192`) – allows deeper chain-of-thought for complex architectural and debugging tasks.
- **No memory-mapped I/O** (`--no-mmap`) – loads the full model into RAM via implicit `--mlock` for deterministic latency.

**Best for**:
Extremely long agentic coding sessions, full-repository analysis, multi-file edits, and long tool-use chains where 160K context and KV quality matter more than raw generation speed.

**Trade-offs**:
`q8_0` KV cache at 160K is VRAM-heavy — adaptive offloading compensates but CPU fallback adds latency at very large contexts. Slightly lower IQ4_NL precision vs Q4_K_M at a smaller file size.

---

## Command

```bash
TURBO_LAYER_ADAPTIVE=1 llama-server \
  -m ~/models/Qwen_Qwen3.6-35B-A3B-IQ4_NL.gguf \
  --host 0.0.0.0 \
  --port 8084 \
  -ngl 999 \
  -c 160000 \
  -n 8192 \
  -b 2048 \
  -ub 2048 \
  --cont-batching \
  --threads 12 \
  --threads-batch 16 \
  --prio 2 \
  --poll 50 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn on \
  --cache-prompt \
  --cache-reuse 512 \
  --ctx-checkpoints 10 \
  --n-cpu-moe 999 \
  --temp 0.6 \
  --min-p 0.05 \
  --top-k 40 \
  --top-p 0.95 \
  --repeat-penalty 1.05 \
  --jinja \
  --reasoning auto \
  --reasoning-budget 8192 \
  --no-mmap
```
