# Qwen3.6-35B A3B MTP — High-Context Q8

## Model Information
- **Model**: Qwen3.6 35B A3B MTP
- **Quantization**: Q4_K_XL GGUF
- **llama.cpp**: Official upstream — https://github.com/ggml-org/llama.cpp
- **Variant**: MTP (Multi-Token Prediction) — supported in official llama.cpp, not in TurboQuant
- **Suitable for**: Long-context coding, agentic tool use, repository-scale analysis

## Runtime Configuration
Optimized for:
- RTX 4050 Laptop GPU (6 GB VRAM)
- CUDA Flash Attention
- MTP speculative decoding (built-in draft heads, no separate model)
- **q4_0 KV cache** (balances VRAM and quality)
- 100K token context window
- Continuous batching
- MoE experts fully offloaded to RAM

## Expected Performance
- Expect ~28–32 tokens/sec during generation (MTP boost included)
- Performance depends on:
  - Context size (speed decreases as context fills)
  - Prompt complexity and prefill length
  - GPU layer offloading
  - q4_0` KV cache uses less VRAM than `q8_0` — leaves more headroom for large contexts
  - Concurrent requests
- Flash Attention and MTP speculative decoding improve throughput on RTX 4050 6 GB
- `--no-mmap` + `--mlock` trades RAM usage for deterministic latency

## Use Cases
Agentic coding, code generation, debugging, code review, architecture decisions.
Optimized for tools like `qwen-code` CLI that send repeated system prompts and accumulate long context from file reads and tool outputs.
* Current setup and tests use PI Agent. Given light system prompt is what gives best results.

Not ideal for: Creative writing, chat, brainstorming. For those use `--temp 0.7`, `--top-p 0.95`, `--top-k 20`.

## Download Model
> **Requires the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)** — install it with:
> ```bash
> pip install huggingface_hub
> ```

```bash
huggingface-cli download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --local-dir ~/models/Qwen3.6-35B-A3B-MTP-GGUF
```

---

## Overview
**MTP High-Context Quality Mode** – Maximises KV cache fidelity and context length using Multi-Token Prediction (MTP) speculative decoding on the official llama.cpp (`github.com/ggml-org/llama.cpp`).

- **MTP speculative decoding** (`--spec-type draft-mtp --spec-draft-n-max 3`) – uses the model's built-in draft heads to predict 2 tokens ahead, boosting throughput without a separate draft model.
- **Draft layers fully on CPU** (`--spec-draft-ngl 0 --gpu-layers-draft 0`) – keeps draft heads off the GPU, freeing VRAM for the main model's attention layers.
- **All MoE experts on CPU** (`--n-cpu-moe 41`) – offloads the sparsely activated 3B experts to RAM, reserving GPU bandwidth for attention layers.
- **q4_0 KV cache** (--cache-type-k q4_0 --cache-type-v q4_0) – balances VRAM efficiency and generation quality; leaves more room for the 100K token context.
- **No memory-mapped I/O** (`--no-mmap`) – loads the full model into RAM via `--mlock` for deterministic latency and to avoid page-fault stalls during long sessions.
- **Large batch size** (`-b 2048`) – improves throughput for prefill-heavy workloads such as long system prompts and document analysis.

> **⚠️ Requires the official llama.cpp** — https://github.com/ggml-org/llama.cpp — TurboQuant does not include MTP support. See the [Compilation](#compilation) section below.

**Best for**:
Long agentic coding sessions, document-scale analysis, multi-turn tool use, and any workload where generation quality and a 100K token context window take priority over maximum KV compression.

**Trade-offs**:
q4_0 KV cache uses less VRAM than q8_0, allowing larger effective context on 6 GB GPU; quality remains very good for coding and tool use. MTP draft heads run on CPU, so gains are most noticeable when the GPU is the bottleneck, not the CPU.

---

## Compilation

This model uses **Multi-Token Prediction (MTP)**, which is supported in the official llama.cpp upstream. TurboQuant does not include MTP support, so this model requires building from the official repo latest release.

Upstream repo: https://github.com/ggml-org/llama.cpp

Full guide: https://unsloth.ai/docs/models/qwen3.6#mtp-guide

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y

git clone https://github.com/ggml-org/llama.cpp

cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON

cmake --build llama.cpp/build --config Release -j --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split

cp llama.cpp/build/bin/llama-* llama.cpp
```

The resulting binary used in the command below is at:
`~/src/llama.cpp/build/bin/llama-server`

---

## Command
- Modify bin llama-server path to match your workspace
```bash
/home/work/src/llama.cpp/build/bin/llama-server \
  --metrics \
  -m ~/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --host 127.0.0.1 \
  --port 8084 \
  -ngl 41 \
  --spec-draft-ngl 0 \
  --gpu-layers-draft 0 \
  --flash-attn on \
  --threads 10 \
  --threads-batch 12 \
  --prio 3 \
  --poll 0 \
  --n-cpu-moe 41 \
  -b 2048 \
  -ub 512 \
  --cont-batching \
  -c 100000 \
  --cache-ram 0 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --cache-prompt \
  --cache-reuse 512 \
  --ctx-checkpoints 2 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0.75 \
  -n 16384 \
  --jinja \
  --reasoning auto \
  --reasoning-budget 8192 \
  --temp 0.6 \
  --min-p 0.05 \
  --top-k 20 \
  --top-p 0.95 \
  --repeat-penalty 1.0 \
  --mlock \
  --no-mmap
```
