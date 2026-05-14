# Qwen3.6-35b-moe Coding Optimized

## Model Information
- Model: Qwen3.6 35B A3B
- Quantization: Q4_K_M GGUF
- Optimized for local inference with TurboQuant llama.cpp
- Suitable for long-context coding and general reasoning workloads

## Runtime Configuration
Optimized for:
- RTX 4050 Laptop GPU (6 GB VRAM)
- TurboQuant KV cache
- CUDA Flash Attention
- Large context inference
- Continuous batching
- Long coding sessions

## Expected Performance
- Expect ~17.45 tokens/sec during generation
- Performance depends on:
  - Context size
  - Prompt complexity
  - GPU layer offloading
  - TurboQuant KV cache compression
  - Concurrent requests
- TurboQuant KV cache and Flash Attention improve throughput and memory efficiency on RTX 4050 6 GB GPUs
- Larger context sizes may reduce effective generation speed over time


## Use cases

Agentic coding, code generation, debugging, code review, architecture decisions.
Optimized for tools like qwen-code cli that send repeated system prompts and accumulate long context from file reads and tool outputs.

Not ideal for: Creative writing, chat, brainstorming. For those use `temp 0.7`, `top-p 0.95`, `top-k 20`.

## Download Model

> **Requires the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)** — install it with:
> ```bash
> pip install huggingface_hub
> ```

```bash
hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir ~/models
```
