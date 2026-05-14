## Overview
**Turbo Adaptive Fit** – Maximises context length and throughput on VRAM‑constrained GPUs (RTX 4050 6GB and similar).

- **TurboQuant adaptive layering** (`TURBO_LAYER_ADAPTIVE=1` + `-ngl auto`) – dynamically shifts layers between GPU/CPU to fit the model into available VRAM.
- **Auto‑fitting** (`--fit on --fit-target 500`) – reserves only ~500 MB VRAM headroom, leaving the rest for a huge `65536` token context.
- **Aggressive KV cache compression** (`q4_0`) – doubles context size compared to standard `q8_0` at a minor precision cost.
- **All MoE experts on CPU** (`--n-cpu-moe 999`) – offloads the sparsely activated 3B experts to RAM, freeing GPU memory for attention layers.

**Best for**:
Very long agentic coding sessions, repository‑scale analysis, multi‑turn tool use, and batch processing where context length matters more than absolute KV precision.

**Trade‑offs**:
Slightly lower KV cache fidelity (`q4_0`) – still safe for code generation and reasoning; generation speed remains ~16–19 tok/s on RTX 4050.

## Command
TURBO_LAYER_ADAPTIVE=1 llama-server \
  -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8084 \
  -ngl auto \
  --fit on \
  --fit-target 500 \
  -c 65536 \
  -n 16384 \
  -b 512 \
  -ub 512 \
  --cont-batching \
  --threads 10 \
  --threads-batch 14 \
  --prio 2 \
  --poll 0 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --flash-attn on \
  --cache-prompt \
  --cache-reuse 512 \
  --ctx-checkpoints 10 \
  --n-cpu-moe 999 \
  --temp 0.6 \
  --min-p 0.05 \
  --top-k 40 \
  --repeat-penalty 1.05 \
  --mlock \
  --jinja \
  --reasoning auto
