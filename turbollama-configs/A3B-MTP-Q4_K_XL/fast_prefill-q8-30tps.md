#!/bin/bash
# Average initial prefill prompt takes 1 minute, after that
# the interactions is fast. Currently tested with Pi agent and Hermes Agent.
args=(
    -m ~/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
    # --- Networking ---
    --host 127.0.0.1
    --port 8084
    --metrics
    # --- Performance (hardware & speed) ---
    -ngl 41
    --spec-draft-ngl 0
    --n-cpu-moe 40
    -b 32768
    -ub 3072
    --ctx-size 70000
    -n 8192
    --mlock
    --no-mmap
    --cache-type-k q8_0
    --cache-type-v q8_0
    --cache-reuse 2048
    --cache-prompt
    --keep -1
    --ctx-checkpoints 2
    --flash-attn on
    --cont-batching
    --parallel 1
    # --- Model behavior ---
    --temp 0.6
    --top-p 0.95
    --top-k 20
    --min-p 0.05
    --repeat-penalty 1.0
    --reasoning auto
    --reasoning-budget 2048
    --jinja
    # --- Speculative decoding ---
    --spec-type draft-mtp
    --spec-draft-n-max 3
    --spec-draft-p-min 0.6
    # --- Misc ---
    --warmup
)

/home/work/src/llama.cpp/build/bin/llama-server "${args[@]}"
