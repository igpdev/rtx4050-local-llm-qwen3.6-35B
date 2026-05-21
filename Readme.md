# RTX 4050 Local LLM Setup — llama.cpp + Qwen3.6 35B A3B

Local LLM inference on RTX 4050 6GB VRAM. Uses two different runtimes: **TurboQuant** (a custom llama.cpp fork) and the **official llama.cpp** upstream, which includes MTP support.

## Overview

| Model               | Quant   | Context | KV Cache | Expected Speed | llama.cpp Build  | Primary Use Case |
|---------------------|---------|---------|----------|----------------|------------------|------------------|
| Qwen3.6 35B A3B     | Q4_K_M  | 65k     | q4_0     | ~17 t/s        | TurboQuant       | Agentic coding   |
| Qwen3.6 35B A3B     | IQ4_NL  | 160k    | q8_0     | ~23 t/s        | TurboQuant       | Agentic coding   |
| Qwen3.6 35B A3B MTP | Q4_K_XL | 100k    | q8_0     | ~30 t/s        | llama.cpp latest | Agentic coding   |

> All models run on the same hardware. The MTP model runs on the official llama.cpp upstream (not TurboQuant) — see [Official llama.cpp](#official-llamacpp) below.

---

## System Specifications

### Laptop
- **Model**: MSI Cyborg 15.6"
- **CPU**: Intel Core i7-13620H (13th Gen, 10 cores / 16 threads, 6P + 4E, AVX2)
- **RAM**: 64 GB (upgraded from 16 GB)
- **Storage**: 1 TB NVMe SSD (upgraded from 512 GB)
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU
- **OS**: Ubuntu 24.04 LTS · Kernel 6.17 · NVIDIA Proprietary Drivers · GCC 12.4.0

### GPU Details

#### NVIDIA RTX 4050 Laptop GPU
- Ada Lovelace architecture
- 6 GB VRAM
- CUDA + Tensor Cores
- Compute Capability: 8.9

#### Integrated GPU
- Intel UHD Graphics

---

## llama.cpp Builds

Two separate runtimes are used depending on the model.

### TurboQuant llama.cpp

Used for: **Qwen3.6 35B A3B Q4_K_M** and **Qwen3.6 35B A3B IQ4_NL**

Repository: https://github.com/CarapaceUDE/turboquant-llama

Optimized for NVIDIA CUDA, Ada Lovelace GPUs, Flash Attention for all quant types, and native CPU performance.

#### Active Backends
- ggml-cpu
- ggml-cuda

#### Enabled Features
- GGML_CUDA=ON
- GGML_CUDA_FA=ON
- GGML_CUDA_FA_ALL_QUANTS=ON
- GGML_CUDA_GRAPHS=ON
- GGML_CUDA_NCCL=ON
- GGML_NATIVE=ON
- GGML_OPENMP=ON

#### Disabled Backends
- Vulkan · HIP / ROCm · OpenCL · Metal · SYCL · WebGPU

#### Build Configuration
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build -j$(nproc)
```

#### Rebuild Script
```bash
cat > rebuild.sh <<'REBUILD'
rm -rf build
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build -j$(nproc)
REBUILD
chmod +x rebuild.sh
```

#### Binary Verification
```bash
./build/bin/llama-cli --version
```
Expected output:
```
ggml_cuda_init: found 1 CUDA devices
Device 0: NVIDIA GeForce RTX 4050 Laptop GPU
compute capability 8.9
```

---

### Official llama.cpp

Used for: **Qwen3.6 35B A3B MTP Q4_K_XL**

The official upstream llama.cpp includes MTP (Multi-Token Prediction) support. TurboQuant does not, so this model requires building from the official repo.

Upstream repo: https://github.com/ggml-org/llama.cpp

#### Build Configuration
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

Binary location after build: `~/src/llama.cpp/build/bin/llama-server`

---

## Models & Configs

### Qwen3.6 35B A3B — Q4_K_M
- [`qwen3.6-35-config-big-context.md`](./turbollama-configs/qwen3.6-35b/A3B-Q4_K_M/qwen3.6-35-config-big-context.md) — big context launch config
- [`qwen3.6-35-config-balanced.md`](./turbollama-configs/qwen3.6-35b/A3B-Q4_K_M/qwen3.6-35-config-balanced.md) — balanced launch config
- [`Qwen3.6-35B-A3B-UD-Q4_K_M-model-overview.md`](./turbollama-configs/qwen3.6-35b/A3B-Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M-model-overview.md) — model overview & download instructions

### Qwen3.6 35B A3B — IQ4_NL
- [`ultra-context-q8-23tps.md`](./turbollama-configs/qwen3.6-35b/A3B-IQ4_NL/ultra-context-q8-23tps.md) — ultra-long-context launch config (160K tokens, q8_0 KV)

### Qwen3.6 35B A3B MTP — Q4_K_XL
- [`high-context-q8-30tps.md`](./turbollama-configs/qwen3.6-35b/A3B-MTP-Q4_K_XL/high-context-q8-30tps.md) — MTP launch config & compilation instructions

---

## Screenshots

![Code](images/img_code.png)
![Info](images/img_info.png)
![Qwen CLI](images/img_qwencli.png)
![Web](images/img_web.png)
