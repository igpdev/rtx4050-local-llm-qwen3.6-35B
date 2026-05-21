# llama.cpp Build Instructions

Two separate runtimes are used depending on the model.

---

## TurboQuant llama.cpp

Used for: **Qwen3.6 35B A3B Q4_K_M** and **Qwen3.6 35B A3B IQ4_NL**

Repository: https://github.com/CarapaceUDE/turboquant-llama

Optimized for NVIDIA CUDA, Ada Lovelace GPUs, Flash Attention for all quant types, and native CPU performance.

### Active Backends
- ggml-cpu
- ggml-cuda

### Enabled Features
- GGML_CUDA=ON
- GGML_CUDA_FA=ON
- GGML_CUDA_FA_ALL_QUANTS=ON
- GGML_CUDA_GRAPHS=ON
- GGML_CUDA_NCCL=ON
- GGML_NATIVE=ON
- GGML_OPENMP=ON

### Disabled Backends
- Vulkan · HIP / ROCm · OpenCL · Metal · SYCL · WebGPU

### Build Configuration
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

### Rebuild Script
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

### Binary Verification
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

## Official llama.cpp

Used for: **Qwen3.6 35B A3B MTP Q4_K_XL**

The official upstream llama.cpp includes MTP (Multi-Token Prediction) support. TurboQuant does not, so this model requires building from the official repo.

Upstream repo: https://github.com/ggml-org/llama.cpp

Full guide: https://unsloth.ai/docs/models/qwen3.6#mtp-guide

### Build Configuration
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
