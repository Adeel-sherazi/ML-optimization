# Performance Benchmarks

Standardized benchmarks for ML optimization techniques.

## Benchmark Results

### Matrix Operations

**Hardware**: Intel i7-10700K, 32GB RAM

| Size | Naive (ms) | Blocked (ms) | NumPy (ms) | Speedup |
|------|------------|--------------|------------|---------|
| 64   | 45.23      | 12.34        | 0.87       | 52x     |
| 128  | 361.45     | 89.23        | 3.21       | 113x    |
| 256  | -          | 712.34       | 15.67      | 45x     |
| 512  | -          | 5834.12      | 78.34      | 74x     |

### Model Compression

**Model**: ResNet-50 (25.6M parameters)

| Technique         | Size (MB) | Accuracy | Inference (ms) | Speedup |
|-------------------|-----------|----------|----------------|---------|
| FP32 Baseline     | 102.4     | 76.1%    | 12.3          | 1.0x    |
| Dynamic INT8      | 25.6      | 76.0%    | 6.2           | 2.0x    |
| Static INT8       | 25.6      | 75.9%    | 5.1           | 2.4x    |
| INT8 + Pruning    | 12.8      | 75.2%    | 3.8           | 3.2x    |

### CUDA Kernels

**Hardware**: NVIDIA RTX 3090

| Operation     | CPU (ms) | CUDA (ms) | Speedup |
|---------------|----------|-----------|---------|
| Vector Add 1M | 2.34     | 0.12      | 19.5x   |
| MatMul 1024²  | 856.2    | 12.3      | 69.6x   |
| Conv2d        | 234.5    | 3.2       | 73.3x   |
| ReLU 10M      | 45.6     | 0.8       | 57.0x   |

### Video Pipeline

**Model**: MobileNetV2, 224x224 input

| Batch Size | Throughput (FPS) | Latency (ms) | GPU Util |
|------------|------------------|--------------|----------|
| 1          | 45.2             | 22.1         | 45%      |
| 4          | 156.8            | 25.5         | 78%      |
| 8          | 268.4            | 29.8         | 92%      |
| 16         | 312.7            | 51.2         | 98%      |

### TensorRT Optimization

**Model**: ResNet-50, Batch Size 8

| Framework      | Precision | Throughput (FPS) | Latency (ms) |
|----------------|-----------|------------------|--------------|
| PyTorch        | FP32      | 89.2             | 89.7         |
| PyTorch        | FP16      | 178.4            | 44.8         |
| TensorRT       | FP32      | 245.6            | 32.6         |
| TensorRT       | FP16      | 534.2            | 15.0         |
| TensorRT       | INT8      | 892.7            | 9.0          |

## Running Benchmarks

### Matrix Operations

```bash
cd 01-math-fundamentals/matrix_ops
python gemm.py
```

### Model Compression

```bash
cd 02-model-compression/quantization
python int8_quantization.py
```

### CUDA Kernels

```bash
cd 03-cuda-kernels/basics
python cuda_basics.py
```

### Video Pipeline

```bash
cd 05-video-pipeline/streaming
python realtime_inference.py
```

### End-to-End System

```bash
cd 06-end-to-end-systems/image_classification
python optimized_classifier.py
```

## Profiling Commands

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(input_data)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### NVIDIA Nsight Compute

```bash
ncu --set full -o profile python script.py
ncu-ui profile.ncu-rep
```

### NVIDIA Nsight Systems

```bash
nsys profile -o timeline python script.py
nsys-ui timeline.qdrep
```

## Hardware Specifications

### CPU System
- Processor: Intel Core i7-10700K @ 3.8GHz
- RAM: 32GB DDR4-3200
- OS: Ubuntu 22.04 LTS

### GPU System
- GPU: NVIDIA RTX 3090 (24GB)
- CUDA: 11.8
- cuDNN: 8.6.0
- TensorRT: 8.6.0

## Reproducing Results

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run benchmarks:
   ```bash
   python benchmarks/run_all_benchmarks.py
   ```

3. View results:
   ```bash
   cat benchmarks/results.json
   ```

## Notes

- All benchmarks run with warmup iterations
- Results averaged over 100 iterations
- CUDA timings include device synchronization
- Memory transfer times excluded for GPU benchmarks
- TensorRT benchmarks use optimal batch size per precision
