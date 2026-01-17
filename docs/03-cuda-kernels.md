# CUDA Programming Guide

Production CUDA programming for ML optimization.

## Overview

CUDA (Compute Unified Device Architecture) enables parallel computing on NVIDIA GPUs.

## GPU Architecture Basics

### Memory Hierarchy (Fast → Slow)

1. **Registers**: Per-thread, fastest
2. **Shared Memory**: Per-block, ~100x slower than registers
3. **L1/L2 Cache**: Automatic
4. **Global Memory**: All threads, slowest (~1000x slower)
5. **Constant Memory**: Read-only, cached
6. **Texture Memory**: Optimized for 2D locality

### Thread Hierarchy

```
Grid
  └── Block (up to 1024 threads)
       └── Warp (32 threads)
            └── Thread
```

## Basic CUDA Kernel

```cuda
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Launch with 256 threads per block
int threads = 256;
int blocks = (n + threads - 1) / threads;
vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
```

## Optimization Techniques

### 1. Memory Coalescing

**Problem**: Scattered memory access is slow

**Solution**: Access contiguous memory locations

```cuda
// Bad: Strided access
c[idx * stride] = a[idx * stride] + b[idx * stride];

// Good: Coalesced access
c[idx] = a[idx] + b[idx];
```

### 2. Shared Memory

Use shared memory for frequently accessed data:

```cuda
__global__ void matmul_shared(float* A, float* B, float* C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    // Load tile into shared memory
    tile_A[ty][tx] = A[...];
    tile_B[ty][tx] = B[...];
    __syncthreads();
    
    // Compute using shared memory
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += tile_A[ty][k] * tile_B[k][tx];
    }
}
```

### 3. Occupancy Optimization

**Occupancy** = Active warps / Maximum warps

Factors affecting occupancy:
- Registers per thread
- Shared memory per block
- Threads per block

**Best Practice**: Aim for >50% occupancy

### 4. Bank Conflicts

Shared memory is divided into banks (32 on modern GPUs).

```cuda
// Bad: Bank conflict
__shared__ float data[32][32];
float val = data[threadIdx.x][0];  // All threads access bank 0

// Good: No conflict
float val = data[0][threadIdx.x];  // Different banks
```

### 5. Warp Divergence

All threads in a warp execute the same instruction.

```cuda
// Bad: Divergent branches
if (threadIdx.x < 16) {
    // Only half of warp executes
}

// Good: Aligned with warp size
if (threadIdx.x < 32) {
    // Full warp executes
}
```

## PyTorch CUDA Extensions

### Creating Custom CUDA Ops

```python
from torch.utils.cpp_extension import load

# Compile and load CUDA extension
custom_op = load(
    name='custom_op',
    sources=['custom_op.cu'],
    extra_cuda_cflags=['-O3']
)

# Use in PyTorch
output = custom_op.forward(input_tensor)
```

### Example: Custom ReLU

```cuda
#include <torch/extension.h>

__global__ void relu_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    int n = input.numel();
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output;
}
```

## Performance Benchmarking

### Using CUDA Events

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<blocks, threads>>>(...);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

### Using Nsight Compute

```bash
# Profile kernel
ncu --set full -o profile python script.py

# View results
ncu-ui profile.ncu-rep
```

## Common Patterns

### Reduction

```cuda
__global__ void reduce_sum(float* input, float* output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

### Scan (Prefix Sum)

Use CUB library for production implementations.

## Memory Transfer Optimization

### Pinned Memory

```python
# Faster host-device transfers
tensor = torch.randn(size).pin_memory()
tensor_gpu = tensor.cuda(non_blocking=True)
```

### Unified Memory

```cuda
// Managed memory (automatic transfer)
float* data;
cudaMallocManaged(&data, size);

// Access from both CPU and GPU
data[0] = 1.0;  // CPU
kernel<<<...>>>(data);  // GPU
```

## Best Practices Checklist

- [ ] Memory access coalesced
- [ ] Shared memory used for reused data
- [ ] Occupancy >50%
- [ ] No bank conflicts
- [ ] Minimal warp divergence
- [ ] Asynchronous transfers
- [ ] Streams for overlap
- [ ] Profiled with Nsight

## Common Pitfalls

1. **Not checking for errors**
   ```cuda
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
       printf("Error: %s\n", cudaGetErrorString(err));
   }
   ```

2. **Forgetting synchronization**
   ```cuda
   cudaDeviceSynchronize();  // Wait for kernel
   ```

3. **Unoptimized memory access patterns**

4. **Too many registers per thread** → Low occupancy

## Performance Targets

| Metric                | Good     | Excellent |
|-----------------------|----------|-----------|
| Memory Bandwidth      | >50%     | >80%      |
| Occupancy             | >50%     | >75%      |
| SM Efficiency         | >60%     | >80%      |
| Warp Execution Eff.   | >80%     | >95%      |

## Further Reading

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [CUB Library](https://nvlabs.github.io/cub/)
