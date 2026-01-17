"""
Basic CUDA kernel implementations with PyTorch.

Demonstrates fundamental CUDA programming concepts.
"""

import torch
import time
from typing import Tuple


# CUDA kernel source code as strings (to be compiled with torch.utils.cpp_extension)
VECTOR_ADD_KERNEL = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto c = torch::zeros_like(a);
    int n = a.numel();
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );
    
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_cuda, "Vector addition (CUDA)");
}
"""

MATRIX_MUL_KERNEL = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul_cuda, "Matrix multiplication (CUDA)");
}
"""


class CUDAKernelExamples:
    """Examples of CUDA kernels with PyTorch fallback."""
    
    @staticmethod
    def vector_add_pytorch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """PyTorch implementation of vector addition."""
        return a + b
    
    @staticmethod
    def matmul_pytorch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """PyTorch implementation of matrix multiplication."""
        return torch.mm(A, B)
    
    @staticmethod
    def relu_cuda_simulation(x: torch.Tensor) -> torch.Tensor:
        """
        Simulated CUDA ReLU implementation.
        In production, this would be a custom CUDA kernel.
        """
        return torch.clamp(x, min=0.0)
    
    @staticmethod
    def benchmark_operation(
        func,
        *args,
        num_iterations: int = 100,
        warmup: int = 10,
        use_cuda: bool = True
    ) -> float:
        """
        Benchmark an operation.
        
        Args:
            func: Function to benchmark
            *args: Arguments to the function
            num_iterations: Number of iterations
            warmup: Number of warmup iterations
            use_cuda: Whether to use CUDA
            
        Returns:
            Average time in milliseconds
        """
        if use_cuda and torch.cuda.is_available():
            args = [arg.cuda() if isinstance(arg, torch.Tensor) else arg for arg in args]
        
        # Warmup
        for _ in range(warmup):
            _ = func(*args)
        
        if use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            result = func(*args)
        
        if use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = (time.time() - start) / num_iterations * 1000
        
        return elapsed


def demonstrate_cuda_basics():
    """Demonstrate basic CUDA concepts with PyTorch."""
    print("CUDA Basics Demonstration")
    print("=" * 70)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n1. CUDA Environment")
    print(f"   CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Device count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Total memory: {total_memory:.2f} GB")
    
    # Vector addition benchmark
    print(f"\n2. Vector Addition Benchmark")
    sizes = [1000, 10000, 100000, 1000000]
    
    print(f"{'Size':<15} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<15}")
    print("-" * 70)
    
    for size in sizes:
        a = torch.randn(size)
        b = torch.randn(size)
        
        # CPU benchmark
        cpu_time = CUDAKernelExamples.benchmark_operation(
            CUDAKernelExamples.vector_add_pytorch,
            a, b,
            use_cuda=False
        )
        
        # GPU benchmark (if available)
        if cuda_available:
            gpu_time = CUDAKernelExamples.benchmark_operation(
                CUDAKernelExamples.vector_add_pytorch,
                a, b,
                use_cuda=True
            )
            speedup = cpu_time / gpu_time
        else:
            gpu_time = -1
            speedup = 0
        
        gpu_str = f"{gpu_time:.4f}" if cuda_available else "N/A"
        speedup_str = f"{speedup:.2f}x" if cuda_available else "N/A"
        print(f"{size:<15} {cpu_time:<15.4f} {gpu_str:<15} {speedup_str:<15}")
    
    # Matrix multiplication benchmark
    print(f"\n3. Matrix Multiplication Benchmark")
    sizes = [128, 256, 512, 1024]
    
    print(f"{'Size':<15} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<15}")
    print("-" * 70)
    
    for size in sizes:
        A = torch.randn(size, size)
        B = torch.randn(size, size)
        
        # CPU benchmark
        cpu_time = CUDAKernelExamples.benchmark_operation(
            CUDAKernelExamples.matmul_pytorch,
            A, B,
            use_cuda=False,
            num_iterations=10
        )
        
        # GPU benchmark (if available)
        if cuda_available:
            gpu_time = CUDAKernelExamples.benchmark_operation(
                CUDAKernelExamples.matmul_pytorch,
                A, B,
                use_cuda=True,
                num_iterations=10
            )
            speedup = cpu_time / gpu_time
        else:
            gpu_time = -1
            speedup = 0
        
        gpu_str = f"{gpu_time:.4f}" if cuda_available else "N/A"
        speedup_str = f"{speedup:.2f}x" if cuda_available else "N/A"
        print(f"{size}x{size:<9} {cpu_time:<15.4f} {gpu_str:<15} {speedup_str:<15}")
    
    print("\n" + "=" * 70)
    print("CUDA Programming Concepts:")
    print("1. Thread Hierarchy: Grid -> Block -> Thread")
    print("2. Memory Hierarchy: Global -> Shared -> Registers")
    print("3. Kernel Launch: kernel<<<blocks, threads>>>(args)")
    print("4. Synchronization: __syncthreads(), cudaDeviceSynchronize()")
    print("5. Memory Coalescing: Access contiguous memory for best performance")


def save_cuda_kernel_templates():
    """Save CUDA kernel templates to files."""
    import os
    
    # Get the directory of the current script
    kernels_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".")
    os.makedirs(kernels_dir, exist_ok=True)
    
    # Save vector add kernel
    with open(os.path.join(kernels_dir, "vector_add.cu"), "w") as f:
        f.write(VECTOR_ADD_KERNEL)
    
    # Save matrix multiply kernel
    with open(os.path.join(kernels_dir, "matmul.cu"), "w") as f:
        f.write(MATRIX_MUL_KERNEL)
    
    print(f"\nCUDA kernel templates saved to {kernels_dir}/")
    print("To compile: use torch.utils.cpp_extension.load()")


if __name__ == "__main__":
    demonstrate_cuda_basics()
    save_cuda_kernel_templates()
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("1. Study memory coalescing patterns")
    print("2. Implement shared memory optimizations")
    print("3. Profile kernels with nvprof/Nsight Compute")
    print("4. Optimize occupancy and memory bandwidth")
