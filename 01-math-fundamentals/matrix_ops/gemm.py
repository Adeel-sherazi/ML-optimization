"""
Production-ready matrix operations implementation.

This module provides optimized matrix operations from scratch,
demonstrating fundamental ML math operations.
"""

import numpy as np
from typing import Optional, Tuple
import time


class MatrixOps:
    """Optimized matrix operations with performance benchmarking."""
    
    @staticmethod
    def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Naive matrix multiplication (for educational purposes).
        
        Args:
            A: Matrix of shape (M, K)
            B: Matrix of shape (K, N)
            
        Returns:
            Result matrix of shape (M, N)
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Incompatible shapes: ({M}, {K}) x ({K2}, {N})"
        
        C = np.zeros((M, N), dtype=A.dtype)
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    C[i, j] += A[i, k] * B[k, j]
        return C
    
    @staticmethod
    def matmul_blocked(A: np.ndarray, B: np.ndarray, block_size: int = 32) -> np.ndarray:
        """
        Cache-optimized blocked matrix multiplication.
        
        Args:
            A: Matrix of shape (M, K)
            B: Matrix of shape (K, N)
            block_size: Size of blocks for cache optimization
            
        Returns:
            Result matrix of shape (M, N)
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Incompatible shapes: ({M}, {K}) x ({K2}, {N})"
        
        C = np.zeros((M, N), dtype=A.dtype)
        
        # Blocked matrix multiplication
        for i in range(0, M, block_size):
            for j in range(0, N, block_size):
                for k in range(0, K, block_size):
                    # Get block boundaries
                    i_end = min(i + block_size, M)
                    j_end = min(j + block_size, N)
                    k_end = min(k + block_size, K)
                    
                    # Multiply blocks
                    C[i:i_end, j:j_end] += A[i:i_end, k:k_end] @ B[k:k_end, j:j_end]
        
        return C
    
    @staticmethod
    def matmul_optimized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        NumPy-optimized matrix multiplication (uses BLAS).
        
        Args:
            A: Matrix of shape (M, K)
            B: Matrix of shape (K, N)
            
        Returns:
            Result matrix of shape (M, N)
        """
        return A @ B
    
    @staticmethod
    def conv2d_im2col(
        input: np.ndarray,
        kernel: np.ndarray,
        stride: int = 1,
        padding: int = 0
    ) -> np.ndarray:
        """
        2D Convolution using im2col algorithm (converts to GEMM).
        
        Args:
            input: Input tensor of shape (N, C, H, W)
            kernel: Kernel of shape (F, C, KH, KW)
            stride: Stride for convolution
            padding: Padding amount
            
        Returns:
            Output tensor of shape (N, F, OH, OW)
        """
        N, C, H, W = input.shape
        F, C_k, KH, KW = kernel.shape
        assert C == C_k, "Channel mismatch"
        
        # Calculate output dimensions
        OH = (H + 2 * padding - KH) // stride + 1
        OW = (W + 2 * padding - KW) // stride + 1
        
        # Pad input
        if padding > 0:
            input = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        
        # im2col transformation
        col = np.zeros((N, C, KH, KW, OH, OW))
        for y in range(KH):
            y_max = y + stride * OH
            for x in range(KW):
                x_max = x + stride * OW
                col[:, :, y, x, :, :] = input[:, :, y:y_max:stride, x:x_max:stride]
        
        # Reshape for GEMM
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)
        kernel_col = kernel.reshape(F, -1).T
        
        # GEMM operation
        output = col @ kernel_col
        
        # Reshape to output format
        output = output.reshape(N, OH, OW, F).transpose(0, 3, 1, 2)
        
        return output


def benchmark_matmul():
    """Benchmark different matrix multiplication implementations."""
    sizes = [64, 128, 256, 512]
    
    print("Matrix Multiplication Benchmarks")
    print("=" * 60)
    print(f"{'Size':<10} {'Naive (ms)':<15} {'Blocked (ms)':<15} {'Optimized (ms)':<15}")
    print("-" * 60)
    
    for size in sizes:
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # Skip naive for large sizes (too slow)
        if size <= 128:
            start = time.time()
            _ = MatrixOps.matmul_naive(A, B)
            naive_time = (time.time() - start) * 1000
        else:
            naive_time = -1
        
        # Blocked
        start = time.time()
        _ = MatrixOps.matmul_blocked(A, B)
        blocked_time = (time.time() - start) * 1000
        
        # Optimized
        start = time.time()
        _ = MatrixOps.matmul_optimized(A, B)
        opt_time = (time.time() - start) * 1000
        
        naive_str = f"{naive_time:.2f}" if naive_time > 0 else "N/A"
        print(f"{size:<10} {naive_str:<15} {blocked_time:<15.2f} {opt_time:<15.2f}")
    
    print("=" * 60)
    print("Note: Optimized version uses BLAS (highly optimized GEMM)")


if __name__ == "__main__":
    # Run benchmarks
    benchmark_matmul()
    
    # Test correctness
    print("\nVerifying correctness...")
    A = np.random.randn(32, 32).astype(np.float32)
    B = np.random.randn(32, 32).astype(np.float32)
    
    naive = MatrixOps.matmul_naive(A, B)
    blocked = MatrixOps.matmul_blocked(A, B)
    optimized = MatrixOps.matmul_optimized(A, B)
    
    print(f"Naive vs Blocked: {np.allclose(naive, blocked)}")
    print(f"Naive vs Optimized: {np.allclose(naive, optimized)}")
    print(f"Blocked vs Optimized: {np.allclose(blocked, optimized)}")
    
    # Test convolution
    print("\nTesting convolution...")
    input_tensor = np.random.randn(2, 3, 32, 32).astype(np.float32)
    kernel = np.random.randn(16, 3, 3, 3).astype(np.float32)
    
    output = MatrixOps.conv2d_im2col(input_tensor, kernel, stride=1, padding=1)
    print(f"Convolution output shape: {output.shape}")
    print(f"Expected shape: (2, 16, 32, 32)")
    print(f"Shape correct: {output.shape == (2, 16, 32, 32)}")
