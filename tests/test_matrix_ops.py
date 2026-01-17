"""
Unit tests for matrix operations.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..['01-math-fundamentals'].matrix_ops.gemm import MatrixOps


class TestMatrixOps:
    """Test matrix operations."""
    
    def test_matmul_naive_correctness(self):
        """Test naive matrix multiplication correctness."""
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)
        
        result = MatrixOps.matmul_naive(A, B)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        
        assert np.allclose(result, expected)
    
    def test_matmul_blocked_correctness(self):
        """Test blocked matrix multiplication correctness."""
        np.random.seed(42)
        A = np.random.randn(64, 64).astype(np.float32)
        B = np.random.randn(64, 64).astype(np.float32)
        
        result = MatrixOps.matmul_blocked(A, B)
        expected = A @ B
        
        assert np.allclose(result, expected, rtol=1e-5)
    
    def test_matmul_optimized_correctness(self):
        """Test optimized matrix multiplication correctness."""
        np.random.seed(42)
        A = np.random.randn(128, 128).astype(np.float32)
        B = np.random.randn(128, 128).astype(np.float32)
        
        result = MatrixOps.matmul_optimized(A, B)
        expected = A @ B
        
        assert np.allclose(result, expected)
    
    def test_conv2d_output_shape(self):
        """Test convolution output shape."""
        input_tensor = np.random.randn(2, 3, 32, 32).astype(np.float32)
        kernel = np.random.randn(16, 3, 3, 3).astype(np.float32)
        
        output = MatrixOps.conv2d_im2col(input_tensor, kernel, stride=1, padding=1)
        
        assert output.shape == (2, 16, 32, 32)
    
    def test_matmul_incompatible_shapes(self):
        """Test that incompatible shapes raise error."""
        A = np.random.randn(3, 4).astype(np.float32)
        B = np.random.randn(5, 6).astype(np.float32)
        
        with pytest.raises(AssertionError):
            MatrixOps.matmul_naive(A, B)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
