#!/usr/bin/env python3
"""
Quick start example demonstrating all ML optimization techniques.
"""

import torch
import numpy as np

# Constants
SEPARATOR_WIDTH = 70


def main():
    print("=" * SEPARATOR_WIDTH)
    print("ML Optimization Quick Start")
    print("=" * SEPARATOR_WIDTH)
    
    # 1. Math Fundamentals
    print("\n1. Testing Matrix Operations...")
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from matrix_ops.gemm import MatrixOps
        
        A = np.random.randn(64, 64).astype(np.float32)
        B = np.random.randn(64, 64).astype(np.float32)
        C = MatrixOps.matmul_optimized(A, B)
        print(f"   ✓ Matrix multiplication: {A.shape} x {B.shape} = {C.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 2. Autodiff
    print("\n2. Testing Automatic Differentiation...")
    try:
        from autodiff.autograd import Tensor
        
        x = Tensor([3.0], requires_grad=True)
        y = x ** 2
        y.backward()
        print(f"   ✓ Gradient of x² at x=3: {x.grad[0]:.1f} (expected: 6.0)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 3. Model Compression
    print("\n3. Testing Model Compression...")
    try:
        import torch.nn as nn
        # Add model compression directory to path
        compression_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '02-model-compression')
        sys.path.insert(0, compression_path)
        from quantization.int8_quantization import QuantizationHelper
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        model_quantized = QuantizationHelper.dynamic_quantization(model)
        print(f"   ✓ Dynamic quantization applied")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 4. CUDA Basics
    print("\n4. Checking CUDA Availability...")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA version: {torch.version.cuda}")
    else:
        print(f"   ℹ CUDA not available (CPU only)")
    
    # 5. TensorRT
    print("\n5. Testing ONNX Export...")
    try:
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10)
        )
        
        dummy_input = torch.randn(1, 3, 32, 32)
        onnx_path = "/tmp/test_model.onnx"
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13
        )
        print(f"   ✓ Model exported to ONNX: {onnx_path}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("Quick Start Complete!")
    print("\nNext Steps:")
    print("1. Explore individual modules in each directory")
    print("2. Run examples: python 01-math-fundamentals/matrix_ops/gemm.py")
    print("3. Check documentation in docs/")
    print("4. Run tests: pytest tests/")
    print("=" * SEPARATOR_WIDTH)


if __name__ == "__main__":
    main()
