#!/usr/bin/env python3
"""
Simple validation script to verify repository structure.
"""

import os
import sys


def check_file(path, description):
    """Check if file exists."""
    exists = os.path.isfile(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists


def check_dir(path, description):
    """Check if directory exists."""
    exists = os.path.isdir(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists


def main():
    print("=" * 70)
    print("ML Optimization Repository Validation")
    print("=" * 70)
    
    all_good = True
    
    # Check core files
    print("\nCore Files:")
    all_good &= check_file("README.md", "Main README")
    all_good &= check_file("requirements.txt", "Requirements")
    all_good &= check_file("setup.py", "Setup script")
    all_good &= check_file("CONTRIBUTING.md", "Contributing guide")
    all_good &= check_file("LICENSE", "License")
    
    # Check module directories
    print("\nModule Directories:")
    all_good &= check_dir("01-math-fundamentals", "Math fundamentals")
    all_good &= check_dir("02-model-compression", "Model compression")
    all_good &= check_dir("03-cuda-kernels", "CUDA kernels")
    all_good &= check_dir("04-tensorrt-optimization", "TensorRT")
    all_good &= check_dir("05-video-pipeline", "Video pipeline")
    all_good &= check_dir("06-end-to-end-systems", "End-to-end systems")
    
    # Check key implementations
    print("\nKey Implementations:")
    all_good &= check_file("01-math-fundamentals/matrix_ops/gemm.py", "Matrix ops")
    all_good &= check_file("01-math-fundamentals/autodiff/autograd.py", "Autodiff")
    all_good &= check_file("02-model-compression/quantization/int8_quantization.py", "Quantization")
    all_good &= check_file("02-model-compression/pruning/magnitude_pruning.py", "Pruning")
    all_good &= check_file("03-cuda-kernels/basics/cuda_basics.py", "CUDA basics")
    all_good &= check_file("04-tensorrt-optimization/model_conversion/pytorch_to_tensorrt.py", "TensorRT")
    all_good &= check_file("05-video-pipeline/streaming/realtime_inference.py", "Video pipeline")
    all_good &= check_file("06-end-to-end-systems/image_classification/optimized_classifier.py", "Classifier")
    
    # Check documentation
    print("\nDocumentation:")
    all_good &= check_dir("docs", "Docs directory")
    all_good &= check_file("docs/01-math-fundamentals.md", "Math docs")
    all_good &= check_file("docs/02-model-compression.md", "Compression docs")
    all_good &= check_file("docs/03-cuda-kernels.md", "CUDA docs")
    
    # Check tests
    print("\nTests:")
    all_good &= check_dir("tests", "Tests directory")
    all_good &= check_file("tests/test_matrix_ops.py", "Matrix tests")
    all_good &= check_file("tests/test_autodiff.py", "Autodiff tests")
    
    # Check CI/CD
    print("\nCI/CD:")
    all_good &= check_file(".github/workflows/ci.yml", "CI workflow")
    
    # Summary
    print("\n" + "=" * 70)
    if all_good:
        print("✓ All checks passed! Repository structure is complete.")
        return 0
    else:
        print("✗ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
