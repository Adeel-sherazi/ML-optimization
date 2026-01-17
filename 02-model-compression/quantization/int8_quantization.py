"""
Production-ready INT8 quantization for PyTorch models.

Demonstrates post-training quantization and quantization-aware training.
"""

import os
import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Tuple
import time
import numpy as np


class SimpleModel(nn.Module):
    """Simple CNN for demonstration."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class QuantizationHelper:
    """Helper class for model quantization operations."""
    
    @staticmethod
    def prepare_for_quantization(model: nn.Module) -> nn.Module:
        """
        Prepare model for quantization-aware training.
        
        Args:
            model: PyTorch model to quantize
            
        Returns:
            Model prepared for quantization
        """
        # Set quantization config
        model.qconfig = quant.get_default_qat_qconfig('fbgemm')
        
        # Prepare for quantization-aware training
        model_prepared = quant.prepare_qat(model)
        
        return model_prepared
    
    @staticmethod
    def convert_to_quantized(model: nn.Module) -> nn.Module:
        """
        Convert prepared model to quantized version.
        
        Args:
            model: Prepared model
            
        Returns:
            Quantized model
        """
        model.eval()
        model_quantized = quant.convert(model)
        return model_quantized
    
    @staticmethod
    def post_training_static_quantization(
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader
    ) -> nn.Module:
        """
        Perform post-training static quantization.
        
        Args:
            model: FP32 model
            calibration_data: Data loader for calibration
            
        Returns:
            Quantized INT8 model
        """
        # Fuse modules for better performance
        model.eval()
        
        # Set quantization configuration
        model.qconfig = quant.get_default_qconfig('fbgemm')
        
        # Prepare for calibration
        model_prepared = quant.prepare(model)
        
        # Calibrate with representative dataset
        with torch.no_grad():
            for batch in calibration_data:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch
                model_prepared(inputs)
        
        # Convert to quantized model
        model_quantized = quant.convert(model_prepared)
        
        return model_quantized
    
    @staticmethod
    def dynamic_quantization(model: nn.Module) -> nn.Module:
        """
        Perform dynamic quantization (weights only).
        
        Args:
            model: FP32 model
            
        Returns:
            Dynamically quantized model
        """
        model_quantized = quant.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        return model_quantized
    
    @staticmethod
    def measure_model_size(model: nn.Module) -> float:
        """
        Measure model size in MB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in megabytes
        """
        torch.save(model.state_dict(), "/tmp/temp_model.pth")
        size_mb = os.path.getsize("/tmp/temp_model.pth") / (1024 * 1024)
        os.remove("/tmp/temp_model.pth")
        return size_mb
    
    @staticmethod
    def benchmark_inference(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup: int = 10
    ) -> float:
        """
        Benchmark model inference time.
        
        Args:
            model: Model to benchmark
            input_shape: Shape of input tensor
            num_iterations: Number of iterations for benchmarking
            warmup: Number of warmup iterations
            
        Returns:
            Average inference time in milliseconds
        """
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        
        avg_time = (time.time() - start_time) / num_iterations * 1000
        return avg_time


def demonstrate_quantization():
    """Demonstrate different quantization techniques."""
    print("INT8 Quantization Demonstration")
    print("=" * 70)
    
    # Create model
    model_fp32 = SimpleModel(num_classes=10)
    model_fp32.eval()
    
    print("\n1. Original FP32 Model")
    print(f"   Model architecture: Simple CNN")
    print(f"   Input shape: (1, 3, 32, 32)")
    
    # Create dummy calibration data
    calibration_data = [torch.randn(8, 3, 32, 32) for _ in range(10)]
    
    # Dynamic quantization
    print("\n2. Dynamic Quantization (weights only)")
    model_dynamic = QuantizationHelper.dynamic_quantization(model_fp32)
    
    # Test inference
    dummy_input = torch.randn(1, 3, 32, 32)
    output_fp32 = model_fp32(dummy_input)
    output_dynamic = model_dynamic(dummy_input)
    
    print(f"   FP32 output mean: {output_fp32.mean().item():.4f}")
    print(f"   Dynamic quantized output mean: {output_dynamic.mean().item():.4f}")
    print(f"   Relative difference: {torch.abs(output_fp32 - output_dynamic).mean().item():.6f}")
    
    # Benchmark
    print("\n3. Performance Comparison")
    print(f"{'Model Type':<25} {'Inference Time (ms)':<25} {'Speedup':<15}")
    print("-" * 70)
    
    fp32_time = QuantizationHelper.benchmark_inference(model_fp32, (1, 3, 32, 32))
    dynamic_time = QuantizationHelper.benchmark_inference(model_dynamic, (1, 3, 32, 32))
    
    print(f"{'FP32':<25} {fp32_time:<25.3f} {'1.00x':<15}")
    print(f"{'Dynamic INT8':<25} {dynamic_time:<25.3f} {f'{fp32_time/dynamic_time:.2f}x':<15}")
    
    print("\n" + "=" * 70)
    print("Note: Static quantization requires calibration data")
    print("Note: QAT (Quantization-Aware Training) provides best accuracy")
    print("\nBest Practices:")
    print("1. Use dynamic quantization for LSTM/Transformer models")
    print("2. Use static quantization for CNNs with calibration data")
    print("3. Use QAT when accuracy drop is significant")
    print("4. Always validate accuracy after quantization")


if __name__ == "__main__":
    demonstrate_quantization()
    
    print("\n" + "=" * 70)
    print("Quantization complete! Key takeaways:")
    print("- INT8 quantization can reduce model size by ~4x")
    print("- Inference speedup varies by hardware (CPU: 2-4x, specialized HW: >10x)")
    print("- Always measure accuracy degradation")
    print("- TensorRT provides additional optimizations for NVIDIA GPUs")
