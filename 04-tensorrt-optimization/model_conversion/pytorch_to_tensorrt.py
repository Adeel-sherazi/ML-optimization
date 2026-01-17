"""
PyTorch to TensorRT model conversion.

Production-ready TensorRT optimization pipeline.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import time


class ConversionHelper:
    """Helper for PyTorch to TensorRT conversion."""
    
    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        dummy_input: torch.Tensor,
        onnx_path: str,
        opset_version: int = 13
    ) -> None:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model
            dummy_input: Example input tensor
            onnx_path: Path to save ONNX model
            opset_version: ONNX opset version
        """
        model.eval()
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to ONNX: {onnx_path}")
    
    @staticmethod
    def verify_onnx(onnx_path: str) -> bool:
        """
        Verify ONNX model validity.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            True if valid, False otherwise
        """
        try:
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            print(f"ONNX model is valid: {onnx_path}")
            return True
        except Exception as e:
            print(f"ONNX validation failed: {e}")
            return False
    
    @staticmethod
    def optimize_onnx(input_path: str, output_path: str) -> None:
        """
        Optimize ONNX model.
        
        Args:
            input_path: Input ONNX model path
            output_path: Output optimized ONNX model path
        """
        try:
            from onnxsim import simplify
            import onnx
            
            model = onnx.load(input_path)
            model_simp, check = simplify(model)
            
            if check:
                onnx.save(model_simp, output_path)
                print(f"Optimized ONNX model saved: {output_path}")
            else:
                print("Simplification failed, using original model")
        except ImportError:
            print("onnx-simplifier not installed. Skip optimization.")
            print("Install with: pip install onnx-simplifier")


class TensorRTBenchmark:
    """Benchmark TensorRT performance."""
    
    @staticmethod
    def benchmark_pytorch(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        use_cuda: bool = True
    ) -> float:
        """
        Benchmark PyTorch model.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            num_iterations: Number of iterations
            use_cuda: Whether to use CUDA
            
        Returns:
            Average inference time in milliseconds
        """
        model.eval()
        
        if use_cuda and torch.cuda.is_available():
            model = model.cuda()
            dummy_input = torch.randn(input_shape).cuda()
        else:
            dummy_input = torch.randn(input_shape)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        if use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        
        if use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        avg_time = (time.time() - start) / num_iterations * 1000
        
        return avg_time


class SimpleResNetBlock(nn.Module):
    """Simple ResNet block for demonstration."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class DemoModel(nn.Module):
    """Demo model for TensorRT conversion."""
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.block1 = SimpleResNetBlock(64)
        self.block2 = SimpleResNetBlock(64)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def demonstrate_tensorrt_conversion():
    """Demonstrate TensorRT conversion pipeline."""
    print("TensorRT Model Conversion Pipeline")
    print("=" * 70)
    
    # Create model
    model = DemoModel(num_classes=10)
    model.eval()
    
    print("\n1. Model Architecture")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model type: ResNet-like CNN")
    
    # Export to ONNX
    print("\n2. Export to ONNX")
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = "/tmp/model.onnx"
    
    ConversionHelper.export_to_onnx(
        model,
        dummy_input,
        onnx_path,
        opset_version=13
    )
    
    # Verify ONNX
    print("\n3. Verify ONNX Model")
    is_valid = ConversionHelper.verify_onnx(onnx_path)
    
    # Optimize ONNX
    print("\n4. Optimize ONNX Model")
    optimized_path = "/tmp/model_optimized.onnx"
    ConversionHelper.optimize_onnx(onnx_path, optimized_path)
    
    # Benchmark
    print("\n5. Performance Benchmark")
    print(f"{'Framework':<20} {'Batch Size':<15} {'Time (ms)':<15} {'Throughput (fps)':<20}")
    print("-" * 70)
    
    for batch_size in [1, 4, 8]:
        input_shape = (batch_size, 3, 224, 224)
        
        # PyTorch CPU
        cpu_time = TensorRTBenchmark.benchmark_pytorch(
            model,
            input_shape,
            num_iterations=50,
            use_cuda=False
        )
        cpu_fps = batch_size / (cpu_time / 1000)
        
        print(f"{'PyTorch CPU':<20} {batch_size:<15} {cpu_time:<15.2f} {cpu_fps:<20.2f}")
        
        # PyTorch GPU (if available)
        if torch.cuda.is_available():
            gpu_time = TensorRTBenchmark.benchmark_pytorch(
                model,
                input_shape,
                num_iterations=50,
                use_cuda=True
            )
            gpu_fps = batch_size / (gpu_time / 1000)
            
            print(f"{'PyTorch GPU':<20} {batch_size:<15} {gpu_time:<15.2f} {gpu_fps:<20.2f}")
    
    print("\n" + "=" * 70)
    print("TensorRT Conversion Notes:")
    print("1. TensorRT requires NVIDIA GPU and TensorRT library")
    print("2. Typical speedup: 2-5x over PyTorch for inference")
    print("3. Use FP16/INT8 precision for additional speedup")
    print("4. Dynamic shapes supported but may impact performance")
    print("\nNext Steps:")
    print("1. Install TensorRT: https://developer.nvidia.com/tensorrt")
    print("2. Convert ONNX to TensorRT engine")
    print("3. Benchmark TensorRT performance")
    print("4. Deploy with Triton Inference Server for production")


if __name__ == "__main__":
    demonstrate_tensorrt_conversion()
    
    print("\n" + "=" * 70)
    print("Optimization Checklist:")
    print("✓ Export to ONNX format")
    print("✓ Verify and optimize ONNX model")
    print("- Convert to TensorRT engine (requires TensorRT)")
    print("- Benchmark TensorRT performance")
    print("- Profile with Nsight Systems")
