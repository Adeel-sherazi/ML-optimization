# Getting Started Guide

Complete guide to setting up and using the ML Optimization repository.

## Prerequisites

### Required
- Python 3.8 or higher
- pip package manager
- Git

### Optional (for full functionality)
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8+
- cuDNN 8.6+
- TensorRT 8.6+ (for TensorRT examples)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Adeel-sherazi/ML-optimization.git
cd ML-optimization
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n ml-opt python=3.10
conda activate ml-opt
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 4. Verify Installation

```bash
python validate_structure.py
```

You should see:
```
✓ All checks passed! Repository structure is complete.
```

## Quick Start Examples

### 1. Matrix Operations

Learn about optimized matrix operations:

```bash
cd 01-math-fundamentals/matrix_ops
python gemm.py
```

**What you'll learn:**
- Naive vs optimized implementations
- Cache-friendly algorithms
- Performance benchmarking

### 2. Automatic Differentiation

Understand backpropagation:

```bash
cd 01-math-fundamentals/autodiff
python autograd.py
```

**What you'll learn:**
- Building computational graphs
- Reverse-mode autodiff
- Simple neural network training

### 3. Model Quantization

Compress models with quantization:

```bash
cd 02-model-compression/quantization
python int8_quantization.py
```

**What you'll learn:**
- INT8 quantization
- Performance vs accuracy tradeoffs
- Hardware acceleration

### 4. Model Pruning

Remove redundant parameters:

```bash
cd 02-model-compression/pruning
python magnitude_pruning.py
```

**What you'll learn:**
- Structured vs unstructured pruning
- Sparsity levels
- Fine-tuning after pruning

### 5. CUDA Programming

GPU acceleration basics:

```bash
cd 03-cuda-kernels/basics
python cuda_basics.py
```

**What you'll learn:**
- CUDA programming model
- Memory hierarchy
- Performance comparison

### 6. TensorRT Conversion

Optimize for deployment:

```bash
cd 04-tensorrt-optimization/model_conversion
python pytorch_to_tensorrt.py
```

**What you'll learn:**
- PyTorch to ONNX conversion
- Model optimization
- Inference benchmarking

### 7. Video Pipeline

Real-time video processing:

```bash
cd 05-video-pipeline/streaming
python realtime_inference.py
```

**What you'll learn:**
- Dynamic batching
- Throughput optimization
- Latency management

### 8. Complete System

End-to-end optimized system:

```bash
cd 06-end-to-end-systems/image_classification
python optimized_classifier.py
```

**What you'll learn:**
- Combining all techniques
- Production deployment
- Performance monitoring

## Learning Path

### Week 1: Foundations
- [ ] Run all math fundamentals examples
- [ ] Understand autodiff engine
- [ ] Read docs/01-math-fundamentals.md

### Week 2: Model Compression
- [ ] Experiment with quantization
- [ ] Try different pruning strategies
- [ ] Read docs/02-model-compression.md

### Week 3: GPU Programming
- [ ] Learn CUDA basics
- [ ] Run kernel benchmarks
- [ ] Read docs/03-cuda-kernels.md

### Week 4: Advanced Topics
- [ ] TensorRT conversion
- [ ] Video pipeline optimization
- [ ] End-to-end system deployment

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_matrix_ops.py -v
```

## Benchmarking

```bash
# Run benchmarks
cd benchmarks
python run_benchmarks.py  # (create this to run all benchmarks)

# Or run individual benchmarks
python ../01-math-fundamentals/matrix_ops/gemm.py
python ../02-model-compression/quantization/int8_quantization.py
```

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Make sure you're in the repository root
cd /path/to/ML-optimization

# Reinstall in development mode
pip install -e .
```

### CUDA Not Available

If CUDA is not detected:

```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Module Not Found

If you get "No module named 'xyz'":

```bash
# Install missing dependencies
pip install -r requirements.txt

# Or install specific package
pip install xyz
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
```

### 2. Make Changes

Edit files, add features, fix bugs.

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Format Code

```bash
black .
isort .
flake8 .
```

### 5. Commit and Push

```bash
git add .
git commit -m "Add feature: description"
git push origin feature/my-feature
```

## Next Steps

### Advanced Topics
1. Implement custom CUDA kernels
2. Create custom TensorRT plugins
3. Optimize specific model architectures
4. Benchmark on different hardware

### Projects
1. Optimize your own model
2. Create deployment pipeline
3. Build production inference server
4. Contribute improvements

## Getting Help

### Documentation
- Main README: [README.md](../README.md)
- Project Overview: [docs/PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- Module docs: See docs/ directory

### Community
- GitHub Issues: Report bugs or ask questions
- Discussions: Share ideas and improvements

### Resources
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

## Tips for Success

1. **Start Simple**: Begin with math fundamentals before advanced topics
2. **Run Examples**: Execute all examples to see them in action
3. **Read Code**: Study the implementations, not just outputs
4. **Experiment**: Modify parameters and observe effects
5. **Profile**: Use profiling tools to understand performance
6. **Benchmark**: Always measure before and after optimization
7. **Document**: Keep notes on what works for your use case

## Common Workflows

### Optimize an Existing Model

1. Benchmark baseline performance
2. Apply quantization
3. Try pruning
4. Convert to TensorRT
5. Benchmark optimized version
6. Validate accuracy

### Deploy a Model

1. Export to ONNX
2. Optimize ONNX graph
3. Convert to TensorRT
4. Create inference pipeline
5. Add batching and preprocessing
6. Monitor performance

### Learn GPU Programming

1. Start with CUDA basics
2. Understand memory hierarchy
3. Implement simple kernels
4. Profile with Nsight
5. Optimize memory access
6. Build custom PyTorch extensions

---

Happy optimizing! 🚀
