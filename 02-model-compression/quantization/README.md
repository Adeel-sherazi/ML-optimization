# Quantization Module

Production-ready model quantization implementations for PyTorch.

## Files

- `int8_quantization.py` - INT8 quantization examples and benchmarks

## Quantization Types

### 1. Dynamic Quantization

Quantizes weights to INT8, activations computed dynamically.

```python
from int8_quantization import QuantizationHelper

model_quantized = QuantizationHelper.dynamic_quantization(model)
```

**Best for**: LSTM, Transformers, models with variable input sizes

### 2. Static Quantization

Quantizes both weights and activations using calibration data.

```python
model_quantized = QuantizationHelper.post_training_static_quantization(
    model,
    calibration_dataloader
)
```

**Best for**: CNNs with fixed input size

### 3. Quantization-Aware Training (QAT)

Simulates quantization during training for best accuracy.

```python
model_prepared = QuantizationHelper.prepare_for_quantization(model)
# Train model_prepared
model_quantized = QuantizationHelper.convert_to_quantized(model_prepared)
```

**Best for**: When accuracy drop is unacceptable

## Running Examples

```bash
python int8_quantization.py
```

This demonstrates:
- Dynamic quantization on a simple CNN
- Performance comparison (FP32 vs INT8)
- Output difference analysis
- Best practices

## Expected Results

- **Model Size**: ~4x reduction (FP32 → INT8)
- **Inference Speed**: 2-4x faster on CPU
- **Accuracy Drop**: <1% with proper calibration

## Hardware Support

| Platform      | INT8 Support | Notes                    |
|---------------|--------------|--------------------------|
| Intel CPU     | ✓            | VNNI instructions        |
| ARM CPU       | ✓            | dot product instructions |
| NVIDIA GPU    | ✓            | Tensor Cores (sm_75+)    |
| AMD GPU       | Partial      | Via ROCm                 |
| Edge Devices  | ✓            | TPU, NPU support         |

## Best Practices

1. Always validate accuracy on validation set
2. Use calibration data representative of production
3. Consider per-channel quantization for better accuracy
4. Profile to ensure speedup is achieved
5. Combine with TensorRT for maximum performance

## Troubleshooting

**No speedup observed?**
- Check if INT8 kernels are being used
- Ensure hardware supports INT8
- Try larger batch sizes
- Profile with `torch.profiler`

**High accuracy drop?**
- Use QAT instead of post-training
- Increase calibration data
- Try per-channel quantization
- Keep critical layers in FP32 (mixed precision)
