# Model Compression Techniques

Production-ready implementations of model compression techniques for deployment.

## Overview

Model compression reduces model size and inference time while maintaining accuracy:

1. **Quantization**: Reduce numerical precision (FP32 → INT8)
2. **Pruning**: Remove redundant parameters
3. **Knowledge Distillation**: Train smaller models to mimic larger ones

## 1. Quantization

### Types of Quantization

#### Post-Training Dynamic Quantization
- Quantizes weights to INT8
- Activations quantized dynamically at runtime
- No calibration data needed
- Best for: LSTM, Transformers

```python
from quantization.int8_quantization import QuantizationHelper

model_quantized = QuantizationHelper.dynamic_quantization(model_fp32)
```

#### Post-Training Static Quantization
- Quantizes both weights and activations
- Requires calibration data
- Better accuracy than dynamic
- Best for: CNNs

```python
model_quantized = QuantizationHelper.post_training_static_quantization(
    model_fp32,
    calibration_dataloader
)
```

#### Quantization-Aware Training (QAT)
- Simulates quantization during training
- Best accuracy preservation
- Requires retraining

```python
model_prepared = QuantizationHelper.prepare_for_quantization(model)
# Train model_prepared
model_quantized = QuantizationHelper.convert_to_quantized(model_prepared)
```

### Expected Results

| Model Type | Size Reduction | Speedup (CPU) | Accuracy Drop |
|------------|----------------|---------------|---------------|
| ResNet-50  | 4x             | 2-4x          | <1%           |
| MobileNet  | 4x             | 2-3x          | <0.5%         |
| BERT       | 4x             | 2-4x          | <2%           |

## 2. Pruning

### Unstructured Pruning

Removes individual weights based on magnitude:

```python
from pruning.magnitude_pruning import PruningHelper

# Prune 30% of weights
PruningHelper.magnitude_pruning(model, amount=0.3, structured=False)
```

**Pros**: Higher compression ratios (50-90% sparsity)  
**Cons**: Requires sparse compute support for speedup

### Structured Pruning

Removes entire filters/channels:

```python
# Prune 20% of filters
PruningHelper.magnitude_pruning(model, amount=0.2, structured=True)
```

**Pros**: Actual speedup on any hardware  
**Cons**: Lower compression ratios (20-40% sparsity)

### Global Pruning

Prunes globally across all layers:

```python
PruningHelper.global_pruning(model, amount=0.4)
```

### Iterative Magnitude Pruning (IMP)

Gradually prune and fine-tune:

```python
model = PruningHelper.iterative_pruning(
    model,
    train_fn=train_function,
    val_fn=validation_function,
    final_sparsity=0.9
)
```

### Expected Results

| Sparsity | Accuracy Drop | Notes                    |
|----------|---------------|--------------------------|
| 50%      | <1%           | Easy to achieve          |
| 70%      | 1-2%          | Requires fine-tuning     |
| 90%      | 2-5%          | Needs careful tuning     |
| 95%+     | >5%           | Model-dependent          |

## 3. Knowledge Distillation

Train a smaller "student" model to mimic a larger "teacher" model.

### Temperature Scaling

Soften teacher predictions to transfer knowledge:

```python
from distillation.knowledge_distillation import DistillationTrainer

trainer = DistillationTrainer(
    teacher_model=large_model,
    student_model=small_model,
    temperature=3.0,
    alpha=0.7
)
```

### Expected Results

| Setup              | Size Reduction | Accuracy Gap |
|--------------------|----------------|--------------|
| Teacher: ResNet-50 | 10x smaller    | 2-3%         |
| Student: MobileNet |                |              |

## Combining Techniques

Maximum compression with minimal accuracy loss:

1. **Distillation** → Train smaller model
2. **Quantization-Aware Training** → Add quantization
3. **Pruning** → Remove redundant weights
4. **TensorRT** → Deploy optimized engine

### Combined Results

- **Size**: 40-100x reduction
- **Speed**: 10-50x faster inference
- **Accuracy**: 3-5% drop (acceptable for many tasks)

## Best Practices

### Quantization
1. Always validate accuracy on validation set
2. Use symmetric quantization for better hardware support
3. Per-channel quantization for better accuracy
4. Consider mixed precision (critical layers in FP16)

### Pruning
1. Start with low sparsity, increase gradually
2. Fine-tune after each pruning iteration
3. Use global pruning for better results
4. Combine with quantization for maximum compression

### Knowledge Distillation
1. Use temperature scaling (T=2-5)
2. Balance hard and soft targets (α=0.5-0.8)
3. Match intermediate features for better transfer
4. Use data augmentation

## Hardware Considerations

| Technique          | CPU | GPU | Mobile | Edge (TPU/NPU) |
|--------------------|-----|-----|--------|----------------|
| FP16 Quantization  | ✗   | ✓   | ✓      | ✓              |
| INT8 Quantization  | ✓   | ✓   | ✓      | ✓              |
| Unstructured Prune | ✗   | ✗*  | ✗      | ✗              |
| Structured Prune   | ✓   | ✓   | ✓      | ✓              |

*Requires sparse tensor cores (Ampere+)

## Debugging Common Issues

### Accuracy Drop Too High
- Use QAT instead of post-training quantization
- Reduce sparsity percentage
- Fine-tune for more epochs
- Check for layer sensitivity (avoid pruning critical layers)

### No Speedup Observed
- Structured pruning instead of unstructured
- Ensure INT8 kernels are being used
- Check batch size (quantization benefits increase with batch size)
- Profile to identify bottlenecks

## References

- [Quantization: Lower Numerical Precision](https://arxiv.org/abs/2004.09602)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
