# Automatic Differentiation Module

Simple automatic differentiation engine similar to PyTorch's autograd.

## Files

- `autograd.py` - Autodiff engine with Tensor class and optimizer

## Features

### Tensor Operations

- Addition: `a + b`
- Multiplication: `a * b`
- Power: `a ** n`
- Matrix multiplication: `a.matmul(b)`
- ReLU activation: `a.relu()`
- Sum reduction: `a.sum()`

### Gradient Computation

```python
from autograd import Tensor

# Forward pass
x = Tensor([3.0], requires_grad=True)
y = x ** 2
print(f"y = {y.data[0]}")  # 9.0

# Backward pass
y.backward()
print(f"dy/dx = {x.grad[0]}")  # 6.0
```

### Optimizer

```python
from autograd import Tensor, SGD

# Parameters
W = Tensor(np.random.randn(3, 4), requires_grad=True)
b = Tensor(np.zeros(4), requires_grad=True)

# Optimizer
optimizer = SGD([W, b], lr=0.01)

# Training step
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Running Examples

```bash
python autograd.py
```

This will run through several test cases demonstrating:
- Simple gradients (x²)
- Chain rule
- Matrix multiplication gradients
- Simple 2-layer neural network

## How It Works

1. **Forward Pass**: Build computational graph
2. **Backward Pass**: Topological sort + reverse-mode autodiff
3. **Gradient Accumulation**: Support for multiple backward passes

## Key Concepts

- Computational graph
- Reverse-mode automatic differentiation
- Chain rule
- Gradient accumulation

## Limitations

This is a simple educational implementation. For production:
- Use PyTorch, TensorFlow, or JAX
- This implementation doesn't support:
  - Advanced operations (convolution, pooling)
  - GPU acceleration
  - Efficient memory management
  - Advanced optimizers (Adam, RMSprop, etc.)
