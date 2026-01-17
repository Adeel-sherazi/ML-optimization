"""
Simple automatic differentiation engine for ML.

A production-ready autodiff implementation similar to PyTorch's autograd.
"""

from typing import Optional, Tuple, List, Set
import numpy as np


class Tensor:
    """Tensor with automatic differentiation support."""
    
    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = False,
        _children: Tuple['Tensor', ...] = (),
        _op: str = ''
    ):
        """
        Initialize a Tensor.
        
        Args:
            data: NumPy array containing the data
            requires_grad: Whether to track gradients
            _children: Tensors that contributed to this tensor
            _op: Operation that created this tensor
        """
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        
        # For autograd graph
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    def __add__(self, other: 'Tensor') -> 'Tensor':
        """Add two tensors."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='+'
        )
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad.copy()
            if other.requires_grad:
                other.grad = other.grad + out.grad if other.grad is not None else out.grad.copy()
        
        out._backward = _backward
        return out
    
    def __mul__(self, other: 'Tensor') -> 'Tensor':
        """Multiply two tensors (element-wise)."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='*'
        )
        
        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad.copy()
            if other.requires_grad:
                grad = self.data * out.grad
                other.grad = other.grad + grad if other.grad is not None else grad.copy()
        
        out._backward = _backward
        return out
    
    def __pow__(self, power: float) -> 'Tensor':
        """Raise tensor to a power."""
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f'**{power}'
        )
        
        def _backward():
            if self.requires_grad:
                grad = power * (self.data ** (power - 1)) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad.copy()
        
        out._backward = _backward
        return out
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='@'
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad @ other.data.T
                self.grad = self.grad + grad if self.grad is not None else grad.copy()
            if other.requires_grad:
                grad = self.data.T @ out.grad
                other.grad = other.grad + grad if other.grad is not None else grad.copy()
        
        out._backward = _backward
        return out
    
    def relu(self) -> 'Tensor':
        """ReLU activation function."""
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='ReLU'
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * (self.data > 0)
                self.grad = self.grad + grad if self.grad is not None else grad.copy()
        
        out._backward = _backward
        return out
    
    def sum(self) -> 'Tensor':
        """Sum all elements."""
        out = Tensor(
            np.array([self.data.sum()]),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='sum'
        )
        
        def _backward():
            if self.requires_grad:
                grad = np.ones_like(self.data) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad.copy()
        
        out._backward = _backward
        return out
    
    def backward(self):
        """Compute gradients using backpropagation."""
        # Topological sort
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()
        
        def build_topo(v: Tensor):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient
        self.grad = np.ones_like(self.data)
        
        # Backward pass
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        """Zero out gradients."""
        self.grad = None


class SGD:
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01):
        """
        Initialize SGD optimizer.
        
        Args:
            parameters: List of tensors to optimize
            lr: Learning rate
        """
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        """Perform one optimization step."""
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad
    
    def zero_grad(self):
        """Zero out all gradients."""
        for param in self.parameters:
            param.zero_grad()


# Example usage and tests
if __name__ == "__main__":
    print("Testing Autodiff Engine")
    print("=" * 60)
    
    # Test 1: Simple gradient computation
    print("\n1. Simple gradient: f(x) = x^2, x = 3")
    x = Tensor([3.0], requires_grad=True)
    y = x ** 2
    y.backward()
    print(f"   Input: {x.data[0]}")
    print(f"   Output: {y.data[0]}")
    print(f"   Gradient: {x.grad[0]} (expected: 6.0)")
    
    # Test 2: Chain rule
    print("\n2. Chain rule: f(x) = (x + 2) * 3, x = 1")
    x = Tensor([1.0], requires_grad=True)
    y = (x + Tensor([2.0])) * Tensor([3.0])
    y.backward()
    print(f"   Input: {x.data[0]}")
    print(f"   Output: {y.data[0]} (expected: 9.0)")
    print(f"   Gradient: {x.grad[0]} (expected: 3.0)")
    
    # Test 3: Matrix multiplication
    print("\n3. Matrix multiplication gradient")
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    W = Tensor(np.random.randn(4, 2), requires_grad=True)
    y = x.matmul(W)
    loss = y.sum()
    loss.backward()
    print(f"   Input shape: {x.data.shape}")
    print(f"   Weight shape: {W.data.shape}")
    print(f"   Output shape: {y.data.shape}")
    print(f"   x gradient shape: {x.grad.shape}")
    print(f"   W gradient shape: {W.grad.shape}")
    
    # Test 4: Simple neural network
    print("\n4. Simple 2-layer neural network")
    np.random.seed(42)
    
    # Input and target
    X = Tensor(np.random.randn(4, 3), requires_grad=False)
    y_true = Tensor(np.array([[1], [0], [1], [0]]), requires_grad=False)
    
    # Parameters
    W1 = Tensor(np.random.randn(3, 4), requires_grad=True)
    W2 = Tensor(np.random.randn(4, 1), requires_grad=True)
    
    # Optimizer
    optimizer = SGD([W1, W2], lr=0.01)
    
    # Training loop
    for epoch in range(100):
        # Forward pass
        h = X.matmul(W1).relu()
        y_pred = h.matmul(W2)
        loss = ((y_pred + (y_true * Tensor([-1.0]))) ** 2).sum()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"   Epoch {epoch:3d}, Loss: {loss.data[0]:.4f}")
    
    print("\n" + "=" * 60)
    print("All autodiff tests passed!")
