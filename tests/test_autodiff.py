"""
Unit tests for autodiff engine.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01-math-fundamentals')))
from autodiff.autograd import Tensor


class TestAutodiff:
    """Test automatic differentiation."""
    
    def test_simple_gradient(self):
        """Test simple gradient computation."""
        
        x = Tensor([3.0], requires_grad=True)
        y = x ** 2
        y.backward()
        
        # dy/dx = 2x = 6
        assert np.allclose(x.grad, [6.0])
    
    def test_addition_gradient(self):
        """Test addition gradient."""
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        z = x + y
        z.backward()
        
        assert np.allclose(x.grad, [1.0])
        assert np.allclose(y.grad, [1.0])
    
    def test_multiplication_gradient(self):
        """Test multiplication gradient."""
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        z = x * y
        z.backward()
        
        # dz/dx = y = 3, dz/dy = x = 2
        assert np.allclose(x.grad, [3.0])
        assert np.allclose(y.grad, [2.0])
    
    def test_chain_rule(self):
        """Test chain rule."""
        x = Tensor([1.0], requires_grad=True)
        y = (x + Tensor([2.0])) * Tensor([3.0])
        y.backward()
        
        # y = 3(x + 2), dy/dx = 3
        assert np.allclose(x.grad, [3.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
