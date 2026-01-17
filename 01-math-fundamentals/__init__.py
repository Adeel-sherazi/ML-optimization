"""Math fundamentals module for ML optimization."""

from .matrix_ops.gemm import MatrixOps
from .autodiff.autograd import Tensor, SGD

__all__ = ['MatrixOps', 'Tensor', 'SGD']
