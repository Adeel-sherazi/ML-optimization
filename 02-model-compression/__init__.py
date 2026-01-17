"""Model compression module."""

from .quantization.int8_quantization import QuantizationHelper
from .pruning.magnitude_pruning import PruningHelper

__all__ = ['QuantizationHelper', 'PruningHelper']
