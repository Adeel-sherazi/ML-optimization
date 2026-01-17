"""
Production-ready model pruning implementation.

Demonstrates structured and unstructured pruning techniques.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Tuple, List
import numpy as np


class PruningHelper:
    """Helper class for model pruning operations."""
    
    @staticmethod
    def magnitude_pruning(
        model: nn.Module,
        amount: float = 0.3,
        structured: bool = False
    ) -> nn.Module:
        """
        Apply magnitude-based pruning to model.
        
        Args:
            model: PyTorch model to prune
            amount: Fraction of parameters to prune (0.0 to 1.0)
            structured: If True, use structured pruning; else unstructured
            
        Returns:
            Pruned model
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if structured and isinstance(module, nn.Conv2d):
                    # Structured pruning (prune entire filters)
                    prune.ln_structured(
                        module,
                        name='weight',
                        amount=amount,
                        n=2,
                        dim=0  # Prune output channels
                    )
                else:
                    # Unstructured pruning (prune individual weights)
                    prune.l1_unstructured(
                        module,
                        name='weight',
                        amount=amount
                    )
        
        return model
    
    @staticmethod
    def global_pruning(
        model: nn.Module,
        amount: float = 0.3
    ) -> nn.Module:
        """
        Apply global magnitude pruning across all layers.
        
        Args:
            model: PyTorch model to prune
            amount: Global fraction of parameters to prune
            
        Returns:
            Globally pruned model
        """
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        
        return model
    
    @staticmethod
    def iterative_pruning(
        model: nn.Module,
        train_fn,
        val_fn,
        initial_sparsity: float = 0.1,
        final_sparsity: float = 0.9,
        num_iterations: int = 5
    ) -> nn.Module:
        """
        Apply iterative magnitude pruning (IMP - Iterative Magnitude Pruning).
        
        Args:
            model: PyTorch model
            train_fn: Training function
            val_fn: Validation function
            initial_sparsity: Starting sparsity level
            final_sparsity: Target sparsity level
            num_iterations: Number of pruning iterations
            
        Returns:
            Pruned model
        """
        current_sparsity = initial_sparsity
        sparsity_increment = (final_sparsity - initial_sparsity) / num_iterations
        
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}, Sparsity: {current_sparsity:.2%}")
            
            # Prune
            PruningHelper.magnitude_pruning(model, amount=current_sparsity)
            
            # Fine-tune
            train_fn(model)
            
            # Evaluate
            accuracy = val_fn(model)
            print(f"  Accuracy: {accuracy:.2%}")
            
            current_sparsity += sparsity_increment
        
        return model
    
    @staticmethod
    def remove_pruning(model: nn.Module) -> nn.Module:
        """
        Remove pruning reparameterization and make pruning permanent.
        
        Args:
            model: Pruned model
            
        Returns:
            Model with pruning made permanent
        """
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass  # No pruning on this module
        
        return model
    
    @staticmethod
    def get_sparsity(model: nn.Module) -> Tuple[float, dict]:
        """
        Calculate model sparsity.
        
        Args:
            model: PyTorch model
            
        Returns:
            Tuple of (overall_sparsity, layer_sparsity_dict)
        """
        total_params = 0
        zero_params = 0
        layer_sparsity = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                layer_zeros = torch.sum(weight == 0).item()
                layer_total = weight.numel()
                
                total_params += layer_total
                zero_params += layer_zeros
                
                layer_sparsity[name] = layer_zeros / layer_total if layer_total > 0 else 0
        
        overall_sparsity = zero_params / total_params if total_params > 0 else 0
        
        return overall_sparsity, layer_sparsity
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Tuple[int, int]:
        """
        Count total and non-zero parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Tuple of (total_params, non_zero_params)
        """
        total = sum(p.numel() for p in model.parameters())
        non_zero = sum(torch.count_nonzero(p).item() for p in model.parameters())
        
        return total, non_zero


class SimpleCNN(nn.Module):
    """Simple CNN for pruning demonstration."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def demonstrate_pruning():
    """Demonstrate different pruning techniques."""
    print("Model Pruning Demonstration")
    print("=" * 70)
    
    # Create model
    model = SimpleCNN()
    
    # Original model stats
    total_params, non_zero_params = PruningHelper.count_parameters(model)
    print(f"\n1. Original Model")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Non-zero parameters: {non_zero_params:,}")
    
    # Unstructured pruning
    print(f"\n2. Unstructured Magnitude Pruning (30%)")
    model_unstructured = SimpleCNN()
    PruningHelper.magnitude_pruning(model_unstructured, amount=0.3, structured=False)
    
    sparsity, layer_sparsity = PruningHelper.get_sparsity(model_unstructured)
    total, non_zero = PruningHelper.count_parameters(model_unstructured)
    
    print(f"   Overall sparsity: {sparsity:.2%}")
    print(f"   Non-zero parameters: {non_zero:,}")
    print(f"   Compression ratio: {total_params / non_zero:.2f}x")
    
    print(f"\n   Layer-wise sparsity:")
    for name, sparse in layer_sparsity.items():
        if sparse > 0:
            print(f"      {name}: {sparse:.2%}")
    
    # Structured pruning
    print(f"\n3. Structured Pruning (20% of filters)")
    model_structured = SimpleCNN()
    PruningHelper.magnitude_pruning(model_structured, amount=0.2, structured=True)
    
    sparsity, _ = PruningHelper.get_sparsity(model_structured)
    total, non_zero = PruningHelper.count_parameters(model_structured)
    
    print(f"   Overall sparsity: {sparsity:.2%}")
    print(f"   Non-zero parameters: {non_zero:,}")
    print(f"   Note: Structured pruning enables actual speedup")
    
    # Global pruning
    print(f"\n4. Global Pruning (40%)")
    model_global = SimpleCNN()
    PruningHelper.global_pruning(model_global, amount=0.4)
    
    sparsity, layer_sparsity = PruningHelper.get_sparsity(model_global)
    total, non_zero = PruningHelper.count_parameters(model_global)
    
    print(f"   Overall sparsity: {sparsity:.2%}")
    print(f"   Non-zero parameters: {non_zero:,}")
    print(f"   Compression ratio: {total_params / non_zero:.2f}x")
    
    # Test forward pass
    print(f"\n5. Forward Pass Verification")
    dummy_input = torch.randn(1, 3, 32, 32)
    
    output_original = model(dummy_input)
    output_pruned = model_unstructured(dummy_input)
    
    print(f"   Original output shape: {output_original.shape}")
    print(f"   Pruned output shape: {output_pruned.shape}")
    print(f"   Output difference: {torch.abs(output_original - output_pruned).mean().item():.6f}")
    
    print("\n" + "=" * 70)
    print("Pruning Best Practices:")
    print("1. Unstructured pruning: Better compression, needs sparse compute support")
    print("2. Structured pruning: Lower compression, but actual speedup on any hardware")
    print("3. Global pruning: Better than layer-wise for similar sparsity")
    print("4. Iterative pruning: Prune gradually during training for best accuracy")
    print("5. Fine-tuning: Always fine-tune after pruning to recover accuracy")


if __name__ == "__main__":
    demonstrate_pruning()
    
    print("\n" + "=" * 70)
    print("Pruning Summary:")
    print("- Typical sparsity: 50-90% depending on model and task")
    print("- Always measure accuracy impact")
    print("- Combine with quantization for maximum compression")
    print("- Use lottery ticket hypothesis for training from scratch")
