# Contributing to ML Optimization

Thank you for your interest in contributing to this project! This document provides guidelines for contributions.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. Create a detailed bug report including:
   - Environment (OS, Python version, CUDA version)
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and stack traces

### Suggesting Enhancements

1. Check existing issues and pull requests
2. Create an issue describing:
   - Use case and motivation
   - Proposed solution
   - Alternative approaches considered

### Pull Requests

1. **Fork the repository** and create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for new functionality

4. **Update documentation** as needed

5. **Run tests and linting**
   ```bash
   pytest tests/
   black .
   isort .
   flake8 .
   ```

6. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: description"
   ```

7. **Push and create pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting
- Use type hints where appropriate
- Maximum line length: 100 characters

### Code Quality

```python
# Good: Clear, documented, type-hinted
def matrix_multiply(
    A: np.ndarray,
    B: np.ndarray
) -> np.ndarray:
    """
    Multiply two matrices.
    
    Args:
        A: Matrix of shape (M, K)
        B: Matrix of shape (K, N)
        
    Returns:
        Result matrix of shape (M, N)
    """
    return A @ B
```

### CUDA Code

- Clear kernel naming
- Document thread/block configuration
- Include performance notes
- Error checking

```cuda
// Good: Well-documented kernel
__global__ void optimized_kernel(
    const float* input,
    float* output,
    int n
) {
    // Thread configuration: 256 threads per block
    // Expected performance: ~500 GB/s on A100
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}
```

### Documentation

- **Docstrings**: All public functions and classes
- **Inline comments**: For complex logic
- **README**: Update if adding new modules
- **Examples**: Include usage examples

### Testing

- Unit tests for all new functionality
- Integration tests for complex features
- Performance benchmarks for optimizations
- Minimum 80% code coverage

```python
def test_matrix_multiply():
    """Test matrix multiplication correctness."""
    A = np.random.randn(64, 64)
    B = np.random.randn(64, 64)
    
    result = matrix_multiply(A, B)
    expected = A @ B
    
    assert np.allclose(result, expected)
```

## Project Structure

```
ML-optimization/
├── 01-math-fundamentals/     # Core math implementations
├── 02-model-compression/      # Quantization, pruning, distillation
├── 03-cuda-kernels/           # CUDA programming
├── 04-tensorrt-optimization/  # TensorRT integration
├── 05-video-pipeline/         # Video processing
├── 06-end-to-end-systems/     # Complete systems
├── tests/                     # Unit tests
├── docs/                      # Documentation
└── benchmarks/                # Performance benchmarks
```

## Adding New Modules

1. Create module directory in appropriate section
2. Add implementation with documentation
3. Add unit tests in `tests/`
4. Add documentation in `docs/`
5. Update main README.md
6. Add example usage

## Performance Requirements

- Optimized implementations should be >2x faster than naive
- Include benchmarks comparing naive vs optimized
- Document hardware requirements
- Profile with appropriate tools (nvprof, Nsight, etc.)

## Documentation Requirements

Each module should include:

1. **README or docs**: Purpose, usage, performance
2. **Docstrings**: All public APIs
3. **Examples**: Working code samples
4. **Benchmarks**: Performance comparisons

## Review Process

1. Automated checks (tests, linting) must pass
2. Code review by maintainer
3. Documentation review
4. Performance validation (if applicable)
5. Approval and merge

## Getting Help

- Open an issue for questions
- Join discussions for design decisions
- Tag maintainers for urgent matters

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Credited for significant features

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue for any questions about contributing!
