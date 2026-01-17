# Matrix Operations Module

Production-ready matrix operation implementations demonstrating optimization techniques.

## Files

- `gemm.py` - General Matrix Multiply (GEMM) implementations and benchmarks

## Features

### Implementations

1. **Naive Matrix Multiplication**: O(n³) triple-loop implementation
2. **Blocked Matrix Multiplication**: Cache-optimized with blocking
3. **Optimized Matrix Multiplication**: BLAS-accelerated (NumPy)
4. **im2col Convolution**: Convert convolution to GEMM

### Running Examples

```bash
python gemm.py
```

This will:
- Run benchmarks for different matrix sizes
- Compare naive, blocked, and optimized implementations
- Verify correctness across all implementations
- Test convolution implementation

## Expected Output

```
Matrix Multiplication Benchmarks
============================================================
Size       Naive (ms)      Blocked (ms)    Optimized (ms) 
------------------------------------------------------------
64         45.23           12.34           0.87           
128        361.45          89.23           3.21           
256        N/A             712.34          15.67          
512        N/A             5834.12         78.34          
============================================================
```

## Key Concepts

- **Cache Blocking**: Improves cache utilization
- **Memory Layout**: Row-major (C-order) for NumPy
- **BLAS**: Basic Linear Algebra Subprograms for optimal performance
- **im2col**: Lowering convolution to matrix multiplication

## Performance Tips

1. Use NumPy/BLAS for production code
2. Blocking size should fit in L1/L2 cache (typically 32-64)
3. Ensure memory is contiguous (`np.ascontiguousarray`)
4. Consider GPU acceleration for large matrices
