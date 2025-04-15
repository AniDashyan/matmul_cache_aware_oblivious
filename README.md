# Performance Comparison of Matrix Multiplication Algorithms

## Overview

This project implements and compares three matrix multiplication algorithms:
1. **Naive Matrix Multiplication**: Standard implementation of matrix multiplication.
2. **Cache-Aware Matrix Multiplication**: Matrix multiplication optimized for cache by dividing matrices into sub-blocks sized according to the cache line or L1 cache capacity.
3. **Recursive Divide-and-Conquer Matrix Multiplication**: A recursive method that divides matrices into smaller quadrants without explicitly referencing cache size, but still adapts to varying cache sizes efficiently.

The purpose of this project is to compare the performance of these approaches, focusing on how cache-aware techniques (blocking and recursion) can reduce cache misses and improve execution time compared to the naive approach.

## Build & Run

To build and run the project:

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/AniDashyan/matmul_cache_aware_oblivious
   cd matmul_cache_aware_oblivious
   ```

2. Use `cmake` to generate the build files:
   ```bash
   cmake -S . -B build
   ```

3. Build the project:
   ```bash
   cmake --build build --config Release
   ```

4. Run the executable with the desired matrix dimensions:
   ```bash
   ./build/matmul --row [N] --col [M]
   ```
   - `N`: Number of rows in the matrices.
   - `M`: Number of columns in the matrices.

## Example Output

```
Matrix Multiplication Performance (size = 1000x2000): 
---------------------------------------- 
Method                   Time (ms) 
---------------------------------------- 
Naive                    1 seconds 
Blocked (blockSize=64)   790 milliseconds 
Recursive                1 seconds 
---------------------------------------- 
Cache Information: 
L1D Size: 49152 bytes 
Line Size: 64 bytes 
```

## Explanation

### Naive Matrix Multiplication
The naive approach performs the standard matrix multiplication algorithm using three nested loops. This approach results in significant cache misses, especially for larger matrices, as it does not take into account how data is loaded and stored in the CPU cache.

### Cache-Aware Matrix Multiplication
To optimize for cache, the matrices are split into sub-blocks that fit the cache line or L1 cache. This technique significantly reduces cache misses, especially when the matrices are large, as it ensures that data loaded into the cache is reused before it is evicted. The result is fewer cache misses and improved performance.

### Recursive Divide-and-Conquer Matrix Multiplication
This method divides matrices into smaller quadrants and recursively multiplies them. As the problem size decreases through recursion, the sub-problems eventually fit into the cache without the need for explicitly referencing cache size or block dimensions. This approach adapts well to varying cache sizes and performs efficiently, especially for larger matrices.

### Performance Comparison
By comparing the execution times, we can observe that both the cache-aware blocked approach and the recursive divide-and-conquer approach outperform the naive implementation, with the blocked approach typically showing the best performance due to its explicit cache optimization.
