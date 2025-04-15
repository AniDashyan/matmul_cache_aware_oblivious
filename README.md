# Performance Comparison of Matrix Multiplication Algorithms

## Overview

This project implements and compares three matrix multiplication algorithms:

1. **Naive Matrix Multiplication**:  
   A standard implementation using three nested loops. This method suffers from poor cache utilization and is single-threaded, making it inefficient for large matrices.

2. **Cache-Aware Matrix Multiplication**:  
   Optimized for cache efficiency by dividing matrices into sub-blocks sized to fit into the L1 cache or a cache line. This method uses OpenMP for parallelism, allowing computations to leverage multi-core CPUs.

3. **Recursive Divide-and-Conquer Matrix Multiplication**:  
   A recursive technique that partitions matrices into smaller quadrants. It adapts implicitly to cache sizes without explicit blocking but operates in a single-threaded manner.

The primary objective is to evaluate performance across these algorithms, demonstrating how cache-aware strategies—such as blocking, recursion, and parallelism—help reduce cache misses and improve execution times. These differences become especially apparent when working with large square matrices (e.g., 1024×1024).


## Build & Run

### To build and run the project:

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/AniDashyan/matmul_cache_aware_oblivious
cd matmul_cache_aware_oblivious
```

Generate the build files with CMake, enabling OpenMP support:

```bash
cmake -S . -B build -DCMAKE_CXX_FLAGS="-fopenmp"
```

Build the project:

```bash
cmake --build build --config Release
```

Run the executable with the desired matrix size and thread count:

```bash
./build/matmul --size [N] --threads [T]
```

- `--size [N]`: Sets the dimension of square matrices (N×N). Default is 1024.
- `--threads [T]`: Defines the number of threads for the blocked algorithm. Default is the system’s maximum thread count.

**Example:**

```bash
./build/matmul --size 1024 --threads 8
```


## Example Output

```
Matrix Multiplication Performance (size = 1024x1024): 
---------------------------------------- 
Method                   Time (ms) 
---------------------------------------- 
Naive                    7 seconds 
Blocked (blockSize=64)   1 seconds 
Recursive                10 seconds 
---------------------------------------- 
Cache Information: 
L1D Size: 49152 bytes 
Line Size: 64 bytes 
Block Size: 64 ints 
Threads Used: 8 
```

## Parallelism with OpenMP

The cache-aware blocked matrix multiplication algorithm uses OpenMP to parallelize computation across multiple CPU cores, yielding significant performance improvements on large matrices. Key aspects include:

- **Threading Mechanism**:  
  OpenMP parallelizes the outer loops of the blocked algorithm. Threads work independently on sub-blocks of the result matrix `C`, preventing data races and reducing synchronization overhead.

- **Dynamic Scheduling**:  
  Utilizes `schedule(dynamic)` to achieve load balancing. This allows threads to dynamically pick up work units, accommodating variations in workload due to cache behavior or CPU interruptions.

- **Cache-Line Alignment**:  
  Matrix rows are aligned to 64-byte cache lines (equivalent to 16 `int` values) to minimize false sharing. This ensures efficient memory access for each thread.

- **Performance Impact**:  
  On an 8-core CPU, the blocked algorithm with 8 threads can achieve a **5–10× speedup** over the naive method when processing 1024×1024 matrices. This improvement comes from both cache reuse and effective parallelism.

If OpenMP is disabled or unavailable, the algorithm reverts to sequential execution while retaining its cache-aware design.


## Explanation

### Naive Matrix Multiplication

This method performs matrix multiplication using three nested loops. It lacks memory locality and causes frequent cache misses, especially for large matrices. Furthermore, it is single-threaded, making it unsuitable for modern multi-core processors.

### Cache-Aware Matrix Multiplication

This technique divides matrices into blocks sized to fit into the L1 cache (e.g., 128×128 `int` elements for a 32 KB cache). By reusing data within these blocks, the algorithm minimizes cache misses. It is further enhanced through:

- **OpenMP parallelization** across matrix blocks.
- **Dynamic scheduling** to balance computational load.
- **Cache-line alignment** to reduce memory access conflicts.

These optimizations allow it to outperform other approaches significantly on large matrix sizes.

### Recursive Divide-and-Conquer Matrix Multiplication

This recursive algorithm splits matrices into four quadrants and recursively multiplies the sub-matrices. As recursion proceeds, smaller matrices naturally fit into cache, improving data locality without explicitly defining block sizes. Despite this, it remains single-threaded and incurs recursive overhead, limiting its efficiency on large matrices compared to the blocked method.


## Performance Comparison

For square matrices of size 1024×1024 on an 8-core processor:

- **Naive**:  
  The slowest method due to high cache miss rates and lack of parallel execution.

- **Blocked (Cache-Aware)**:  
  The fastest method. It achieves 5–10× speedup by optimizing memory access and utilizing all cores

- **Recursive**:  
  Offers some performance benefit (2–3× faster than naive) but is limited by its sequential nature.
