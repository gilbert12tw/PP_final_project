# CUDA-Enhanced Knapsack Solver by Dynamic Programming

A high-performance CUDA implementation of the knapsack problem using various optimization techniques including Meet-in-the-Middle (MITM) and Multiple-Choice Knapsack Problem (MCKP) approaches.

## Features

- Multiple solution implementations:
  - Sequential CPU implementation
  - Basic GPU implementation
  - MCKP-based GPU optimization
  - MITM with dual GPU acceleration
- Binary I/O for efficient data handling
- Comprehensive test suite with varying problem sizes
- Utilities for test case generation and verification

## Prerequisites

- CUDA Toolkit
- GCC/G++ compiler
- Python 3.x (for testing)
- Multiple GPUs (for MITM implementation)

## Installation

1. Load required modules:
```bash
module load cuda
```

2. Compile the utilities:
```bash
cd utilities
make
```

3. Compile the solvers:
```bash
nvcc -O3 seq.cpp -o seq
nvcc -O3 gpu.cu -o gpu
nvcc -O3 knapsack_mckp.cu -o knapsack_mckp
nvcc -O3 knapsack_mitm.cu -o knapsack_mitm
```

## Usage

### Generate Test Cases

Generate test cases with specified parameters:
```bash
./utilities/gen <output_file> <n> <m>
```
- `n`: Total number of items
- `m`: Maximum weight capacity

### Run Solutions

Run different implementations:

```bash
# Sequential CPU version
./seq <input_file> <output_file>

# Basic GPU version
./gpu <input_file> <output_file>

# MCKP version
./knapsack_mckp <input_file> <output_file>

# MITM version (requires 2 GPUs)
./knapsack_mitm <input_file> <output_file>
```

### Verify Results

Read binary files:
```bash
./utilities/reader <binary_file>
```

### Run Tests

Execute test suite:
```bash
# For single GPU implementations
python3 judge_srun.py <executable> 1

# For MITM implementation
python3 judge_srun.py knapsack_mitm 2
```

## Test Cases

The project includes test cases of varying sizes:

- Small (n ≤ 50)
- Medium (50 < n ≤ 500)
- Large (500 < n ≤ 10000)
- GPU-optimized (n > 10000)
- Extra large (n ≥ 1000000)

## File Formats

### Input Format (*.in)
- Binary file containing 4-byte integers
- First two integers: n (items count), m (capacity)
- Followed by n pairs of integers: (weight, value)

### Output Format (*.out)
- Binary file containing a single 4-byte integer
- Represents the maximum achievable value

## Performance

The implementations show significant performance improvements:

- Sequential: Baseline performance
- GPU: ~30x speedup for large cases
- MCKP: ~50x speedup for large cases
- MITM: ~100x speedup for large cases with dual GPU