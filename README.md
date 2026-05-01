# OpenMPMatrixMultiplication

The goal is to experiment with OpenMP for blocked matrix multiplication for different number of blocks size and threads.

## Build (Ubuntu 20.04)

```bash
gcc -O3 -march=native -fopenmp main.c -o matmul
```

## Run

```bash
# Usage
./matmul <matrix_size_N> <block_size_B> [num_threads] [repetitions]

# Example
./matmul 1024 32 8 5
```

## Performance sweep examples

```bash
for t in 1 2 4 8; do
  for b in 8 16 32 64; do
    ./matmul 1024 $b $t 3
  done
done
```

## Sample output

```text
=== OpenMP Blocked Matrix Multiplication ===
Matrix size (N)      : 512
Block size (B)       : 32
OpenMP threads       : 4
Repetitions          : 3
Best time (s)        : 0.022450
Average time (s)     : 0.023180
Performance (GFLOPS) : 11.956
Checksum(C)          : 17142482947.668106