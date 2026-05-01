The goal is to experiment with OpenMP for blocked matrix multiplication for different block sizes and thread counts.

## Build (Ubuntu 20.04)

```bash
gcc -O3 -march=native -fopenmp main.c -o matmul
```

## Run benchmark sweep

```bash
# Usage
./matmul <N> [repetitions] [block_sizes_csv] [thread_counts_csv]

# Default sweep (blocks: 8,16,32,64; threads: 1,2,4,8)
./matmul 1024

# Custom sweep
./matmul 1024 3 8,16,32,64,128 1,2,4,8
```

## What the program reports

For each `(block_size, thread_count)` combination it prints a table with:
- Best execution time (seconds)
- Average execution time (seconds)
- GFLOPS
- Speedup vs 1 thread for the same block size
- Parallel efficiency (`speedup / threads`)

## Example output (format)

```text
=== OpenMP Blocked Matrix Multiplication Sweep ===
N=512, repetitions=3
Checksum(C) from last run: 17142482947.668106

| Block | Threads | Best Time (s) | Avg Time (s) | GFLOPS | Speedup vs 1T | Efficiency |
|------:|--------:|--------------:|-------------:|-------:|--------------:|-----------:|
|    32 |       1 |      0.081005 |     0.082131 |   3.31 |         1.000 |      1.000 |
|    32 |       2 |      0.043512 |     0.044007 |   6.16 |         1.862 |      0.931 |
|    32 |       4 |      0.022450 |     0.023180 |  11.96 |         3.608 |      0.902 |
```

## Suggested assignment workflow

1. Run one sweep for your chosen `N`.
2. Copy the printed table directly into your report.
3. Add short observations on which block size and thread count gave best performance.