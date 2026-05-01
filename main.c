#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

static inline double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_matrix(double *M, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n * n; i++) {
        M[i] = (double)(rand() % 1000) / 100.0;
    }
}

static void zero_matrix(double *M, int n) {
    memset(M, 0, sizeof(double) * (size_t)n * (size_t)n);
}

static void blocked_matmul(const double *A, const double *B, double *C, int n, int block_size) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                int i_end = (ii + block_size < n) ? ii + block_size : n;
                int j_end = (jj + block_size < n) ? jj + block_size : n;
                int k_end = (kk + block_size < n) ? kk + block_size : n;

                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        double a = A[i * n + k];
                        for (int j = jj; j < j_end; j++) {
                            C[i * n + j] += a * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

static double checksum(const double *M, int n) {
    double s = 0.0;
    for (int i = 0; i < n * n; i++) {
        s += M[i];
    }
    return s;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s <matrix_size_N> <block_size_B> [num_threads] [repetitions]\n"
            "Example: %s 1024 32 8 5\n",
            prog, prog);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        usage(argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int block = atoi(argv[2]);
    int threads = (argc >= 4) ? atoi(argv[3]) : omp_get_max_threads();
    int reps = (argc == 5) ? atoi(argv[4]) : 3;

    if (n <= 0 || block <= 0 || threads <= 0 || reps <= 0) {
        fprintf(stderr, "Error: all arguments must be positive integers.\n");
        return 1;
    }

    omp_set_num_threads(threads);

    size_t total = (size_t)n * (size_t)n;
    double *A = (double *)malloc(sizeof(double) * total);
    double *B = (double *)malloc(sizeof(double) * total);
    double *C = (double *)malloc(sizeof(double) * total);

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed for N=%d\n", n);
        free(A); free(B); free(C);
        return 1;
    }

    fill_matrix(A, n, 42U);
    fill_matrix(B, n, 1337U);

    double best = 1e300;
    double total_time = 0.0;

    for (int r = 0; r < reps; r++) {
        zero_matrix(C, n);
        double t0 = now_seconds();
        blocked_matmul(A, B, C, n, block);
        double t1 = now_seconds();

        double dt = t1 - t0;
        total_time += dt;
        if (dt < best) best = dt;
    }

    double avg = total_time / (double)reps;
    double flops = 2.0 * (double)n * (double)n * (double)n;
    double gflops = flops / best / 1e9;

    printf("=== OpenMP Blocked Matrix Multiplication ===\n");
    printf("Matrix size (N)      : %d\n", n);
    printf("Block size (B)       : %d\n", block);
    printf("OpenMP threads       : %d\n", threads);
    printf("Repetitions          : %d\n", reps);
    printf("Best time (s)        : %.6f\n", best);
    printf("Average time (s)     : %.6f\n", avg);
    printf("Performance (GFLOPS) : %.3f\n", gflops);
    printf("Checksum(C)          : %.6f\n", checksum(C, n));

    free(A);
    free(B);
    free(C);
    return 0;
}