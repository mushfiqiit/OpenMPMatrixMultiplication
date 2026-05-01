#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define MAX_LIST 64

typedef struct {
    int threads;
    int block;
    double best_sec;
    double avg_sec;
    double gflops;
    double speedup_vs_1t;
    double efficiency;
} Result;

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

static int min_int(int a, int b) { return (a < b) ? a : b; }

static void blocked_matmul(const double *A, const double *B, double *C, int n, int block_size) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                int i_end = min_int(ii + block_size, n);
                int j_end = min_int(jj + block_size, n);
                int k_end = min_int(kk + block_size, n);

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
    for (int i = 0; i < n * n; i++) s += M[i];
    return s;
}

static int parse_list(const char *s, int *out, int max_items) {
    if (!s || !*s) return 0;
    char *copy = strdup(s);
    if (!copy) return 0;

    int count = 0;
    char *tok = strtok(copy, ",");
    while (tok && count < max_items) {
        int v = atoi(tok);
        if (v > 0) out[count++] = v;
        tok = strtok(NULL, ",");
    }
    free(copy);
    return count;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <N> [repetitions] [block_sizes_csv] [thread_counts_csv]\n"
        "Example: %s 1024 3 8,16,32,64 1,2,4,8\n",
        prog, prog);
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
        usage(argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int reps = (argc >= 3) ? atoi(argv[2]) : 3;
    if (n <= 0 || reps <= 0) {
        fprintf(stderr, "Error: N and repetitions must be positive integers.\n");
        return 1;
    }

    int blocks[MAX_LIST] = {8, 16, 32, 64};
    int threads[MAX_LIST] = {1, 2, 4, 8};
    int nb = 4;
    int nt = 4;

    if (argc >= 4) {
        nb = parse_list(argv[3], blocks, MAX_LIST);
        if (nb <= 0) {
            fprintf(stderr, "Error: invalid block_sizes_csv\n");
            return 1;
        }
    }
    if (argc >= 5) {
        nt = parse_list(argv[4], threads, MAX_LIST);
        if (nt <= 0) {
            fprintf(stderr, "Error: invalid thread_counts_csv\n");
            return 1;
        }
    }

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

    Result *results = (Result *)malloc(sizeof(Result) * (size_t)nb * (size_t)nt);
    if (!results) {
        fprintf(stderr, "Result allocation failed\n");
        free(A); free(B); free(C);
        return 1;
    }

    int rcount = 0;
    double last_checksum = 0.0;
    for (int bi = 0; bi < nb; bi++) {
        int block = blocks[bi];
        for (int ti = 0; ti < nt; ti++) {
            int th = threads[ti];
            omp_set_num_threads(th);

            double best = 1e300;
            double sum = 0.0;
            for (int r = 0; r < reps; r++) {
                zero_matrix(C, n);
                double t0 = now_seconds();
                blocked_matmul(A, B, C, n, block);
                double t1 = now_seconds();
                double dt = t1 - t0;
                if (dt < best) best = dt;
                sum += dt;
            }

            double avg = sum / (double)reps;
            double flops = 2.0 * (double)n * (double)n * (double)n;
            double gflops = flops / best / 1e9;
            last_checksum = checksum(C, n);

            results[rcount].threads = th;
            results[rcount].block = block;
            results[rcount].best_sec = best;
            results[rcount].avg_sec = avg;
            results[rcount].gflops = gflops;
            results[rcount].speedup_vs_1t = 0.0;
            results[rcount].efficiency = 0.0;
            rcount++;
        }
    }

    for (int bi = 0; bi < nb; bi++) {
        double base = 0.0;
        for (int i = 0; i < rcount; i++) {
            if (results[i].block == blocks[bi] && results[i].threads == 1) {
                base = results[i].best_sec;
                break;
            }
        }
        for (int i = 0; i < rcount; i++) {
            if (results[i].block != blocks[bi]) continue;
            if (base > 0.0) {
                results[i].speedup_vs_1t = base / results[i].best_sec;
                results[i].efficiency = results[i].speedup_vs_1t / (double)results[i].threads;
            }
        }
    }

    printf("=== OpenMP Blocked Matrix Multiplication Sweep ===\n");
    printf("N=%d, repetitions=%d\n", n, reps);
    printf("Checksum(C) from last run: %.6f\n\n", last_checksum);

    printf("| Block | Threads | Best Time (s) | Avg Time (s) | GFLOPS | Speedup vs 1T | Efficiency |\n");
    printf("|------:|--------:|--------------:|-------------:|-------:|--------------:|-----------:|\n");
    for (int i = 0; i < rcount; i++) {
        printf("| %5d | %7d | %13.6f | %12.6f | %6.2f | %13.3f | %10.3f |\n",
               results[i].block,
               results[i].threads,
               results[i].best_sec,
               results[i].avg_sec,
               results[i].gflops,
               results[i].speedup_vs_1t,
               results[i].efficiency);
    }

    free(results);
    free(A);
    free(B);
    free(C);
    return 0;
}