#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Function to read matrix from file
void read_matrix(const char* filename, double** matrix, int* rows, int* cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    fscanf(file, "%d %d", rows, cols);
    *matrix = (double*)malloc((*rows) * (*cols) * sizeof(double));
    
    for (int i = 0; i < (*rows) * (*cols); i++) {
        fscanf(file, "%lf", &(*matrix)[i]);
    }
    
    fclose(file);
}

// Serial matrix multiplication for comparison
void serial_matmul(double* A, double* B, double* C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = A[i * n + k];
            for (int j = 0; j < p; j++) {
                C[i * p + j] += a_ik * B[k * p + j];
            }
        }
    }
}

// Parallel matrix multiplication with block-row partitioning
void parallel_matmul(double* A, double* B, double* C, 
                     int m, int n, int p, int rank, int size) {
    
    // Calculate rows per process
    int rows_per_proc = m / size;
    int remainder = m % size;
    
    // Calculate start row and number of rows for this process
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    // Allocate local matrices
    double* local_A = (double*)malloc(local_rows * n * sizeof(double));
    double* local_C = (double*)calloc(local_rows * p, sizeof(double));
    
    // Prepare send counts and displacements for Scatterv
    int* sendcounts = NULL;
    int* displs = NULL;
    
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * n;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }
    
    // Scatter rows of A to all processes
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 local_A, local_rows * n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    // Broadcast entire matrix B to all processes
    MPI_Bcast(B, n * p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform local matrix multiplication (optimized loop order)
    for (int i = 0; i < local_rows; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = local_A[i * n + k];
            for (int j = 0; j < p; j++) {
                local_C[i * p + j] += a_ik * B[k * p + j];
            }
        }
    }
    
    // Prepare receive counts and displacements for Gatherv
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * p;
            displs[i] = (i * rows_per_proc + (i < remainder ? i : remainder)) * p;
        }
    }
    
    // Gather results back to root
    MPI_Gatherv(local_C, local_rows * p, MPI_DOUBLE,
                C, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Cleanup
    free(local_A);
    free(local_C);
    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }
}

int main(int argc, char** argv) {
    int rank, size;
    double *A = NULL, *B = NULL, *C = NULL, *C_serial = NULL;
    int m, n, p, k;
    double start_time, end_time, parallel_time, serial_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Root process reads matrices
    if (rank == 0) {
        printf("=== MPI Matrix Multiplication ===\n");
        printf("Number of processes: %d\n\n", size);
        
        read_matrix("matrix_a.txt", &A, &m, &n);
        read_matrix("matrix_b.txt", &B, &k, &p);
        
        if (n != k) {
            fprintf(stderr, "Matrix dimensions incompatible: A(%dx%d) B(%dx%d)\n", 
                    m, n, k, p);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        printf("Matrix A: %d x %d\n", m, n);
        printf("Matrix B: %d x %d\n", k, p);
        printf("Result C: %d x %d\n\n", m, p);
        
        C = (double*)calloc(m * p, sizeof(double));
        C_serial = (double*)calloc(m * p, sizeof(double));
    }
    
    // Broadcast dimensions to all processes
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Allocate B on all processes
    if (rank != 0) {
        B = (double*)malloc(n * p * sizeof(double));
    }
    
    // === PARALLEL EXECUTION ===
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    parallel_matmul(A, B, C, m, n, p, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    parallel_time = end_time - start_time;
    
    if (rank == 0) {
        printf("Parallel execution time: %.6f seconds\n", parallel_time);
        
        // === SERIAL EXECUTION (for comparison) ===
        printf("\nRunning serial version for comparison...\n");
        start_time = MPI_Wtime();
        
        serial_matmul(A, B, C_serial, m, n, p);
        
        end_time = MPI_Wtime();
        serial_time = end_time - start_time;
        
        printf("Serial execution time: %.6f seconds\n", serial_time);
        printf("\n=== Performance Metrics ===\n");
        printf("Speedup: %.2fx\n", serial_time / parallel_time);
        printf("Efficiency: %.2f%%\n", (serial_time / parallel_time / size) * 100);
        
        // Verify correctness (check a few elements)
        int errors = 0;
        for (int i = 0; i < m && errors < 10; i++) {
            for (int j = 0; j < p && errors < 10; j++) {
                double diff = C[i * p + j] - C_serial[i * p + j];
                if (diff < 0) diff = -diff;
                if (diff > 1e-6) {
                    printf("Mismatch at C[%d][%d]: parallel=%.2f serial=%.2f\n",
                           i, j, C[i * p + j], C_serial[i * p + j]);
                    errors++;
                }
            }
        }
        
        if (errors == 0) {
            printf("\n✓ Verification passed: Results match!\n");
        } else {
            printf("\n✗ Verification failed: %d mismatches found\n", errors);
        }
        
        // Cleanup
        free(A);
        free(C);
        free(C_serial);
    }
    
    if (rank != 0 || B) {
        free(B);
    }
    
    MPI_Finalize();
    return 0;
}