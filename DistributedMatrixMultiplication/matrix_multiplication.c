#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

#define BLOCK_SIZE 64

// --- Helper: Robust File Reader (Dimensions Only) ---
int read_matrix_dims(const char *filename, int *rows, int *cols) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return 0;
    if (fscanf(fp, "%d %d", rows, cols) != 2) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return 1;
}

// --- Helper: Optimized Buffered File Reader ---
// Reads whole file into memory first, then parses. Much faster than fscanf.
void read_matrix_data(const char *filename, int rows, int cols, int *buffer) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("Error opening file"); exit(1); }

    // 1. Get file size
    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);
    rewind(fp);

    // 2. Read entire file into buffer
    char *file_buf = (char *)malloc(filesize + 1);
    if (!file_buf) { perror("Memory allocation failed"); exit(1); }
    fread(file_buf, 1, filesize, fp);
    file_buf[filesize] = '\0';
    fclose(fp);

    // 3. Parse numbers from buffer
    char *ptr = file_buf;
    char *endptr;
    
    // Skip the first two numbers (dimensions) since we already have them
    strtol(ptr, &endptr, 10); ptr = endptr;
    strtol(ptr, &endptr, 10); ptr = endptr;

    long total_elements = (long)rows * cols;
    long idx = 0;

    while (idx < total_elements && *ptr) {
        // Skip whitespace/newlines manually to find next digit or sign
        while (*ptr && !isdigit(*ptr) && *ptr != '-') ptr++;
        
        if (*ptr) {
            buffer[idx++] = (int)strtol(ptr, &endptr, 10);
            ptr = endptr;
        }
    }

    free(file_buf);
}

// --- Helper: Serial Matrix Multiplication (Reference) ---
void serial_multiply(int rA, int cA, int cB, int *A, int *B, int *C) {
    memset(C, 0, (long)rA * cB * sizeof(int));

    for (int i = 0; i < rA; i++) {
        int * restrict c_row = &C[i * cB];
        int * restrict a_row = &A[i * cA];
        for (int k = 0; k < cA; k++) {
            int a_val = a_row[k];
            int * restrict b_row = &B[k * cB];
            for (int j = 0; j < cB; j++) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int rows_A, cols_A, rows_B, cols_B;
    int *A_full = NULL, *B_full = NULL, *C_full = NULL;
    int *C_serial = NULL;
    int *A_local, *C_local;
    double start_par, end_par, start_ser, end_ser;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- Step 1: Read and Broadcast Dimensions ---
    if (rank == 0) {
        if (!read_matrix_dims("matrix_a.txt", &rows_A, &cols_A) || 
            !read_matrix_dims("matrix_b.txt", &rows_B, &cols_B)) {
            fprintf(stderr, "Error reading matrix dimensions.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (cols_A != rows_B) {
            fprintf(stderr, "Matrix dimension mismatch: %d != %d\n", cols_A, rows_B);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        printf("--- Assignment 1: Distributed Matrix Multiplication ---\n");
        
        printf("Matrix A: %dx%d | Matrix B: %dx%d\n", rows_A, cols_A, rows_B, cols_B);
    }

    MPI_Bcast(&rows_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rows_B, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols_B, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- Step 2: Serial Baseline (Rank 0 only) ---
    if (rank == 0) {
        A_full = (int *)malloc((long)rows_A * cols_A * sizeof(int));
        B_full = (int *)malloc((long)rows_B * cols_B * sizeof(int));
        C_serial = (int *)malloc((long)rows_A * cols_B * sizeof(int));
        
        printf("[Rank 0] Reading Data (Buffered)...\n");
        read_matrix_data("matrix_a.txt", rows_A, cols_A, A_full);
        read_matrix_data("matrix_b.txt", rows_B, cols_B, B_full);

        printf("[Rank 0] Starting Serial Execution...\n");
        start_ser = MPI_Wtime();
        serial_multiply(rows_A, cols_A, cols_B, A_full, B_full, C_serial);
        end_ser = MPI_Wtime();
        printf("[Rank 0] Serial Time: %f seconds\n", end_ser - start_ser);
    }
    
    // --- Step 3: Parallel Setup ---
    if (rank != 0) {
        B_full = (int *)malloc((long)rows_B * cols_B * sizeof(int));
    }

    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int *recvcounts = malloc(size * sizeof(int));
    int *displs_C = malloc(size * sizeof(int));

    int base_rows = rows_A / size;
    int remainder = rows_A % size;
    int offset_A = 0, offset_C = 0;

    for (int i = 0; i < size; i++) {
        int r = base_rows + (i < remainder ? 1 : 0);
        sendcounts[i] = r * cols_A;
        recvcounts[i] = r * cols_B;
        displs[i] = offset_A;
        displs_C[i] = offset_C;
        offset_A += sendcounts[i];
        offset_C += recvcounts[i];
    }

    int local_rows = sendcounts[rank] / cols_A;
    A_local = (int *)malloc((long)local_rows * cols_A * sizeof(int));
    C_local = (int *)calloc((long)local_rows * cols_B, sizeof(int));

    // --- Step 4: Parallel Execution ---
    
    if (rank == 0) {
        printf("[Rank 0] Starting Parallel Execution...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD); 
    start_par = MPI_Wtime();

    // Distribute A
    MPI_Scatterv(A_full, sendcounts, displs, MPI_INT, 
                 A_local, local_rows * cols_A, MPI_INT, 
                 0, MPI_COMM_WORLD);
    
    // Broadcast B
    MPI_Bcast(B_full, (long)rows_B * cols_B, MPI_INT, 0, MPI_COMM_WORLD);

    // --- OPTIMIZED COMPUTE: Tiled (Blocked) Matrix Multiplication ---
    // Loops over blocks (ii, kk, jj) then inside blocks (i, k, j)
    // This keeps active data inside the CPU L1/L2 cache.
    for (int ii = 0; ii < local_rows; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < cols_A; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < cols_B; jj += BLOCK_SIZE) {
                
                // Handle edges of the matrix that don't fit a full block
                int i_max = (ii + BLOCK_SIZE > local_rows) ? local_rows : ii + BLOCK_SIZE;
                int k_max = (kk + BLOCK_SIZE > cols_A)     ? cols_A     : kk + BLOCK_SIZE;
                int j_max = (jj + BLOCK_SIZE > cols_B)     ? cols_B     : jj + BLOCK_SIZE;

                for (int i = ii; i < i_max; i++) {
                    int * restrict c_row = &C_local[i * cols_B];
                    int * restrict a_row = &A_local[i * cols_A];
                    
                    for (int k = kk; k < k_max; k++) {
                        int a_val = a_row[k];
                        int * restrict b_row = &B_full[k * cols_B];
                        
                        // Inner loop: Compiler will automatically vectorize this 
                        // thanks to -O3 and -march=native
                        for (int j = jj; j < j_max; j++) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }

    // Gather Results
    if (rank == 0) C_full = (int *)malloc((long)rows_A * cols_B * sizeof(int));
    
    MPI_Gatherv(C_local, local_rows * cols_B, MPI_INT,
                C_full, recvcounts, displs_C, MPI_INT,
                0, MPI_COMM_WORLD);

    end_par = MPI_Wtime();

    // --- Step 5: Final Report (Rank 0) ---
    if (rank == 0) {
        double time_par = end_par - start_par;
        double time_ser = end_ser - start_ser;

        printf("[Rank 0] Parallel Time: %f seconds\n", time_par);
        
        printf("\n--- Results ---\n");
        printf("Serial Time:   %f s\n", time_ser);
        printf("Parallel Time: %f s\n", time_par);
        printf("Speedup:       %.2fx\n", time_ser / time_par);
        printf("Efficiency:    %.2f%%\n", (time_ser / time_par / size) * 100);

        // Verification (Optional but recommended)
        int correct = 1;
        for (long i = 0; i < (long)rows_A * cols_B; i++) {
            if (C_full[i] != C_serial[i]) {
                correct = 0;
                printf("Mismatch at index %ld: Serial %d vs Parallel %d\n", i, C_serial[i], C_full[i]);
                break;
            }
        }
        if (correct) printf("Verification: PASS\n");
        else printf("Verification: FAIL\n");

        FILE *f_out = fopen("result.txt", "w");
        fprintf(f_out, "%d %d\n", rows_A, cols_B);
        for (long i = 0; i < (long)rows_A * cols_B; i++) {
            fprintf(f_out, "%d ", C_full[i]);
            if ((i + 1) % cols_B == 0) fprintf(f_out, "\n");
        }
        fclose(f_out);
        
        free(A_full); free(C_full); free(C_serial);
    }

    free(B_full); free(A_local); free(C_local);
    free(sendcounts); free(displs); free(recvcounts); free(displs_C);

    MPI_Finalize();
    return 0;
}