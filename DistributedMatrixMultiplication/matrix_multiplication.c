#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

// BLOCK_SIZE determines the size of the sub-matrices (tiles).
// 64x64 integers fit well into the L1/L2 cache of most modern CPUs.
// This prevents "cache misses" which slow down processing.
#define BLOCK_SIZE 64

// --- Helper: Robust File Reader (Dimensions Only) ---
// This function strictly reads the first line of the file to get rows/cols.
int read_matrix_dims(const char *filename, int *rows, int *cols) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return 0;
    // Attempt to read two integers (Rows Cols) from the start of the file
    if (fscanf(fp, "%d %d", rows, cols) != 2) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return 1;
}

// --- Helper: Optimized Buffered File Reader ---
// Standard fscanf is slow because it hits the hard drive for every number.
// This function reads the ENTIRE file into RAM (buffer) first, then parses it.
// This is critical for performance with large text files.
void read_matrix_data(const char *filename, int rows, int cols, int *buffer) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("Error opening file"); exit(1); }

    // 1. Get total file size in bytes to allocate the exact memory needed
    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);
    rewind(fp); // Go back to the start of the file

    // 2. Allocate a huge string buffer to hold the file content
    char *file_buf = (char *)malloc(filesize + 1);
    if (!file_buf) { perror("Memory allocation failed"); exit(1); }
    
    // 3. Read the file in one massive chunk (very fast I/O)
    fread(file_buf, 1, filesize, fp);
    file_buf[filesize] = '\0'; // Null-terminate the string
    fclose(fp);

    // 4. Parse numbers from the memory buffer
    char *ptr = file_buf;
    char *endptr;
    
    // Skip the first two numbers (dimensions) as we already have them
    strtol(ptr, &endptr, 10); ptr = endptr;
    strtol(ptr, &endptr, 10); ptr = endptr;

    long total_elements = (long)rows * cols;
    long idx = 0;

    // Loop through the string, finding numbers and converting them to ints
    while (idx < total_elements && *ptr) {
        // Skip non-digit characters (spaces, newlines)
        while (*ptr && !isdigit(*ptr) && *ptr != '-') ptr++;
        
        if (*ptr) {
            // strtol converts string to long integer
            buffer[idx++] = (int)strtol(ptr, &endptr, 10);
            ptr = endptr; // Move pointer to after the number we just read
        }
    }

    free(file_buf); // Clean up the temporary file buffer
}

// --- Helper: Serial Matrix Multiplication (Reference) ---
// This is the standard "slow" version used only to verify correctness.
// It runs on a single core (Rank 0) to produce the "Gold Standard" answer.
void serial_multiply(int rA, int cA, int cB, int *A, int *B, int *C) {
    // Clear result matrix memory
    memset(C, 0, (long)rA * cB * sizeof(int));

    // Standard Triple Loop (Row-Major Order)
    for (int i = 0; i < rA; i++) {
        for (int k = 0; k < cA; k++) {
            int a_val = A[i * cA + k];
            for (int j = 0; j < cB; j++) {
                C[i * cB + j] += a_val * B[k * cB + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int rows_A, cols_A, rows_B, cols_B;
    
    // Pointers for Full Matrices (Used only by Rank 0)
    int *A_full = NULL, *B_full = NULL, *C_full = NULL;
    int *C_serial = NULL; // To store the correct answer for checking
    
    // Pointers for Local Data (Used by ALL Ranks)
    int *A_local, *C_local; 
    
    // Timing variables
    double start_par, end_par, start_ser, end_ser;

    // --- MPI Initialization ---
    MPI_Init(&argc, &argv);               // Start MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get my ID (0, 1, 2...)
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processors

    // --- Step 1: Read and Broadcast Dimensions ---
    // Only Rank 0 (The Master) has access to the files.
    if (rank == 0) {
        // Read dimensions from the text files
        if (!read_matrix_dims("matrix_a.txt", &rows_A, &cols_A) || 
            !read_matrix_dims("matrix_b.txt", &rows_B, &cols_B)) {
            fprintf(stderr, "Error reading matrix dimensions.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Basic Matrix Multiplication Rule: Cols of A must equal Rows of B
        if (cols_A != rows_B) {
            fprintf(stderr, "Matrix dimension mismatch: %d != %d\n", cols_A, rows_B);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        printf("--- Assignment 1: Distributed Matrix Multiplication ---\n");
        printf("Running on %d MPI processes (cores)\n", size); 
        printf("Matrix A: %dx%d | Matrix B: %dx%d\n", rows_A, cols_A, rows_B, cols_B);
    }

    // Rank 0 tells everyone else what the matrix sizes are.
    // We broadcast these 4 integers so everyone can allocate memory correctly.
    MPI_Bcast(&rows_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rows_B, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols_B, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- Step 2: Serial Baseline (Rank 0 only) ---
    // Rank 0 loads the full data and runs a serial version to check accuracy later.
    if (rank == 0) {
        A_full = (int *)malloc((long)rows_A * cols_A * sizeof(int));
        B_full = (int *)malloc((long)rows_B * cols_B * sizeof(int));
        C_serial = (int *)malloc((long)rows_A * cols_B * sizeof(int));
        
        printf("[Rank 0] Reading Data (Buffered)...\n");
        read_matrix_data("matrix_a.txt", rows_A, cols_A, A_full);
        read_matrix_data("matrix_b.txt", rows_B, cols_B, B_full);

        printf("[Rank 0] Starting Serial Execution (for verification)...\n");
        start_ser = MPI_Wtime();
        serial_multiply(rows_A, cols_A, cols_B, A_full, B_full, C_serial);
        end_ser = MPI_Wtime();
        printf("[Rank 0] Serial Time: %f seconds\n", end_ser - start_ser);
    }
    
    // --- Step 3: Parallel Setup ---
    // Everyone needs memory for Matrix B because we replicate B across all nodes.
    if (rank != 0) {
        B_full = (int *)malloc((long)rows_B * cols_B * sizeof(int));
    }

    // Arrays to calculate how many rows of Matrix A each rank gets.
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int *recvcounts = malloc(size * sizeof(int));
    int *displs_C = malloc(size * sizeof(int));

    // Divide rows of A as evenly as possible
    int base_rows = rows_A / size;
    int remainder = rows_A % size;
    int offset_A = 0, offset_C = 0;

    for (int i = 0; i < size; i++) {
        // If there's a remainder, give extra rows to the first few ranks
        int r = base_rows + (i < remainder ? 1 : 0);
        
        // Calculate data size for Scatterv (A) and Gatherv (C)
        sendcounts[i] = r * cols_A; // Elements of A to send
        recvcounts[i] = r * cols_B; // Elements of C to receive
        displs[i] = offset_A;       // Where A's chunk starts
        displs_C[i] = offset_C;     // Where C's chunk starts
        
        offset_A += sendcounts[i];
        offset_C += recvcounts[i];
    }

    // Calculate how many rows *this* specific rank is responsible for
    int local_rows = sendcounts[rank] / cols_A;
    
    // Allocate local memory
    A_local = (int *)malloc((long)local_rows * cols_A * sizeof(int));
    C_local = (int *)calloc((long)local_rows * cols_B, sizeof(int)); // Calloc inits to 0

    // --- Step 4: Parallel Execution ---
    
    if (rank == 0) printf("[Rank 0] Starting Parallel Execution...\n");
    
    MPI_Barrier(MPI_COMM_WORLD); // Wait for everyone to be ready
    start_par = MPI_Wtime();

    // 1. Distribute A: Scatter rows of A to all processes
    MPI_Scatterv(A_full, sendcounts, displs, MPI_INT, 
                 A_local, local_rows * cols_A, MPI_INT, 
                 0, MPI_COMM_WORLD);
    
    // 2. Broadcast B: Send the ENTIRE Matrix B to everyone
    MPI_Bcast(B_full, (long)rows_B * cols_B, MPI_INT, 0, MPI_COMM_WORLD);

    // --- OPTIMIZED COMPUTE: Tiled (Blocked) Matrix Multiplication ---
    // Instead of processing row by row, we process block by block (e.g., 64x64 chunks).
    // This keeps the "active" data inside the CPU cache, making it much faster.
    
    // Outer loops: Iterate over the blocks
    for (int ii = 0; ii < local_rows; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < cols_A; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < cols_B; jj += BLOCK_SIZE) {
                
                // Calculate boundary checks (handle edges if matrix size isn't multiple of 64)
                int i_max = (ii + BLOCK_SIZE > local_rows) ? local_rows : ii + BLOCK_SIZE;
                int k_max = (kk + BLOCK_SIZE > cols_A)     ? cols_A     : kk + BLOCK_SIZE;
                int j_max = (jj + BLOCK_SIZE > cols_B)     ? cols_B     : jj + BLOCK_SIZE;

                // Inner loops: Perform standard multiplication INSIDE the block
                for (int i = ii; i < i_max; i++) {
                    int * restrict c_row = &C_local[i * cols_B];
                    int * restrict a_row = &A_local[i * cols_A];
                    
                    for (int k = kk; k < k_max; k++) {
                        int a_val = a_row[k];
                        int * restrict b_row = &B_full[k * cols_B];
                        
                        // Innermost loop:
                        // "restrict" keyword and flags (-O3) help compiler Vectorize this
                        for (int j = jj; j < j_max; j++) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }

    // 3. Gather Results: Collect all local C chunks back to Rank 0
    if (rank == 0) C_full = (int *)malloc((long)rows_A * cols_B * sizeof(int));
    
    MPI_Gatherv(C_local, local_rows * cols_B, MPI_INT,
                C_full, recvcounts, displs_C, MPI_INT,
                0, MPI_COMM_WORLD);

    end_par = MPI_Wtime(); // Stop the timer

    // --- Step 5: Final Report (Rank 0) ---
    if (rank == 0) {
        double time_par = end_par - start_par;
        double time_ser = end_ser - start_ser;

        printf("\n--- Results ---\n");
        printf("Serial Time:   %f s\n", time_ser);
        printf("Parallel Time: %f s\n", time_par);
        printf("Speedup:       %.2fx\n", time_ser / time_par);
        printf("Efficiency:    %.2f%%\n", (time_ser / time_par / size) * 100);

        // Verification: Compare Parallel Result vs Serial Result
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

        // Write the result to a file
        FILE *f_out = fopen("result.txt", "w");
        fprintf(f_out, "%d %d\n", rows_A, cols_B);
        for (long i = 0; i < (long)rows_A * cols_B; i++) {
            fprintf(f_out, "%d ", C_full[i]);
            if ((i + 1) % cols_B == 0) fprintf(f_out, "\n");
        }
        fclose(f_out);
        
        // Cleanup memory on Rank 0
        free(A_full); free(C_full); free(C_serial);
    }

    // Cleanup local memory on all ranks
    free(B_full); free(A_local); free(C_local);
    free(sendcounts); free(displs); free(recvcounts); free(displs_C);

    MPI_Finalize(); // Shut down MPI
    return 0;
}