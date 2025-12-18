#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

// Note: We did not include the sample data ("matrix_a.txt" and "matrix_b.txt") and "result.txt" in this zip file because they are too large to upload to cscms.sit.

// Program Execution Flow
// 1. Compilation & Launch
// - Command: "mpicc -o matrix_multiplication matrix_multiplication.c -O3 -march=native -std=c99" creates the executable with maximum optimization.
// - Run: "mpirun -np 8 ./matrix_multiplication" launches 8 parallel processes (Ranks 0 to 7).

// 2. Initialization (Rank 0)
// - Rank 0 (Master) reads "matrix_a.txt" and "matrix_b.txt" into RAM using Buffered Reading (fread) for high speed.
// - It broadcasts the matrix dimensions to all other ranks so they can allocate memory.

// 3. Data Distribution (Communication)
// - Scatter: Rank 0 splits Matrix A horizontally and sends a specific chunk of rows to each rank (0–7).
// - Broadcast: Rank 0 sends the entire Matrix B to every rank (required for multiplication).

// 4. Parallel Computation
// - All 8 ranks calculate their assigned rows simultaneously.
// - Optimization: Uses Tiling (64x64 blocks) and i-k-j loop ordering to keep data resident in the CPU cache and enable vectorization.

// 5. Collection & Output
// - Gather: Rank 0 collects the calculated rows from all ranks and stitches them into the final Matrix C.
// - Rank 0 verifies the result against a serial calculation and writes the final output to "result.txt".


// BLOCK_SIZE determines the size of the sub-matrices (tiles) for processing.
// A 64x64 block of integers fits well into the L1/L2 Cache of most modern CPUs.
// This prevents "Cache Misses" (waiting for RAM), which is the main cause of slowness.
#define BLOCK_SIZE 64

// --- Helper: Matrix Dimensions Reader ---
// This function opens the file just to read the first line (Rows and Columns).
// We need this information before we allocate memory for the matrices.
int read_matrix_dims(const char *filename, int *rows, int *cols) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return 0;
    
    // fscanf reads formatted input. We expect two integers at the top.
    if (fscanf(fp, "%d %d", rows, cols) != 2) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return 1;
}

// --- Helper: Optimized Buffered File Reader ---
// Standard 'fscanf' is slow because it asks the Hard Drive for data one number at a time (millions of requests).
// So, this function reads the entire file into a RAM buffer in one go, and then parses the numbers from RAM. This is faster than fscanf.
// (Faster than fscanf by approximately 0.5 to 1 second).
void read_matrix_data(const char *filename, int rows, int cols, int *buffer) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("Error opening file"); exit(1); }

    // 1. Calculate the file size so we know how much RAM to allocate
    fseek(fp, 0, SEEK_END);    // Jump to end of file
    long filesize = ftell(fp); // Gives file size in bytes
    rewind(fp);                // Jump back to start of file

    // 2. Allocate a huge string buffer to hold the raw file content
    char *file_buf = (char *)malloc(filesize + 1);
    if (!file_buf) { perror("Memory allocation failed"); exit(1); }
    
    // 3. Read the file in one massive chunk (This is the fast part)
    fread(file_buf, 1, filesize, fp);
    file_buf[filesize] = '\0'; // Null-terminate the string so it is valid C text
    fclose(fp);

    // 4. Parse the numbers from the memory buffer
    char *ptr = file_buf; // Pointer to current position in text
    char *endptr;         // Pointer to where the number ends
    
    // Skip the first two numbers (dimensions) since we already have them from read_matrix_dims
    strtol(ptr, &endptr, 10); ptr = endptr;
    strtol(ptr, &endptr, 10); ptr = endptr;

    long total_elements = (long)rows * cols;
    long idx = 0;

    // Loop through the text buffer, finding numbers and converting them to ints
    while (idx < total_elements && *ptr) {
        // Fast-forward past any spaces or newlines to find the next digit
        while (*ptr && !isdigit(*ptr) && *ptr != '-') ptr++;
        
        if (*ptr) {
            // strtol: "String to Long". Converts text "123" to integer 123.
            buffer[idx++] = (int)strtol(ptr, &endptr, 10);
            ptr = endptr; // Move our pointer past the number we just read
        }
    }

    free(file_buf); // Clean up the massive text buffer now that we have the ints
}

// --- Helper: Serial Matrix Multiplication ---
// This runs on a single core (Rank 0).
// We use this to verify if our Parallel code is actually working correctly.
void serial_multiply(int rA, int cA, int cB, int *A, int *B, int *C) {
    // Reset the result matrix to all zeros
    memset(C, 0, (long)rA * cB * sizeof(int));

    // Loop Order: i -> k -> j
    // Based on research into memory hierarchy and loop interchange optimization, this loop order (i-k-j) improves performance significantly (often >30%).
    // This is faster than standard math order (i-j-k) because it accesses Matrix B row-by-row (contiguous memory), improving spatial locality.
    for (int i = 0; i < rA; i++) {
        // "restrict" tells the compiler: "These pointers don't overlap."
        // This allows the compiler to apply aggressive optimizations.
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
    
    // "Full" pointers: Used only by Rank 0 to hold the complete matrices.
    int *A_full = NULL, *B_full = NULL, *C_full = NULL;
    int *C_serial = NULL; // Holds the reference answer
    
    // "Local" pointers: Used by ALL ranks to hold their specific chunk of work.
    int *A_local, *C_local;
    double start_par, end_par, start_ser, end_ser; // Timers

    // --- MPI Initialization ---
    MPI_Init(&argc, &argv);               // Start the Parallel Environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Determine the rank of the calling process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Determine the total number of processes

    // --- Step 1: Read and Broadcast Dimensions ---
    // Only the Master (Rank 0) reads the files.
    if (rank == 0) {
        if (!read_matrix_dims("matrix_a.txt", &rows_A, &cols_A) || 
            !read_matrix_dims("matrix_b.txt", &rows_B, &cols_B)) {
            fprintf(stderr, "Error reading matrix dimensions.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Math Rule: Columns of A must equal Rows of B
        if (cols_A != rows_B) {
            fprintf(stderr, "Matrix dimension mismatch: %d != %d\n", cols_A, rows_B);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        printf("--- Assignment 1: Distributed Matrix Multiplication ---\n");
        printf("Running on %d MPI processes (cores)\n", size); 
        printf("Matrix A: %dx%d | Matrix B: %dx%d\n", rows_A, cols_A, rows_B, cols_B);
    }

    // Rank 0 knows the sizes, but Rank 1, 2, 3... do not.
    // MPI_Bcast sends these 4 integers from Rank 0 to EVERYONE.
    MPI_Bcast(&rows_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rows_B, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols_B, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- Step 2: Serial Baseline (Rank 0 only) ---
    // Rank 0 loads data and solves it alone first to get the baseline speed.
    if (rank == 0) {
        A_full = (int *)malloc((long)rows_A * cols_A * sizeof(int));
        B_full = (int *)malloc((long)rows_B * cols_B * sizeof(int));
        C_serial = (int *)malloc((long)rows_A * cols_B * sizeof(int));
        
        printf("[Rank 0] Reading Data (Buffered)...\n");
        read_matrix_data("matrix_a.txt", rows_A, cols_A, A_full);
        read_matrix_data("matrix_b.txt", rows_B, cols_B, B_full);

        printf("[Rank 0] Starting Serial Execution...\n");
        start_ser = MPI_Wtime(); // Start Timer
        serial_multiply(rows_A, cols_A, cols_B, A_full, B_full, C_serial);
        end_ser = MPI_Wtime();   // Stop Timer
        printf("[Rank 0] Serial Time: %f seconds\n", end_ser - start_ser);
    }
    
    // --- Step 3: Parallel Setup ---
    // Every rank needs space for Matrix B because we replicate B everywhere.
    if (rank != 0) {
        B_full = (int *)malloc((long)rows_B * cols_B * sizeof(int));
    }

    // Arrays to calculate how to chop up Matrix A
    int *sendcounts = malloc(size * sizeof(int)); // Array defining the number of elements to send to each process
    int *displs = malloc(size * sizeof(int));     // Array specifying the displacement (offset) into the send buffer
    int *recvcounts = malloc(size * sizeof(int)); // Array defining the number of elements to receive from each process
    int *displs_C = malloc(size * sizeof(int));   // Array specifying the displacement (offset) into the receive buffer

    // Calculate chunks. We divide rows of A among the processors.
    int base_rows = rows_A / size;
    int remainder = rows_A % size;
    int offset_A = 0, offset_C = 0;

    for (int i = 0; i < size; i++) {
        // If rows do not divide evenly, give 1 extra row to the first few ranks
        int r = base_rows + (i < remainder ? 1 : 0);
        
        sendcounts[i] = r * cols_A; // Rows * Columns = Total Integers
        recvcounts[i] = r * cols_B; // Result is Rows * Columns of B
        displs[i] = offset_A;
        displs_C[i] = offset_C;
        
        offset_A += sendcounts[i];
        offset_C += recvcounts[i];
    }

    // Determine the number of rows assigned to this specific rank
    int local_rows = sendcounts[rank] / cols_A;
    
    // Allocate memory for specific chunk of A and result chunk C
    A_local = (int *)malloc((long)local_rows * cols_A * sizeof(int));
    C_local = (int *)calloc((long)local_rows * cols_B, sizeof(int)); // calloc sets to 0

    // --- Step 4: Parallel Execution ---
    
    if (rank == 0) {
        printf("[Rank 0] Starting Parallel Execution...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD); // Wait until everyone is ready to start
    start_par = MPI_Wtime();     // Start Parallel Timer

    // MPI_Scatterv: Rank 0 cuts Matrix A into pieces and sends them to everyone.
    MPI_Scatterv(A_full, sendcounts, displs, MPI_INT, 
                 A_local, local_rows * cols_A, MPI_INT, 
                 0, MPI_COMM_WORLD);
    
    // MPI_Bcast: Rank 0 sends the ENTIRE Matrix B to everyone.
    // (Every rank needs the full B to multiply against their rows of A)
    MPI_Bcast(B_full, (long)rows_B * cols_B, MPI_INT, 0, MPI_COMM_WORLD);

    // --- OPTIMIZED COMPUTE: Tiled (Blocked) Matrix Multiplication ---
    // Instead of doing long rows, we work on small 64x64 squares ("tiles") (Around 7 seconds to 10 seconds faster).
    // This ensures data remains resident in the CPU Cache for repeated reuse.
    
    // Outer 3 loops: Iterate over the blocks (Steps of 64)
    for (int ii = 0; ii < local_rows; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < cols_A; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < cols_B; jj += BLOCK_SIZE) {
                
                // Calculate the edges of the current block
                // (Necessary if the matrix size isn't perfectly divisible by 64)
                int i_max = (ii + BLOCK_SIZE > local_rows) ? local_rows : ii + BLOCK_SIZE;
                int k_max = (kk + BLOCK_SIZE > cols_A)     ? cols_A     : kk + BLOCK_SIZE;
                int j_max = (jj + BLOCK_SIZE > cols_B)     ? cols_B     : jj + BLOCK_SIZE;

                // Inner 3 loops: Do the actual math inside the block
                for (int i = ii; i < i_max; i++) {
                    int * restrict c_row = &C_local[i * cols_B];
                    int * restrict a_row = &A_local[i * cols_A];
                    
                    for (int k = kk; k < k_max; k++) {
                        int a_val = a_row[k];
                        int * restrict b_row = &B_full[k * cols_B];
                        
                        // Innermost loop: This runs millions of times.
                        // Because we are using blocks, 'c_row' and 'b_row' are
                        // likely already in the cache, making this super fast.
                        for (int j = jj; j < j_max; j++) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }

    // MPI_Gatherv: Rank 0 collects all the calculated chunks of C from everyone.
    if (rank == 0) C_full = (int *)malloc((long)rows_A * cols_B * sizeof(int));
    
    MPI_Gatherv(C_local, local_rows * cols_B, MPI_INT,
                C_full, recvcounts, displs_C, MPI_INT,
                0, MPI_COMM_WORLD);

    end_par = MPI_Wtime(); // Stop Parallel Timer

    // --- Step 5: Final Report (Rank 0) ---
    // Only Rank 0 prints the results to the terminal.
    if (rank == 0) {
        double time_par = end_par - start_par;
        double time_ser = end_ser - start_ser;

        printf("[Rank 0] Parallel Time: %f seconds\n", time_par);
        
        printf("\n--- Results ---\n");
        printf("Serial Time:   %f s\n", time_ser);
        printf("Parallel Time: %f s\n", time_par);
        printf("Speedup:       %.2fx\n", time_ser / time_par);
        printf("Efficiency:    %.2f%%\n", (time_ser / time_par / size) * 100);

        // Verification: Compare every single number in Parallel vs Serial
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

        // Write the final result matrix to a text file
        FILE *f_out = fopen("result.txt", "w");
        fprintf(f_out, "%d %d\n", rows_A, cols_B);
        for (long i = 0; i < (long)rows_A * cols_B; i++) {
            fprintf(f_out, "%d ", C_full[i]);
            // Print a newline at the end of every row
            if ((i + 1) % cols_B == 0) fprintf(f_out, "\n");
        }
        fclose(f_out);
        
        // Free the heavy "Full" matrices
        free(A_full); free(C_full); free(C_serial);
    }

    // Free the "Local" memory on all ranks
    free(B_full); free(A_local); free(C_local);
    free(sendcounts); free(displs); free(recvcounts); free(displs_C);

    MPI_Finalize(); // Shut down MPI
    return 0;
}