/**
 * matrix_add_parallel.c
 *
 * Performs parallel matrix addition (A + B) for 1000x1000 matrices using MPI.
 *
 * Strategy:
 * 1. Process 0 reads both matrices from 'matrixA.txt' and 'matrixB.txt'.
 * 2. Process 0 calculates how many rows to send to each process (handles remainders).
 * 3. Process 0 "scatters" (using MPI_Scatterv) chunks of A and B to all processes.
 * 4. All processes (including 0) compute their local sum.
 * 5. Process 0 "gathers" (using MPI_Gatherv) the results.
 * 6. Process 0 writes the final matrix to 'result_parallel.txt' and
 * prints the execution time to stdout.
 *
 * Compile:
 * mpicc -O3 -o parallel matrix_add_parallel.c
 * Run:
 * mpirun -np 8 ./parallel
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> // For clock_gettime, although MPI_Wtime is used

#define ROWS 10000
#define COLS 10000

// --- Helper functions (same as sequential version) ---

int** allocate_matrix() {
    int* data = (int*)malloc(ROWS * COLS * sizeof(int));
    if (data == NULL) return NULL;
    int** row_ptrs = (int**)malloc(ROWS * sizeof(int*));
    if (row_ptrs == NULL) {
        free(data);
        return NULL;
    }
    for (int i = 0; i < ROWS; i++) {
        row_ptrs[i] = &(data[i * COLS]);
    }
    return row_ptrs;
}

void free_matrix(int** matrix) {
    if (matrix) {
        free(matrix[0]);
        free(matrix);
    }
}

void read_matrix(const char* filename, int** matrix) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (fscanf(file, "%d", &matrix[i][j]) != 1) {
                fprintf(stderr, "Error reading matrix data from %s\n", filename);
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
}

void write_matrix(const char* filename, int** matrix) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening output file");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            fprintf(file, "%d ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

// --- Main MPI Logic ---
int main(int argc, char** argv) {
    int rank, size;
    int** matrixA = NULL;
    int** matrixB = NULL;
    int** matrixResult = NULL;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- Start Timer (All processes sync here) ---
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // --- Root Process (rank 0): Allocate and Read Data ---
    if (rank == 0) {
        matrixA = allocate_matrix();
        matrixB = allocate_matrix();
        matrixResult = allocate_matrix();
        if (matrixA == NULL || matrixB == NULL || matrixResult == NULL) {
            fprintf(stderr, "Root matrix allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        read_matrix("matrixA.txt", matrixA);
        read_matrix("matrixB.txt", matrixB);
    }

    // --- All Processes: Calculate row distribution ---
    int rows_per_proc = ROWS / size;
    int remainder = ROWS % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int local_size = local_rows * COLS; // Total elements for this process

    // --- All Processes: Allocate local buffers ---
    int* local_A = (int*)malloc(local_size * sizeof(int));
    int* local_B = (int*)malloc(local_size * sizeof(int));
    int* local_Result = (int*)malloc(local_size * sizeof(int));

    if (local_A == NULL || local_B == NULL || local_Result == NULL) {
        fprintf(stderr, "Rank %d: Local buffer allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- Root Process (rank 0): Prepare Scatterv parameters ---
    int* send_counts = NULL;
    int* displs = NULL; 

    if (rank == 0) {
        send_counts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));

        int current_displ = 0;
        for (int i = 0; i < size; i++) {
            int rows_for_this_proc = rows_per_proc + (i < remainder ? 1 : 0);
            send_counts[i] = rows_for_this_proc * COLS;
            displs[i] = current_displ;
            current_displ += send_counts[i];
        }
    }
    
    // --- Scatter Data ---
    MPI_Scatterv(
        (rank == 0) ? matrixA[0] : NULL, 
        send_counts,                     
        displs,                          
        MPI_INT,                         
        local_A,                         
        local_size,                      
        MPI_INT,                         
        0,                               
        MPI_COMM_WORLD
    );

    MPI_Scatterv(
        (rank == 0) ? matrixB[0] : NULL,
        send_counts,
        displs,
        MPI_INT,
        local_B,
        local_size,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // --- All Processes: Local Computation ---
    for (int i = 0; i < local_size; i++) {
        local_Result[i] = local_A[i] + local_B[i];
    }

    // --- Gather Data ---
    MPI_Gatherv(
        local_Result,                    
        local_size,                      
        MPI_INT,                         
        (rank == 0) ? matrixResult[0] : NULL, 
        send_counts,                     
        displs,                          
        MPI_INT,                         
        0,                               
        MPI_COMM_WORLD
    );

    // --- Root Process (rank 0): Write Result ---
    if (rank == 0) {
        write_matrix("result_parallel.txt", matrixResult);
    }

    // --- Free ALL Memory (must be done *before* final barrier) ---
    if (rank == 0) {
        free_matrix(matrixA);
        free_matrix(matrixB);
        free_matrix(matrixResult);
        free(send_counts);
        free(displs);
    }
    free(local_A);
    free(local_B);
    free(local_Result);

    // --- Stop Timer (All processes sync here) ---
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // --- Root Process (rank 0): Print Time ---
    if (rank == 0) {
        // Print time to stdout for the comparison program
        printf("%f\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}