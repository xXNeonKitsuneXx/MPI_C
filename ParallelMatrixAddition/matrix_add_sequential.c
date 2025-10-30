/**
 * matrix_add_sequential_mpi.c
 *
 * This is a "sequential" program written to run within an MPI environment.
 * Only process 0 will perform the matrix reading, addition, and writing.
 * All other processes will initialize and finalize but do no work.
 *
 * This is useful for comparing against:
 * 1. The TRUE sequential version (to see MPI init/finalize overhead).
 * 2. The FULL parallel version (to see MPI communication overhead).
 *
 * It still prints its execution time to stdout for the compare script.
 *
 * Compile:
 * mpicc -O3 -o sequential_mpi matrix_add_sequential_mpi.c
 * Run:
 * mpirun -np 4 ./sequential_mpi
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> // For high-resolution timing

#define ROWS 10000
#define COLS 10000

// --- Helper functions (same as other versions) ---

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
    double start_time, end_time;

    // Standard MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // All processes synchronize here before the timer starts
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // *** THE SEQUENTIAL PART ***
    // Only rank 0 does any of the actual work.
    if (rank == 0) {
        // 1. Allocate memory
        int** matrixA = allocate_matrix();
        int** matrixB = allocate_matrix();
        int** matrixResult = allocate_matrix();

        if (matrixA == NULL || matrixB == NULL || matrixResult == NULL) {
            fprintf(stderr, "Matrix allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 2. Read input matrices
        read_matrix("matrixA.txt", matrixA);
        read_matrix("matrixB.txt", matrixB);

        // 3. Perform addition
        // Note: We use clock_gettime for the *computation* part
        // but MPI_Wtime for the *total* time.
        // Let's stick to MPI_Wtime for the whole block for consistency.
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                matrixResult[i][j] = matrixA[i][j] + matrixB[i][j];
            }
        }

        // 4. Write result
        write_matrix("result_sequential.txt", matrixResult);

        // 5. Free memory
        free_matrix(matrixA);
        free_matrix(matrixB);
        free_matrix(matrixResult);
    }
    // All other processes (rank > 0) do nothing.
    // They just wait at the barrier.

    // All processes sync up here before stopping the timer
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // 6. Only rank 0 prints the time
    if (rank == 0) {
        // Print time to stdout for the comparison program
        printf("%f\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
