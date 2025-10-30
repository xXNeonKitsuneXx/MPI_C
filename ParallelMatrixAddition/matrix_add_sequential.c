/**
 * matrix_add_sequential.c
 *
 * Performs sequential matrix addition (A + B) for 1000x1000 matrices.
 * Reads from 'matrixA.txt' and 'matrixB.txt'.
 * Writes the result to 'result_sequential.txt'.
 *
 * This program prints its execution time for the addition to stdout,
 * which is used by the 'matrix_add_compare' program.
 *
 * Compile:
 * gcc -O3 -o sequential matrix_add_sequential.c
 * Run:
 * ./sequential
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h> // For high-resolution timing

#define ROWS 10000
#define COLS 10000

/**
 * Allocates a 2D matrix contiguously in memory.
 * This is important for performance and allows easy MPI operations
 * on the whole data block if needed (though not used in this file).
 *
 * @return A pointer to an array of row pointers (int**).
 */
int** allocate_matrix() {
    // Allocate memory for the data block (all elements)
    int* data = (int*)malloc(ROWS * COLS * sizeof(int));
    if (data == NULL) return NULL;

    // Allocate memory for the row pointers
    int** row_ptrs = (int**)malloc(ROWS * sizeof(int*));
    if (row_ptrs == NULL) {
        free(data);
        return NULL;
    }

    // Point the row pointers to the correct locations in the data block
    for (int i = 0; i < ROWS; i++) {
        row_ptrs[i] = &(data[i * COLS]);
    }
    return row_ptrs;
}

/**
 * Frees a contiguously allocated matrix.
 *
 * @param matrix The matrix to free.
 */
void free_matrix(int** matrix) {
    if (matrix) {
        // Free the main data block
        free(matrix[0]);
        // Free the array of row pointers
        free(matrix);
    }
}

/**
 * Reads matrix data from a file into the allocated matrix.
 *
 * @param filename The name of the file to read.
 * @param matrix The matrix to populate.
 */
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

/**
 * Writes the matrix data to a file.
 *
 * @param filename The name of the file to write.
 * @param matrix The matrix containing the data.
 */
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

int main() {
    // 1. Allocate memory
    int** matrixA = allocate_matrix();
    int** matrixB = allocate_matrix();
    int** matrixResult = allocate_matrix();

    if (matrixA == NULL || matrixB == NULL || matrixResult == NULL) {
        fprintf(stderr, "Matrix allocation failed\n");
        return 1;
    }

    // 2. Read input matrices
    // Assuming matrixA.txt and matrixB.txt are in the same directory
    read_matrix("matrixA.txt", matrixA);
    read_matrix("matrixB.txt", matrixB);

    // 3. Perform addition and time it
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start); // Start timer

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            matrixResult[i][j] = matrixA[i][j] + matrixB[i][j];
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end); // Stop timer

    // Calculate elapsed time in seconds
    double time_spent = (end.tv_sec - start.tv_sec);
    time_spent += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    // 4. Write result
    write_matrix("result_sequential.txt", matrixResult);

    // 5. Free memory
    free_matrix(matrixA);
    free_matrix(matrixB);
    free_matrix(matrixResult);

    // 6. Print time to stdout for the comparison program
    // This is the only output to stdout
    printf("%f\n", time_spent);

    return 0;
}
