/**
 * matrix_add_compare.c
 *
 * This program automates the comparison between the sequential and
 * parallel matrix addition programs.
 *
 * 1. Compiles 'matrix_add_sequential.c' into './sequential'
 * 2. Compiles 'matrix_add_parallel.c' into './parallel'
 * 3. Executes './sequential' and captures its stdout (time).
 * 4. Executes 'mpirun -np 8 ./parallel' and captures its stdout (time).
 * 5. Prints a formatted comparison.
 *
 * Note:
 * - This assumes 'gcc' and 'mpicc' are in your system's PATH.
 * - This assumes you have the input files 'matrixA.txt' and 'matrixB.txt'.
 * - The number of parallel processes is hardcoded to 8.
 *
 * Compile:
 * gcc -o compare matrix_add_compare.c
 * Run:
 * ./compare
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper function to capture the time (a double) from a command
double capture_time(const char* command) {
    FILE* pipe;
    char buffer[128];
    double time = 0.0;

    // Open a pipe to the command
    pipe = popen(command, "r");
    if (pipe == NULL) {
        fprintf(stderr, "Failed to run command: %s\n", command);
        exit(EXIT_FAILURE);
    }

    // Read the output from the command (should be just one line with a double)
    if (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        // Remove trailing newline if present
        char* newline = strchr(buffer, '\n');
        if (newline) *newline = '\0';
        
        // Scan the double from the buffer
        if (sscanf(buffer, "%lf", &time) != 1) {
            fprintf(stderr, "Could not parse time from command output: %s\n", buffer);
        }
    } else {
        fprintf(stderr, "No output from command: %s\n", command);
    }


    // Close the pipe
    if (pclose(pipe) == -1) {
        perror("Error closing pipe");
    }

    return time;
}

int main() {
    // 1. Compile the programs
    printf("Compiling sequential version...\n");
    if (system("gcc -O3 -Wall matrix_add_sequential.c -o matrix_add_sequential") != 0) {
        fprintf(stderr, "Sequential compilation failed.\n");
        return 1;
    }

    printf("Compiling parallel version...\n");
    if (system("mpicc -O3 -Wall matrix_add_parallel.c -o matrix_add_parallel") != 0) {
        fprintf(stderr, "Parallel compilation failed.\n");
        return 1;
    }

    printf("Compilation complete.\n\n");

    // 2. Run sequential version and capture time
    printf("Running sequential version...\n");
    double sequential_time = capture_time("./matrix_add_sequential");
    if (sequential_time == 0.0) {
        fprintf(stderr, "Sequential run failed or returned 0.0s time.\n");
    }

    // 3. Run parallel version and capture time
    // We'll use 8 processes as a standard example
    const char* parallel_command = "mpirun -np 8 ./matrix_add_parallel";
    printf("Running parallel version (8 processes)...\n");
    double parallel_time = capture_time(parallel_command);
    if (parallel_time == 0.0) {
        fprintf(stderr, "Parallel run failed or returned 0.0s time.\n");
    }

    // 4. Print comparison
    printf("\n--- Matrix Addition Comparison (1000x1000) ---\n");
    printf("Sequential Time: %f seconds\n", sequential_time);
    printf("Parallel Time:   %f seconds (with 8 processes)\n", parallel_time);
    
    if (parallel_time > 0 && sequential_time > 0) {
        double speedup = sequential_time / parallel_time;
        printf("Speedup:         %.2fx\n", speedup);
    } else {
        printf("Speedup:         N/A (one of the times was 0.0)\n");
    }
    printf("\n");
    printf("Sequential result is in 'result_sequential.txt'\n");
    printf("Parallel result is in 'result_parallel.txt'\n");
    return 0;
}
