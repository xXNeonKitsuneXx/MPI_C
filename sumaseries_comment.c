//Write a program to sum a series of 1 - 100 using MPI

#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    // --- MPI Initialization ---
    MPI_Init(&argc, &argv); // start mpi
    int rank, size;
    
    // Get the rank (ID) of the current process and the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get size

    // Define the range for the summation
    const int START_NUM = 1;
    const int END_NUM = 100;
    const int TOTAL_ELEMENTS = END_NUM - START_NUM + 1;

    // --- Task Partitioning ---
    // Calculate how many numbers each process should handle
    int chunk_size = TOTAL_ELEMENTS / size;
    int remainder = TOTAL_ELEMENTS % size;

    // Determine the starting and ending index for this process
    // The master (rank 0) handles the remainder for simpler distribution logic.
    int local_start_index = rank * chunk_size + (rank < remainder ? rank : remainder);
    int local_end_index = local_start_index + chunk_size + (rank < remainder ? 1 : 0);
    
    // Convert indices back to actual numbers in the 1-100 range
    int local_start_num = START_NUM + local_start_index;
    int local_end_num = START_NUM + local_end_index - 1; 

    // Adjust for the last process if the division isn't perfect
    if (rank == size - 1) {
        // Ensure the last process always includes the END_NUM
        local_end_num = END_NUM;
    } else if (local_start_num > END_NUM) {
        // Handle cases where more processes are launched than elements (e.g., 200 processes for 100 numbers)
        local_start_num = END_NUM + 1;
        local_end_num = END_NUM;
    }


    // --- Local Sum Calculation ---
    long long partial_sum = 0;
    for (int i = local_start_num; i <= local_end_num; i++) {
        partial_sum += i;
    }

    printf("Rank %d calculated sum from %d to %d (Partial Sum: %lld)\n", 
           rank, local_start_num, local_end_num, partial_sum);

    // --- MPI Reduction ---
    long long final_sum = 0;
    
    // MPI_Reduce collects all 'partial_sum' values, applies the MPI_SUM operation,
    // and stores the result in 'final_sum' on the root process (rank 0).
    // Reduce(send_buffer, recv_buffer, count, datatype, operation, root, communicator)
    MPI_Reduce(&partial_sum, &final_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // --- Output Result ---
    if (rank == 0) {
        printf("\n============================================\n");
        printf("Final Sum of numbers from %d to %d is: %lld\n", START_NUM, END_NUM, final_sum);
        printf("Expected Result (100*101/2): 5050\n");
        printf("============================================\n");
    }

    // --- MPI Finalization ---
    MPI_Finalize();
    return 0;
}
