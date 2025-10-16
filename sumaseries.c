//Write a program to sum a series of 1 - 100 using MPI

#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // start mpi
    int rank, size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get size

    const int START_NUM = 1;
    const int END_NUM = 100;
    const int TOTAL_ELEMENTS = END_NUM - START_NUM + 1;

    int chunk_size = TOTAL_ELEMENTS / size;
    int remainder = TOTAL_ELEMENTS % size;

    int local_start_index = rank * chunk_size + (rank < remainder ? rank : remainder);
    int local_end_index = local_start_index + chunk_size + (rank < remainder ? 1 : 0);
    
    int local_start_num = START_NUM + local_start_index;
    int local_end_num = START_NUM + local_end_index - 1; 

    if (rank == size - 1) {
        local_end_num = END_NUM;
    } else if (local_start_num > END_NUM) {
        local_start_num = END_NUM + 1;
        local_end_num = END_NUM;
    }

    long long partial_sum = 0;
    for (int i = local_start_num; i <= local_end_num; i++) {
        partial_sum += i;
    }

    printf("Rank %d calculated sum from %d to %d (Partial Sum: %lld)\n", 
           rank, local_start_num, local_end_num, partial_sum);

    long long final_sum = 0;
    
    MPI_Reduce(&partial_sum, &final_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // --- Output Result ---
    if (rank == 0) {
        printf("\n============================================\n");
        printf("Final Sum of numbers from %d to %d is: %lld\n", START_NUM, END_NUM, final_sum);
        printf("Expected Result (100*101/2): 5050\n");
        printf("============================================\n");
    }

    MPI_Finalize(); // end mpi
    return 0;
}
