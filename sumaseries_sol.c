//Write a program to sum a series of 1 - 100 using MPI

#include <stdio.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // start mpi

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get size

    const int N = 100;
    int start,end;
    int local_sum = 0, total_sum = 0;

    int chunk_size = N / size;
    start = rank * chunk_size + 1;
    end = (rank == size - 1) ? N : start + chunk_size - 1;

    for (int i = start; i <= end; i++){
        local_sum += i;
    }

    printf("Rank %d: sum of[%d .. %d]=%d\n", rank, start, end, local_sum);
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Total Sum from 1 to %d is %d\n", N, total_sum);
    }
    

    MPI_Finalize(); // end mpi
    return 0;
}
