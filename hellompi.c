//Write a C MPI program
//4 processes
//Rank 1,2,3 send message to rank 0
//Rank 0 receives message from rank 1,2,3 and prints them

#include <stdio.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); //start mpi

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); //get size

    const int MASTER = 0;
    if(rank == MASTER) {
        char msg[100];
        MPI_Status status;
        printf("Rank 0 waiting for Messages...\n");

        for(int src = 1; src < size; src++) {
            MPI_Recv(msg, 100, MPI_CHAR, src, 0, MPI_COMM_WORLD, &status);
            printf("Received message from rank %d: %s\n", src, msg);
            // printf("Received message from rank %d: %s\n", status.MPI_SOURCE, msg);
        }
    }
    else {
        char msg[100];
        sprintf(msg, "Hello rank 0. I'm rank %d.", rank);

        MPI_Send(msg, strlen(msg)+1, MPI_CHAR, MASTER, 0, MPI_COMM_WORLD);
        printf("Rank %d sent message to rank 0.\n", rank);
    }
    MPI_Finalize(); //end mpi
    return 0;
}