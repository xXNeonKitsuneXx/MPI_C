#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 1000

int read_matrix(const char *filename, double **matrix, int *rows, int *cols) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("File open failed");
        return 1;
    }

    int r = 0, c = 0;
    char line[MAX_LINE];
    while (fgets(line, MAX_LINE, fp)) {
        if (r == 0) {
            char *token = strtok(line, " ");
            while (token) {
                c++;
                token = strtok(NULL, " ");
            }
        }
        r++;
    }
    rewind(fp);

    *rows = r;
    *cols = c;
    *matrix = (double *)malloc(r * c * sizeof(double));

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            fscanf(fp, "%lf", &((*matrix)[i * c + j]));
        }
    }
    fclose(fp);
    return 0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) printf("Usage: %s matrixA.txt matrixB.txt result_par.txt\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    double *A = NULL, *B = NULL, *C = NULL;
    int rows = 0, cols = 0;

    if (rank == 0) {
        read_matrix(argv[1], &A, &rows, &cols);
        read_matrix(argv[2], &B, &rows, &cols);
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_elements = rows * cols;
    int chunk = total_elements / size;
    int remainder = total_elements % size;

    int local_count = (rank < remainder) ? chunk + 1 : chunk;
    double *localA = (double *)malloc(local_count * sizeof(double));
    double *localB = (double *)malloc(local_count * sizeof(double));
    double *localC = (double *)malloc(local_count * sizeof(double));

    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = (i < remainder) ? chunk + 1 : chunk;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, localA, local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, sendcounts, displs, MPI_DOUBLE, localB, local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start = MPI_Wtime();
    for (int i = 0; i < local_count; i++) {
        localC[i] = localA[i] + localB[i];
    }
    double end = MPI_Wtime();

    if (rank == 0)
        C = (double *)malloc(total_elements * sizeof(double));

    MPI_Gatherv(localC, local_count, MPI_DOUBLE, C, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double total_time = end - start;
        FILE *fp = fopen(argv[3], "w");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fprintf(fp, "%.2lf ", C[i * cols + j]);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "Processing time (parallel): %.6f seconds\n", total_time);
        fclose(fp);
        free(A); free(B); free(C); free(sendcounts); free(displs);
    }

    free(localA); free(localB); free(localC);
    MPI_Finalize();
    return 0;
}