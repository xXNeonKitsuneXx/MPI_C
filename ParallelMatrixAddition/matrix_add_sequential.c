#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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
    if (argc != 4) {
        printf("Usage: %s matrixA.txt matrixB.txt result_sequential.txt\n", argv[0]);
        return 1;
    }

    double *A, *B, *C;
    int rows, cols;

    if (read_matrix(argv[1], &A, &rows, &cols) || read_matrix(argv[2], &B, &rows, &cols)) {
        return 1;
    }

    C = (double *)malloc(rows * cols * sizeof(double));

    clock_t start = clock();
    for (int i = 0; i < rows * cols; i++) {
        C[i] = A[i] + B[i];
    }
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    FILE *fp = fopen(argv[3], "w");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fp, "%.2lf ", C[i * cols + j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "Processing time (sequential): %.6f seconds\n", time_spent);
    fclose(fp);

    free(A); free(B); free(C);
    return 0;
}