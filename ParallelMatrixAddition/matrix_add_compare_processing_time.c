#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    system("mpirun -np 4 ./matrix_add_parallel matrixA.txt matrixB.txt result_parallel.txt");
    system("./matrix_add_sequential matrixA.txt matrixB.txt result_sequential.txt");

    FILE *fp1 = fopen("result_sequential.txt", "r");
    FILE *fp2 = fopen("result_parallel.txt", "r");
    FILE *fp_out = fopen("comparison_result.txt", "w");

    if (!fp1 || !fp2 || !fp_out) {
        perror("File open failed");
        return 1;
    }

    char line[256];
    double time_seq = 0.0, time_par = 0.0;

    while (fgets(line, sizeof(line), fp1)) {
        if (strstr(line, "Processing time"))
            sscanf(line, "Processing time (sequential): %lf", &time_seq);
    }
    while (fgets(line, sizeof(line), fp2)) {
        if (strstr(line, "Processing time"))
            sscanf(line, "Processing time (parallel): %lf", &time_par);
    }

    fprintf(fp_out, "Sequential time: %.6f seconds\n", time_seq);
    fprintf(fp_out, "Parallel time: %.6f seconds\n", time_par);
    fprintf(fp_out, "Speedup: %.2fx\n", time_seq / time_par);

    fclose(fp1);
    fclose(fp2);
    fclose(fp_out);

    printf("Comparison complete. See comparison_result.txt\n");
    return 0;
}