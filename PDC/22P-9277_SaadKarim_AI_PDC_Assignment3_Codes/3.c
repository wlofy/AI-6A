#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int local_number;
    int *all_numbers = NULL;
    int max_value;
    double sum, average;
    double start_time, end_time, allgather_time, reduce_time;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if number of processes is 6
    if (size != 6) {
        if (rank == 0) {
            printf("This program requires exactly 6 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Seed random number generator
    srand(time(NULL) + rank);
    local_number = (rand() % 100) + 1; // Random number between 1 and 100
    printf("Process %d generated number: %d\n", rank, local_number);

    // Allocate array to store all numbers
    all_numbers = (int *)malloc(size * sizeof(int));

    // Time MPI_Allgather
    start_time = MPI_Wtime();
    MPI_Allgather(&local_number, 1, MPI_INT, all_numbers, 1, MPI_INT, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    allgather_time = end_time - start_time;

    // Print collected numbers
    printf("Process %d collected numbers: ", rank);
    for (int i = 0; i < size; i++) {
        printf("%d ", all_numbers[i]);
    }
    printf("\n");

    // Time MPI_Reduce for maximum
    start_time = MPI_Wtime();
    MPI_Reduce(&local_number, &max_value, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    reduce_time = end_time - start_time;

    // Process 0 prints the maximum
    if (rank == 0) {
        printf("Maximum value: %d\n", max_value);
    }

    // Compute average using MPI_Allreduce
    sum = local_number;
    MPI_Allreduce(&sum, &average, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    average /= size;

    // All processes print the average
    printf("Process %d average value: %.2f\n", rank, average);

    // Print timing information
    printf("Process %d: MPI_Allgather time = %.6f seconds, MPI_Reduce time = %.6f seconds\n", 
           rank, allgather_time, reduce_time);

    // Free allocated memory
    free(all_numbers);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}