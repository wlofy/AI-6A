#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int array_size = 16;
    int chunk_size = 4; // Assuming 4 processes
    int *global_array = NULL;
    int local_array[4];

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if number of processes is 4
    if (size != 4) {
        if (rank == 0) {
            printf("This program requires exactly 4 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Process 0 initializes the array
    if (rank == 0) {
        global_array = (int *)malloc(array_size * sizeof(int));
        for (int i = 0; i < array_size; i++) {
            global_array[i] = i + 1; // Example: 1, 2, ..., 16
        }
        printf("Process 0 initial array: ");
        for (int i = 0; i < array_size; i++) {
            printf("%d ", global_array[i]);
        }
        printf("\n");
    }

    // Scatter the array to all processes
    MPI_Scatter(global_array, chunk_size, MPI_INT, local_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process multiplies its chunk by 2
    for (int i = 0; i < chunk_size; i++) {
        local_array[i] *= 2;
    }
    printf("Process %d local array after multiplication: ", rank);
    for (int i = 0; i < chunk_size; i++) {
        printf("%d ", local_array[i]);
    }
    printf("\n");

    // Gather the updated chunks back to process 0
    MPI_Gather(local_array, chunk_size, MPI_INT, global_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Process 0 prints the final array
    if (rank == 0) {
        printf("Process 0 final array: ");
        for (int i = 0; i < array_size; i++) {
            printf("%d ", global_array[i]);
        }
        printf("\n");
        free(global_array);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}