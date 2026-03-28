#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, value;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Process 0 initializes the variable
    if (rank == 0) {
        value = 42; // Example value
        printf("Process 0 broadcasting value: %d\n", value);
    }

    // Broadcast the value from process 0 to all processes
    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process prints the received value
    printf("Process %d received value: %d\n", rank, value);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}