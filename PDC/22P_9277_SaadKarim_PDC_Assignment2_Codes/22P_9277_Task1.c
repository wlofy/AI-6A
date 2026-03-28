#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Initialize MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total processes
    int data = rank * 10; // Calculate data based on rank
    printf("Process %d has data %d\n", rank, data); // Print rank and data
    MPI_Finalize(); // Finalize MPI
    return 0;
}