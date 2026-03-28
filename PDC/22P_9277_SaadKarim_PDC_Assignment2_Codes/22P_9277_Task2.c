#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Init MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get size
    int data = rank; // Send data
    int recv_data; // Receive data
    MPI_Request request; // Request handle
    MPI_Status status;   // Status object

    if (rank == 0) {
        MPI_Isend(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &request); // Non-blocking send
        MPI_Wait(&request, &status); // Wait for send
    } else if (rank == 1) {
        MPI_Irecv(&recv_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request); // Non-blocking receive
        MPI_Wait(&request, &status); // Wait for receive
        printf("Process 1 received data %d from Process 0\n", recv_data); // Print result
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}