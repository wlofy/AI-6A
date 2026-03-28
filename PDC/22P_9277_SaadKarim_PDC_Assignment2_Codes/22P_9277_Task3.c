#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes
    int data = rank; // Data to be sent (set to the process rank)
    int tag = 123;   // Message tag to identify the communication
    MPI_Status status; // Status object for receive operation

    if (rank == 0) {
        MPI_Send(&data, 1, MPI_INT, 1, tag, MPI_COMM_WORLD); // Send 'data' to process 1 with the specified tag
    } else if (rank == 1) {
        int recv_data; // Variable to store received data
        MPI_Recv(&recv_data, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status); // Receive data from process 0 with the matching tag
        printf("Received data %d with tag %d\n", recv_data, status.MPI_TAG); // Print the received data and the tag
    }

    MPI_Finalize();
    return 0;
}