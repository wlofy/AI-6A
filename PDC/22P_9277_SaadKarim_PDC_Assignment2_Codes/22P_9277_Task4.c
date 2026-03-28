#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes
    int data = rank; // Data to send (set to the process rank)
    int next = (rank + 1) % size; // Calculate the rank of the next process in a circular topology
    int prev = (rank - 1 + size) % size; // Calculate the rank of the previous process in a circular topology
    int recv_data; // Variable to store the received data
    MPI_Status status; // Status object for the receive operation

    MPI_Sendrecv(&data, 1, MPI_INT, next, 0, &recv_data, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status); // Perform a combined send and receive operation:
    printf("Process %d received %d from process %d\n", rank, recv_data, prev); // Print the received data and the source process
    MPI_Finalize(); 
    return 0;
}