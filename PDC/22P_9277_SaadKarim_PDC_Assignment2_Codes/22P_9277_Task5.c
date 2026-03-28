#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total processes

    int data = 100;
    MPI_Status status;

    // Blocking communication timing
    MPI_Barrier(MPI_COMM_WORLD); // Wait for all processes
    double start_block = MPI_Wtime(); // Start timer

    if (rank == 0) {
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD); // Send to process 1 (blocking)
        MPI_Recv(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status); // Receive from process 1 (blocking)
    } else if (rank == 1) {
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status); // Receive from process 0 (blocking)
        MPI_Send(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD); // Send to process 0 (blocking)
    }

    MPI_Barrier(MPI_COMM_WORLD); // Wait for all processes
    double end_block = MPI_Wtime(); // End timer

    if (rank == 0) {
        printf("Blocking communication time: %f seconds\n", end_block - start_block);
    }

    // Non-blocking communication timing
    MPI_Request send_req, recv_req;
    data = 200;

    MPI_Barrier(MPI_COMM_WORLD); // Wait for all processes
    double start_nonblock = MPI_Wtime(); // Start timer

    if (rank == 0) {
        MPI_Isend(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &send_req); // Send to process 1 (non-blocking)
        MPI_Irecv(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &recv_req); // Receive from process 1 (non-blocking)
        MPI_Wait(&send_req, &status); // Wait for send to complete
        MPI_Wait(&recv_req, &status); // Wait for receive to complete
    } else if (rank == 1) {
        MPI_Irecv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &recv_req); // Receive from process 0 (non-blocking)
        MPI_Isend(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &send_req); // Send to process 0 (non-blocking)
        MPI_Wait(&recv_req, &status); // Wait for receive to complete
        MPI_Wait(&send_req, &status); // Wait for send to complete
    }

    MPI_Barrier(MPI_COMM_WORLD); // Wait for all processes
    double end_nonblock = MPI_Wtime(); // End timer

    if (rank == 0) {
        printf("Non-blocking communication time: %f seconds\n", end_nonblock - start_nonblock);
    }

    MPI_Finalize();
    return 0;
}