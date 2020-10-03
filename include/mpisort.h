#include <mpi.h>

int MPI_Sort (
    const void * sendbuf,
    const int count,
    MPI_Datatype datatype,
    const void * recvbuf );

int MPI_Sort_bykey (
    const void * sendkeys,
    const void * sendvals,
    const int count,
    MPI_Datatype keytype,
    MPI_Datatype valtype,
    const void * recvkeys,
    const void * recvvals );
