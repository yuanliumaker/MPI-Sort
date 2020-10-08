#include <mpi.h>

int MPI_Sort_bykey_uint8_t (
    const KEY_T * sendkeys,
    const void * sendvals,
    const int sendcount,
    MPI_Datatype keytype,
    MPI_Datatype valtype,
    KEY_T * recvkeys,
    void * recvvals,
    const int recvcount,
    MPI_Comm comm);

int MPI_Sort (
    const void * sendbuf,
    const int sendcount,
    MPI_Datatype datatype,
    void * recvbuf,
    const int recvcount,
    MPI_Comm comm)
{
    if (MPI_UNSIGNED_CHAR == datatype || MPI_BYTE == datatype)
	return MPI_Sort_bykey_uint8_t (
	    sendbuf, 0, sendcount,
	    datatype, -1, recvbuf, 0, recvcount, comm);

    return -1;
}
