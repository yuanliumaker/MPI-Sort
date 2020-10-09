#include <stdint.h>
#include <mpi.h>

int MPI_Sort_bykey_uint8_t (
       const uint8_t * sendkeys,
       const void * sendvals0,
       const void * sendvals1,
       const int sendcount,
       MPI_Datatype valtype0,
       MPI_Datatype valtype1,
       uint8_t * recvkeys,
       void * recvvals0,
       void * recvvals1,
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
    enum { DONTCARE = -1 };

    if (MPI_IN_PLACE == sendbuf)
	sendbuf = recvbuf;

    if (MPI_UNSIGNED_CHAR == datatype || MPI_BYTE == datatype)
	return MPI_Sort_bykey_uint8_t (
	    sendbuf, 0, 0, sendcount,
	    DONTCARE, DONTCARE, recvbuf, 0, 0, recvcount, comm);

    return -1;
}

int MPI_Sort_bykey (
    const void * sendkeys,
    const void * sendvals,
    const int sendcount,
    MPI_Datatype keytype,
    MPI_Datatype valtype,
    void * recvkeys,
    void * recvvals,
    const int recvcount,
    MPI_Comm comm)
{
    enum { DONTCARE = -1 };

    if (MPI_IN_PLACE == sendkeys)
	sendkeys = recvkeys;

    if (MPI_UNSIGNED_CHAR == keytype || MPI_BYTE == keytype)
	return MPI_Sort_bykey_uint8_t (
	    sendkeys, 0, sendvals, sendcount,
	    DONTCARE, valtype, recvkeys, 0, recvvals, recvcount, comm);

    return -1;
}
