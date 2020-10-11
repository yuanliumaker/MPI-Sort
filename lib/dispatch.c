#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include <assert.h>
#include <mpi.h>

#include "macros.h"

#define DECLARE(NAME, TYPE)			\
    int NAME (					\
	const TYPE  * sendkeys,			\
	const void * sendvals0,			\
	const void * sendvals1,			\
	const int sendcount,			\
	MPI_Datatype valtype0,			\
	MPI_Datatype valtype1,			\
	TYPE * recvkeys,			\
	void * recvvals0,			\
	void * recvvals1,			\
	const int recvcount,			\
	MPI_Comm comm);

DECLARE(dsort_uint8_t, uint8_t);
DECLARE(dsort_uint16_t, int16_t);
DECLARE(dsort_uint32_t, int32_t);

enum { DONTCARE_TYPE = -1 };

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
    DIE_UNLESS(sendcount >= 0 && recvcount >= 0);

    if (MPI_IN_PLACE == sendkeys)
	sendkeys = recvkeys;

    if (MPI_UNSIGNED_CHAR == keytype || MPI_INT16_T == keytype || MPI_BYTE == keytype)
	return dsort_uint8_t (
	    sendkeys, 0, sendvals, sendcount,
	    DONTCARE_TYPE, valtype, recvkeys, 0, recvvals, recvcount, comm);

    if (MPI_UNSIGNED_SHORT == keytype || MPI_UINT16_T == keytype)
	return dsort_uint16_t (
	    sendkeys, 0, sendvals, sendcount,
	    DONTCARE_TYPE, valtype, recvkeys, 0, recvvals, recvcount, comm);

    if (MPI_UNSIGNED == keytype || MPI_UINT32_T == keytype)
	return dsort_uint32_t (
	    sendkeys, 0, sendvals, sendcount,
	    DONTCARE_TYPE, valtype, recvkeys, 0, recvvals, recvcount, comm);

    return MPI_ERR_INTERN;
}

int MPI_Sort (
    const void * sendbuf,
    const int sendcount,
    MPI_Datatype datatype,
    void * recvbuf,
    const int recvcount,
    MPI_Comm comm)
{
    return
	MPI_Sort_bykey(sendbuf, 0, sendcount, datatype, DONTCARE_TYPE, recvbuf, 0, recvcount, comm);
}
