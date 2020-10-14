#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include <assert.h>
#include <mpi.h>

#include "macros.h"

int rmanip_to_unsigned (
	 MPI_Datatype type,
	 const ptrdiff_t count,
	 void * const inout );

int rmanip_from_unsigned (
	 MPI_Datatype type,
	 const ptrdiff_t count,
	 void * const inout );

#define DECLARE(NAME, TYPE)			\
    int NAME (					\
	const int stable,			\
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
    int RADIX = 1;
    READENV(RADIX, atoi);

    int STABLE = 0;
    READENV(STABLE, atoi);
    STABLE = !!STABLE;

    DIE_UNLESS(sendcount >= 0 && recvcount >= 0);

    if (MPI_IN_PLACE == sendkeys)
	sendkeys = recvkeys;

    ptrdiff_t esz = 0;

    if (recvvals)
    {
	int s;
	MPI_CHECK(MPI_Type_size(valtype, &s));
	esz = s;
    }

    if (MPI_UNSIGNED_CHAR == keytype || MPI_INT8_T == keytype || MPI_BYTE == keytype)
	return dsort_uint8_t (
	    STABLE, sendkeys, 0, sendvals, sendcount,
	    DONTCARE_TYPE, valtype, recvkeys, 0, recvvals, recvcount, comm);

    if (MPI_UNSIGNED_SHORT == keytype || MPI_UINT16_T == keytype)
	if (RADIX)
	{
	    /* radix sort upon dsort_uint16_t */
	    uint8_t * tmpk = malloc(sizeof(uint8_t) * MAX(recvcount, sendcount));
	    void * tmpv0 = malloc(sizeof(uint16_t) * recvcount);
	    void * tmpv1 = NULL;

	    if (recvvals)
		tmpv1 = malloc(esz * recvcount);

	    /* extract lower half */
	    for(ptrdiff_t i = 0; i < sendcount; ++i)
		tmpk[i] = *(0 + (uint8_t *)(i + (uint16_t *)sendkeys));

	    /* TODO: intermediate results should be partitioned homogenously,
	       ignoring sendcount and recvcount */
	    dsort_uint8_t (
		STABLE, tmpk, sendkeys, sendvals, sendcount,
		MPI_UNSIGNED_SHORT, valtype, 0, tmpv0, tmpv1, recvcount, comm);

	    /* extract higher half  */
	    for(ptrdiff_t i = 0; i < recvcount; ++i)
		tmpk[i] = *(1 + (uint8_t *)(i + (uint16_t *)tmpv0));

	    dsort_uint8_t (
		1, tmpk, tmpv0, tmpv1, recvcount,
		MPI_UNSIGNED_SHORT, valtype, 0, recvkeys, recvvals, recvcount, comm);

	    if (tmpv1)
		free(tmpv1);

	    free(tmpv0);
	    free(tmpk);

	    return MPI_SUCCESS;
	}
	else
	    return dsort_uint16_t (
		STABLE, sendkeys, 0, sendvals, sendcount,
		DONTCARE_TYPE, valtype, recvkeys, 0, recvvals, recvcount, comm);

    if (MPI_UNSIGNED == keytype || MPI_UINT32_T == keytype)
	if (RADIX)
	{
	    /* radix sort upon dsort_uint16_t */
	    uint16_t * tmpk = malloc(sizeof(uint16_t) * MAX(recvcount, sendcount));
	    void * tmpv0 = malloc(sizeof(uint32_t) * recvcount);
	    void * tmpv1 = NULL;

	    if (recvvals)
		tmpv1 = malloc(esz * recvcount);

	    /* extract lower half */
	    for(ptrdiff_t i = 0; i < sendcount; ++i)
		tmpk[i] = *(0 + (uint16_t *)(i + (uint32_t *)sendkeys));

	    dsort_uint16_t (
		STABLE, tmpk, sendkeys, sendvals, sendcount,
		MPI_UNSIGNED, valtype, 0, tmpv0, tmpv1, recvcount, comm);

	    /* extract higher half */
	    for(ptrdiff_t i = 0; i < recvcount; ++i)
		tmpk[i] = *(1 + (uint16_t *)(i + (uint32_t *)tmpv0));

	    dsort_uint16_t (
		1, tmpk, tmpv0, tmpv1, recvcount,
		MPI_UNSIGNED, valtype, 0, recvkeys, recvvals, recvcount, comm);

	    if (tmpv1)
		free(tmpv1);

	    free(tmpv0);
	    free(tmpk);

	    return MPI_SUCCESS;
	}
	else
	    return dsort_uint32_t (
		0, sendkeys, 0, sendvals, sendcount,
		DONTCARE_TYPE, valtype, recvkeys, 0, recvvals, recvcount, comm);

    if (MPI_UNSIGNED_LONG == keytype || MPI_UINT64_T == keytype)
	/* we enforce radix sort for 64 bit digits */
	{
	    __extension__ void extract (
		const int word,
		const ptrdiff_t count,
		const uint64_t * const restrict in,
		uint16_t * const restrict out )
	    {
		for(ptrdiff_t i = 0; i < count; ++i)
		    out[i] = *(word + (uint16_t *)(i + (uint64_t *)in));
	    }

	    uint16_t * tmpk = malloc(sizeof(uint16_t) * MAX(recvcount, sendcount));

	    void * tmpv0 = malloc(sizeof(uint64_t) * recvcount);

	    void * tmpv1 = NULL;

	    if (recvvals)
		tmpv1 = malloc(esz * recvcount);

	    extract(0, sendcount, sendkeys, tmpk);

	    dsort_uint16_t(
		STABLE, tmpk, sendkeys, sendvals, sendcount,
		MPI_UNSIGNED_LONG, valtype, 0, tmpv0, tmpv1, recvcount, comm);

	    extract(1, recvcount, tmpv0, tmpk);

	    dsort_uint16_t(
		1, tmpk, tmpv0, tmpv1, recvcount,
		MPI_UNSIGNED_LONG, valtype, 0, recvkeys, recvvals, recvcount, comm);

	    extract(2, recvcount, recvkeys, tmpk);

	    dsort_uint16_t(
		1, tmpk, recvkeys, recvvals, recvcount,
		MPI_UNSIGNED_LONG, valtype, 0, tmpv0, tmpv1, recvcount, comm);

	    extract(3, recvcount, tmpv0, tmpk);

	    dsort_uint16_t(
		1, tmpk, tmpv0, tmpv1, recvcount,
		MPI_UNSIGNED_LONG, valtype, 0, recvkeys, recvvals, recvcount, comm);

	    if (tmpv1)
		free(tmpv1);

	    free(tmpv0);
	    free(tmpk);

	    return MPI_SUCCESS;
	}

    if (MPI_FLOAT == keytype)
    {
	__extension__ void flip (
	    const ptrdiff_t n,
	    uint32_t * inout )
	{
	    for (ptrdiff_t i = 0; i < n; ++i)
	    {
		int32_t intbits = inout[i];

		intbits ^= (intbits >> 31) & 0x7fffffff;

		inout[i] = (uint32_t)(intbits - 2147483648ll);
	    }
	}

	__extension__ void unflip (
	    const ptrdiff_t n,
	    uint32_t * inout )
	{
	    for (ptrdiff_t i = 0; i < n; ++i)
	    {
		int32_t intbits = inout[i] + 2147483648ll;

		intbits ^= (intbits >> 31) & 0x7fffffff;

		inout[i] = (uint32_t)intbits;
	    }
	}

	flip(sendcount, (uint32_t *)sendkeys);

	MPI_CHECK(
	    MPI_Sort_bykey(sendkeys, sendvals, sendcount,
			   MPI_UNSIGNED, valtype, recvkeys, recvvals, recvcount, comm));

	if (recvkeys != sendkeys)
	    unflip(sendcount, (uint32_t *)sendkeys);

	unflip(recvcount, recvkeys);

	return MPI_SUCCESS;
    }

    if (MPI_CHAR == keytype || MPI_SHORT == keytype || MPI_INTEGER == keytype)
    {
	MPI_CHECK(rmanip_to_unsigned(keytype, sendcount, (void *)sendkeys));

	const MPI_Datatype newtype =
	    MPI_UNSIGNED_CHAR * (MPI_CHAR == keytype || MPI_INT8_T == keytype)
	    | MPI_UNSIGNED_SHORT * (MPI_SHORT == keytype || MPI_INT16_T == keytype)
	    | MPI_UNSIGNED * (MPI_INTEGER == keytype || MPI_INT32_T == keytype)
	    | MPI_UNSIGNED_LONG * (MPI_LONG == keytype || MPI_INT64_T == keytype);

	MPI_CHECK(MPI_Sort_bykey(sendkeys, sendvals, sendcount, newtype, valtype,
				 recvkeys, recvvals, recvcount, comm));

	if (recvkeys != sendkeys)
	    MPI_CHECK(rmanip_from_unsigned(keytype, sendcount, (void *)sendkeys));

	MPI_CHECK(rmanip_from_unsigned(keytype, recvcount, recvkeys));

	return MPI_SUCCESS;
    }

    return MPI_ERR_TYPE;
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
