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

    /* squeeze types into fewer ones */
    {
	if (MPI_IN_PLACE == sendkeys)
	    sendkeys = recvkeys;

	if (MPI_CHAR == keytype)
	    keytype = MPI_INT8_T;

	if (MPI_SHORT == keytype)
	    keytype = MPI_INT16_T;

	if (MPI_INTEGER == keytype)
	    keytype = MPI_INT32_T;

	if (MPI_LONG == keytype)
	    keytype = MPI_INT64_T;

	if (MPI_BYTE == keytype)
	    keytype = MPI_UINT8_T;

	if (MPI_UNSIGNED_SHORT == keytype)
	    keytype = MPI_UINT16_T;

	if (MPI_UNSIGNED == keytype)
	    keytype = MPI_UINT32_T;

	if (MPI_UNSIGNED_LONG == keytype)
	    keytype = MPI_UINT64_T;
    }

    ptrdiff_t esz = 0;

    if (recvvals)
    {
	int s;
	MPI_CHECK(MPI_Type_size(valtype, &s));
	esz = s;
    }

    if (MPI_UINT8_T == keytype)
	return dsort_uint8_t (
	    STABLE, sendkeys, 0, sendvals, sendcount,
	    DONTCARE_TYPE, valtype, recvkeys, 0, recvvals, recvcount, comm);

    if (MPI_UINT16_T == keytype)
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
		MPI_UINT16_T, valtype, 0, tmpv0, tmpv1, recvcount, comm);

	    /* extract higher half  */
	    for(ptrdiff_t i = 0; i < recvcount; ++i)
		tmpk[i] = *(1 + (uint8_t *)(i + (uint16_t *)tmpv0));

	    dsort_uint8_t (
		1, tmpk, tmpv0, tmpv1, recvcount,
		MPI_UINT16_T, valtype, 0, recvkeys, recvvals, recvcount, comm);

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

    if (MPI_UINT32_T == keytype)
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
		MPI_UINT32_T, valtype, 0, tmpv0, tmpv1, recvcount, comm);

	    /* extract higher half */
	    for(ptrdiff_t i = 0; i < recvcount; ++i)
		tmpk[i] = *(1 + (uint16_t *)(i + (uint32_t *)tmpv0));

	    dsort_uint16_t (
		1, tmpk, tmpv0, tmpv1, recvcount,
		MPI_UINT32_T, valtype, 0, recvkeys, recvvals, recvcount, comm);

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

    if (MPI_UINT64_T == keytype)
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
		MPI_UINT64_T, valtype, 0, tmpv0, tmpv1, recvcount, comm);

	    extract(1, recvcount, tmpv0, tmpk);

	    dsort_uint16_t(
		1, tmpk, tmpv0, tmpv1, recvcount,
		MPI_UINT64_T, valtype, 0, recvkeys, recvvals, recvcount, comm);

	    extract(2, recvcount, recvkeys, tmpk);

	    dsort_uint16_t(
		1, tmpk, recvkeys, recvvals, recvcount,
		MPI_UINT64_T, valtype, 0, tmpv0, tmpv1, recvcount, comm);

	    extract(3, recvcount, tmpv0, tmpk);

	    dsort_uint16_t(
		1, tmpk, tmpv0, tmpv1, recvcount,
		MPI_UINT64_T, valtype, 0, recvkeys, recvvals, recvcount, comm);

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
	    int32_t * inout )
	{
	    for (ptrdiff_t i = 0; i < n; ++i)
		inout[i] ^= (inout[i] >> 31) & 0x7fffffff;
	}

	flip(sendcount, (void *)sendkeys);

	MPI_CHECK(
	    MPI_Sort_bykey(sendkeys, sendvals, sendcount,
			   MPI_INT32_T, valtype, recvkeys, recvvals, recvcount, comm));

	if (recvkeys != sendkeys)
	    flip(sendcount, (void *)sendkeys);

	flip(recvcount, recvkeys);

	return MPI_SUCCESS;
    }

    if (MPI_DOUBLE == keytype)
    {
	__extension__ void flip (
	    const ptrdiff_t n,
	    int64_t * inout )
	{
	    for (ptrdiff_t i = 0; i < n; ++i)
		inout[i] ^= (inout[i] >> 63) & 0x7fffffffffffffff;
	}

	flip(sendcount, (void *)sendkeys);

	MPI_CHECK(
	    MPI_Sort_bykey(sendkeys, sendvals, sendcount,
			   MPI_INT64_T, valtype, recvkeys, recvvals, recvcount, comm));

	if (recvkeys != sendkeys)
	    flip(sendcount, (void *)sendkeys);

	flip(recvcount, recvkeys);

	return MPI_SUCCESS;
    }

    if (MPI_INT8_T == keytype
	|| MPI_INT16_T == keytype
	|| MPI_INT32_T == keytype
	|| MPI_INT64_T == keytype)
    {
	MPI_CHECK(rmanip_to_unsigned(keytype, sendcount, (void *)sendkeys));

	const MPI_Datatype newtype =
	    MPI_UINT8_T * (MPI_INT8_T == keytype)
	    | MPI_UINT16_T * (MPI_INT16_T == keytype)
	    | MPI_UINT32_T * (MPI_INT32_T == keytype)
	    | MPI_UINT64_T * (MPI_INT64_T == keytype);

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
