#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include <assert.h>
#include <mpi.h>

#include "macros.h"
#include "rmanip.h"
#include "xtract.h"

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

static int dispatch_unsigned (
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
    int MPI_SORT_RADIX = 1;
    READENV(MPI_SORT_RADIX, atoi);

    int MPI_SORT_STABLE = 0;
    READENV(MPI_SORT_STABLE, atoi);

    int MPI_SORT_DRANGE = 1;
    READENV(MPI_SORT_DRANGE, atoi);

    if (MPI_UINT8_T == keytype)
	return dsort_uint8_t (
	    MPI_SORT_STABLE, sendkeys, 0, sendvals, sendcount,
	    DONTCARE_TYPE, valtype, recvkeys, 0, recvvals, recvcount, comm);

    rmanip_t range_new ;
    const MPI_Datatype keytype_old = keytype;

    if (MPI_SORT_DRANGE)
    {
	range_new = rmanip_contract(comm, keytype, sendcount, (void *)sendkeys);
	keytype = range_new.type_new;
    }

    int err = MPI_ERR_TYPE;

    /* we enforce radix sort for 64 bit digits */
    MPI_SORT_RADIX |= (MPI_UINT64_T == keytype);

    if (!MPI_SORT_RADIX)
    {
	if (MPI_UINT16_T == keytype)
	    err = dsort_uint16_t(
		MPI_SORT_STABLE, sendkeys, 0, sendvals, sendcount,
		DONTCARE_TYPE, valtype, recvkeys, 0, recvvals, recvcount, comm);
	else if (MPI_UINT32_T == keytype)
	    err = dsort_uint32_t(
		0, sendkeys, 0, sendvals, sendcount,
		DONTCARE_TYPE, valtype, recvkeys, 0, recvvals, recvcount, comm);
    }
    else /* if RADIX */
    {
	__extension__ ptrdiff_t getsize(MPI_Datatype t)
	{
	    int s;
	    MPI_CHECK(MPI_Type_size(t, &s));
	    return s;
	}

	ptrdiff_t vtsz = 0;

	if (recvvals)
	    vtsz = getsize(valtype);

	const ptrdiff_t ktsz = getsize(keytype);

	/* TODO: intermediate results should be partitioned homogenously,
	   ignoring sendcount and recvcount */
	void * tmpk = malloc
	    ((MPI_INT16_T == keytype ? sizeof(uint8_t) : sizeof(uint16_t)) * MAX(recvcount, sendcount));

	void * tmpv0 = malloc(ktsz * recvcount);
	void * tmpv1 = recvvals ? malloc(vtsz * recvcount) : NULL;

	if (MPI_UINT16_T == keytype)
	{
	    xtract_8_16(0, sendcount, sendkeys, tmpk);

	    dsort_uint8_t(
		MPI_SORT_STABLE, tmpk, sendkeys, sendvals, sendcount,
		MPI_UINT16_T, valtype, 0, tmpv0, tmpv1, recvcount, comm);

	    xtract_8_16(1, recvcount, tmpv0, tmpk);

	    dsort_uint8_t(
		1, tmpk, tmpv0, tmpv1, recvcount,
		MPI_UINT16_T, valtype, 0, recvkeys, recvvals, recvcount, comm);

	    err = MPI_SUCCESS;
	}
	else if (MPI_UINT32_T == keytype)
	{
	    xtract_16_32(0, sendcount, sendkeys, tmpk);

	    dsort_uint16_t(
		MPI_SORT_STABLE, tmpk, sendkeys, sendvals, sendcount,
		MPI_UINT32_T, valtype, 0, tmpv0, tmpv1, recvcount, comm);

	    xtract_16_32(1, recvcount, tmpv0, tmpk);

	    dsort_uint16_t(
		1, tmpk, tmpv0, tmpv1, recvcount,
		MPI_UINT32_T, valtype, 0, recvkeys, recvvals, recvcount, comm);

	    err = MPI_SUCCESS;
	}
	else if (MPI_UINT64_T == keytype)
	{
	    xtract_16_64(0, sendcount, sendkeys, tmpk);

	    dsort_uint16_t(
		MPI_SORT_STABLE, tmpk, sendkeys, sendvals, sendcount,
		MPI_UINT64_T, valtype, 0, tmpv0, tmpv1, recvcount, comm);

	    xtract_16_64(1, recvcount, tmpv0, tmpk);

	    dsort_uint16_t(
		1, tmpk, tmpv0, tmpv1, recvcount,
		MPI_UINT64_T, valtype, 0, recvkeys, recvvals, recvcount, comm);

	    xtract_16_64(2, recvcount, recvkeys, tmpk);

	    dsort_uint16_t(
		1, tmpk, recvkeys, recvvals, recvcount,
		MPI_UINT64_T, valtype, 0, tmpv0, tmpv1, recvcount, comm);

	    xtract_16_64(3, recvcount, tmpv0, tmpk);

	    dsort_uint16_t(
		1, tmpk, tmpv0, tmpv1, recvcount,
		MPI_UINT64_T, valtype, 0, recvkeys, recvvals, recvcount, comm);
	}

	if (tmpv1)
	    free(tmpv1);

	free(tmpv0);
	free(tmpk);
    }

    if (MPI_SORT_DRANGE)
    {
	if (sendkeys != recvkeys)
	    rmanip_expand(range_new, keytype_old, sendcount, (void *)sendkeys);

	rmanip_expand(range_new, keytype_old, recvcount, recvkeys);
    }

    return err;
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
    DIE_UNLESS(sendcount >= 0 && recvcount >= 0);

    /* squeeze type range into fewer ones */
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

    if (MPI_UINT8_T == keytype
	|| MPI_UINT16_T == keytype
	|| MPI_UINT32_T == keytype
	|| MPI_UINT64_T == keytype)
	return dispatch_unsigned(sendkeys, sendvals, sendcount,
				 keytype, valtype,
				 recvkeys, recvvals, recvcount, comm);

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

	MPI_CHECK(dispatch_unsigned(sendkeys, sendvals, sendcount, newtype, valtype,
				    recvkeys, recvvals, recvcount, comm));

	if (recvkeys != sendkeys)
	    MPI_CHECK(rmanip_from_unsigned(keytype, sendcount, (void *)sendkeys));

	MPI_CHECK(rmanip_from_unsigned(keytype, recvcount, recvkeys));

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
