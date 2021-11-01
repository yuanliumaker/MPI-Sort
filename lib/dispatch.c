#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <assert.h>

#include <mpi.h>

#include "macros.h"
#include "drange.h"
#include "xtract.h"

#define _CAT(a, b) a ## b
#define CAT(a, b) _CAT(a, b)

#define RADIX_SIGNATURE(TYPE)					\
	int CAT(radix_, TYPE) (						\
		const int stable,						\
		const TYPE * sendkeys,					\
		const void * sendvals0,					\
		const void * sendvals1,					\
		const int sendcount,					\
		MPI_Datatype valtype0,					\
		MPI_Datatype valtype1,					\
		TYPE * recvkeys,						\
		void * recvvals0,						\
		void * recvvals1,						\
		const int recvcount,					\
		MPI_Comm comm)

#define SPARSE_SIGNATURE(TYPE)					\
	int CAT(sparse_, TYPE) (					\
		const int stable,						\
		TYPE * sendkeys,						\
		void * sendvals,						\
		const int sendcount,					\
		MPI_Datatype valtype,					\
		TYPE * recvkeys,						\
		void * recvvals,						\
		const int recvcount,					\
		MPI_Comm comm)

RADIX_SIGNATURE(uint8_t);
RADIX_SIGNATURE(uint16_t);

SPARSE_SIGNATURE(uint16_t);
SPARSE_SIGNATURE(uint32_t);
SPARSE_SIGNATURE(uint64_t);

/* return copy of array */
static void * mkcpy (
	MPI_Datatype t,
	const ptrdiff_t c,
	const void * p )
{
	int tsz;
	MPI_CHECK(MPI_Type_size(t, &tsz));

	/* byte count */
	const ptrdiff_t bc = tsz * c;

	void * retval = NULL;
	DIE_UNLESS(retval = malloc(bc));

	memcpy(retval, p, bc);

	return retval;
}

static int dispatch_unsigned (
	void * sendkeys,
	void * sendvals,
	const int sendcount,
	MPI_Datatype keytype,
	MPI_Datatype valtype,
	void * recvkeys,
	void * recvvals,
	const int recvcount,
	MPI_Comm comm)
{
	const int kinplace = MPI_IN_PLACE == sendkeys;
	const int vinplace = MPI_IN_PLACE == sendvals;

	if (kinplace)
		sendkeys = mkcpy(keytype, sendcount, recvkeys);

	if (vinplace)
		sendvals = mkcpy(valtype, sendcount, recvvals);

	int MPI_SORT_RADIX = 1;
	READENV(MPI_SORT_RADIX, atoi);

	int MPI_SORT_STABLE = 0;
	READENV(MPI_SORT_STABLE, atoi);

	int MPI_SORT_DRANGE = 1;
	READENV(MPI_SORT_DRANGE, atoi);

	drange_t range_new ;
	const MPI_Datatype keytype_old = keytype;
	/* range contraction at runtime */
	if (MPI_SORT_DRANGE)
	{
		range_new = drange_contract(comm, keytype, sendcount, (void *)sendkeys);

		keytype = range_new.type_new;
	}

	int err = MPI_ERR_TYPE;

	/* we enforce radix sort for 8 bits keys */
	MPI_SORT_RADIX |= (MPI_UINT8_T == keytype);

	/* we discourage radix sort for 16 bits integers keys */
	MPI_SORT_RADIX = MAX(0, MPI_SORT_RADIX - (MPI_UINT16_T == keytype));

	/* no radix sort for 32 and 64 bits keys */
	MPI_SORT_RADIX *= (MPI_UINT32_T != keytype);
	MPI_SORT_RADIX *= (MPI_UINT64_T != keytype);

	if (!MPI_SORT_RADIX)
	{
		if (MPI_UINT16_T == keytype)
			err = sparse_uint16_t (
				MPI_SORT_STABLE, sendkeys, sendvals, sendcount,
				valtype, recvkeys, recvvals, recvcount, comm);
		else if (MPI_UINT32_T == keytype)
			err = sparse_uint32_t (
				MPI_SORT_STABLE, sendkeys, sendvals, sendcount,
				valtype, recvkeys, recvvals, recvcount, comm);
		else if (MPI_UINT64_T == keytype)
			err = sparse_uint64_t (
				MPI_SORT_STABLE, sendkeys, sendvals, sendcount,
				valtype, recvkeys, recvvals, recvcount, comm);
		else
			__builtin_trap();
	}
	else /* if RADIX */
	{
		if (MPI_UINT8_T == keytype)
			err = radix_uint8_t (
				MPI_SORT_STABLE, sendkeys, 0, sendvals, sendcount,
				MPI_DATATYPE_NULL, valtype, recvkeys, 0, recvvals, recvcount, comm);
		else if (MPI_UINT16_T == keytype)
		{
			err = radix_uint16_t(
				MPI_SORT_STABLE, sendkeys, 0, sendvals, sendcount,
				MPI_DATATYPE_NULL, valtype, recvkeys, 0, recvvals, recvcount, comm);
		}
		else
			__builtin_trap();
	}

	if (MPI_SORT_DRANGE)
	{
		if (sendkeys != recvkeys)
			drange_expand(range_new, keytype_old, sendcount, (void *)sendkeys);

		drange_expand(range_new, keytype_old, recvcount, recvkeys);
	}

	if (vinplace)
		free((void *)sendvals);

	if (kinplace)
		free((void *)sendkeys);

	return err;
}

__attribute__ ((visibility("default")))
int MPI_Sort_bykey (
	void * sendkeys,
	void * sendvals,
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

		if (MPI_UNSIGNED_CHAR == keytype)
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
		MPI_CHECK(drange_to_unsigned(keytype, sendcount, (void *)sendkeys));

		const MPI_Datatype newtype = (MPI_Datatype)
			( (ptrdiff_t)MPI_UINT8_T * (MPI_INT8_T == keytype)
			  | (ptrdiff_t)MPI_UINT16_T * (MPI_INT16_T == keytype)
			  | (ptrdiff_t)MPI_UINT32_T * (MPI_INT32_T == keytype)
			  | (ptrdiff_t)MPI_UINT64_T * (MPI_INT64_T == keytype));

		MPI_CHECK(dispatch_unsigned(sendkeys, sendvals, sendcount, newtype, valtype,
									recvkeys, recvvals, recvcount, comm));

		if (recvkeys != sendkeys)
			MPI_CHECK(drange_from_unsigned(keytype, sendcount, (void *)sendkeys));

		MPI_CHECK(drange_from_unsigned(keytype, recvcount, recvkeys));

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

__attribute__((visibility ("default")))
int MPI_Sort (
	void * sendbuf,
	const int sendcount,
	MPI_Datatype datatype,
	void * recvbuf,
	const int recvcount,
	MPI_Comm comm)
{
	return
		MPI_Sort_bykey(sendbuf, 0, sendcount, datatype, MPI_DATATYPE_NULL, recvbuf, 0, recvcount, comm);
}
