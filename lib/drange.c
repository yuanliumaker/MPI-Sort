

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <limits.h>
#include <mpi.h>

#include "macros.h"
#include "drange.h"



// 将uint8_t 类型的值从有符号转换为无符号（+127+1）并存回原数组
static int _to_uint8_t (
       const ptrdiff_t n,
       uint8_t * const inout )
{
	for(ptrdiff_t i = 0; i < n; ++i)
	{
	 uint8_t val = inout[i];

	 val += INT8_MAX;//127
	 val += 1;

	 inout[i] = val;
	}

	return MPI_SUCCESS;
}
// 同理
static int _to_uint16_t (
       const ptrdiff_t n,
       uint16_t * const inout )
{
	for(ptrdiff_t i = 0; i < n; ++i)
	{
	 uint16_t val = inout[i];

	 val += INT16_MAX;
	 val += 1;

	 inout[i] = val;
	}

	return MPI_SUCCESS;
}

static int _to_uint32_t (
       const ptrdiff_t n,
       uint32_t * const inout )
{
	for(ptrdiff_t i = 0; i < n; ++i)
	{
	 uint32_t val = inout[i];

	 val += INT32_MAX;
	 val += 1;

	 inout[i] = val;
	}

	return MPI_SUCCESS;
}

static int _to_uint64_t (
       const ptrdiff_t n,
       uint64_t * const inout )
{
	for(ptrdiff_t i = 0; i < n; ++i)
	{
	 uint64_t val = inout[i];

	 val += INT64_MAX;
	 val += 1;

	 inout[i] = val;
	}

	return MPI_SUCCESS;
}
// 将相应类型的数转成无符号类型
int drange_to_unsigned (
	 MPI_Datatype uint_t,
	 const ptrdiff_t count,
	 void * const inout)
{
	if (MPI_INT8_T == uint_t)
		    return _to_uint8_t(count, inout);
if (MPI_INT16_T == uint_t)
		    return _to_uint16_t(count, inout);
if (MPI_INT32_T == uint_t)
		    return _to_uint32_t(count, inout);
if (MPI_INT64_T == uint_t)
		    return _to_uint64_t(count, inout);


	return MPI_ERR_TYPE;
}




static int _from_uint8_t (
       const ptrdiff_t n,
       uint8_t * const inout )
{
	for(ptrdiff_t i = 0; i < n; ++i)
	{
	 uint8_t val = inout[i];

	 val -= 1;
	 val -= INT8_MAX;

	 inout[i] = val;
	}

	return MPI_SUCCESS;
}

static int _from_uint16_t (
       const ptrdiff_t n,
       uint16_t * const inout )
{
	for(ptrdiff_t i = 0; i < n; ++i)
	{
	 uint16_t val = inout[i];

	 val -= 1;
	 val -= INT16_MAX;

	 inout[i] = val;
	}

	return MPI_SUCCESS;
}

static int _from_uint32_t (
       const ptrdiff_t n,
       uint32_t * const inout )
{
	for(ptrdiff_t i = 0; i < n; ++i)
	{
	 uint32_t val = inout[i];

	 val -= 1;
	 val -= INT32_MAX;

	 inout[i] = val;
	}

	return MPI_SUCCESS;
}

static int _from_uint64_t (
       const ptrdiff_t n,
       uint64_t * const inout )
{
	for(ptrdiff_t i = 0; i < n; ++i)
	{
	 uint64_t val = inout[i];

	 val -= 1;
	 val -= INT64_MAX;

	 inout[i] = val;
	}

	return MPI_SUCCESS;
}


int drange_from_unsigned (
	 MPI_Datatype uint_t,
	 const ptrdiff_t count,
	 void * const inout)
{
	if (MPI_INT8_T == uint_t)
		    return _from_uint8_t(count, inout);
if (MPI_INT16_T == uint_t)
		    return _from_uint16_t(count, inout);
if (MPI_INT32_T == uint_t)
		    return _from_uint32_t(count, inout);
if (MPI_INT64_T == uint_t)
		    return _from_uint64_t(count, inout);


	return MPI_ERR_TYPE;
}


// 作用：找到一个uint16_t的数组的最大值和最小值，并在可能的情况下将该数组的值映射到较小的数据类型uint8_t,以减少数据的大小


static drange_t _contract_uint16_t (
       MPI_Comm comm,
       const ptrdiff_t count,
       void * const inout)
{
	uint16_t minval = 0, maxval = (uint16_t)ULONG_MAX;

if (count)
	{
	const uint16_t * const restrict in = inout;

	minval = in[0];
	maxval = in[0];

	for (ptrdiff_t i = 1; i < count; ++i)
	{
	const uint16_t val = in[i];

	minval = MIN(minval, val);
	maxval = MAX(maxval, val);
	}
	}

	MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &minval, 1, MPI_UINT16_T, MPI_MIN, comm));
	MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &maxval, 1, MPI_UINT16_T, MPI_MAX, comm));

	drange_t retval = { .minval_old = minval, .maxval_old = maxval, .type_new = MPI_UINT16_T, .err = MPI_SUCCESS };

	const uint64_t rangec = retval.maxval_old - retval.minval_old;
	// 如数组的最大值和最小值的差小于等于uint8_max 则进行映射，映射过程是将每个值减去最小值
	
	if (UINT8_MAX >= rangec)
	{
		const uint16_t * in = inout;
		uint8_t * out = inout;

		for (ptrdiff_t i = 0; i < count; ++i)
			out[i] = (uint8_t)(in[i] - minval);

		retval.type_new = MPI_UINT8_T;

		return retval;
	}
	return retval;
}
static drange_t _contract_uint32_t (
       MPI_Comm comm,
       const ptrdiff_t count,
       void * const inout)
{
	uint32_t minval = 0, maxval = (uint32_t)ULONG_MAX;

if (count)
	{
	const uint32_t * const restrict in = inout;

	minval = in[0];
	maxval = in[0];

	for (ptrdiff_t i = 1; i < count; ++i)
	{
	const uint32_t val = in[i];

	minval = MIN(minval, val);
	maxval = MAX(maxval, val);
	}
	}

	MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &minval, 1, MPI_UINT32_T, MPI_MIN, comm));
	MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &maxval, 1, MPI_UINT32_T, MPI_MAX, comm));

	drange_t retval = { .minval_old = minval, .maxval_old = maxval, .type_new = MPI_UINT32_T, .err = MPI_SUCCESS };

	const uint64_t rangec = retval.maxval_old - retval.minval_old;

	
	if (UINT8_MAX >= rangec)
	{
		const uint32_t * in = inout;
		uint8_t * out = inout;

		for (ptrdiff_t i = 0; i < count; ++i)
			out[i] = (uint8_t)(in[i] - minval);

		retval.type_new = MPI_UINT8_T;

		return retval;
	}

	if (UINT16_MAX >= rangec)
	{
		const uint32_t * in = inout;
		uint16_t * out = inout;

		for (ptrdiff_t i = 0; i < count; ++i)
			out[i] = (uint16_t)(in[i] - minval);

		retval.type_new = MPI_UINT16_T;

		return retval;
	}

	

	


	return retval;
}
static drange_t _contract_uint64_t (
       MPI_Comm comm,
       const ptrdiff_t count,
       void * const inout)
{
	uint64_t minval = 0, maxval = (uint64_t)ULONG_MAX;

if (count)
	{
	const uint64_t * const restrict in = inout;

	minval = in[0];
	maxval = in[0];

	for (ptrdiff_t i = 1; i < count; ++i)
	{
	const uint64_t val = in[i];

	minval = MIN(minval, val);
	maxval = MAX(maxval, val);
	}
	}

	MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &minval, 1, MPI_UINT64_T, MPI_MIN, comm));
	MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &maxval, 1, MPI_UINT64_T, MPI_MAX, comm));

	drange_t retval = { .minval_old = minval, .maxval_old = maxval, .type_new = MPI_UINT64_T, .err = MPI_SUCCESS };

	const uint64_t rangec = retval.maxval_old - retval.minval_old;

	
	if (UINT8_MAX >= rangec)
	{
		const uint64_t * in = inout;
		uint8_t * out = inout;

		for (ptrdiff_t i = 0; i < count; ++i)
			out[i] = (uint8_t)(in[i] - minval);

		retval.type_new = MPI_UINT8_T;

		return retval;
	}

	if (UINT16_MAX >= rangec)
	{
		const uint64_t * in = inout;
		uint16_t * out = inout;

		for (ptrdiff_t i = 0; i < count; ++i)
			out[i] = (uint16_t)(in[i] - minval);

		retval.type_new = MPI_UINT16_T;

		return retval;
	}

	if (UINT32_MAX >= rangec)
	{
		const uint64_t * in = inout;
		uint32_t * out = inout;

		for (ptrdiff_t i = 0; i < count; ++i)
			out[i] = (uint32_t)(in[i] - minval);

		retval.type_new = MPI_UINT32_T;

		return retval;
	}

	


	return retval;
}


drange_t drange_contract (
	 MPI_Comm comm,
	 MPI_Datatype uint_t,
	 const ptrdiff_t count,
	 void * const inout)
{
	if (MPI_UINT16_T == uint_t)
		    return _contract_uint16_t(comm, count, inout);
if (MPI_UINT32_T == uint_t)
		    return _contract_uint32_t(comm, count, inout);
if (MPI_UINT64_T == uint_t)
		    return _contract_uint64_t(comm, count, inout);


	/* bypass range contraction for uint8_t */
	if (MPI_UINT8_T == uint_t)
	   return (drange_t){ .minval_old = 0, .maxval_old = 255, .type_new = MPI_UINT8_T, .err = MPI_SUCCESS };

	return (drange_t) { .err = MPI_ERR_TYPE };
}



static void _expand_uint16_t (
	const drange_t r,
	const ptrdiff_t count,
	void * const inout)
{
	const uint16_t minval = r.minval_old;
	const uint64_t rangec = r.maxval_old - r.minval_old;

	
	if (UINT8_MAX >= rangec)
	{
		const uint8_t * in = inout;
		uint16_t * out = inout;

		for (ptrdiff_t i = count - 1; i >= 0; --i)
		{
			const uint16_t v = in[i];

			out[i] = v + minval;
		}

		return;
	}

	

	

	

}
static void _expand_uint32_t (
	const drange_t r,
	const ptrdiff_t count,
	void * const inout)
{
	const uint32_t minval = r.minval_old;
	const uint64_t rangec = r.maxval_old - r.minval_old;

	
	if (UINT8_MAX >= rangec)
	{
		const uint8_t * in = inout;
		uint32_t * out = inout;

		for (ptrdiff_t i = count - 1; i >= 0; --i)
		{
			const uint32_t v = in[i];

			out[i] = v + minval;
		}

		return;
	}

	if (UINT16_MAX >= rangec)
	{
		const uint16_t * in = inout;
		uint32_t * out = inout;

		for (ptrdiff_t i = count - 1; i >= 0; --i)
		{
			const uint32_t v = in[i];

			out[i] = v + minval;
		}

		return;
	}

	

	

}
static void _expand_uint64_t (
	const drange_t r,
	const ptrdiff_t count,
	void * const inout)
{
	const uint64_t minval = r.minval_old;
	const uint64_t rangec = r.maxval_old - r.minval_old;

	
	if (UINT8_MAX >= rangec)
	{
		const uint8_t * in = inout;
		uint64_t * out = inout;

		for (ptrdiff_t i = count - 1; i >= 0; --i)
		{
			const uint64_t v = in[i];

			out[i] = v + minval;
		}

		return;
	}

	if (UINT16_MAX >= rangec)
	{
		const uint16_t * in = inout;
		uint64_t * out = inout;

		for (ptrdiff_t i = count - 1; i >= 0; --i)
		{
			const uint64_t v = in[i];

			out[i] = v + minval;
		}

		return;
	}

	if (UINT32_MAX >= rangec)
	{
		const uint32_t * in = inout;
		uint64_t * out = inout;

		for (ptrdiff_t i = count - 1; i >= 0; --i)
		{
			const uint64_t v = in[i];

			out[i] = v + minval;
		}

		return;
	}

	

}


void drange_expand (
     	 const drange_t r,
	 MPI_Datatype uint_t,
	 const ptrdiff_t count,
	 void * const inout)
{
	if (MPI_UINT16_T == uint_t)
		    return _expand_uint16_t(r, count, inout);
if (MPI_UINT32_T == uint_t)
		    return _expand_uint32_t(r, count, inout);
if (MPI_UINT64_T == uint_t)
		    return _expand_uint64_t(r, count, inout);

}
