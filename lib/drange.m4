divert(-1)
define(`forloop',
dnl iter:
dnl $1: stmt
dnl $2: var
dnl $3: value
define(`iter',dnl
`pushdef(`$2', `$3')dnl
$1
popdef(`$2')')
dnl foreach:
dnl stmt
dnl itervar
dnl iterval
dnl define(`foreach',
#debugmode(ta)
define(`foreach',dnl
`iter(`$1', $2, `$3')dnl
ifelse(eval($# >= 4), 1, `foreach(`$1', `$2', shift(shift(shift($@))))')')

       `pushdef(`$1', `$2')_forloop(`$1', `$2', `$3', `$4')popdef(`$1')')

define(`unsigned_types',
``MPI_UINT16_T, uint16_t',
`MPI_UINT32_T, uint32_t',
`MPI_UINT64_T, uint64_t'')

define(`signed_to_unsigned_types',
``MPI_INT8_T, int8_t, uint8_t, INT8_MAX',
`MPI_INT16_T, int16_t, uint16_t, INT16_MAX',
`MPI_INT32_T, int32_t, uint32_t, INT32_MAX',
`MPI_INT64_T, int64_t, uint64_t, INT64_MAX'')

define(first, $1)
define(second, $2)
define(third, $3)

define(type, uint`'$1`'_t)

divert(0)

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <mpi.h>

#include "macros.h"
#include "drange.h"

define(to_unsigned, `
static int _to_`'$3 (
       const ptrdiff_t n,
       $3 * const inout )
{
	for(ptrdiff_t i = 0; i < n; ++i)
	{
	 $3 val = inout[i];

	 val += $4;
	 val += 1;

	 inout[i] = val;
	}

	return MPI_SUCCESS;
}')

foreach(`to_unsigned(tuple)', tuple, signed_to_unsigned_types)

int drange_to_unsigned (
	 MPI_Datatype type,
	 const ptrdiff_t count,
	 void * const inout)
{
	foreach(`if (first(tuple) == type)
		    return _to_`'third(tuple)(count, inout);', tuple, signed_to_unsigned_types)

	return MPI_ERR_TYPE;
}

define(from_unsigned, `
static int _from_`'$3 (
       const ptrdiff_t n,
       $3 * const inout )
{
	for(ptrdiff_t i = 0; i < n; ++i)
	{
	 $3 val = inout[i];

	 val -= 1;
	 val -= $4;

	 inout[i] = val;
	}

	return MPI_SUCCESS;
}')

foreach(`from_unsigned(tuple)', tuple, signed_to_unsigned_types)

int drange_from_unsigned (
	 MPI_Datatype type,
	 const ptrdiff_t count,
	 void * const inout)
{
	foreach(`if (first(tuple) == type)
		    return _from_`'third(tuple)(count, inout);', tuple, signed_to_unsigned_types)

	return MPI_ERR_TYPE;
}

define(rcontract,
`static drange_t _contract_`'type($1) (
       MPI_Comm comm,
       const ptrdiff_t count,
       void * const inout)
{
	type($1) minval, maxval;
	{
	const type($1) * const restrict in = inout;

	minval = in[0];
	maxval = in[0];

	for (ptrdiff_t i = 1; i < count; ++i)
	{
	const type($1) val = in[i];

	minval = MIN(minval, val);
	maxval = MAX(maxval, val);
	}
	}

	MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &minval, 1, MPI_UINT`'$1`'_T, MPI_MIN, comm));
	MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &maxval, 1, MPI_UINT`'$1`'_T, MPI_MAX, comm));

	drange_t retval = { .minval_old = minval, .maxval_old = maxval, .type_new = MPI_UINT`'$1`'_T, .err = MPI_SUCCESS };

	const uint64_t rangec = retval.maxval_old - retval.minval_old;

	foreach(`
	ifelse(eval(bitdepth < $1),1,
	if (UINT`'bitdepth`'_MAX >= rangec)
	{
		const type($1) * in = inout;
		type(bitdepth) * out = inout;

		for (ptrdiff_t i = 0; i < count; ++i)
			out[i] = (type(bitdepth))(in[i] - minval);

		retval.type_new = MPI_UINT`'bitdepth`'_T;

		return retval;
	})', bitdepth, 8, 16, 32, 64)

	return retval;
}')

foreach(`rcontract(tuple)', tuple, 16, 32, 64)

drange_t drange_contract (
	 MPI_Comm comm,
	 MPI_Datatype type,
	 const ptrdiff_t count,
	 void * const inout)
{
	foreach(`if (first(tuple) == type)
		    return _contract_`'second(tuple)(comm, count, inout);', tuple, unsigned_types)

	/* bypass range contraction for uint8_t */
	if (MPI_UINT8_T == type)
	   return (drange_t){ .minval_old = 0, .maxval_old = 255, .type_new = MPI_UINT8_T, .err = MPI_SUCCESS };

	return (drange_t) { .err = MPI_ERR_TYPE };
}

define(rexpand,
`static void _expand_`'type($1) (
	const drange_t r,
	const ptrdiff_t count,
	void * const inout)
{
	const type($1) minval = r.minval_old;
	const uint64_t rangec = r.maxval_old - r.minval_old;

	foreach(`
	ifelse(eval(bitdepth < $1),1,
	if (UINT`'bitdepth`'_MAX >= rangec)
	{
		const type(bitdepth) * in = inout;
		type($1) * out = inout;

		for (ptrdiff_t i = count - 1; i >= 0; --i)
		{
			const type($1) v = in[i];

			out[i] = v + minval;
		}

		return;
	})', bitdepth, 8, 16, 32, 64)
}')

foreach(`rexpand(tuple)', tuple, 16, 32, 64)

void drange_expand (
     	 const drange_t r,
	 MPI_Datatype type,
	 const ptrdiff_t count,
	 void * const inout)
{
	foreach(`if (first(tuple) == type)
		    return _expand_`'second(tuple)(r, count, inout);', tuple, unsigned_types)
}
