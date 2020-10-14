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
``MPI_UNSIGNED_CHAR, uint8_t',
`MPI_UNSIGNED_SHORT, uint16_t',
`MPI_UNSIGNED, uint32_t',
`MPI_UNSIGNED_LONG, uint64_t'')

define(`signed_to_unsigned_types',
``MPI_CHAR, int8_t, uint8_t, INT8_MAX',
`MPI_SHORT, int16_t, uint16_t, INT16_MAX',
`MPI_INTEGER, int32_t, uint32_t, INT32_MAX',
`MPI_LONG, int64_t, uint64_t, INT64_MAX'')

define(first, $1)
define(second, $2)
define(third, $3)
divert(0)

#include <stddef.h>
#include <stdint.h>

#include <mpi.h>

#include "macros.h"

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

int rmanip_to_unsigned (
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

int rmanip_from_unsigned (
	 MPI_Datatype type,
	 const ptrdiff_t count,
	 void * const inout)
{
	foreach(`if (first(tuple) == type)
		    return _from_`'third(tuple)(count, inout);', tuple, signed_to_unsigned_types)

	return MPI_ERR_TYPE;
}


dnl  define(rcontract, `
dnl static rmanip_t _contract_`'$2 (
dnl        MPI_Comm comm,
dnl        const ptrdiff_t count,
dnl        void * const inout)
dnl {
dnl 	$2 minval, maxval;
dnl 	{
dnl 	const $2 * const restrict in = inout;
dnl 	minval = in[0];
dnl 	maxval = in[0];
dnl 	for (ptrdiff_t i = 1; i < count; ++i)
dnl 	{
dnl 	const $2 val = in[i];
dnl 	minval = MIN(minval, val);
dnl 	maxval = MAX(maxval, val);
dnl 	}
dnl 	}
dnl 	MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &minval, 1, $1, MPI_MIN, comm));
dnl 	MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &maxval, 1, $1, MPI_MAX, comm));
dnl
dnl
dnl }')
dnl
dnl foreach(`rcontract(tuple)', tuple, supported_types)
dnl
dnl typedef struct
dnl {
dnl 	ptrdiff_t minval_old, maxval_old;
dnl 	MPI_Datatype type_new;
dnl 	MPI_Error err;
dnl } rmanip_t;
dnl
dnl
dnl rmanip_t rmanip_contract_inplace (
dnl 	 MPI_Comm comm,
dnl 	 MPI_Datatype type,
dnl 	 const ptrdiff_t count
dnl 	 void * const inout)
dnl {
dnl 	foreach(`if (first(tuple) == type)
dnl 		    return _cntrct_`'second(tuple)(comm, count, inout);', tuple, supported_types)
dnl
dnl 	return (rmanip_t) { .err = MPI_ERR_TYPE };
dnl }
