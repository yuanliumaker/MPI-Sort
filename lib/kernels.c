#include <stdint.h>

#include "kernels.h"

void mux4 (
    const int * const restrict in0,
    const int * const restrict in1,
    const int * const restrict in2,
    const int * const restrict in3,
    const ptrdiff_t n,
    int * const restrict out)
{
    for (ptrdiff_t i = 0; i < n; ++i)
    {
	out[0 + 4 * i] = in0[i];
	out[1 + 4 * i] = in1[i];
	out[2 + 4 * i] = in2[i];
	out[3 + 4 * i] = in3[i];
    }
}

#define NZCOUNT_KERNEL(T)			\
    T nzcount_ ## T (				\
	const int * const restrict in,		\
	const ptrdiff_t count)			\
    {						\
	T s = 0;				\
						\
	for (ptrdiff_t i = 0; i < count; ++i)	\
	    s += !!in[i];			\
						\
	return s;				\
    }

#include <stdint.h>

NZCOUNT_KERNEL(uint8_t)
NZCOUNT_KERNEL(uint16_t)
NZCOUNT_KERNEL(uint32_t)
NZCOUNT_KERNEL(uint64_t)

ptrdiff_t nzcount (
    const int * const restrict in,
    const ptrdiff_t count )
{
    if (count < 256)
	return nzcount_uint8_t(in, count);

    if (count < 65536)
	return nzcount_uint16_t(in, count);

    if (count < 4294967296)
	return nzcount_uint32_t(in, count);

    return nzcount_uint64_t(in, count);
}

int maxval (
    const int * const restrict in,
    const ptrdiff_t count )
{
    int lmax = in[0];

    for (ptrdiff_t i = 1; i < count; ++i)
	lmax = MAX(lmax, in[i]);

    return lmax;
}

ptrdiff_t exscan (
    const ptrdiff_t count,
    const ptrdiff_t * const restrict in,
    ptrdiff_t * const restrict out )
{
    ptrdiff_t s = 0;

    for (ptrdiff_t i = 0; i < count; ++i)
    {
	out[i] = s;
	s += in[i];
    }

    return s;
}

ptrdiff_t exscan_int32 (
    const ptrdiff_t count,
    const int * const restrict in,
    ptrdiff_t * const restrict out )
{
    ptrdiff_t s = 0;

    for (ptrdiff_t i = 0; i < count; ++i)
    {
	out[i] = s;
	s += in[i];
    }

    return s;
}

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "util.h"

void counting_sort (
    const int minval,
    const int supval,
    const ptrdiff_t samplecount,
    const int * const restrict samples,
    ptrdiff_t * const restrict histo,
    ptrdiff_t * const restrict start,
    ptrdiff_t * const restrict order )
{
    const int d = supval - minval;

    for (ptrdiff_t i = 0; i < samplecount; ++i)
    {
	const int s = samples[i] - minval;
	if (s < 0 || s >= d)
	    printf("ooops %d max is %d\n", s, d);
    }

    for (ptrdiff_t i = 0; i < samplecount; ++i)
	++histo[samples[i] - minval];

    exscan(d, histo, start);

    for (ptrdiff_t i = 0; i < samplecount; ++i)
	order[start[samples[i] - minval]++] = i;

    exscan(d, histo, start);
}
