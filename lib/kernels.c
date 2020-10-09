#include <stdint.h>

#include "kernels.h"

ptrdiff_t fill (
    const KEY_T v,
    const ptrdiff_t count,
    KEY_T * const restrict out )
{
    for (ptrdiff_t i = 0; i < count; ++i)
	out[i] = v;

    return count;
}

#define NZCOUNT_KERNEL(T)			\
    static T nzcount_ ## T (			\
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
    const ptrdiff_t * const in,
    ptrdiff_t * const out )
{
    ptrdiff_t s = 0;

    for (ptrdiff_t i = 0; i < count; ++i)
    {
	const ptrdiff_t v = in[i];

	out[i] = s;

	s += v;
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
#include <assert.h>
#include "util.h"

ptrdiff_t counting_sort (
    const int minval,
    const int supval,
    const ptrdiff_t samplecount,
    const KEY_T * const restrict samples,
    ptrdiff_t * const restrict histo,
    ptrdiff_t * const restrict start,
    ptrdiff_t * const restrict order )
{
    const int d = supval - minval;

    for (ptrdiff_t i = 0; i < samplecount; ++i)
    {
	const int s = samples[i] - minval;
	assert(s >= 0 && s < d);
    }

    for (ptrdiff_t i = 0; i < samplecount; ++i)
	++histo[samples[i] - minval];

    exscan(d, histo, start);

    for (ptrdiff_t i = 0; i < samplecount; ++i)
	order[start[samples[i] - minval]++] = i;

    return exscan(d, histo, start);
}

#include <assert.h>

ptrdiff_t lowerbound (
    const ptrdiff_t * first,
    const ptrdiff_t * last,
    const ptrdiff_t val)
{
    const ptrdiff_t * const head = first;
    const ptrdiff_t * it;
    ptrdiff_t count, step;
    count = last - first;

    while (count > 0)
    {
	it = first;
	step = count / 2;

	it += step;
	if (*it < val)
	{
	    first = ++it;
	    count -= step + 1;
	}
	else
	    count = step;
    }

    assert(head <= first);

    return first - head;
}

#define GATHER_KERNEL(T)			\
    static void gather_ ## T (			\
	const ptrdiff_t count,			\
	const T * restrict const in,		\
	const ptrdiff_t * restrict const idx,	\
	T * restrict const out )		\
    {						\
	for (ptrdiff_t i = 0; i < count; ++i)	\
	    out[i] = in[idx[i]];		\
    }

#include <stdint.h>

GATHER_KERNEL(uint8_t)
GATHER_KERNEL(uint16_t)
GATHER_KERNEL(uint32_t)
GATHER_KERNEL(uint64_t)

void gather (
    const ptrdiff_t size,
    const ptrdiff_t count,
    const void * const in,
    const ptrdiff_t * const idx,
    void * const out )
{
    if (1 == size)
	return gather_uint8_t(count, in, idx, out);

    if (2 == size)
	return gather_uint16_t(count, in, idx, out);

    if (4 == size)
	return gather_uint32_t(count, in, idx, out);

    if (8 == size)
	return gather_uint64_t(count, in, idx, out);

    /* generic impl */
    for (ptrdiff_t i = 0; i < count; ++i)
	memcpy(size * i + (char *)out,
	       size * idx[i] + (char *)in,
	       size);
}

ptrdiff_t rle (
    const KEY_T * const in,
    const ptrdiff_t n,
    KEY_T * const vs,
    ptrdiff_t * ls)
{
    if (!n)
	return 0;

    ptrdiff_t c = 0;

    KEY_T v = in[0];

    if (vs && ls)
    {
	ptrdiff_t p = 0;

	for (ptrdiff_t i = 1; i < n; ++i)
	    if (v != in[i])
	    {
		vs[c] = v;
		ls[c++] = i - p;

		v = in[i];
		p = i;
	    }

	vs[c] = v;
	ls[c++] = n - p;
    }
    else
    {
	for (ptrdiff_t i = 1; i < n; ++i)
	{
	    if (v != in[i])
	    {
		++c;
		v = in[i];
	    }
	}
	++c;
    }

    return c;
}
