static range_t range_keys (
    const KEY_T * const restrict in,
    const ptrdiff_t count)
{
    KEY_T lmin = in[0], lmax = in[0];

    for (ptrdiff_t i = 1; i < count; ++i)
    {
	const int s = in[i];

	lmin = MIN(lmin, s);
	lmax = MAX(lmax, s);
    }

    return (range_t){ lmin, 1 + (ptrdiff_t)lmax };
}

static ptrdiff_t exscan (
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

#include <stdlib.h>
#include <assert.h>

static ptrdiff_t counting_sort (
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

static ptrdiff_t rle (
    const KEY_T * const restrict in,
    const ptrdiff_t n,
    KEY_T * const restrict vs,
    ptrdiff_t * const restrict ls)
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

static ptrdiff_t fill (
    const KEY_T v,
    const ptrdiff_t count,
    KEY_T * const restrict out )
{
    for (ptrdiff_t i = 0; i < count; ++i)
	out[i] = v;

    return count;
}
