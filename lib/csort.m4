divert(-1)
dnl forloop(var, from, to, stmt) - simple version
define(`forloop', `pushdef(`$1', `$2')_forloop($@)popdef(`$1')')
define(`_forloop',
       `$4`'ifelse($1, `$3', `', `define(`$1', incr($1))$0($@)')')
divert(0)

define(N, 8)

define(M, eval(1<<16))

static ptrdiff_t counting_sort (
    const unsigned int minval,
    const unsigned int supval,
    const ptrdiff_t samplecount,
    const KEY_T * const restrict samples,
    ptrdiff_t * const restrict histo,
    ptrdiff_t * const restrict start,
    ptrdiff_t * const restrict order )
{
    const unsigned int d = supval - minval;

    if (samplecount < (1ull << 32))
    {
	const ptrdiff_t nicecount = ifelse(eval((N & (N - 1)) == 0), 1, `samplecount & ~eval(N-1)', `N * (samplecount / N)');

    	uint32_t h[65537];
    	memset(h, 0, sizeof(h));

    	for (int i = 0; i < nicecount; i += N)
    	{dnl
		forloop(C, 0, eval(N - 1),`
		const KEY_T r`'C = samples[i + C] - minval;')dnl

		forloop(C, 0, eval(N - 1),`
		assert(r`'C >= 0 && r`'C < d - 1);')dnl

		forloop(C, 0, eval(N - 1), `
		++h[r`'C];')dnl
    	}

    	for (int i = nicecount ; i < samplecount; ++i)
	    ++h[samples[i] - minval];

    	int s = 0;
    	for (int i = 0; i < d; ++i)
    	{
		const int v = h[i];

		h[i] = s;
    		start[i] = s;

    		s += v;
		histo[i] = v;
     	}

    	for (int i = 0; i < nicecount; i += N)
    	{
		forloop(C, 0, eval(N - 1),`
		const KEY_T r`'C = samples[i + C] - minval;')

        	forloop(C, 0, eval(N - 1),`
		const int slot`'C = h[r`'C]++;')

        	forloop(C, 0, eval(N - 1),`
		order[slot`'C] = i + C;')
    	}

    	for (int i = nicecount; i < samplecount ; ++i)
	    order[h[samples[i] - minval]++] = i;

    	return s;
    }

#ifndef NDEBUG
    for (ptrdiff_t i = 0; i < samplecount; ++i)
    {
	const int s = samples[i] - minval;
	assert(s >= 0 && s < d);
    }
#endif

    for (ptrdiff_t i = 0; i < samplecount; ++i)
	++histo[samples[i] - minval];

    exscan(d, histo, start);

    for (ptrdiff_t i = 0; i < samplecount; ++i)
	order[start[samples[i] - minval]++] = i;

    return exscan(d, histo, start);
}
