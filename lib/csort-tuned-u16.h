

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
	const ptrdiff_t nicecount = samplecount & ~7;

    	uint32_t h[65537];
    	memset(h, 0, sizeof(h));

    	for (int i = 0; i < nicecount; i += 8)
    	{		
		const KEY_T r0 = samples[i + 0] - minval;
		const KEY_T r1 = samples[i + 1] - minval;
		const KEY_T r2 = samples[i + 2] - minval;
		const KEY_T r3 = samples[i + 3] - minval;
		const KEY_T r4 = samples[i + 4] - minval;
		const KEY_T r5 = samples[i + 5] - minval;
		const KEY_T r6 = samples[i + 6] - minval;
		const KEY_T r7 = samples[i + 7] - minval;
		
		assert(r0 >= 0 && r0 < d - 1);
		assert(r1 >= 0 && r1 < d - 1);
		assert(r2 >= 0 && r2 < d - 1);
		assert(r3 >= 0 && r3 < d - 1);
		assert(r4 >= 0 && r4 < d - 1);
		assert(r5 >= 0 && r5 < d - 1);
		assert(r6 >= 0 && r6 < d - 1);
		assert(r7 >= 0 && r7 < d - 1);
		
		++h[r0];
		++h[r1];
		++h[r2];
		++h[r3];
		++h[r4];
		++h[r5];
		++h[r6];
		++h[r7];    	}

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

    	for (int i = 0; i < nicecount; i += 8)
    	{
		
		const KEY_T r0 = samples[i + 0] - minval;
		const KEY_T r1 = samples[i + 1] - minval;
		const KEY_T r2 = samples[i + 2] - minval;
		const KEY_T r3 = samples[i + 3] - minval;
		const KEY_T r4 = samples[i + 4] - minval;
		const KEY_T r5 = samples[i + 5] - minval;
		const KEY_T r6 = samples[i + 6] - minval;
		const KEY_T r7 = samples[i + 7] - minval;

        	
		const int slot0 = h[r0]++;
		const int slot1 = h[r1]++;
		const int slot2 = h[r2]++;
		const int slot3 = h[r3]++;
		const int slot4 = h[r4]++;
		const int slot5 = h[r5]++;
		const int slot6 = h[r6]++;
		const int slot7 = h[r7]++;

        	
		order[slot0] = i + 0;
		order[slot1] = i + 1;
		order[slot2] = i + 2;
		order[slot3] = i + 3;
		order[slot4] = i + 4;
		order[slot5] = i + 5;
		order[slot6] = i + 6;
		order[slot7] = i + 7;
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
