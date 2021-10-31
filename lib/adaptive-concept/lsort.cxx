#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <limits>
#include <algorithm>
#include <utility>

template < typename C, typename I, typename V >
static void gatherciv (
		const C count,
		const V * __restrict__ const in,
		const I * __restrict__ const idx,
		V * __restrict__ const out )
    {
		for (C i = 0; i < count; ++i)
				out[i] = in[idx[i]];
    }

template < typename C, typename I >
static void gather (
	const size_t vsz,
	const C count,
	const void * __restrict__ const in,
	const I * __restrict__ const idx,
	void * __restrict__ const out )
    {
		switch (vsz)
		{
		case 1 :
			gatherciv(count, in
		}
    }

template < typename T >
static void sortk (
	const ptrdiff_t c,
	T * k )
{
	std::sort(k, k + c);
}

template < typename K, typename C >
static void sortkcv_indirect (
	const size_t vsz,
	const C c,
	K * const k,
	void * const v )
{
	typedef std::pair<K, C> KI_t;

	KI_t * t = (KI_t *)malloc(sizeof(*t) * c);

	for (C i = 0; i < c; ++i)
		t[i] = (KI_t){ .first = k[i], .second = i };

	std::sort(t, t + c);

	void * v2 = malloc(vsz * c);
	memcpy(v2, v, vsz * c);

	enum { BUNCH = 1 << 12 };

	for (ptrdiff_t base = 0; base < c; base += BUNCH)
	{
		const C n = (C)std::min((ptrdiff_t)c - base, (ptrdiff_t)BUNCH);

		const KI_t * iki = t + base;
		
		K * ok = k + base;
		for (C i = 0; i < n; ++i)
			ok[i] = iki[i].first;

		C ord[BUNCH];
		for (C i = 0; i < n; ++i)
			ord[i] = iki[i].second;

		gather(vsz, n, v2, ord, v + base);
	}

	free(t);
	free(v2);
}

template < typename K, typename C, typename V >
static void sortkcv_direct (
	const C c,
	K * k,
	V * v )
{
	typedef std::pair<K, V> KV_t;

	KV_t * t = (KV_t *)malloc(sizeof(*t) * c);

	for (C i = 0; i < c; ++i)
		t[i] = (KV_t){ .first = k[i], .second = v[i] };

	std::sort(t, t + c);

	enum { BUNCH = 1 << 12 };

	for (ptrdiff_t base = 0; base < c; base += BUNCH)
	{
		const C n = (C)std::min((ptrdiff_t)c - base, (ptrdiff_t)BUNCH);

		KV_t * ikv = t + base;

		K * ok = k + base;
		for (C i = 0; i < n; ++i)
			ok[i] = ikv[i].first;

		V * ov = v + base;
		for (C i = 0; i < n; ++i)
			ov[i] = ikv[i].second;
	}

	free(t);
}

template < typename K, typename C >
static void sortkcv (
	const ptrdiff_t vsz,
	const C c,
	K * k,
	void * v )
{
	if (vsz <= sizeof(c))
		switch (vsz)
		{
		case 1 :
			sortkcv_direct(c, k, (uint8_t *)v);
			break;
		case 2 :
			sortkcv_direct(c, k, (uint16_t *)v);
			break;
		case 4 :
			sortkcv_direct(c, k, (uint32_t *)v);
			break;
		case 8 :
			sortkcv_direct(c, k, (uint64_t *)v);
			break;
		default:
			sortkcv_indirect(vsz, c, k, v);
		}
	else
		sortkcv_indirect(vsz, c, k, v);
}

template < typename K >
static void sortkc (
	const ptrdiff_t vsz,
	const ptrdiff_t c,
	K * k,
	void * v )
{
	if ((ptrdiff_t)std::numeric_limits<uint32_t>::max() >= c)
		sortkcv(vsz, (uint32_t)c, k, v);
	else
		sortkcv(vsz, (uint64_t)c, k, v);
}

extern "C"
void lsortu(
	const ptrdiff_t ksz,
	const ptrdiff_t vsz,
	const ptrdiff_t cnt,
	void * k,
	void * v)
{
	switch (ksz)
	{
	case 8 :
		if (v)
			sortkc(vsz, cnt, (uint64_t *)k, v);
		else
			sortk(cnt, (uint64_t *)k);

		break;

	case 4 :
		if (v)
			sortkc(vsz, cnt, (uint32_t *)k, v);
		else
			sortk(cnt, (uint32_t *)k);

		break;

	case 2 :
		if (v)
			sortkc(vsz, cnt, (uint16_t *)k, v);
		else
			sortk(cnt, (uint16_t *)k);

		break;

	case 1 :
		if (v)
			sortkc(vsz, cnt, (uint8_t *)k, v);
		else
			sortk(cnt, (uint16_t *)k);

		break;

	default:
		fprintf(stderr,	"error in lsortu: unsupported element size (%d)\n",	ksz);
		exit(EXIT_FAILURE);
	}
}
