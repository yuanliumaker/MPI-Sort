#include <cstdio>
#include <cstddef>
#include <cstring>
#include <cassert>

#include <limits>
#include <algorithm>
#include <utility>

#include <stdint.h>

template < typename C, typename I, typename V >
static void gather (
	const C count,
	const V * __restrict__ const in,
	const I * __restrict__ const idx,
	V * __restrict__ const out )
{
	for (C i = 0; i < count; ++i)
		out[i] = in[idx[i]];
}

template < typename C, typename I >
static void gather_values (
	const size_t vsz,
	const C count,
	const void * __restrict__ const in,
	const I * __restrict__ const idx,
	void * __restrict__ const out )
{
	switch (vsz)
	{
	case 1 :
		gather(count, (uint8_t *)in, idx, (uint8_t *)out);
		break;

	case 2 :
		gather(count, (uint16_t *)in, idx, (uint16_t *)out);
		break;

	case 4 :
		gather(count, (uint32_t *)in, idx, (uint32_t *)out);
		break;

	case 8 :
		gather(count, (uint64_t *)in, idx, (uint64_t *)out);
		break;

	default :
		/* generic */
		for (ptrdiff_t i = 0; i < count; ++i)
			memcpy(vsz * i + (char *)out,
				   vsz * idx[i] + (char *)in,
				   vsz);
	}
}

template < typename K, typename C >
static void sort_kv_indirect (
	const int s,
	const size_t vsz,
	const C c,
	K * const k,
	void * const v )
{
	typedef std::pair<K, C> KI_t;

	KI_t * t = (KI_t *)malloc(sizeof(*t) * c);

	for (C i = 0; i < c; ++i)
		t[i] = std::make_pair(k[i], i);

	if (s)
		std::stable_sort(t, t + c);
	else
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

		gather_values(vsz, n, v2, ord, vsz * base + (char *)v);
	}

	free(t);
	free(v2);
}

template < typename K, typename C, typename V >
static void sort_kv_direct (
	const int s,
	const C c,
	K * k,
	V * v )
{
	typedef std::pair<K, V> KV_t;

	KV_t * t = (KV_t *)malloc(sizeof(*t) * c);

	for (C i = 0; i < c; ++i)
		t[i] = std::make_pair(k[i], v[i]);

	if (s)
		std::stable_sort(t, t + c);
	else
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
static void sort_bykey_t (
	const int s,
	const ptrdiff_t vsz,
	const C c,
	K * k,
	void * v )
{
	if (vsz <= sizeof(c))
		switch (vsz)
		{
		case 1 :
			sort_kv_direct(s, c, k, (uint8_t *)v);
			break;

		case 2 :
			sort_kv_direct(s, c, k, (uint16_t *)v);
			break;

		case 4 :
			sort_kv_direct(s, c, k, (uint32_t *)v);
			break;

		case 8 :
			sort_kv_direct(s, c, k, (uint64_t *)v);
			break;

		default:
			sort_kv_indirect(s, vsz, c, k, v);
		}
	else
		sort_kv_indirect(s, vsz, c, k, v);
}

template < typename K >
static void sort_bykey (
	const int s,
	const ptrdiff_t vsz,
	const ptrdiff_t c,
	K * k,
	void * v )
{
	/* why signed integers? we want to trigger gatherdd
	   this is based on a speculation of mine,
	   that finally gatherdd is finally faster than scalar loads on
	   microarchs in 2020+ */

	if ((ptrdiff_t)std::numeric_limits<int32_t>::max() >= c)
		sort_bykey_t(s, vsz, (int32_t)c, k, v);
	else
		sort_bykey_t(s, vsz, (int64_t)c, k, v);
}

template < typename T >
static void sort (
	const int s,
	const ptrdiff_t c,
	T * k )
{
	if (s)
		std::stable_sort(k, k + c);
	else
		std::sort(k, k + c);
}

extern "C"
void lsort (
	//s : stable requested if s nonzero
	const int s,
	//ksz : key size in bytes
	const ptrdiff_t ksz,
	//vsz : value size in bytes 
	const ptrdiff_t vsz,
	//cnt : item count of both k and v
	const ptrdiff_t cnt,
	//k : keys array
	void * k,
	//v : values array
	void * v)
{
	switch (ksz)
	{
	case 1 :
		if (v)
			sort_bykey(s, vsz, cnt, (uint8_t *)k, v);
		else
			sort(s, cnt, (uint8_t *)k);

		break;

	case 2 :
		if (v)
			sort_bykey(s, vsz, cnt, (uint16_t *)k, v);
		else
			sort(s, cnt, (uint16_t *)k);

		break;

	case 4 :
		if (v)
			sort_bykey(s, vsz, cnt, (uint32_t *)k, v);
		else
			sort(s, cnt, (uint32_t *)k);

		break;

	case 8 :
		if (v)
			sort_bykey(s, vsz, cnt, (uint64_t *)k, v);
		else
			sort(s, cnt, (uint64_t *)k);

		break;

	default:
		fprintf(stderr,	"error in lsortu: unsupported element size (%d)\n",	ksz);
		exit(EXIT_FAILURE);
	}
}
