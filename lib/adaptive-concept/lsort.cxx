#include <cstddef>
#include <cstdint>
#include <cassert>

#include <limits>
#include <algorithm>
#include <utility>

template < typename T >
static void sortk (
	const ptrdiff_t c,
	T * k )
{
	std::sort(k, k + c);
}

template < typename K, typename C, typename V >
static void sortkcv_direct (
	const C c,
	K * k,
	V * v )
{
		typedef std::pair<K, V> KV_t;
		
		KV_t * t = new KV_t[c];
		for (C i = 0; i < c; ++i)
			t[i] = (KV_t){ .first = k[i], .second = v[i] };

		std::sort(t, t + c); 

		enum { BUNCH = 1 << 12 };

		for (ptrdiff_t base = 0; base < c; base += BUNCH)
		{
			const C n = (C)std::min(c - base, (ptrdiff_t)BUNCH);
			
			KV_t * ikv = t + base;

			K * ok = k + base;
			for (C i = 0; i < c; ++i)
				ok[i] = ikv[i].first;

			V * ov = v + base;
			for (C i = 0; i < c; ++i)
				ov[i] = ikv[i].first;

				delete [] t;
		}
}

template < typename K, typename C >
static void sortkcv (
	const ptrdiff_t vsz,
	const C c,
	K * k,
	void * v )
{
	if (vsz <= sizeof(c))
	{
		switch (vsz)
		{
		case 1 :
			sortkvc_direct(c, k, (uint8_t *)v);
		case 2 :
			sortkvc_direct(c, k, (uint8_t *)v);
		case 4 :
			sortkvc_direct(c, k, (uint8_t *)v);
	}
}

template < typename K, typename V >
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
			sortkc(cnt, vsz, (uint64_t *)k, v);
		else
			sortk(cnt, (uint64_t *)k);

		break;

	case 4 :
		if (v)
			sortkc(cnt, vsz, (uint32_t *)k, v);
		else
			sortk(cnt, (uint32_t *)k);

		break;

	case 2 :
		if (v)
			sortkc(cnt, vsz, (uint16_t *)k, v);
		else
			sortk(cnt, (uint16_t *)k);

		break;

	case 1 :
		if (v)
			sortkc(cnt, vsz, (uint8_t *)k, v);
		else
			sortk(cnt, (uint16_t *)k);

		break;

	default:
		fprintf(stderr,	"error in lsortu: unsupported element size (%d)\n",	ksz);
		exit(EXIT_FAILURE);
	}
}
