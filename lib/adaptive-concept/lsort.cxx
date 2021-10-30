#include <cstddef>
#include <cstdint>
#include <cassert>
#include <algorithm>

template < typename T >
static void sortk (
	T * keys,
	const ptrdiff_t n )
{
	std::sort(keys, keys + n);
#ifndef NDEBUG
	for (ptrdiff_t i = 1; i < n; ++i)
		assert(keys[i] >= keys[i - 1]);
#endif
}

extern "C"
void lsortu(
	const ptrdiff_t e,
	const ptrdiff_t c,
	void * k )
{
	switch (e)
	{
	case 8 :
		sortk((uint64_t *)k, c);
		break;
	case 4 :
		sortk((uint32_t *)k, c);
		break;
	case 2 :
		sortk((uint16_t *)k, c);
		break;
	case 1 :
		sortk((uint8_t *)k, c);
		break;
	default:
		fprintf(
			stderr,
			"error in lsort: unsupported element size (%d)\n",
			e);
		exit(EXIT_FAILURE);
	}
}
