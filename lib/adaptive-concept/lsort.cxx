#include <cstddef>
#include <cstdint>
#include <cassert>
#include <algorithm>

extern "C"
void lsort(
	uint32_t * keys,
	const ptrdiff_t n)
{
	std::sort(keys, keys + n);
#ifndef NDEBUG
	for (ptrdiff_t i = 1; i < n; ++i)
		assert(keys[i] >= keys[i - 1]);
#endif
}
