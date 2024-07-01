#include <assert.h>

#include "common.h"

ptrdiff_t exscan_inplace (
    const ptrdiff_t count,
    ptrdiff_t * const inout)
{
    ptrdiff_t s = 0;

    for (ptrdiff_t i = 0; i < count; ++i)
    {
	const ptrdiff_t v = inout[i];

	inout[i] = s;

	s += v;
    }

    return s;
}

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
// 作用：用于从输入数组in 中根据索引数组idx 收集元素并存储到输出数组out 中，count为输入数组长度
#define GATHER_KERNEL(T)			\
    static void gather_ ## T (			\
	const ptrdiff_t count,			\
	const T * restrict const in,		\
	const ptrdiff_t * restrict const idx,	\
	T * restrict const out )		\
    {						\
	_Pragma("GCC unroll (4)")		\
	for (ptrdiff_t i = 0; i < count; ++i)	\
	    out[i] = in[idx[i]];		\
    }

#include <stdint.h>

GATHER_KERNEL(uint8_t)
GATHER_KERNEL(uint16_t)
GATHER_KERNEL(uint32_t)
GATHER_KERNEL(uint64_t)

#include <string.h>

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
    // 一般实现，将in 复制到out 中，基于索引数组，size 是数组中的元素大小
    for (ptrdiff_t i = 0; i < count; ++i)
	memcpy(size * i + (char *)out,
	       size * idx[i] + (char *)in,
	       size);
}
