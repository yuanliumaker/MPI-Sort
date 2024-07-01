// 找最值
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
// 计算数组in 的累加和 out存储前缀和，返回数组in 的累加和s
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

#if (_BITCOUNT_ == 8) && defined(_TUNED_)
#include "csort-tuned-u8.h"
#elif (_BITCOUNT_ == 16) && defined(_TUNED_)
#include "csort-tuned-u16.h"
#else
// 计数排序
// minval：样本最小值
// supval: 样本最大值+1
// samplecount: 样本数量
// samples 指向样本数组的指针
// histo : 直方图
// order: 排序顺序
// start: 排序后的结果数组
static ptrdiff_t counting_sort (
	const unsigned int minval,
	const unsigned int supval,
	const ptrdiff_t samplecount,
	const KEY_T * const restrict samples,
	ptrdiff_t * const restrict histo,
	ptrdiff_t * const restrict start,
	ptrdiff_t * const restrict order )
{
	const int d = supval - minval;
// 确保每个样本值都在范围[minval,supval)内
#ifndef NDEBUG
	for (ptrdiff_t i = 0; i < samplecount; ++i)
	{
		const int s = samples[i] - minval;
		assert(s >= 0 && s < d);
	}
#endif
// 构建直方图，记录每个样本值出现的次数
	for (ptrdiff_t i = 0; i < samplecount; ++i)
		++histo[samples[i] - minval];
	// 对该直方图进行前缀和计算
	exscan(d, histo, start);

	for (ptrdiff_t i = 0; i < samplecount; ++i)
		order[start[samples[i] - minval]++] = i;

	return exscan(d, histo, start);
}
#endif 
// 游程编码（run-length encoding），用于压缩数据。
// in:输入数组，n: 输入数组的长度 vs:输出数组，存储值，ls:输出数组，存储长度
// 如in=[1,1,1,2,2,3,3,3,3,4]
// vs=[1,2,3,4]
// ls=[3,2,4,1]
// c 为压缩后的长度
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
