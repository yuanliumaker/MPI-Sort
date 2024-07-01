#include <stddef.h>

typedef struct { ptrdiff_t begin, end; } range_t;
// 计算输入数组 inout 的累加和，count 为输入数组的长度
ptrdiff_t exscan_inplace (
	const ptrdiff_t count,
	ptrdiff_t * const inout);
// 计算有序数组中第一个不小于给定值val 的位置 first数组首地址，last=first+sizeof(array)/sizeof(array[0])
ptrdiff_t lowerbound (
	const ptrdiff_t * first,
	const ptrdiff_t * last,
	const ptrdiff_t val);
// 将输入数组in 按给定索引数组order 复制到输出数组out 中，cout数组大小，element_size 数组中元素大小
void gather (
	const ptrdiff_t element_size,
	const ptrdiff_t count,
	const void * const in,
	const ptrdiff_t * const order,
	void * const out );

