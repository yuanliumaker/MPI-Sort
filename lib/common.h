#include <stddef.h>

typedef struct { ptrdiff_t begin, end; } range_t;
// ������������ inout ���ۼӺͣ�count Ϊ��������ĳ���
ptrdiff_t exscan_inplace (
	const ptrdiff_t count,
	ptrdiff_t * const inout);
// �������������е�һ����С�ڸ���ֵval ��λ�� first�����׵�ַ��last=first+sizeof(array)/sizeof(array[0])
ptrdiff_t lowerbound (
	const ptrdiff_t * first,
	const ptrdiff_t * last,
	const ptrdiff_t val);
// ����������in ��������������order ���Ƶ��������out �У�cout�����С��element_size ������Ԫ�ش�С
void gather (
	const ptrdiff_t element_size,
	const ptrdiff_t count,
	const void * const in,
	const ptrdiff_t * const order,
	void * const out );

