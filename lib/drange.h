#include <stddef.h>
// ������Ӧ���͵���ת����Ӧ���޷�������
int drange_to_unsigned (
    MPI_Datatype uint_t,
    const ptrdiff_t count,
    void * const inout );
// ���޷�������ת���з��� cout ���鳤�ȣ�inout �������� uint_t �������������
int drange_from_unsigned (
    MPI_Datatype uint_t,
    const ptrdiff_t count,
    void * const inout );

typedef struct
{
  uint64_t minval_old, maxval_old;
  MPI_Datatype type_new;
  int err;
} drange_t;
// ������������ (uint_16,uint_32,uint64_t)ӳ�䵽��С���������ͣ��Լ������ݴ�С
drange_t drange_contract (
    MPI_Comm comm,
    MPI_Datatype uint_t,
    const ptrdiff_t count,
    void * const inout);
// ����� ���������������󵽽ϴ����������
void drange_expand (
    const drange_t r,
    MPI_Datatype uint_t,
    const ptrdiff_t count,
    void * const inout);
