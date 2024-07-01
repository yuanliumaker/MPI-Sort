#include <stddef.h>
// 将各相应类型的数转成相应的无符号类型
int drange_to_unsigned (
    MPI_Datatype uint_t,
    const ptrdiff_t count,
    void * const inout );
// 各无符号类型转成有符号 cout 数组长度，inout 输入数组 uint_t 输入数组的类型
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
// 将各类型数组 (uint_16,uint_32,uint64_t)映射到较小的数据类型，以减少数据大小
drange_t drange_contract (
    MPI_Comm comm,
    MPI_Datatype uint_t,
    const ptrdiff_t count,
    void * const inout);
// 逆过程 将各类型数据扩大到较大的数据类型
void drange_expand (
    const drange_t r,
    MPI_Datatype uint_t,
    const ptrdiff_t count,
    void * const inout);
