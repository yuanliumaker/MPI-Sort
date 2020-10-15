#include <stddef.h>

int drange_to_unsigned (
    MPI_Datatype uint_t,
    const ptrdiff_t count,
    void * const inout );

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

drange_t drange_contract (
    MPI_Comm comm,
    MPI_Datatype uint_t,
    const ptrdiff_t count,
    void * const inout);

void drange_expand (
    const drange_t r,
    MPI_Datatype uint_t,
    const ptrdiff_t count,
    void * const inout);
