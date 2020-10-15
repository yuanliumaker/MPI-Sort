#include <stddef.h>

int rmanip_to_unsigned (
    MPI_Datatype uint_t,
    const ptrdiff_t count,
    void * const inout );

int rmanip_from_unsigned (
    MPI_Datatype uint_t,
    const ptrdiff_t count,
    void * const inout );

typedef struct
{
  uint64_t minval_old, maxval_old;
  MPI_Datatype type_new;
  int err;
} rmanip_t;

rmanip_t rmanip_contract (
    MPI_Comm comm,
    MPI_Datatype uint_t,
    const ptrdiff_t count,
    void * const inout);

void rmanip_expand (
    const rmanip_t r,
    MPI_Datatype uint_t,
    const ptrdiff_t count,
    void * const inout);
