#include <stddef.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

typedef struct { ptrdiff_t begin, end; } range_t;

ptrdiff_t exscan_inplace (
    const ptrdiff_t count,
    ptrdiff_t * const inout);

ptrdiff_t lowerbound (
    const ptrdiff_t * first,
    const ptrdiff_t * last,
    const ptrdiff_t val);

void gather (
    const ptrdiff_t element_size,
    const ptrdiff_t count,
    const void * const in,
    const ptrdiff_t * const order,
    void * const out );

