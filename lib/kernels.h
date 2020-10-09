#include <stddef.h>

ptrdiff_t fill (
    const KEY_T v,
    const ptrdiff_t count,
    KEY_T * const restrict out );

#include "range.h"

ptrdiff_t nzcount (
    const int * const restrict in,
    const ptrdiff_t count );

int maxval (
    const int * const restrict in,
    const ptrdiff_t count );

ptrdiff_t exscan (
    const ptrdiff_t count,
    const ptrdiff_t * const in,
    ptrdiff_t * const out );

ptrdiff_t exscan_int32 (
    const ptrdiff_t count,
    const int * const restrict in,
    ptrdiff_t * const restrict out );

ptrdiff_t counting_sort (
    const int minval,
    const int supval,
    const ptrdiff_t samplecount,
    const KEY_T * const restrict samples,
    ptrdiff_t * const restrict histo,
    ptrdiff_t * const restrict start,
    ptrdiff_t * const restrict order );

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

ptrdiff_t rle (
    const KEY_T * const seq,
    const ptrdiff_t count,
    KEY_T * const values,
    ptrdiff_t * lengths );
