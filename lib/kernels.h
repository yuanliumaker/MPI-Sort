#include <stddef.h>

void mux4 (
    const int * const restrict in0,
    const int * const restrict in1,
    const int * const restrict in2,
    const int * const restrict in3,
    const ptrdiff_t n,
    int * const restrict out );

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

const ptrdiff_t lowerbound (
    const ptrdiff_t * first,
    const ptrdiff_t * last,
    const ptrdiff_t val);
