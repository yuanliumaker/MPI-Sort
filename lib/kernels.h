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
    const ptrdiff_t * const restrict in,
    ptrdiff_t * const restrict out );

ptrdiff_t exscan_int32 (
    const ptrdiff_t count,
    const int * const restrict in,
    ptrdiff_t * const restrict out );

void counting_sort (
    const int minval,
    const int supval,
    const ptrdiff_t samplecount,
    const int * const restrict samples,
    ptrdiff_t * const restrict histo,
    ptrdiff_t * const restrict start,
    ptrdiff_t * const restrict order );
