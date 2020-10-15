divert(-1)
define(kernel,
`
void xtract_`'$1_`'$2 (
    const int slot,
    const ptrdiff_t count,
    const uint`'$2_t * const restrict in,
    uint`'$1_t * const restrict out )
{
    for(ptrdiff_t i = 0; i < count; ++i)
	out[i] = *(slot + (uint`'$1`'_t *)(i + in));
}
')
divert(0)
#include "xtract.h"

kernel(8, 16)
kernel(16, 32)
kernel(16, 64)
