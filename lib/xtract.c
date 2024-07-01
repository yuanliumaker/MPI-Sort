
#include "xtract.h"


void xtract_8_16 (
    const int slot,
    const ptrdiff_t count,
    const uint16_t * const restrict in,
    uint8_t * const restrict out )
{
    for(ptrdiff_t i = 0; i < count; ++i)
	out[i] = *(slot + (uint8_t *)(i + in));
}


void xtract_16_32 (
    const int slot,
    const ptrdiff_t count,
    const uint32_t * const restrict in,
    uint16_t * const restrict out )
{
    for(ptrdiff_t i = 0; i < count; ++i)
	out[i] = *(slot + (uint16_t *)(i + in));
}


void xtract_16_64 (
    const int slot,
    const ptrdiff_t count,
    const uint64_t * const restrict in,
    uint16_t * const restrict out )
{
    for(ptrdiff_t i = 0; i < count; ++i)
	out[i] = *(slot + (uint16_t *)(i + in));
}

