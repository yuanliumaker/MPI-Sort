#include <stddef.h>
#include <stdint.h>

// 用于从uint16_t类型的输入数组in  提取uint8_t类型的值，并将之存储到out中 slot=1or 0以 确定大小端
void xtract_8_16 (
    const int slot,
    const ptrdiff_t count,
    const uint16_t * const restrict in,
    uint8_t * const restrict out );
//同理
void xtract_16_32 (
    const int slot,
    const ptrdiff_t count,
    const uint32_t * const restrict in,
    uint16_t * const restrict out );
// 同理
void xtract_16_64 (
    const int slot,
    const ptrdiff_t count,
    const uint64_t * const restrict in,
    uint16_t * const restrict out );
