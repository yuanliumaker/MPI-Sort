#include <stddef.h>
#include <stdint.h>

// ���ڴ�uint16_t���͵���������in  ��ȡuint8_t���͵�ֵ������֮�洢��out�� slot=1or 0�� ȷ����С��
void xtract_8_16 (
    const int slot,
    const ptrdiff_t count,
    const uint16_t * const restrict in,
    uint8_t * const restrict out );
//ͬ��
void xtract_16_32 (
    const int slot,
    const ptrdiff_t count,
    const uint32_t * const restrict in,
    uint16_t * const restrict out );
// ͬ��
void xtract_16_64 (
    const int slot,
    const ptrdiff_t count,
    const uint64_t * const restrict in,
    uint16_t * const restrict out );
