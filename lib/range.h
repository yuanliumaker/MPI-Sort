#include <stddef.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

typedef struct { ptrdiff_t begin, end; } range_t;

static range_t range_split (
    const ptrdiff_t w,
    const ptrdiff_t wn,
    const ptrdiff_t tn)
{
    const ptrdiff_t share = tn / wn;
    const ptrdiff_t rmnd = tn % wn;

    const ptrdiff_t s = share * w + (w < rmnd ? w : rmnd);
    const ptrdiff_t c = share + (w < rmnd);

    return (range_t){ s, s + c };
}

static range_t range_keys (
    const KEY_T * const restrict in,
    const SIZE_T count)
{
    KEY_T lmin = in[0], lmax = in[0];

    for (SIZE_T i = 1; i < count; ++i)
    {
	const int s = in[i];

	lmin = MIN(lmin, s);
	lmax = MAX(lmax, s);
    }

    return (range_t){ lmin, 1 + (ptrdiff_t)lmax };
}

static ptrdiff_t range_count (
    const range_t e )
{
    return e.end - e.begin;
}
