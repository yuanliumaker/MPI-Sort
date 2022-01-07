#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <assert.h>

#include <mpi.h>

#include "util.h"

#define SWAPK(name, type, op)								\
	void name (void * restrict inout, const ptrdiff_t n)	\
	{														\
		type * const restrict v = (type *)inout;			\
															\
		for (ptrdiff_t i = 0; i < n; ++i)					\
			v[i] = op(v[i]);								\
	}

SWAPK(swapk16, uint16_t, __builtin_bswap16);
SWAPK(swapk32, uint32_t, __builtin_bswap32);
SWAPK(swapk64, uint64_t, __builtin_bswap64);

int main (
	const int argc,
	const char * argv [])
{
	/* bytes to skip from the head of the input file */
	ptrdiff_t CPD_SKIPB = 0;
	READENV(CPD_SKIPB, atoll);

	/* element size in bytes, elements are to be swapped */
	ptrdiff_t CPD_SWAPB = 1;
	READENV(CPD_SWAPB, atoll);

	MPI_CHECK(MPI_Init((int *)&argc, (char ***)&argv));

	MPI_Comm comm = MPI_COMM_WORLD;

	int r, rc;
	MPI_CHECK(MPI_Comm_rank(comm, &r));
	MPI_CHECK(MPI_Comm_size(comm, &rc));

	if (3 != argc)
	{
		if (!r)
			fprintf(stderr,
					"usage: %s <path/to/input> <path/to/output>\n",
					argv[0]);

		MPI_CHECK(MPI_Finalize());

		return EXIT_FAILURE;
	}

	if (!r)
		printf("%s: MPI comm size is %d\n", argv[0], rc);

	if (!r)
		CHECK(1 == CPD_SWAPB || 2 == CPD_SWAPB || 4 == CPD_SWAPB || 8 == CPD_SWAPB,
			  "%s: error: CPD_SWAPB (%zd) is not 1, or 2, or 4, or 8\n",
			  argv[0], CPD_SWAPB);

	/* input byte count */
	const ptrdiff_t ibc = count_slices_pathname(argv[1], 1);

	ptrdiff_t CPD_COUNTB = ibc - CPD_SKIPB;
	READENV(CPD_COUNTB, atoll);

	/* output byte count */
	const ptrdiff_t obc = CPD_COUNTB;

	CHECK(!(obc % CPD_SWAPB),
		"%s: error: output footprint (%zd B) not multiple of CPD_SWAPB (%zd)\n",
		argv[0], obc, CPD_SWAPB);

	const double tbegin = MPI_Wtime();

	MPI_File fin, fout;
	MPI_CHECK(MPI_File_open(comm, (char *)argv[1],
		MPI_MODE_RDONLY, MPI_INFO_NULL, &fin));

	MPI_CHECK(MPI_File_open(comm, argv[2],
		MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fout));

	MPI_CHECK(MPI_File_set_size(fout, obc));

	ptrdiff_t CPD_BUFB = 4 << 20;
	READENV(CPD_BUFB, atoll);

	const ptrdiff_t ec = MAX(1, (CPD_BUFB + CPD_SWAPB - 1) / CPD_SWAPB);
	const ptrdiff_t bc = CPD_SWAPB * ec;

	void * const buf = malloc(bc);

	__typeof__(swapk16) * ker = swapk16;

	if (4 == CPD_SWAPB)
		ker = swapk32;

	if (8 == CPD_SWAPB)
		ker = swapk64;

	/* round robin */
	for (ptrdiff_t base = 0; base < obc; base += bc * rc)
	{
		const ptrdiff_t off = MIN(obc, base + bc * (r + 0));
		const ptrdiff_t cnt = MAX(0, MIN(obc, base + bc * (r + 1)) - off);

		MPI_CHECK(MPI_File_read_at_all(fin, CPD_SKIPB + off, buf,
									   cnt, MPI_BYTE, MPI_STATUS_IGNORE));

		if (1 < CPD_SWAPB)
			ker(buf, cnt / CPD_SWAPB);

		MPI_CHECK(MPI_File_write_at_all(fout, off, buf,
										cnt, MPI_BYTE, MPI_STATUS_IGNORE));
	}

	free(buf);

	MPI_CHECK(MPI_File_close(&fout));
	MPI_CHECK(MPI_File_close(&fin));

	const double tend = MPI_Wtime();

	MPI_CHECK(MPI_Finalize());

	if (!r)
		fprintf(stderr,
				"%s: copied <%s> to <%s> in %.3f s (%.3f GB/s). bye.\n",
				argv[0], argv[1], argv[2], tend - tbegin, obc * 1e-9 / (tend - tbegin));

	return EXIT_SUCCESS;
}
