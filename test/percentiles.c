#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <assert.h>
#include <string.h>
#include <math.h>

#include <posix-util.h>

#include <mpi.h>
#include <mpi-sort.h>

#include "util.h"

void printi8 (const void * const ptr) { printf("%hd", *(int8_t *)ptr); }
void printu8 (const void * const ptr) { printf("%hu", *(uint8_t *)ptr); }
void printi16 (const void * const ptr) { printf("%hd", *(int16_t *)ptr); }
void printu16 (const void * const ptr) { printf("%hu", *(uint16_t *)ptr); }
void printi32 (const void * const ptr) { printf("%d", *(int32_t *)ptr); }
void printu32 (const void * const ptr) { printf("%u", *(uint32_t *)ptr); }
void printi64 (const void * const ptr) { printf("%lld", *(int64_t *)ptr); }
void printu64 (const void * const ptr) { printf("%llu", *(uint64_t *)ptr); }
void printf32 (const void * const ptr) { printf("%f", *(float *)ptr); }
void printf64 (const void * const ptr) { printf("%lf", *(double *)ptr); }

int main (
	const int argc,
	const char * argv [])
{
	MPI_CHECK(MPI_Init((int *)&argc, (char ***)&argv));

	MPI_Comm comm = MPI_COMM_WORLD;

	int r, rc;
	MPI_CHECK(MPI_Comm_rank(comm, &r));
	MPI_CHECK(MPI_Comm_size(comm, &rc));

	if (argc < 3)
	{
		if (!r)
			fprintf(stderr,
					"usage: %s <int8|uint8|int16|uint16|int32|uint32|int64|uint64|float|double> "
					"<path/to/data.raw> percentile0 [percentile1 [...]]\n",
					argv[0]);

		MPI_CHECK(MPI_Finalize());

		return EXIT_FAILURE;
	}

	ptrdiff_t esz = -1;
	MPI_Datatype type = -1;
	__typeof__(printu8) * printel = NULL;

	if (!strcmp("int8", argv[1]))
	{
		esz = 1;
		type = MPI_CHAR;
		printel = printi8;
	}
	else if (!strcmp("uint8", argv[1]))
	{
		esz = 1;
		type = MPI_UNSIGNED_CHAR;
		printel = printu8;
	}
	else if (!strcmp("int16", argv[1]))
	{
		esz = 2;
		type = MPI_SHORT;
		printel = printi16;
	}
	else if (!strcmp("uint16", argv[1]))
	{
		esz = 2;
		type = MPI_UNSIGNED_SHORT;
		printel = printu16;
	}
	else if (!strcmp("int32", argv[1]))
	{
		esz = 4;
		type = MPI_INTEGER;
		printel = printi32;
	}
	else if (!strcmp("uint32", argv[1]))
	{
		esz = 4;
		type = MPI_UNSIGNED;
		printel = printu32;
	}
	else if (!strcmp("int64", argv[1]))
	{
		esz = 8;
		type = MPI_LONG;
		printel = printi64;
	}
	else if (!strcmp("uint64", argv[1]))
	{
		esz = 8;
		type = MPI_UNSIGNED_LONG;
		printel = printu64;
	}
	else if (!strcmp("float", argv[1]))
	{
		esz = 4;
		type = MPI_FLOAT;
		printel = printf32;
	}
	else if (!strcmp("double", argv[1]))
	{
		esz = 8;
		type = MPI_DOUBLE;
		printel = printf64;
	}
	else
	{
		if (!r)
			fprintf(stderr,
					"ERROR: unrecognized type (%s)\n",
					argv[1]);

		MPI_CHECK(MPI_Finalize());

		return EXIT_FAILURE;
	}

	/* item count */
	ptrdiff_t ic = count_slices_pathname(argv[2], esz);

	const double tbegin = MPI_Wtime();

	if (!r)
		fprintf(stderr, "processing %zd elements\n", ic);

	/* homogeneous blocksize */
	const ptrdiff_t bsz = ((ic + rc - 1) / rc);

	/* local element range */
	ptrdiff_t rangelo = (r + 0) * bsz;
	ptrdiff_t rangehi = (r + 1) * bsz;

	rangehi = MIN(rangehi, ic - 0);
	rangelo = MIN(rangelo, ic - 0);

	const ptrdiff_t rangec = MAX(0, rangehi - rangelo);

	void * keys = NULL;

	/* read keys */
	{
		const double t0 = MPI_Wtime();

		MPI_File f = NULL;

		MPI_CHECK(MPI_File_open
				  (comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f));

		POSIX_CHECK(keys = malloc(rangec * esz));

		MPI_CHECK(MPI_File_read_at_all
				  (f, rangelo * esz, keys, rangec, type, MPI_STATUS_IGNORE));

		MPI_CHECK(MPI_File_close(&f));

		MPI_CHECK(MPI_Barrier(comm));

		const double t1 = MPI_Wtime();

		if (!r)
			fprintf(stderr,
					"%s: <%s> loaded in %.3f s\n",
					argv[0], argv[2], t1 - t0);
	}

	/* sorting */
	{
		const double t0 = MPI_Wtime();

		MPI_CHECK(MPI_Sort(MPI_IN_PLACE, rangec, type, keys, rangec, comm));

		const double t1 = MPI_Wtime();

		if (!r)
			fprintf(stderr,
					"%s: sorted %zd <%s> elements in %.3f s\n",
					argv[0], ic, argv[1], t1 - t0);
	}

	/* print the percentiles */
	for (ptrdiff_t i = 3; i < argc; ++i)
	{
		const double p = atof(argv[i]) * .01;

		const ptrdiff_t entry = MAX(0, MIN(ic - 1, (ptrdiff_t)round(ic * p)));

		const ptrdiff_t rsrc = entry / bsz;

		MPI_Request req;
		char __attribute__((aligned(32))) buf[8];

		if (!r)
			MPI_CHECK(MPI_Irecv(buf, 1, type, rsrc, i + argc * rsrc, comm, &req));

		/* this rank has the desired entry */
		if (rsrc == r)
			MPI_CHECK(MPI_Send((entry - bsz * rsrc) * esz + (int8_t *)keys,
							   1, type, 0, i + argc * rsrc, comm));

		if (!r)
		{
			MPI_CHECK(MPI_Wait(&req, MPI_STATUS_IGNORE));

			printf("%s ", argv[i]);
			printel(buf);
			printf("\n");
		}
	}

	free(keys);

	const double tend = MPI_Wtime();

	MPI_CHECK(MPI_Finalize());

	if (!r)
		fprintf(stderr,
				"%s: served %d queries on %zd entries in %.3f s. Bye.\n",
				argv[0], argc - 3, ic, tend - tbegin);

	return EXIT_SUCCESS;
}
