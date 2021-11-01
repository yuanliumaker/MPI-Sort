#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <assert.h>
#include <string.h>

#include <mpi.h>
#include <mpi-sort.h>

#include "util.h"

int main (
	const int argc,
	const char * argv [])
{
	int BYKEY_CHECK = 0;
	READENV(BYKEY_CHECK, atoi);

	MPI_CHECK(MPI_Init((int *)&argc, (char ***)&argv));

	MPI_Comm comm = MPI_COMM_WORLD;

	int r, rc;
	MPI_CHECK(MPI_Comm_rank(comm, &r));
	MPI_CHECK(MPI_Comm_size(comm, &rc));

	/* print MPI_TAG_UB */
	{
		int * v = NULL, flag = 0;
		MPI_CHECK(MPI_Comm_get_attr(comm, MPI_TAG_UB, (void **)&v, &flag));

		if (!r)
			printf("MPI_TAG_UB: %d (flag: %d)\n", *v, flag);
	}

	if (argc != 4)
	{
		if (!r)
			fprintf(stderr,
					"usage: %s <int8|uint8|int16|uint16|int32|uint32|int64|uint64|float|double> <path/to/input> <path/to/output>\n",
					argv[0]);

		MPI_CHECK(MPI_Finalize());

		return EXIT_FAILURE;
	}

	MPI_Datatype type = MPI_DATATYPE_NULL;
	ptrdiff_t esz = -1;

	if (!strcmp("int8", argv[1]))
	{
		esz = 1;
		type = MPI_CHAR;
	}
	else if (!strcmp("uint8", argv[1]))
	{
		esz = 1;
		type = MPI_UNSIGNED_CHAR;
	}
	else if (!strcmp("int16", argv[1]))
	{
		esz = 2;
		type = MPI_SHORT;
	}
	else if (!strcmp("uint16", argv[1]))
	{
		esz = 2;
		type = MPI_UNSIGNED_SHORT;
	}
	else if (!strcmp("int32", argv[1]))
	{
		esz = 4;
		type = MPI_INTEGER;
	}
	else if (!strcmp("uint32", argv[1]))
	{
		esz = 4;
		type = MPI_UNSIGNED;
	}
	else if (!strcmp("int64", argv[1]))
	{
		esz = 8;
		type = MPI_LONG;
	}
	else if (!strcmp("uint64", argv[1]))
	{
		esz = 8;
		type = MPI_UINT64_T;
	}
	else if (!strcmp("float", argv[1]))
	{
		esz = 4;
		type = MPI_FLOAT;
	}
	else if (!strcmp("double", argv[1]))
	{
		esz = 8;
		type = MPI_DOUBLE;
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
	ptrdiff_t ic = 0;

	/* count items */
	{
		MPI_File f;
		MPI_CHECK(MPI_File_open(comm, (char *)argv[2],
								MPI_MODE_RDONLY, MPI_INFO_NULL, &f));

		MPI_Offset fsz;
		MPI_CHECK(MPI_File_get_size(f, &fsz));
		assert(0 == fsz % esz);

		ic = fsz / esz;
	}

	if (!r)
		printf("processing %zd elements\n", ic);

	/* homogeneous blocksize */
	const ptrdiff_t bsz = ((ic + rc - 1) / rc);

	/* local element range */
	ptrdiff_t rangelo = (r + 0) * bsz;
	ptrdiff_t rangehi = (r + 1) * bsz;

	rangehi = MIN(rangehi, ic - 0);
	rangelo = MIN(rangelo, ic - 0);

	const ptrdiff_t rangec = MAX(0, rangehi - rangelo);

	void * keys = NULL, *sortedkeys = NULL;

	/* read keys */
	{
		MPI_File f = NULL;

		MPI_CHECK(MPI_File_open
				  (comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f));

		keys = malloc(rangec * esz);
		assert(keys);

		MPI_CHECK(MPI_File_read_at_all
				  (f, rangelo * esz, keys, rangec, type, MPI_STATUS_IGNORE));

		MPI_CHECK(MPI_File_close(&f));
	}

	void * values = NULL, *sortedvalues = NULL;

	if (BYKEY_CHECK)
	{
		values = malloc(rangec * esz);
		memcpy(values, keys, rangec * esz);

		sortedvalues = malloc(rangec * esz);
	}

	sortedkeys = malloc(rangec * esz);

	double tbegin = MPI_Wtime();

	int NTIMES = 1;
	READENV(NTIMES, atoi);

	for (int t = 0; t < NTIMES; ++t)
	{
		if (BYKEY_CHECK)
			MPI_CHECK(MPI_Sort_bykey(keys, keys, rangec,
									 type, type,
									 sortedkeys, sortedvalues, rangec, comm));
		else
			MPI_CHECK(MPI_Sort(keys, rangec, type, sortedkeys, rangec, comm));
	}

	double tend = MPI_Wtime();

	if (BYKEY_CHECK)
		if (memcmp(sortedkeys, sortedvalues, rangec * esz))
			fprintf(stderr,
					"ERROR: rank %d: keys do not match with values.\n",
					r);

	/* write to file */
	{
		MPI_File f;
		MPI_CHECK(MPI_File_open
				  (comm, argv[3], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f));

		MPI_CHECK(MPI_File_set_size(f, ic * esz));

		MPI_CHECK(MPI_File_write_at_all
				  (f, rangelo * esz, sortedkeys, rangec, type, MPI_STATUS_IGNORE));

		MPI_CHECK(MPI_File_close(&f));
	}

	if (BYKEY_CHECK)
	{
		free(sortedvalues);
		free(values);
	}

	free(sortedkeys);
	free(keys);

	MPI_CHECK(MPI_Finalize());

	if (!r)
		printf("%s: sorted %zd entries in %.3f ms. Bye.\n",
			   argv[0], ic, (tend - tbegin) * 1e3);

	return EXIT_SUCCESS;
}
