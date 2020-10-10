#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <assert.h>
#include <string.h>

#include <mpi.h>
#include <mpi-sort.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define READENV(x, op)				\
    do						\
    {						\
	if (getenv(#x))				\
	    x = op(getenv(#x));			\
    }						\
    while(0)

#define MPI_CHECK(stmt)						\
    do								\
    {								\
	const int code = stmt;					\
								\
	if (code != MPI_SUCCESS)				\
	{							\
	    char msg[2048];					\
	    int len = sizeof(msg);				\
	    MPI_Error_string(code, msg, &len);			\
								\
	    fprintf(stderr,					\
		    "ERROR\n" #stmt "%s (%s:%d)\n",		\
			msg, __FILE__, __LINE__);		\
								\
	    fflush(stderr);					\
								\
	    MPI_Abort(MPI_COMM_WORLD, code);			\
	}							\
    }								\
    while(0)

int main (
    const int argc,
    const char * argv [])
{
    int BYKEY = 0;
    READENV(BYKEY, atoi);

    ptrdiff_t ESZ = 1;
    READENV(ESZ, atoi);

    MPI_CHECK(MPI_Init((int *)&argc, (char ***)&argv));

    MPI_Comm comm = MPI_COMM_WORLD;

    int r, rc;
    MPI_CHECK(MPI_Comm_rank(comm, &r));
    MPI_CHECK(MPI_Comm_size(comm, &rc));

    if (argc != 3)
    {
	if (!r)
	    fprintf(stderr,
		    "usage: %s <path/to/input> <path/to/output>\n",
		    argv[0]);

	return EXIT_FAILURE;
    }

    __extension__ ptrdiff_t count_items (
	const char * p)
    {
	ptrdiff_t retval;

	if (!r)
	{
	    MPI_File f;
	    MPI_CHECK(MPI_File_open(MPI_COMM_SELF, (char *)p,
				    MPI_MODE_RDONLY, MPI_INFO_NULL, &f));

	    MPI_Offset fsz;
	    MPI_CHECK(MPI_File_get_size(f, &fsz));
	    assert(0 == fsz % ESZ);

	    retval = fsz / ESZ;

	    MPI_CHECK(MPI_File_close(&f));
	}

	MPI_CHECK(MPI_Bcast(&retval, 1, MPI_INT64_T, 0, MPI_COMM_WORLD));

	return retval;
    }

    __extension__ void * read (
	const char * const p,
	const ptrdiff_t s,
	const ptrdiff_t c,
	MPI_Datatype t)
    {
	MPI_File f = NULL;

	MPI_CHECK(MPI_File_open
		  (comm, (char *)p, MPI_MODE_RDONLY, MPI_INFO_NULL, &f));

	void * buf = malloc(c * ESZ);
	assert(buf);

	MPI_CHECK(MPI_File_read_at_all
		  (f, s * ESZ, buf, c, t, MPI_STATUS_IGNORE));

	MPI_CHECK(MPI_File_close(&f));

	return buf;
    }

    /* element count */
    const ptrdiff_t ec = count_items(argv[1]);

    if (!r)
	printf("processing %zd elements\n", ec);

    /* homogeneous blocksize */
    const ptrdiff_t bsz = ((ec + rc - 1) / rc);
    //const ptrdiff_t bsz = ((ec + rc/2 - 1) / (rc/2));

    /* local element range */
    ptrdiff_t rangelo = (r + 0) * bsz;
    ptrdiff_t rangehi = (r + 1) * bsz;

    rangehi = MIN(rangehi, ec - 0);
    rangelo = MIN(rangelo, ec - 0);

    const ptrdiff_t rangec = MAX(0, rangehi - rangelo);

    const MPI_Datatype type = ESZ == 1 ? MPI_UNSIGNED_CHAR : MPI_UNSIGNED_SHORT;

    void * keys = read(argv[1], rangelo, rangec, type);
    assert(keys);

    void * values = NULL;

    if (BYKEY)
    {
	values = malloc(rangec * ESZ);
	memcpy(values, keys, rangec * ESZ);
    }

    double tbegin = MPI_Wtime();

    if (BYKEY)
	MPI_CHECK(MPI_Sort_bykey(MPI_IN_PLACE, keys, rangec,
				 type, type,
				 keys, values, rangec, comm));
    else
	MPI_CHECK(MPI_Sort(MPI_IN_PLACE, rangec, type, keys, rangec, comm));

    double tend = MPI_Wtime();

    if (BYKEY)
	if (memcmp(keys, values, rangec * ESZ))
	    fprintf(stderr,
		    "ERROR: rank %d: keys do not match with values.\n",
		    r);

    /* write to file */
    {
	MPI_File f;
	MPI_CHECK(MPI_File_open
		  (comm, argv[2], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f));

	MPI_CHECK(MPI_File_write_at_all
		  (f, rangelo * ESZ, keys, rangec, type, MPI_STATUS_IGNORE));

	MPI_CHECK(MPI_File_close(&f));
    }

    if (BYKEY)
	free(values);

    free(keys);

    MPI_CHECK(MPI_Finalize());

    if (!r)
	printf("%s: sorted %zd entries in %.3f ms. Bye.\n", argv[0], ec, (tend - tbegin) * 1e3);

    return EXIT_SUCCESS;
}
