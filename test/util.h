#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>

#define READENV(x, op)							\
	do											\
	{											\
		if (getenv(#x))							\
			x = op(getenv(#x));					\
	}											\
	while(0)

#define TOSTR_(a) #a
#define MKSTR(a) TOSTR_(a)

#define POSIX_CHECK(stmt)							\
	do												\
	{												\
		if (!(stmt))								\
		{											\
			perror(#stmt  " in "					\
				   __FILE__ ":" MKSTR(__LINE__) );	\
													\
			exit(EXIT_FAILURE);						\
		}											\
	}												\
	while(0)

#ifndef MPI_CHECK
#define MPI_CHECK(stmt)								\
	do												\
	{												\
		const int code = stmt;						\
													\
		if (code != MPI_SUCCESS)					\
		{											\
			char msg[2048];							\
			int len = sizeof(msg);					\
			MPI_Error_string(code, msg, &len);		\
													\
			fprintf(stderr,							\
					"ERROR\n" #stmt "%s (%s:%d)\n",	\
					msg, __FILE__, __LINE__);		\
													\
			fflush(stderr);							\
													\
			MPI_Abort(MPI_COMM_WORLD, code);		\
		}											\
	}												\
	while(0)
#endif

#define CHECK(stmt, ...)						\
	do											\
	{											\
		if (!(stmt))							\
		{										\
			fprintf(stderr,						\
					__VA_ARGS__);				\
												\
			exit(EXIT_FAILURE);					\
		}										\
	}											\
	while(0)

static size_t count_slices (
	MPI_File f,
	size_t nbytes_slice)
{
	MPI_Offset fsize;
	MPI_CHECK(MPI_File_get_size(f, &fsize));

	CHECK(fsize % nbytes_slice == 0,
		  "error: fsize (%zd) not multiple of nbytes_slice (%zd)\n",
		  (size_t)fsize, nbytes_slice);

	return fsize / nbytes_slice;
}

inline static size_t count_slices_pathname (
	const char * const pathname,
	const size_t nbytes_slice)
{
	int rank;
	MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

	size_t retval = -1;

	if (rank == 0)
	{
		MPI_File fin;

		MPI_CHECK(MPI_File_open(MPI_COMM_SELF, (char *)pathname,
								MPI_MODE_RDONLY, MPI_INFO_NULL, &fin));

		retval = count_slices(fin, nbytes_slice);

		MPI_CHECK(MPI_File_close(&fin));
	}

	MPI_CHECK(MPI_Bcast(&retval, 1, MPI_INT64_T, 0, MPI_COMM_WORLD));

	return retval;
}
