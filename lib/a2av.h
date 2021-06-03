#include <stddef.h>

#include <mpi.h>

void a2av (
	const void * sendbuf,
	const ptrdiff_t * sendcounts,
	const ptrdiff_t * sdispls,
	MPI_Datatype type,
	void * recvbuf,
	const ptrdiff_t * recvcounts,
	const ptrdiff_t * rdispls,
	MPI_Comm comm );
