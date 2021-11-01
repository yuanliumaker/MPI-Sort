#include <mpi.h>

int MPI_Sort (
	void * sendbuf_destructive,
	int sendcount,
	MPI_Datatype datatype,
	const void * recvbuf,
	const int recvcount,
	MPI_Comm comm);

int MPI_Sort_bykey (
	void * sendkeys_destructive,
	void * sendvals_destructive,
	const int sendcount,
	MPI_Datatype keytype,
	MPI_Datatype valtype,
	void * recvkeys,
	void * recvvals,
	const int recvcount,
	MPI_Comm comm);
