# mpi-sort

The library exposes two functions:

```c
#include <mpi.h>

int MPI_Sort (
	void * sendbuf_destructive,
	int sendcount,
	MPI_Datatype datatype,
	void * recvbuf,
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
```

# Compilation
Optimization flags have to be passed to the makefile via environment variables.
For example: `CC=mpicc CFLAGS=" -march=core-avx2 -Ofast -DNDEBUG " make -C lib`
The makefile in the lib/ subfolder generates both a static and dynamic version of the library.

# Tests
Tests have to be compiled in a similar way as lib.
For example: `CXX=mpicxx CXXFLAGS=" -march=core-avx2 -Ofast -DNDEBUG" CC=mpicc CFLAGS=" -march=core-avx2 -Ofast -DNDEBUG"`
