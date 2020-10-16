# mpi-sort

MPI_Sort and MPI_Sort_bykey are kindly brought to you by a broken shoulder and a 2 weeks sick note.
The library exposes two functions:

```c
#include <mpi.h>

int MPI_Sort (
    const void * sendbuf,
    const int sendcount,
    MPI_Datatype datatype,
    const void * recvbuf,
    const int recvcount,
    MPI_Comm comm);

int MPI_Sort_bykey (
    const void * sendkeys,
    const void * sendvals,
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
