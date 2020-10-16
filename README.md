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

* Compilation
Example: `CFLAGS=" -march=core-avx2 -Ofast -DNDEBUG " make -C lib`