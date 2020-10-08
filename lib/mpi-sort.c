#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <assert.h>
#include <string.h>

#include <mpi.h>

#include "util.h"
#include "kernels.h"

#define CAT(x, y) x ## y
#define NAME(x) CAT(MPI_Sort_bykey_, x)

int NAME(KEY_T) (
    const KEY_T * sendkeys,
    const void * sendvals,
    const int sendcount,
    MPI_Datatype keytype,
    MPI_Datatype valtype,
    KEY_T * recvkeys,
    void * recvvals,
    const int recvcount,
    MPI_Comm comm)
{
    if (MPI_IN_PLACE == sendkeys)
	sendkeys = recvkeys;

    if (MPI_IN_PLACE == sendvals)
	sendvals = recvvals;

    int r, rc;
    MPI_CHECK(MPI_Comm_rank(comm, &r));
    MPI_CHECK(MPI_Comm_size(comm, &rc));

    range_t keyrange;

    /* find key ranges */
    {
	keyrange = range_keys(sendkeys, sendcount);

	MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &keyrange.begin, 1, MPI_INT64_T, MPI_MIN, comm));
	MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &keyrange.end, 1, MPI_INT64_T, MPI_MAX, comm));

	/* more comfortable for exclusive scan */
	++keyrange.end;
    }

    const ptrdiff_t keyrange_count = keyrange.end - keyrange.begin;

    ptrdiff_t *histo = 0, *start = 0, *order = 0;
    DIE_UNLESS(histo = calloc(keyrange_count, sizeof(*histo)));
    DIE_UNLESS(start = malloc(keyrange_count * sizeof(*start)));
    DIE_UNLESS(order = malloc(sendcount * sizeof(*order)));

    /* local sort */
    const ptrdiff_t local_count =
	counting_sort(keyrange.begin, keyrange.end, sendcount, sendkeys, histo, start, order);

    ptrdiff_t global_base = 0;
    MPI_CHECK(MPI_Exscan(&local_count, &global_base, 1, MPI_INT64_T, MPI_SUM, comm));


    ptrdiff_t recvstart[rc + 1], msgstart[rc + 1], msglen[rc];

    /* compute msgstart, msglen, recvstart */
    {
	/* compute rank recv start */
	{
	    ptrdiff_t tmp = recvcount, myend = 0;
	    MPI_CHECK(MPI_Scan(&tmp, &myend, 1, MPI_INT64_T, MPI_SUM, comm));

	    recvstart[0] = 0;
	    MPI_CHECK(MPI_Allgather(&myend, 1, MPI_INT64_T, recvstart + 1, 1, MPI_INT64_T, comm));
	}

	ptrdiff_t * global_bas = calloc(sizeof(*global_bas), keyrange_count);

	/* global_bas : "vertical" exclusive scan */
	{
	    MPI_CHECK(MPI_Exscan(histo, global_bas, keyrange_count, MPI_INT64_T, MPI_SUM, comm));

	    ptrdiff_t * tmp = malloc(keyrange_count * sizeof(*tmp));

	    MPI_CHECK(MPI_Allreduce(histo, tmp, keyrange_count, MPI_INT64_T, MPI_SUM, comm));

	    exscan(keyrange_count, tmp, tmp);

	    for (ptrdiff_t i = 0; i < keyrange_count; ++i)
		global_bas[i] += tmp[i];

	    free(tmp);
	}

#ifndef NDEBUG
	{
	    ptrdiff_t global_count = 0;
	    MPI_CHECK(MPI_Allreduce(&local_count, &global_count, 1, MPI_INT64_T, MPI_SUM, comm));

	    for (int i = 0; i < keyrange_count; ++i)
		assert(global_bas[i] >= 0 && global_bas[i] <= global_count);

	    for (int i = 1; i < keyrange_count; ++i)
		assert(global_bas[i - 1] <= global_bas[i]);

	    for (int rr = 0; rr < rc; ++rr)
		assert(recvstart[rr] >= 0);

	    for (int rr = 1; rr < rc; ++rr)
		assert(recvstart[rr - 1] <= recvstart[rr]);

	    assert(recvstart[rc] == global_count);
	}
#endif

	/* compute msg start */
	msgstart[0] = 0;

	for (int rr = 1; rr < rc; ++rr)
	{
	    const ptrdiff_t key = lowerbound(global_bas, global_bas + keyrange_count, recvstart[rr]);

	    assert(key);
	    assert(global_bas[key - 1] < recvstart[rr] || key == keyrange_count);
	    assert(global_bas[key] >= recvstart[rr] || key == keyrange_count);

	    if (key < keyrange_count)
		msgstart[rr] =
		    start[key - 1] + MIN(
			recvstart[rr] - global_bas[key - 1],
			start[key] - start[key - 1]);
	    else
		msgstart[rr] = start[keyrange_count - 1];

	}

	msgstart[rc] = sendcount;

	/* msg len */
	for (int rr = 0; rr < rc; ++rr)
	    msglen[rr] = msgstart[rr + 1] - msgstart[rr];

#ifndef NDEBUG
	for (int rr = 0; rr < rc; ++rr)
	    assert(msgstart[rr] >= 0);

	for (int rr = 1; rr < rc; ++rr)
	    assert(msgstart[rr - 1] <= msgstart[rr]);

	ptrdiff_t s = 0;

	for (ptrdiff_t rr = 0; rr < rc; ++rr)
	    s += msglen[rr];

	assert(s == sendcount);
#endif
	MPI_CHECK(MPI_Alltoall(MPI_IN_PLACE, 1, MPI_INT64_T, msglen, 1, MPI_INT64_T, comm));

#ifndef NDEBUG
	{
	    ptrdiff_t s = 0;
	    for (ptrdiff_t rr = 0; rr < rc; ++rr)
		s += msglen[rr];

	    assert(s == recvcount);
	}
#endif
	free(global_bas);
    }

    /*printf("rank %d keyrange_count %d, sendcount %d, recvcount %d\n",
      r, keyrange_count, sendcount, recvcount);*/

    ptrdiff_t * global_start = malloc(keyrange_count * sizeof(*global_start));

    if (recvkeys != NULL)
    {
	MPI_CHECK(MPI_Allreduce(start, global_start, keyrange_count, MPI_INT64_T, MPI_SUM, comm));

	ptrdiff_t first = 0, last = keyrange_count - 1;

	if (r)
	    first = lowerbound(global_start, global_start + keyrange_count, recvstart[r]);

	if (r != rc - 1)
	    last = -1 + lowerbound(global_start, global_start + keyrange_count, recvstart[r + 1]);

	KEY_T * dst = recvkeys;

	dst += fill(keyrange.begin + first - 1, global_start[first] - recvstart[r], dst);

	for(ptrdiff_t i = first; i < last; ++i)
	    dst += fill(keyrange.begin + i, global_start[i + 1] - global_start[i], dst);

	dst += fill(keyrange.begin + last, recvstart[r + 1] - global_start[last], dst);

	assert(dst - recvkeys == recvcount);
    }

    free(global_start);

    free(order);
    free(start);
    free(histo);

    return MPI_SUCCESS;
}
