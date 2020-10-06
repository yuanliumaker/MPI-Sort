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
    const void * recvkeys,
    const void * recvvals,
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

    ptrdiff_t * global_bas = malloc(keyrange_count * sizeof(*global_bas));
    MPI_CHECK(MPI_Allreduce(start, global_bas, keyrange_count, MPI_INT64_T, MPI_SUM, comm));

    {
	ptrdiff_t * tmp = calloc(sizeof(*tmp), keyrange_count);
	MPI_CHECK(MPI_Exscan(start, tmp, keyrange_count, MPI_INT64_T, MPI_SUM, comm));

	for (ptrdiff_t i = 0; i < keyrange_count; ++i)
	    global_bas[i] += tmp[i];
    }

    for (int rr = 0; rr < rc; ++rr)
    {
	MPI_CHECK(MPI_Barrier(comm));
	if (rr == r)
	{
	    printf("rank %d (rangecount: %zd\n", r, keyrange_count);
	    for (int d = 0; d < keyrange_count; ++d)
		if (114 == d)
		printf("%zd:%zd ", keyrange.begin + d, global_bas[d]);
	    printf("\n");
	}

	MPI_CHECK(MPI_Barrier(comm));
    }

    const ptrdiff_t * upperbound (
	const ptrdiff_t * first,
	const ptrdiff_t * last,
	const ptrdiff_t val)
    {
	const ptrdiff_t * it;
	ptrdiff_t count, step;
	count = last - first;

	while (count > 0)
	{
	    it = first;
	    step = count / 2;

	    it += step;

	    if (!(val < *it))
	    {
		first = ++it;
		count -= step + 1;
	    }
	    else
		count = step;
	}

	return first;
    }

    ptrdiff_t find_hec_send (const int rr)
    {

    }

#if 0
    const range_t keyrange = range_part(r, rc, nc + 1);

    range_t nrng_ibuf = minmax(ibuf, irng.count);

    /* 1 to get the supremum, 1 for the exclusive scan */
    nrng_ibuf.count += 2;

    /* start_final is indexed from 0 on any MPI task
       howevery globally it starts at global_start */
    ptrdiff_t global_start = 0, local_count = 0;


    MPI_CHECK(MPI_Barrier(comm));

    double tbegin = MPI_Wtime();


#ifndef NDEBUG
    int * nodes_debug = malloc(irng.count * sizeof(*nodes_debug));

    for (ptrdiff_t i = 0 ; i < irng.count; ++i)
    {
	nodes_debug[i] = ibuf[labels[i]];

	assert(nodes_debug[i] >= 0 && nodes_debug[i] < nc);

	if (i)
	    assert(nodes_debug[i - 1] <= nodes_debug[i]);
    }

    __extension__ void sanchk_debug (
	const int * const mynodes,
	const ptrdiff_t count,
	const ptrdiff_t rank )
    {
	for (ptrdiff_t i = 0; i < count; ++i)
	    assert(mynodes[i] >= 0 && mynodes[i] < nc);

	const range_t range = range_part(rank, rc, nc + 1);

	for (ptrdiff_t i = 0; i < count; ++i)
	    assert(mynodes[i] >= range.start && mynodes[i] < range.start + range.count);
    }
#endif

    /* we use order (into labels) to compute the actual labels (i.e. the global element id) */
    for (ptrdiff_t i = 0; i < irng.count; ++i)
	labels[i] = labels[i] / 4 + erng.start;

    /* copy the intersection to histo_final */
    {
	const ptrdiff_t v0 = MAX(nrng_ibuf.start, nrng.start);
	const ptrdiff_t v1 = MIN(nrng_ibuf.start + nrng_ibuf.count, nrng.start + nrng.count);

	memcpy(histo_final + v0 - nrng.start,
	       histo + v0 - nrng_ibuf.start,
	       sizeof(*histo) * MAX(0, v1 - v0));
    }

    /* identify non-zero entries in my histogram that go to remote nodes */
    {
	const int vmin = nrng_ibuf.start;
	const int vsup = vmin + nrng_ibuf.count;
	const int vmax = vsup - 1;
	const int vlen = vmax - vmin;

	/* histogram entry count to send around */
	int hec_send[rc], hec_recv[rc];

	/* build hec_send, receive hec_recv */
	{
	    for (int rr = 0; rr < rc; ++rr)
	    {
		const range_t range = range_part(rr, rc, nc + 1);

		const int slot0 = MIN(MAX(range.start - vmin, 0), vlen);
		const int slot1 = MIN(MAX(range.start + range.count - vmin, 0), vlen);

		int c = 0;
		for (int s = slot0; s < slot1; ++s)
		    c += !!histo[s];

		hec_send[rr] = c * (rr != r);
	    }

	    MPI_CHECK(MPI_Alltoall(hec_send, 1, MPI_INT, hec_recv, 1, MPI_INT, comm));
	}

	/* send and receive histogram entries */
	{
	    ptrdiff_t msgstart_send[rc], msgstart_recv[rc];

	    const ptrdiff_t itemc_send = exscan_int32(rc, hec_send, msgstart_send);
	    const ptrdiff_t itemc_recv = exscan_int32(rc, hec_recv, msgstart_recv);

	    typedef struct { int key, val; } keyval_t;

	    MPI_Datatype KEYVAL;
	    MPI_CHECK(MPI_Type_contiguous(sizeof(keyval_t), MPI_BYTE, &KEYVAL));
	    MPI_CHECK(MPI_Type_commit(&KEYVAL));

	    keyval_t * msgheader_recv;
	    DIE_UNLESS(msgheader_recv = malloc(sizeof(keyval_t) * itemc_recv));

	    const int msgc_recv = nzcount(hec_recv, rc);
	    MPI_Request reqs[msgc_recv];
	    int source_recv[msgc_recv];

	    /* post recv requests */
	    {
		int c = 0;
		for (int rr = 0; rr < rc; ++rr)
		    if (r != rr && hec_recv[rr])
		    {
			source_recv[c] = rr;

			MPI_CHECK(MPI_Irecv(msgheader_recv + msgstart_recv[rr], hec_recv[rr], KEYVAL,
					    rr, r + rc * rr, comm, reqs + c++));
		    }

		assert(msgc_recv == c);
	    }

	    /* populate and send nonzero entries */
	    for (int rr = 0; rr < rc; ++rr)
		if (r != rr && hec_send[rr])
		{
		    const range_t range = range_part(rr, rc, nc + 1);

		    const int slot0 = MIN( MAX(range.start - vmin, 0), vlen);
		    const int slot1 = MIN( MAX(range.start + range.count - vmin, 0), vlen);

		    keyval_t * msg;
		    DIE_UNLESS(msg = malloc(sizeof(keyval_t) * hec_send[rr]));

		    int c = 0;

		    for (int s = slot0; s < slot1; ++s)
		    {
			const int h = histo[s];

			if (h)
			    msg[c++] = (keyval_t){ .key = s + vmin, .val = h };
		    }

		    MPI_CHECK(MPI_Send(msg, c, KEYVAL, rr, rr + rc * r, comm));

		    free(msg);
		}

	    /* receive nonzero entries, update my histogram */
	    for(int imsg = 0; imsg < msgc_recv; ++imsg)
	    {
		MPI_CHECK(MPI_Wait(reqs + imsg, MPI_STATUS_IGNORE));

		const int rr = source_recv[imsg];

		const keyval_t * const msg = msgheader_recv + msgstart_recv[rr];

		const int n = hec_recv[rr];
		for (int i = 0; i < n; ++i)
		{
		    const keyval_t kv = msg[i];

		    const ptrdiff_t entry = kv.key - nrng.start;

		    histo_final[entry] += kv.val;
		}
	    }

	    MPI_CHECK(MPI_Type_free(&KEYVAL));

	    /* exclusive scan on global histogram */
	    {
		local_count = exscan(nrng.count, histo_final, start_final);

		MPI_CHECK(MPI_Exscan(&local_count, &global_start, 1, MPI_INT64_T, MPI_SUM, comm));

		DIE_UNLESS(labels_final = malloc(sizeof(*labels_final) * local_count));
	    }

	    ptrdiff_t * offsets;
	    DIE_UNLESS(offsets = malloc(sizeof(*offsets) * nrng.count));
	    memcpy(offsets, start_final, sizeof(*offsets) * nrng.count);

	    /* populate n2e with the local items */
	    {
		const ptrdiff_t v0 = MAX(nrng_ibuf.start, nrng.start);
		const ptrdiff_t v1 = MIN(nrng_ibuf.start + nrng_ibuf.count, nrng.start + nrng.count);

		for (int v = v0; v < v1; ++v)
		{
		    const ptrdiff_t idst = v - nrng.start;
		    const ptrdiff_t isrc = v - nrng_ibuf.start;

		    memcpy(labels_final + offsets[idst], labels + start[isrc], sizeof(*labels_final) * histo[isrc]);

		    offsets[idst] += histo[isrc];
		}
	    }

	    /* send and receive labels around */
	    {
		struct { ptrdiff_t * msg; MPI_Request req; } stage[rc];

		memset(stage, 0, sizeof(stage));

		__extension__ void post_and_send (const int d)
		{
		    /* post recv */
		    {
			const int rsrc = (r + d) % rc;

			ptrdiff_t lblc = 0;

			/* compute message label count */
			for (ptrdiff_t i = 0; i < hec_recv[rsrc]; ++i)
			    lblc += msgheader_recv[msgstart_recv[rsrc] + i].val;

			DIE_UNLESS(stage[d].msg = malloc(sizeof(ptrdiff_t) * lblc));

			/* post recv */
			MPI_CHECK(MPI_Irecv(stage[d].msg, lblc, MPI_INT64_T, rsrc, d, comm, &stage[d].req));
		    }

		    /* send */
		    {
			const int rdst = (r - d + rc) % rc;

			const range_t range = range_part(rdst, rc, nc + 1);

			const int slot0 = MIN(MAX(range.start - vmin, 0), vlen);
			const int slot1 = MIN(MAX(range.start + range.count - vmin, 0), vlen);

			MPI_CHECK(MPI_Send(labels + start[slot0], start[slot1] - start[slot0], MPI_INT64_T, rdst, d, comm));
		    }
		}

		__extension__ void wait_and_update (const int d)
		{
		    MPI_CHECK(MPI_Wait(&stage[d].req, MPI_STATUS_IGNORE));

		    const int rsrc = (r + d) % rc;

		    /* scatter labels from message */
		    const ptrdiff_t n = hec_recv[rsrc];

		    ptrdiff_t base = 0;
		    for (ptrdiff_t i = 0; i < n; ++i)
		    {
			const keyval_t header = msgheader_recv[msgstart_recv[rsrc] + i];

			const ptrdiff_t slot = header.key - nrng.start;

			memcpy(labels_final + offsets[slot],
			       stage[d].msg + base, header.val * sizeof(ptrdiff_t));

			offsets[slot] += header.val;
			base += header.val;
		    }

		    free(stage[d].msg);
		    stage[d].msg = NULL;
		    stage[d].req = -1;
		}

		if (CCO)
		    for (int d = 1; d < rc; ++d)
		    {
			post_and_send(d);

			if (CCO < d)
			    wait_and_update(d - CCO);

			if (rc - 1 == d)
			    for (int i = CCO - 1; i >= 0; --i)
				wait_and_update(d - i);
		    }
		else
		    for (int d = 1; d < rc; ++d)
		    {
			post_and_send(d);
			wait_and_update(d);
		    }

#ifndef NDEBUG
		for (ptrdiff_t n = 0; n < nrng.count; ++n)
		    assert(offsets[n] == start_final[n] + histo_final[n]);
#endif
	    }

	    free(offsets);
	    free(msgheader_recv);
	}
    }

    double tend = MPI_Wtime();

    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &tbegin, 1, MPI_DOUBLE, MPI_MIN, comm));
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &tend  , 1, MPI_DOUBLE, MPI_MAX, comm));

    /* verify against reference data */
    if (VERIFY)
    {
	/* n2e_start.raw */
	{
	    ptrdiff_t * ref = read("n2e_start.raw", nrng, MPI_INT64_T);

	    /* in order to verify we need to subtract the base */
	    for (ptrdiff_t i = 0; i < nrng.count; ++i)
		ref[i] -= global_start;

	    if(memcmp(start_final, ref, sizeof(*ref) * nrng.count))
		printf("ERROR: VERIFICATION FAILED: start does not match with n2e_start.raw\n");

	    MPI_CHECK(MPI_Barrier(comm));

	    free(ref);

	    if (!r)
		printf("SUCCESSFULLY VERIFIED AGAINST n2e_start.raw\n");

	}

	/* n2e.raw */
	{
	    ptrdiff_t * n2e_ref =
		read("n2e.raw", (range_t){.start = global_start, .count = local_count}, MPI_INT64_T);

	    for (ptrdiff_t i = 0; i < nrng.count; ++i)
	    {
		const ptrdiff_t base = start_final[i];
		const ptrdiff_t count = histo_final[i];

		__extension__ int compar (const void * a, const void  *b) { return *(ptrdiff_t *)b - *(ptrdiff_t *)a; }

		qsort(labels_final + base, count, sizeof(ptrdiff_t), compar);
		qsort(n2e_ref + base, count, sizeof(ptrdiff_t), compar);

		if (memcmp(labels_final + base, n2e_ref + base, sizeof(ptrdiff_t) * count))
		{
		    printf("VERIFICATION FAILED: rank %d: n2e does not match with n2e.raw at position %zd\n", r, i + nrng.start);

		    return EXIT_FAILURE;
		}
	    }

	    free(n2e_ref);

	    MPI_CHECK(MPI_Barrier(comm));

	    if (!r)
		printf("SUCCESSFULLY VERIFIED AGAINST n2e.raw\n");

	}
    }

#ifndef NDEBUG
    free(nodes_debug);
#endif
    free(labels_final);
    free(start_final);
    free(start);
    free(histo_final);
    free(histo);
    free(labels);
    free(ibuf);

    MPI_CHECK(MPI_Finalize());

    if (!r)
	printf("n2e: found %zd entries in %.3f ms. Bye.\n", 4 * ec, (tend - tbegin) * 1e3);

    return EXIT_SUCCESS;
#endif

    return MPI_SUCCESS;
}
