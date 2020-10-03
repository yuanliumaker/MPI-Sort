#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <assert.h>
#include <string.h>
#include <unistd.h>

#include <mpi.h>

#include "util.h"
#include "kernels.h"

int main (
    const int argc,
    const char * argv [])
{
    int CCO = 1, VERIFY = 1;
    READENV(CCO, atoi);
    READENV(VERIFY, atoi);

    MPI_CHECK(MPI_Init((int *)&argc, (char ***)&argv));

    MPI_Comm comm = MPI_COMM_WORLD;

    int r, rc;
    MPI_CHECK(MPI_Comm_rank(comm, &r));
    MPI_CHECK(MPI_Comm_size(comm, &rc));

    if (argc != 2)
    {
	fprintf(stderr,
		"usage: %s <path/to/meshdir>\n",
		argv[0]);

	return EXIT_FAILURE;
    }

    if (chdir(argv[1]))
	perror("changing working directory");

    __extension__ ptrdiff_t count_items (
	const char * p,
	const ptrdiff_t esz)
    {
	ptrdiff_t retval;

	if (!r)
	{
	    MPI_File f;
	    MPI_CHECK(MPI_File_open(MPI_COMM_SELF, (char *)p,
				    MPI_MODE_RDONLY, MPI_INFO_NULL, &f));

	    MPI_Offset fsz;
	    MPI_CHECK(MPI_File_get_size(f, &fsz));
	    DIE_UNLESS(0 == fsz % esz);

	    retval = fsz / esz;

	    MPI_CHECK(MPI_File_close(&f));
	}

	MPI_CHECK(MPI_Bcast(&retval, 1, MPI_INT64_T, 0, MPI_COMM_WORLD));

	return retval;
    }

    __extension__ void * read (
	const char * const p,
	const range_t r,
	MPI_Datatype t)
    {
	MPI_File f = NULL;

	MPI_CHECK(MPI_File_open
		  (comm, (char *)p, MPI_MODE_RDONLY, MPI_INFO_NULL, &f));

	int esz;
	MPI_CHECK(MPI_Type_size(t, &esz));

	void * buf;
	DIE_UNLESS(buf = malloc(r.count * (ptrdiff_t)esz));

	MPI_CHECK(MPI_File_read_at_all
		  (f, r.start * (ptrdiff_t)esz, buf, r.count, t, MPI_STATUS_IGNORE));

	MPI_CHECK(MPI_File_close(&f));

	return buf;
    }

    const ptrdiff_t nc = count_items("x.raw", sizeof(float));
    const ptrdiff_t ec = count_items("i0.raw", sizeof(int));

    const ptrdiff_t ic = ec * 4;

    if (!r)
	printf("nodes %zd elements %zd indices %zd\n", nc, ec, ic);

    const range_t erng = range_part(r, rc, ec);
    const range_t irng = (range_t) { .start = erng.start * 4, .count = erng.count * 4 };

    int * ibuf;
    DIE_UNLESS(ibuf = malloc(irng.count * sizeof(*ibuf)));

    /* load indices */
    {
	assert(erng.count * 4 == irng.count);

	const int * stream[4] =
	    {
		read("i0.raw", erng, MPI_INT),
		read("i1.raw", erng, MPI_INT),
		read("i2.raw", erng, MPI_INT),
		read("i3.raw", erng, MPI_INT)
	    };

	mux4(stream[0], stream[1], stream[2], stream[3], erng.count, ibuf);

	for (int c = 3; c >= 0; --c)
	    free((void *)stream[c]);
    }

    const range_t nrng = range_part(r, rc, nc + 1);

    range_t nrng_ibuf = minmax(ibuf, irng.count);

    /* 1 to get the supremum, 1 for the exclusive scan */
    nrng_ibuf.count += 2;

    ptrdiff_t *labels, *labels_final = NULL, *histo, *histo_final, *start, *start_final;
    DIE_UNLESS(labels = malloc(irng.count * sizeof(*labels)));
    DIE_UNLESS(histo = calloc(nrng_ibuf.count, sizeof(*histo)));
    DIE_UNLESS(histo_final = calloc(nrng.count, sizeof(*histo_final)));
    DIE_UNLESS(start = malloc(sizeof(*start) * nrng_ibuf.count));
    DIE_UNLESS(start_final = calloc(nrng.count, sizeof(*start_final)));

    /* entries in start_final are local, but globally they start at global_start */
    ptrdiff_t global_start = 0, local_count = 0;

    MPI_CHECK(MPI_Barrier(comm));

    double tbegin = MPI_Wtime();

    /* we temporarily use labels to store the order */
    counting_sort(
	nrng_ibuf.start, nrng_ibuf.start + nrng_ibuf.count, irng.count,
	ibuf, histo, start, labels);

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
}
