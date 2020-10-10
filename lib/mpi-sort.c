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
    const void * sendvals0,
    const void * sendvals1,
    const int sendcount,
    MPI_Datatype valtype0,
    MPI_Datatype valtype1,
    KEY_T * recvkeys,
    void * recvvals0,
    void * recvvals1,
    const int recvcount,
    MPI_Comm comm)
{
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

    ptrdiff_t * global_start = malloc(keyrange_count * sizeof(*global_start));
    MPI_CHECK(MPI_Allreduce(start, global_start, keyrange_count, MPI_INT64_T, MPI_SUM, comm));

    ptrdiff_t recvstart_rank[rc + 1];

    /* compute recvstart_rank */
    {
	ptrdiff_t tmp = recvcount, myend = 0;
	MPI_CHECK(MPI_Scan(&tmp, &myend, 1, MPI_INT64_T, MPI_SUM, comm));

	recvstart_rank[0] = 0;
	MPI_CHECK(MPI_Allgather(&myend, 1, MPI_INT64_T, recvstart_rank + 1, 1, MPI_INT64_T, comm));
    }

    if (recvvals0 || recvvals1)
    {
	KEY_T * sortedkeys = malloc(sizeof(*sortedkeys) * sendcount);
	gather(sizeof(KEY_T), sendcount, sendkeys, order, sortedkeys);

	ptrdiff_t * recv_start = calloc(sizeof(*recv_start), keyrange_count);

#ifndef NDEBUG
	ptrdiff_t * recv_histo = calloc(sizeof(*recv_histo), keyrange_count);
#endif
	/* compute recv_start */
	{
#if 1
	    const ptrdiff_t t0 = recvstart_rank[r + 0];
	    const ptrdiff_t t1 = recvstart_rank[r + 1];

	    const ptrdiff_t k0 =
		lowerbound(global_start, global_start + keyrange_count, t0);

	    const ptrdiff_t k1 =
		lowerbound(global_start, global_start + keyrange_count, t1);

	    const ptrdiff_t first = MAX(0, k0 - 1);
	    const ptrdiff_t last = MAX(0, k1 - 1);

	    recv_start[first] = global_start[k0] - t0;

	    for(ptrdiff_t i = first + 1; i < last; ++i)
		recv_start[i] = global_start[i + 1] - global_start[i];

	    recv_start[last] += t1 - global_start[MAX(k0, k1 - 1)];

#else
	    ptrdiff_t first = 0, last = keyrange_count - 1;

	    if (r)
 		first = lowerbound(global_start, global_start + keyrange_count, recvstart_rank[r]);

	    if (r != rc - 1)
	    {
		last = -1 + lowerbound(global_start, global_start + keyrange_count, recvstart_rank[r + 1]);
		//last = MAX(first, last);
	    }


	    if (first)
		recv_start[first - 1] = global_start[first] - recvstart_rank[r];

	    for(ptrdiff_t i = first; i < last; ++i)
		recv_start[i] = global_start[i + 1] - global_start[i];

	    recv_start[last] += recvstart_rank[r + 1] - global_start[MAX(first, last)];

	    if (1 == r)
		printf("first: %zd, last %zd -> count %zd %zd (contribs: %zd, %zd)\n",
		       first, last, recv_start[0], recv_start[1],
		       global_start[first] - recvstart_rank[r],
		       recvstart_rank[r + 1] - global_start[last]
		    );

	    assert(first - 1 >= 0 || !r);
	    assert(last + 2 <= keyrange_count || rc - 1 == r);
#endif
#ifndef NDEBUG
	    {
		memcpy(recv_histo, recv_start, sizeof(*recv_histo) * keyrange_count);

		ptrdiff_t s = 0;
		for (ptrdiff_t i = 0; i < keyrange_count; ++i)
		    s += recv_histo[i];

		if (recvcount != s)
		    printf("%zd vs %zd oopsa rank %d: targets: %zd and %zd, got keys: %zd -> %zd and %zd -> %zd (keyrange count %zd)\n",
			   recvcount, s, r, recvstart_rank[r], recvstart_rank[r + 1], first, global_start[first], last, global_start[last], keyrange_count);

		assert(recvcount == recvstart_rank[r + 1] - recvstart_rank[r]);
		assert(recvcount == s);
	    }
#endif
	    exscan(keyrange_count, recv_start, recv_start);
	}

	ptrdiff_t send_msgstart[rc + 1], send_msglen[rc], send_headlen[rc], recv_msglen[rc], recv_headlen[rc];
	/* compute msgstart, msglen */
	{
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
		    assert(recvstart_rank[rr] >= 0);

		for (int rr = 1; rr < rc; ++rr)
		    assert(recvstart_rank[rr - 1] <= recvstart_rank[rr]);

		assert(recvstart_rank[rc] == global_count);
	    }
#endif

	    /* compute send msg start */
	    send_msgstart[0] = 0;

	    for (int rr = 1; rr < rc; ++rr)
	    {
		const ptrdiff_t key =
		    lowerbound(global_bas, global_bas + keyrange_count, recvstart_rank[rr]);

		assert(recvstart_rank[rr]);

		/* i have no items for rank rr */
		if (!key)
		{
		    send_msgstart[rr] = 0;

		    continue;
		}

		assert(global_bas[key - 1] < recvstart_rank[rr] || key == keyrange_count);
		assert(global_bas[key] >= recvstart_rank[rr] || key == keyrange_count);

		if (key < keyrange_count)
		    send_msgstart[rr] =
			start[key - 1] + MIN(
			    recvstart_rank[rr] - global_bas[key - 1],
			    start[key] - start[key - 1]);
		else
		    send_msgstart[rr] = start[keyrange_count - 1];
	    }

	    send_msgstart[rc] = sendcount;

	    /* msg len */
	    for (int rr = 0; rr < rc; ++rr)
		send_msglen[rr] = send_msgstart[rr + 1] - send_msgstart[rr];

	    /* TODO: compute header length with binary searches */
	    for (int rr = 0; rr < rc; ++rr)
	    {
		send_headlen[rr] = rle(sortedkeys + send_msgstart[rr], send_msglen[rr], NULL, NULL);
		assert(send_headlen[rr] >= 0 && send_headlen[rr] <= keyrange_count);
	    }

#ifndef NDEBUG
	    for (int rr = 0; rr < rc; ++rr)
		assert(send_msgstart[rr] >= 0);

	    for (int rr = 1; rr < rc; ++rr)
		assert(send_msgstart[rr - 1] <= send_msgstart[rr]);

	    ptrdiff_t s = 0;

	    for (ptrdiff_t rr = 0; rr < rc; ++rr)
		s += send_msglen[rr];

	    assert(s == sendcount);
#endif

	    MPI_CHECK(MPI_Alltoall(send_headlen, 1, MPI_INT64_T, recv_headlen, 1, MPI_INT64_T, comm));
	    MPI_CHECK(MPI_Alltoall(send_msglen, 1, MPI_INT64_T, recv_msglen, 1, MPI_INT64_T, comm));

#ifndef NDEBUG
	    {
		ptrdiff_t s = 0;
		for (ptrdiff_t rr = 0; rr < rc; ++rr)
		    s += recv_msglen[rr];

		assert(s == recvcount);
	    }
#endif
	    free(global_bas);
	}

	/* send/recv messages around */
	{
	    int esz0 = 0, esz1 = 0;

	    if (recvvals0)
		MPI_CHECK(MPI_Type_size(valtype0, &esz0));

	    if (recvvals1)
		MPI_CHECK(MPI_Type_size(valtype1, &esz1));

	    const ptrdiff_t esz01 = esz0 + esz1;

	    MPI_Datatype VALUE01;
	    MPI_CHECK(MPI_Type_contiguous(esz01, MPI_BYTE, &VALUE01));
	    MPI_CHECK(MPI_Type_commit(&VALUE01));

	    MPI_Datatype MPI_KEY_T;
	    MPI_CHECK(MPI_Type_contiguous(sizeof(KEY_T), MPI_BYTE, &MPI_KEY_T));
	    MPI_CHECK(MPI_Type_commit(&MPI_KEY_T));

	    typedef struct
	    {
		void * values;
		KEY_T * keys;
		ptrdiff_t * lengths;
		MPI_Request requests[3];
	    } message_t;

	    message_t recv_msg[rc];

	    memset(recv_msg, 0, sizeof(recv_msg));

			__extension__ void homocheck(int val, uint8_t * d, ptrdiff_t n)
			{
			    for(ptrdiff_t i = 0; i < n; ++i)
			    {
				if (!(d[i] == val))
				    printf("HOMOCHECK rank %d oopsa %d vs %d at %d\n",
					   r, d[i], val, i);

				assert(d[i] == val);
			    }
			}

	    __extension__ void post_and_send (const int d)
	    {
		/* post recv */
		{
		    const int rsrc = (r + d) % rc;

		    const ptrdiff_t mlen = recv_msglen[rsrc];

		    if (mlen)
		    {
			message_t * m = recv_msg + d;

			const ptrdiff_t hlen = recv_headlen[rsrc];
			m->keys = malloc(sizeof(KEY_T) * hlen);
			MPI_CHECK(MPI_Irecv(m->keys, hlen, MPI_KEY_T, rsrc, 0 + 3 * d, comm, m->requests + 0));

			m->lengths = malloc(sizeof(ptrdiff_t) * hlen);
			MPI_CHECK(MPI_Irecv(m->lengths, hlen, MPI_INT64_T, rsrc, 1 + 3 * d, comm, m->requests + 1));

			const ptrdiff_t mlen = recv_msglen[rsrc];
			m->values = malloc(esz01 * mlen);
			MPI_CHECK(MPI_Irecv(m->values, mlen, VALUE01, rsrc, 2 + 3 * d, comm, m->requests + 2));
		    }
		}

		/* send */
		{
		    const int rdst = (r - d + rc) % rc;

		    const ptrdiff_t mlen = send_msglen[rdst];
		    assert(mlen >= 0 && mlen <= sendcount);

		    if (mlen)
		    {
			const ptrdiff_t hlen = send_headlen[rdst];
			const ptrdiff_t mstart = send_msgstart[rdst];
			assert(hlen >= 0 && hlen <= keyrange_count);
			assert(mstart >= 0 && mstart < sendcount);

			KEY_T * keys = malloc(sizeof(*keys) * hlen);
			ptrdiff_t * lengths = malloc(sizeof(*lengths) * hlen);

			const ptrdiff_t check = rle(sortedkeys + mstart, mlen, keys, lengths);
#ifndef NDEBUG

			{
			KEY_T * asd = sortedkeys + mstart;
			for(ptrdiff_t l = 0; l < check; ++l)
			{
			    homocheck(keys[l], asd, lengths[l]);
			    asd += lengths[l];
			}
			}

			{
			    assert(check == hlen);

			    for (ptrdiff_t i = 0; i < check; ++i)
				assert(keys[i] >= keyrange.begin && keys[i] < keyrange.end - 1);

			    ptrdiff_t s = 0;
			    for (ptrdiff_t i = 0; i < check; ++i)
				s += lengths[i];

			    assert(mlen == s);
			}
#endif

			MPI_CHECK(MPI_Send(keys, hlen, MPI_KEY_T, rdst, 0 + 3 * d, comm));
			MPI_CHECK(MPI_Send(lengths, hlen, MPI_INT64_T, rdst, 1 + 3 * d, comm));


			void * values = malloc(esz01 * mlen);

			if (recvvals0)
			    gather(esz0, mlen, sendvals0, order + mstart, values);

			if (recvvals1)
			    gather(esz1, mlen, sendvals1, order + mstart, esz0 * mlen + values);

#ifndef NDEBUG
			{
			    KEY_T * asd2 = values;
			    for(ptrdiff_t l = 0; l < check; ++l)
			    {
				homocheck(keys[l], asd2, lengths[l]);
				asd2 += lengths[l];
			    }

			    KEY_T * f = values;
			    assert(asd2 - f == mlen);
			}

#endif
			MPI_CHECK(MPI_Send(values, mlen, VALUE01, rdst, 2 + 3 * d, comm));

			free(lengths);
			free(keys);
			free(values);
		    }
		}
	    }

	    __extension__ void wait_and_update (const int d)
	    {
		const int rsrc = (r + d) % rc;

		if (!recv_msglen[rsrc])
		    return;

		message_t * m = recv_msg + d;

		MPI_CHECK(MPI_Waitall(3, m->requests, MPI_STATUSES_IGNORE));

		const ptrdiff_t hlen = recv_headlen[rsrc];
		const ptrdiff_t mlen = recv_msglen[rsrc];

		const void * v0 = m->values;
		const void * v1 = esz0 * mlen + m->values;

		for (int l = 0; l < hlen; ++l)
		{
		    const KEY_T k = m->keys[l] - keyrange.begin;
		    const ptrdiff_t c = m->lengths[l];

		    if (recvvals0)
			memcpy(recvvals0 + esz0 * recv_start[k], v0, esz0 * c);

		    if (recvvals1)
			memcpy(recvvals1 + esz1 * recv_start[k], v1, esz1 * c);

		    homocheck(m->keys[l], v1, c);
		    v0 += esz0 * c;
		    v1 += esz1 * c;
		    recv_start[k] += c;
#ifndef NDEBUG
		    /*{
			const uint8_t * data = recvvals1;
			for(int i = 0; i < c; ++i)
			{
			    if (!(data[i] == m->keys[l]))
				printf("rank %d oopsa %d vs %d at %d\n",
				       r, data[i], m->keys[l], i);
			    assert(data[i] == m->keys[l]);
			}
		    }*/
		    assert(recv_histo[k]);
		    recv_histo[k] -= c;
		    assert(recv_histo[k] >= 0);
#endif
		}

#ifndef NDEBUG
		{
		    ptrdiff_t s = 0;
		    for (ptrdiff_t l = 0; l < hlen; ++l)
			s += m->lengths[l];

		    assert(mlen == s);
		}
#endif

		free(m->values);
		free(m->lengths);
		free(m->keys);

		memset(m, 0, sizeof(*m));
	    }

	    int CCO = 0;
	    READENV(CCO, atoi);

	    for (int d = 0; d < rc; ++d)
	    {
		post_and_send(d);

		if (CCO - 1 < d)
		    wait_and_update(d - CCO);

		if (rc - 1 == d)
		    for (int i = CCO - 1; i >= 0; --i)
			wait_and_update(d - i);
	    }

#ifndef NDEBUG
	    int fail = 0;
	    for (ptrdiff_t i = 0; i < keyrange_count; ++i)
	    {
		if (recv_histo[i])
		{
		    fail = 1;
		    printf("oopsa rank %d histo[%d] = %zd\n", r, i, recv_histo[i]);
		}
	    }
	    assert(!fail);
	    free(recv_histo);
#endif
	    MPI_CHECK(MPI_Type_free(&MPI_KEY_T));
	    MPI_CHECK(MPI_Type_free(&VALUE01));
	}

	free(recv_start);
	free(sortedkeys);
    }

    if (recvkeys != NULL)
    {
#if 1
	const ptrdiff_t t0 = recvstart_rank[r + 0];
	const ptrdiff_t t1 = recvstart_rank[r + 1];

	const ptrdiff_t k0 =
	    lowerbound(global_start, global_start + keyrange_count, t0);

	const ptrdiff_t k1 =
	    lowerbound(global_start, global_start + keyrange_count, t1);

	const ptrdiff_t first = MAX(0, k0 - 1);
	const ptrdiff_t last = MAX(0, k1 - 1);

	ptrdiff_t first_count = global_start[k0] - t0;
	ptrdiff_t last_count = t1 - global_start[MAX(k0, k1 - 1)];

	if (first == last)
	{
	    first_count += last_count;
	    last_count = 0;
	}

	KEY_T * dst = recvkeys;

	assert(global_start[first] >= 0);
	assert(recvstart_rank[r] >= 0);

	assert(dst - recvkeys + first_count <= recvcount);
	dst += fill(keyrange.begin + first, first_count, dst);

	for(ptrdiff_t i = first + 1; i < last; ++i)
	{
	    assert(dst - recvkeys + global_start[i + 1] - global_start[i] <= recvcount);
	    dst += fill(keyrange.begin + i, global_start[i + 1] - global_start[i], dst);
	}

	assert(dst - recvkeys + last_count <= recvcount);
	dst += fill(keyrange.begin + last, last_count, dst);

	assert(dst - recvkeys == recvcount);

#else
	ptrdiff_t first = 0, last = keyrange_count - 1;

	if (r)
	    first = lowerbound(global_start, global_start + keyrange_count, recvstart_rank[r]);

	if (r != rc - 1)
	{
	    last = -1 + lowerbound(global_start, global_start + keyrange_count, recvstart_rank[r + 1]);
	    //last = MAX(last, first);
	}

	KEY_T * dst = recvkeys;

	assert(global_start[first] >= 0);
	assert(recvstart_rank[r] >= 0);

	__extension__ ptrdiff_t cap(int n) { return MIN(n, recvcount - (dst - recvkeys)); }

	dst += fill(keyrange.begin + first - 1, cap(global_start[first] - recvstart_rank[r]), dst);

	for(ptrdiff_t i = first; i < last; ++i)
	    dst += fill(keyrange.begin + i, cap(global_start[i + 1] - global_start[i]), dst);

	dst += fill(keyrange.begin + last, cap(recvstart_rank[r + 1] - global_start[last]), dst);
#endif
	assert(dst - recvkeys == recvcount);
    }

    free(global_start);
    free(order);
    free(start);
    free(histo);

    return MPI_SUCCESS;
}
