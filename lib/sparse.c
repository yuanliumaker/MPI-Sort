#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <assert.h>
#include <string.h>

#include "macros.h"
#include "a2av.h"

#define _CAT(a, b) a ## b
#define CAT(a, b) _CAT(a, b)

#define KEY_T CAT(CAT(uint, _KEYBITS_), _t)
#define MPI_KEY CAT(CAT(MPI_UINT, _KEYBITS_ ), _T)

void lsort (
	const int stable,
	const ptrdiff_t key_bytesize,
	const ptrdiff_t value_bytesize,
	const ptrdiff_t count,
	void * keys,
	void * values);

static ptrdiff_t exscan (
	const ptrdiff_t count,
	const ptrdiff_t * const in,
	ptrdiff_t * const out )
{
	ptrdiff_t s = 0;

	for (ptrdiff_t i = 0; i < count; ++i)
	{
		const ptrdiff_t v = in[i];

		out[i] = s;

		s += v;
	}

	return s;
}

static ptrdiff_t lb (
	const KEY_T * first,
	ptrdiff_t count,
	const KEY_T val)
{
	const KEY_T * const head = first;
	const KEY_T * it;
	ptrdiff_t step;

	while (count > 0)
	{
		it = first;
		step = count / 2;

		it += step;
		if (*it < val)
		{
			first = ++it;
			count -= step + 1;
		}
		else
			count = step;
	}

	assert(head <= first);

	return first - head;
}

static ptrdiff_t ub (
	const KEY_T * first,
	ptrdiff_t count,
	const KEY_T val)
{
	const KEY_T * const head = first;
	const KEY_T * it;
	ptrdiff_t step;

	while (count > 0)
	{
		it = first;
		step = count / 2;

		it += step;
		if (*it <= val)
		{
			first = ++it;
			count -= step + 1;
		}
		else
			count = step;
	}

	assert(head <= first);

	return first - head;
}

void CAT(sparse_u, _KEYBITS_) (
		const int stable,
		KEY_T * sendkeys,
		void * sendvals,
		const int sendcount,
		MPI_Datatype valtype,
		KEY_T * recvkeys,
		void * recvvals,
		const int recvcount,
		MPI_Comm comm)
{
	int rank, rankcount;
	MPI_CHECK(MPI_Comm_rank(comm, &rank));
	MPI_CHECK(MPI_Comm_size(comm, &rankcount));

	const ptrdiff_t rankcountp1 = rankcount + 1;

	int vsz;
	MPI_CHECK(MPI_Type_size(valtype, &vsz));

	lsort(stable, sizeof(KEY_T), vsz, sendcount, sendkeys, sendvals);

	KEY_T krmin = sendkeys[0], krmax = sendkeys[sendcount - 1];

	/* find key ranges */
	{
		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &krmin, 1, MPI_UINT64_T, MPI_MIN, comm));
		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &krmax, 1, MPI_UINT64_T, MPI_MAX, comm));
	}

	ptrdiff_t recvstart_rank[rankcountp1];

	/* compute recvstart_rank */
	{
		ptrdiff_t tmp = recvcount, myend = 0;
		MPI_CHECK(MPI_Scan(&tmp, &myend, 1, MPI_INT64_T, MPI_SUM, comm));

		recvstart_rank[0] = 0;
		MPI_CHECK(MPI_Allgather(&myend, 1, MPI_INT64_T, recvstart_rank + 1, 1, MPI_INT64_T, comm));
	}

	KEY_T global_startkey[rankcount];
	ptrdiff_t global_count[rankcount];

	/* find approximate splitters location */
	{
		size_t curkey = krmin, qcount = 0;

		for (KEY_T b = _KEYBITS_ - 1; b < _KEYBITS_; --b)
		{
			const KEY_T delta = ((KEY_T)1) << b;

			if (krmax - krmin < delta)
				continue;

			const KEY_T newkey = MIN(krmax, curkey + delta);

			KEY_T query[rankcount];
			MPI_CHECK(MPI_Allgather(&newkey, 1, MPI_KEY, query, 1, MPI_KEY, comm));

			ptrdiff_t partials[rankcount];
			for (int r = 0; r < rankcount; ++r)
				partials[r] = lb(sendkeys, sendcount, query[r]);

			ptrdiff_t answer = 0;
			MPI_CHECK(MPI_Reduce_scatter_block
					  (partials, &answer, 1, MPI_INT64_T, MPI_SUM, comm));

			if (answer <= recvstart_rank[rank])
			{
				curkey = newkey;
				qcount = answer;
			}
		}

		MPI_CHECK(MPI_Allgather(&curkey, 1, MPI_KEY, global_startkey, 1, MPI_KEY, comm));
		MPI_CHECK(MPI_Allgather(&qcount, 1, MPI_INT64_T, global_count, 1, MPI_INT64_T, comm));

		for (int r = 0; r < rankcount; ++r)
			assert(global_count[r] <= recvstart_rank[r]);
	}

	ptrdiff_t sstart[rankcountp1];

	/* compute sstart */
	{
		sstart[0] = 0;

		for (int r = 1; r < rankcount; ++r)
			sstart[r] = lb(sendkeys, sendcount, global_startkey[r]);

		sstart[rankcount] = sendcount;

		ptrdiff_t q[rankcount];
		memset(q, 0, sizeof(q));

		for (int r = 1; r < rankcount; ++r)
			/* mismatch -- we need to take some more */
			if (global_count[r] != recvstart_rank[r])
				q[r] = ub(sendkeys, sendcount, global_startkey[r]) - sstart[r];

		ptrdiff_t qstart[rankcount];
		memset(qstart, 0, sizeof(qstart));
		MPI_CHECK(MPI_Exscan(q, qstart, rankcount, MPI_INT64_T, MPI_SUM, comm));

		for (int r = 1; r < rankcount; ++r)
			if (global_count[r] != recvstart_rank[r])
				sstart[r] += MAX(0, MIN(q[r], recvstart_rank[r] - global_count[r] - qstart[r]));
	}

#ifndef NDEBUG
	ptrdiff_t check[rankcountp1];
	MPI_CHECK(MPI_Scan(sstart, check, rankcountp1, MPI_INT64_T, MPI_SUM, comm));

	if (rankcount == rank + 1)
		for (int r = 0; r <= rank; ++r)
			assert(recvstart_rank[r] == check[r]);
#endif

	/* send around */
	{
		ptrdiff_t scount[rankcount], rcount[rankcount];

		for (int r = 0; r < rankcount; ++r)
			scount[r] = sstart[r + 1] - sstart[r];

		MPI_CHECK(MPI_Alltoall(scount, 1, MPI_INT64_T, rcount, 1, MPI_INT64_T, comm));

		ptrdiff_t rstart[rankcount];
		const ptrdiff_t __attribute__((unused)) check = exscan(rankcount, rcount, rstart);
		assert(check == recvcount);

		/* keys */
		a2av(sendkeys, scount, sstart, MPI_KEY, recvkeys, rcount, rstart, comm);
		
		/* values */
		if (recvvals)
			a2av(sendvals, scount, sstart, valtype, recvvals, rcount, rstart, comm);
	}

	/* sort once more */
	lsort(stable, vsz, sizeof(KEY_T), recvcount, recvkeys, recvvals);

#ifndef NDEBUG
	for(ptrdiff_t i = 1; i < recvcount; ++i)
		assert(recvkeys[i - 1] <= recvkeys[i]);
#endif
}
