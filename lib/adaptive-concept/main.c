#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include <assert.h>
#include <string.h>

#include <mpi.h>

#include <mpi-util.h>
#include <common-util.h>
#include <posix-util.h>

#include "a2av.h"

#define _CAT(a, b) a ## b
#define CAT(a, b) _CAT(a, b)
typedef CAT(CAT(uint, _KEYBITS_), _t) KEY_T;

#define MPI_KEY_T CAT(CAT(MPI_UINT, _KEYBITS_ ), _T)

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

int VERBOSE = 0;

void sparse_sort (
	KEY_T * sendkeys,
	const ptrdiff_t sendcount,
	KEY_T * recvkeys,
	const ptrdiff_t recvcount,
	MPI_Comm comm )
{
	READENV(VERBOSE, atoi);

	int rank, rankcount;
	MPI_CHECK(MPI_Comm_rank(comm, &rank));
	MPI_CHECK(MPI_Comm_size(comm, &rankcount));

	const ptrdiff_t rankcountp1 = rankcount + 1;

	lsort(0, sizeof(KEY_T), 0, sendcount, sendkeys, NULL);

	KEY_T krmin = sendkeys[0], krmax = sendkeys[sendcount - 1];

	/* find key ranges */
	{
		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &krmin, 1, MPI_KEY_T, MPI_MIN, comm));
		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &krmax, 1, MPI_KEY_T, MPI_MAX, comm));
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
			MPI_CHECK(MPI_Allgather(&newkey, 1, MPI_KEY_T, query, 1, MPI_KEY_T, comm));

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

		MPI_CHECK(MPI_Allgather(&curkey, 1, MPI_KEY_T, global_startkey, 1, MPI_KEY_T, comm));
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

		a2av(sendkeys, scount, sstart, MPI_KEY_T, recvkeys, rcount, rstart, comm);
	}

	/* sort once more */
	lsort(0, sizeof(KEY_T), 0, recvcount, recvkeys, NULL);

#ifndef NDEBUG
	for(ptrdiff_t i = 1; i < recvcount; ++i)
		assert(recvkeys[i - 1] <= recvkeys[i]);
#endif
}

int main (
	const int argc,
	const char * argv [])
{
	MPI_CHECK(MPI_Init((int *)&argc, (char ***)&argv));

	MPI_Comm comm = MPI_COMM_WORLD;

	int r, rc;
	MPI_CHECK(MPI_Comm_rank(comm, &r));
	MPI_CHECK(MPI_Comm_size(comm, &rc));

	/* print MPI_TAG_UB */
	{
		int * v = NULL, flag = 0;
		MPI_CHECK(MPI_Comm_get_attr(comm, MPI_TAG_UB, (void **)&v, &flag));

		if (!r)
			printf("MPI_TAG_UB: %d (flag: %d)\n", *v, flag);
	}

	if (3 != argc)
	{
		if (!r)
			fprintf(stderr,
					"usage: %s <path/to/in-uint%d.raw> <path/to/output.raw>\n",
					argv[0], _KEYBITS_);

		MPI_CHECK(MPI_Finalize());

		return EXIT_FAILURE;
	}

	const ptrdiff_t esz = _KEYBITS_ / 8;

	/* item count */
	ptrdiff_t ic = 0;

	/* count items */
	{
		MPI_File f;
		MPI_CHECK(MPI_File_open(comm, (char *)argv[1],
								MPI_MODE_RDONLY, MPI_INFO_NULL, &f));

		MPI_Offset fsz;
		MPI_CHECK(MPI_File_get_size(f, &fsz));
		assert(0 == fsz % esz);

		ic = fsz / esz;
	}

	if (!r)
		printf("processing %zd elements\n", ic);

	workload_t load = divide_workload(r, rc, ic);

	const ptrdiff_t rangec = load.count;

	void * keys = NULL;

	/* read keys */
	{
		MPI_File f = NULL;

		MPI_CHECK(MPI_File_open
				  (comm, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &f));

		POSIX_CHECK(keys = malloc(rangec * esz));

		MPI_CHECK(MPI_File_read_at_all
				  (f, load.start * esz, keys, rangec, MPI_KEY_T, MPI_STATUS_IGNORE));

		MPI_CHECK(MPI_File_close(&f));
	}

	double tbegin = MPI_Wtime();

	void * sortedkeys = NULL;
	POSIX_CHECK(sortedkeys = malloc(rangec * esz));

	int NTIMES = 1;
	READENV(NTIMES, atoi);

	for (int t = 0; t < NTIMES; ++t)
		sparse_sort(keys, load.count, sortedkeys, load.count, comm);

	double tend = MPI_Wtime();

	/* write to file */
	{
		MPI_File f;
		MPI_CHECK(MPI_File_open
				  (comm, argv[2], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f));

		MPI_CHECK(MPI_File_set_size(f, ic * esz));

		MPI_CHECK(MPI_File_write_at_all
				  (f, load.start * esz, sortedkeys, rangec, MPI_KEY_T, MPI_STATUS_IGNORE));

		MPI_CHECK(MPI_File_close(&f));
	}

	free(keys);

	MPI_CHECK(MPI_Finalize());

	if (!r)
		printf("%s: sorted %zd entries in %.3f ms. Bye.\n",
			   argv[0], ic, (tend - tbegin) * 1e3);

	return EXIT_SUCCESS;
}
