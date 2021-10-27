#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <assert.h>
#include <string.h>

#include <mpi.h>

#include <mpi-util.h>
#include <common-util.h>
#include <posix-util.h>

#include "../a2av.h"

typedef struct { ptrdiff_t begin, end; } range_t;

range_t range_keys (
	const uint32_t * const restrict in,
	const ptrdiff_t count)
{
	uint32_t lmin = in[0], lmax = in[0];

	for (ptrdiff_t i = 1; i < count; ++i)
	{
		const int s = in[i];

		lmin = MIN(lmin, s);
		lmax = MAX(lmax, s);
	}

	return (range_t){ lmin, 1 + (ptrdiff_t)lmax };
}

ptrdiff_t exscan (
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

ptrdiff_t lb_u32 (
	const uint32_t * first,
	ptrdiff_t count,
	const uint32_t val)
{
	const uint32_t * const head = first;
	const uint32_t * it;
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

ptrdiff_t ub_u32 (
	const uint32_t * first,
	ptrdiff_t count,
	const uint32_t val)
{
	const uint32_t * const head = first;
	const uint32_t * it;
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

void dsort_uint32 (
	uint32_t * sendkeys,
	const ptrdiff_t sendcount,
	uint32_t * recvkeys,
	const ptrdiff_t recvcount,
	MPI_Comm comm )
{
	READENV(VERBOSE, atoi);

	int rank, rankcount;
	MPI_CHECK(MPI_Comm_rank(comm, &rank));
	MPI_CHECK(MPI_Comm_size(comm, &rankcount));

	const ptrdiff_t rankcountp1 = rankcount + 1;

	/* local sort */
	__extension__ void lsort(uint32_t * keys)
	{
		__extension__ int compar (
			const void * a,
			const void * b )
		{
			return (int)((ptrdiff_t)*(uint32_t *)a - (ptrdiff_t)*(uint32_t *)b);
		}

		qsort(keys, sendcount, sizeof(*keys), compar);
#ifndef NDEBUG
		for (ptrdiff_t i = 1; i < sendcount; ++i)
			assert(keys[i] >= keys[i - 1]);
#endif
	}

	lsort(sendkeys);

	range_t keyrange = (range_t) { .begin = sendkeys[0], .end = sendkeys[sendcount - 1] + 1 };

	/* find key ranges */
	{
		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &keyrange.begin, 1, MPI_INT64_T, MPI_MIN, comm));
		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &keyrange.end, 1, MPI_INT64_T, MPI_MAX, comm));
	}

	ptrdiff_t recvstart_rank[rankcountp1];

	/* compute recvstart_rank */
	{
		ptrdiff_t tmp = recvcount, myend = 0;
		MPI_CHECK(MPI_Scan(&tmp, &myend, 1, MPI_INT64_T, MPI_SUM, comm));

		recvstart_rank[0] = 0;
		MPI_CHECK(MPI_Allgather(&myend, 1, MPI_INT64_T, recvstart_rank + 1, 1, MPI_INT64_T, comm));
	}

	ptrdiff_t global_startkey[rankcountp1], global_count[rankcountp1];

	/* find an adaptive histogram with rankcount bins */
	{
		ptrdiff_t curkey = keyrange.begin, qcount = 0;

		for (ptrdiff_t b = 31; b >= 0; --b)
		{
			const ptrdiff_t delta = 1ll << b;

			if (keyrange.end - keyrange.begin < delta)
				continue;

			const ptrdiff_t newkey = MIN(keyrange.end - 1, curkey + delta);

			ptrdiff_t query[rankcount];
			MPI_CHECK(MPI_Allgather(&newkey, 1, MPI_INT64_T, query, 1, MPI_INT64_T, comm));

			ptrdiff_t partials[rankcount];
			for (int r = 0; r < rankcount; ++r)
				partials[r] = lb_u32(sendkeys, sendcount, query[r]);
			
			ptrdiff_t answer = 0;
			MPI_CHECK(MPI_Reduce_scatter_block
					  (partials, &answer, 1, MPI_INT64_T, MPI_SUM, comm));

			if (answer <= recvstart_rank[rank])
			{
				curkey = newkey;
				qcount = answer;
			}
		}
			
		MPI_CHECK(MPI_Allgather(&curkey, 1, MPI_INT64_T, global_startkey, 1, MPI_INT64_T, comm));
		global_startkey[rankcount] = keyrange.end;

		MPI_CHECK(MPI_Allgather(&qcount, 1, MPI_INT64_T, global_count, 1, MPI_INT64_T, comm));
		global_count[rankcount] = sendcount;
	}

	ptrdiff_t sstart[rankcountp1];
	sstart[0] = 0;
	sstart[rankcount] = sendcount;

	/* refinement stage */
	for (int r = 1; r < rankcount; ++r)
	{
		const ptrdiff_t key = global_startkey[r];
		const ptrdiff_t gcount = global_count[r];
		const ptrdiff_t target = recvstart_rank[r];

		sstart[r] = lb_u32(sendkeys, sendcount, key);

		/* adjustment */
		if (gcount != target)
		{
			assert(sstart[r] == 0 || gcount);
			assert(gcount < target);
			
			const ptrdiff_t q = ub_u32(sendkeys, sendcount, key) - sstart[r];
			ptrdiff_t qstart = 0;
			MPI_CHECK(MPI_Exscan(&q, &qstart, 1, MPI_INT64_T, MPI_SUM, comm));

			sstart[r] += MAX(0, MIN(q, target - gcount - qstart));
		}
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

		a2av(sendkeys, scount, sstart, MPI_UNSIGNED, recvkeys, rcount, rstart, comm);
	}

	/* sort once more */
	lsort(recvkeys);

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
					"usage: %s <path/to/in-uint32.raw> <path/to/output.raw>\n",
					argv[0]);

		MPI_CHECK(MPI_Finalize());

		return EXIT_FAILURE;
	}

	const ptrdiff_t esz = 4;

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
				  (f, load.start * esz, keys, rangec, MPI_UNSIGNED, MPI_STATUS_IGNORE));

		MPI_CHECK(MPI_File_close(&f));
	}

	double tbegin = MPI_Wtime();

	void * sortedkeys = NULL;
	POSIX_CHECK(sortedkeys = malloc(rangec * esz));

	int NTIMES = 1;
	READENV(NTIMES, atoi);

	for (int t = 0; t < NTIMES; ++t)
		dsort_uint32(keys, load.count, sortedkeys, load.count, comm);

	double tend = MPI_Wtime();

	/* write to file */
	{
		MPI_File f;
		MPI_CHECK(MPI_File_open
				  (comm, argv[2], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f));

		MPI_CHECK(MPI_File_set_size(f, ic * esz));

		MPI_CHECK(MPI_File_write_at_all
				  (f, load.start * esz, sortedkeys, rangec, MPI_UNSIGNED, MPI_STATUS_IGNORE));

		MPI_CHECK(MPI_File_close(&f));
	}

	free(keys);

	MPI_CHECK(MPI_Finalize());

	if (!r)
		printf("%s: sorted %zd entries in %.3f ms. Bye.\n",
			   argv[0], ic, (tend - tbegin) * 1e3);

	return EXIT_SUCCESS;
}
