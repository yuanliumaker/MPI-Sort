#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <assert.h>
#include <string.h>

#include <mpi.h>

#include <mpi-util.h>
#include <common-util.h>
#include <posix-util.h>

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

ptrdiff_t exscan_inplace (
    const ptrdiff_t count,
    ptrdiff_t * const inout)
{
    ptrdiff_t s = 0;

    for (ptrdiff_t i = 0; i < count; ++i)
    {
		const ptrdiff_t v = inout[i];

		inout[i] = s;

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

ptrdiff_t lb_i64 (
	const int64_t * first,
	ptrdiff_t count,
	const int64_t val)
{
	const int64_t * const head = first;
	const int64_t * it;
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

int VERBOSE = 0;

void dsort_uint32 (
	const ptrdiff_t recvcount,
	const ptrdiff_t sendcount,
	uint32_t * const keys,
	MPI_Comm comm )
{
	READENV(VERBOSE, atoi);

	int rank, rankcount;
	MPI_CHECK(MPI_Comm_rank(comm, &rank));
	MPI_CHECK(MPI_Comm_size(comm, &rankcount));

	const ptrdiff_t rankcountp1 = rankcount + 1;

	/* local sort */
	{
		__extension__ int compar (
			const void * a,
			const void * b )
		{
			return *(uint32_t *)a - *(uint32_t *)b;
		}

		qsort(keys, sendcount, sizeof(*keys), compar);
	}

	range_t keyrange = (range_t) { .begin = keys[0], .end = keys[sendcount - 1] + 1 };

	/* find key ranges */
	{
		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &keyrange.begin, 1, MPI_INT64_T, MPI_MIN, comm));
		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &keyrange.end, 1, MPI_INT64_T, MPI_MAX, comm));
	}

	if (VERBOSE)
		for (int rr = 0; rr < rankcount; ++rr)
		{
			MPI_CHECK(MPI_Barrier(comm));

			if (rr == rank)
			{
				printf("rank %d:", rank);
				for (int i = 0; i < recvcount; ++i)
					printf(" %u", keys[i]);
				printf("\n");

			}
				fflush(stdout);
			MPI_CHECK(MPI_Barrier(comm));
		}

	ptrdiff_t recvstart_rank[rankcountp1];

	/* compute recvstart_rank */
	{
		ptrdiff_t tmp = recvcount, myend = 0;
		MPI_CHECK(MPI_Scan(&tmp, &myend, 1, MPI_INT64_T, MPI_SUM, comm));

		recvstart_rank[0] = 0;
		MPI_CHECK(MPI_Allgather(&myend, 1, MPI_INT64_T, recvstart_rank + 1, 1, MPI_INT64_T, comm));
	}

	ptrdiff_t local_histo[rankcountp1], local_start[rankcountp1];
	ptrdiff_t global_startkey_hi[rankcountp1], global_startkey_lo[rankcountp1];
	ptrdiff_t global_count_hi[rankcountp1], global_count_lo[rankcountp1];

	/* find an adaptive histogram with rankcount bins */
	{
		{
			ptrdiff_t curkey = keyrange.end, qcount = recvstart_rank[rankcount];

			for (ptrdiff_t b = 31; b >= 0; --b)
			{
				const ptrdiff_t delta = 1ll << b;

				if (keyrange.end - keyrange.begin < delta)
					continue;

				const ptrdiff_t newkey = MAX(0, curkey - delta);

				ptrdiff_t query[rankcount];
				MPI_CHECK(MPI_Allgather(&newkey, 1, MPI_INT64_T, query, 1, MPI_INT64_T, comm));

				ptrdiff_t counts[rankcount];
				for (int rr = 0; rr < rankcount; ++rr)
					/* we count all keys SMALLER THAN query */
					counts[rr] = lb_u32(keys, sendcount, query[rr]);

				ptrdiff_t newcount = 0;
				MPI_CHECK(MPI_Reduce_scatter_block
						  (counts, &newcount, 1, MPI_INT64_T, MPI_SUM, comm));

				if (newcount >= recvstart_rank[rank])
				{
					curkey = newkey;
					qcount = newcount;
				}
			}

			MPI_CHECK(MPI_Allgather(&curkey, 1, MPI_INT64_T, global_startkey_hi, 1, MPI_INT64_T, comm));
			global_startkey_hi[rankcount] = keyrange.end;

			MPI_CHECK(MPI_Allgather(&qcount, 1, MPI_INT64_T, global_count_hi, 1, MPI_INT64_T, comm));
			global_count_hi[rankcount] = sendcount;
		}

		{
			ptrdiff_t curkey = keyrange.begin, qcount = 0;

			for (ptrdiff_t b = 31; b >= 0; --b)
			{
				const ptrdiff_t delta = 1ll << b;

				if (keyrange.end - keyrange.begin < delta)
					continue;

				const ptrdiff_t newkey = MIN(keyrange.end, curkey + delta);

				ptrdiff_t query[rankcount];
				MPI_CHECK(MPI_Allgather(&newkey, 1, MPI_INT64_T, query, 1, MPI_INT64_T, comm));

				ptrdiff_t counts[rankcount];
				for (int rr = 0; rr < rankcount; ++rr)
					/* we count all keys SMALLER THAN query */
					counts[rr] = lb_u32(keys, sendcount, query[rr]);

				ptrdiff_t newcount = 0;
				MPI_CHECK(MPI_Reduce_scatter_block
						  (counts, &newcount, 1, MPI_INT64_T, MPI_SUM, comm));

				if (newcount <= recvstart_rank[rank])
				{
					curkey = newkey;
					qcount = newcount;
				}
			}

			MPI_CHECK(MPI_Allgather(&curkey, 1, MPI_INT64_T, global_startkey_lo, 1, MPI_INT64_T, comm));
			global_startkey_lo[rankcount] = keyrange.end;

			MPI_CHECK(MPI_Allgather(&qcount, 1, MPI_INT64_T, global_count_lo, 1, MPI_INT64_T, comm));
			global_count_lo[rankcount] = sendcount;
		}

		if (VERBOSE)
			for (int rr = 0; rr < rankcount; ++rr)
			{
				MPI_CHECK(MPI_Barrier(comm));

				if (rr == rank)
				{
					printf("rank %d: start: %zd -> global key hi/lo %zd %zd global count hi/lo %zd %zd",
						   rank, recvstart_rank[rank],
						   global_startkey_lo[rank], global_startkey_hi[rank],
						   global_count_lo[rank], global_count_hi[rank]);

					if (global_startkey_lo[rank] != global_startkey_hi[rank])
						printf("    must add %zd",
							   recvstart_rank[rr] - global_count_lo[rankcount]);

					printf("\n");
					fflush(stdout);
				}
				MPI_CHECK(MPI_Barrier(comm));
			}

		ptrdiff_t sstart[rankcount + 1];
		sstart[0] = 0;
		sstart[rankcount] = sendcount;

		/* conflict resolution */
		for (int rr = 1; rr < rankcount; ++rr)
		{
			const ptrdiff_t key = global_startkey_lo[rr];
			sstart[rr] = lb_u32(keys, sendcount, key);

			const ptrdiff_t gcount = global_count_lo[rr];
			const ptrdiff_t target = recvstart_rank[rr];

			if (gcount != target)
			{
				assert(gcount < target);
				const ptrdiff_t q = ub_u32(keys, sendcount, key) - sstart[rr];
				ptrdiff_t qstart = 0;
				MPI_CHECK(MPI_Exscan(&q, &qstart, 1, MPI_INT64_T, MPI_SUM, comm));

				const ptrdiff_t old = sstart[rr];
				sstart[rr] += MAX(0, MIN(q, target - gcount - qstart));
				//printf("sstart[%d] = %zd\n", rr, sstar
			}
		}

		ptrdiff_t gscount[rankcount];
		for (ptrdiff_t rr = 0; rr < rankcount; ++rr)
		{
			gscount[rr] = sstart[rr + 1] - sstart[rr];
			//printf("gscount %d = %zd\n", rr, gscount[rr]);
		}

		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, gscount, rankcount, MPI_INT64_T, MPI_SUM, comm));

		ptrdiff_t gsstart[rankcount + 1];
		gsstart[0] = 0;
		for (ptrdiff_t rr = 1; rr <= rankcount; ++rr)
			gsstart[rr] = gsstart[rr - 1] + gscount[rr - 1];

		if (!rank)
		{
			printf("gsstart: ");
			for (int rr = 0; rr <= rankcount; ++rr)
				printf(" %-3d", gsstart[rr]);
			printf("\n");

			printf("target : ");
			for (int rr = 0; rr <= rankcount; ++rr)
				printf(" %-3d", recvstart_rank[rr]);
			printf("\n");
		}

/*
		for (int rr = 0; rr < rankcount; ++rr)
			local_start[rr] = lb_u32(keys, sendcount, global_startkey[rr]);

		local_start[rankcount] = sendcount;

		for (int rr = 0; rr < rankcount; ++rr)
			local_histo[rr] = local_start[rr + 1] - local_start[rr];

		local_histo[rankcount] = 0;
*/	}
#if 0

	const ptrdiff_t keyrange_count = keyrange.end - keyrange.begin;

	ptrdiff_t send_msgstart[rankcountp1], send_msglen[rankcount], send_headlen[rankcount], recv_msglen[rankcount], recv_headlen[rankcount];

	/* compute msgstarts, msglens, and headlens */
	{
		ptrdiff_t * global_bas = calloc(sizeof(*global_bas), rankcountp1);

		/* global_bas : "vertical" exclusive scan */
		{
			MPI_CHECK(MPI_Exscan(local_histo, global_bas, rankcountp1, MPI_INT64_T, MPI_SUM, comm));

			ptrdiff_t * tmp = NULL;
			POSIX_CHECK(tmp = malloc(rankcountp1 * sizeof(*tmp)));

			/* TODO: optimize as bcast global_bas + histo from rank rc - 1 */
			MPI_CHECK(MPI_Allreduce(local_histo, tmp, rankcountp1, MPI_INT64_T, MPI_SUM, comm));

			exscan_inplace(rankcountp1, tmp);

			for (ptrdiff_t i = 0; i < rankcountp1; ++i)
				global_bas[i] += tmp[i];

			free(tmp);
		}


#ifndef NDEBUG
		{
			const ptrdiff_t local_count = sendcount;

			ptrdiff_t global_count = 0;
			MPI_CHECK(MPI_Allreduce(&local_count, &global_count, 1, MPI_INT64_T, MPI_SUM, comm));

			for (int i = 0; i <= rankcount; ++i)
				assert(global_bas[i] >= 0 && global_bas[i] <= global_count);

			for (int i = 1; i <= rankcount; ++i)
				assert(global_bas[i - 1] <= global_bas[i]);

			for (int rr = 0; rr <= rankcount; ++rr)
				assert(recvstart_rank[rr] >= 0);

			for (int rr = 1; rr <= rankcount; ++rr)
				assert(recvstart_rank[rr - 1] <= recvstart_rank[rr]);

			assert(recvstart_rank[rankcount] == global_count);
		}
#endif

		/* compute send msg start */
		send_msgstart[0] = 0;
#if 1
		for (int rr = 1; rr < rankcount; ++rr)
		{
			//const ptrdiff_t key = local_start[rr];
			ptrdiff_t key = lb_i64(global_bas, rankcountp1, recvstart_rank[rr]);

			if (key)
				key--;

			//send_msgstart[rr] = global_bas[key] + local_start[rr];

			//assert(global_bas[key - 1] < recvstart_rank[rr] || key == keyrange_count);
			//assert(global_bas[key] >= recvstart_rank[rr] || key == keyrange_count);

		}
		send_msgstart[rankcount] = sendcount;
#endif

		if (VERBOSE)
			for (int c = 0; c < 4; ++c)
			for (int rr = 0; rr < rankcount; ++rr)
			{
				MPI_CHECK(MPI_Barrier(comm));

				if (rr == rank)
				{
					if (0 == c)
					{
						printf("rank %d:", rank);
						for (int i = 0; i <= rankcount; ++i)
							printf(" %-3zd", global_bas[i]);
						printf("\n");
					}
					/*else if (1 == c)
					{
						printf("rank %d local count:", rank);
						for (int i = 0; i <= rankcount; ++i)
							printf(" %03zd", local_histo[i]);
						printf("\n");
					}
					else if (2 == c)
					{
						printf("rank %d local start:", rank);
						for (int i = 0; i <= rankcount; ++i)
							printf(" %03zd", local_start[i]);
						printf("\n");
						}*/

					else if (3 == c)
					{
						printf("rank %d msg_start:", rank);
						for (int i = 0; i <= rankcount; ++i)
							printf(" %03zd", send_msgstart[rr]);
						printf("\n");
					}
						fflush(stdout);
				}
				MPI_CHECK(MPI_Barrier(comm));
			}

		free(global_bas);
	}
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

	/* homogeneous blocksize */
	//const ptrdiff_t bsz = ((ic + rc - 1) / rc);
	workload_t load = divide_workload(r, rc, ic);

	/* local element range */
	//ptrdiff_t rangelo = (r + 0) * bsz;
	//ptrdiff_t rangehi = (r + 1) * bsz;

	//rangehi = MIN(rangehi, ic - 0);
	//rangelo = MIN(rangelo, ic - 0);
	const ptrdiff_t rangelo = load.start;
	const ptrdiff_t rangehi = load.start + load.count;

	const ptrdiff_t rangec = MAX(0, rangehi - rangelo);

	void * keys = NULL;

	/* read keys */
	{
		MPI_File f = NULL;

		MPI_CHECK(MPI_File_open
				  (comm, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &f));

		keys = malloc(rangec * esz);
		assert(keys);

		MPI_CHECK(MPI_File_read_at_all
				  (f, rangelo * esz, keys, rangec, MPI_UNSIGNED, MPI_STATUS_IGNORE));

		MPI_CHECK(MPI_File_close(&f));
	}

	double tbegin = MPI_Wtime();

	int NTIMES = 1;
	READENV(NTIMES, atoi);

	for (int t = 0; t < NTIMES; ++t)
		dsort_uint32(rangec, rangec, keys, comm);

	double tend = MPI_Wtime();

	/* write to file */
	{
		MPI_File f;
		MPI_CHECK(MPI_File_open
				  (comm, argv[2], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f));

		MPI_CHECK(MPI_File_set_size(f, ic * esz));

		MPI_CHECK(MPI_File_write_at_all
				  (f, rangelo * esz, keys, rangec, MPI_UNSIGNED, MPI_STATUS_IGNORE));

		MPI_CHECK(MPI_File_close(&f));
	}

	free(keys);

	MPI_CHECK(MPI_Finalize());

	if (!r)
		printf("%s: sorted %zd entries in %.3f ms. Bye.\n",
			   argv[0], ic, (tend - tbegin) * 1e3);

	return EXIT_SUCCESS;
}
