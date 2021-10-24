#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <assert.h>
#include <string.h>

#include <mpi.h>

#include <mpi-util.h>
#include <common-util.h>
#include <posix-util.h>

typedef uint16_t KEY_T;

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

#include "../common.h"
#include "../xtract.h"
#include "../csort-tuned-u16.h"

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

void dsort_uint32(
	const ptrdiff_t count, 
	uint32_t * const keys, 
	MPI_Comm comm)
{
	int rc, r;
	MPI_CHECK(MPI_Comm_size(comm, &rc));
	MPI_CHECK(MPI_Comm_rank(comm, &r));

		ptrdiff_t globalcount = 0;
		MPI_CHECK( MPI_Allreduce(&count, &globalcount, 1, MPI_INT64_T, MPI_SUM, comm));

#if 1
	__extension__ int compar (
		const void * a,
		const void * b )
	{
		return *(uint32_t *)a - *(uint32_t *)b;
	}

	qsort(keys, count, sizeof(*keys), compar);
#else
	uint16_t * tmp = NULL;
	POSIX_CHECK(tmp = malloc(count * sizeof(*tmp)));

	ptrdiff_t *histo = NULL, *start = NULL, *order = NULL;
	POSIX_CHECK(histo = malloc((count + 1) * sizeof(*histo)));
	POSIX_CHECK(start = malloc((count + 1) * sizeof(*start)));
	POSIX_CHECK(order = malloc((count + 1) * sizeof(*order)));
	
	const unsigned int minval = 0;
	const unsigned int supval = 65536;

	xtract_16_32 (0, count, keys, tmp);
	
	counting_sort (minval, supval, count, tmp, histo, start, order);

	uint32_t * shuffled = NULL;
	POSIX_CHECK(shuffled = malloc(sizeof(*shuffled) * count));
	
	for(ptrdiff_t i = 0; i < count; ++i)
		shuffled[i] = keys[order[i]];

	xtract_16_32 (1, count, shuffled, tmp);

	counting_sort (minval, supval, count, tmp, histo, start, order);

	for(ptrdiff_t i = 0; i < count; ++i)
		keys[i] = shuffled[order[i]];
	
	free(shuffled);
	free(order);
	free(start);
	free(histo);
	free(tmp);
#endif

	ptrdiff_t start = 0;
	MPI_CHECK(MPI_Exscan(&count, &start, 1, MPI_INT64_T, MPI_SUM, comm));

	ptrdiff_t pos = 0;
	ptrdiff_t res = 0;

	for (int32_t b = 31; b >= 0; --b)
	{
		const uint32_t delta = 1 << b;

		const uint32_t candidate = pos + delta;

		ptrdiff_t query[rc];
		MPI_CHECK(MPI_Allgather(&candidate, 1, MPI_INT64_T, query, 1, MPI_INT64_T, comm));

		ptrdiff_t results[rc];
		for (int rr = 0; rr < rc; ++rr)
		{
			const ptrdiff_t idx = lb_u32(keys, count, query[rr]);
			
			if (idx != count)
				results[rr] = MAX(0, idx - 1);
			else
				results[rr] = count;
		}

		ptrdiff_t candres = 0;
		MPI_CHECK(MPI_Reduce_scatter_block (
					  results, &candres, 1, MPI_INT64_T, MPI_SUM, comm));

		if (candres <= start)
		{
			pos = candidate;
			res = candres;
		}
	}

	for (int rr = 0; rr < rc; ++rr)
	{
	MPI_CHECK(MPI_Barrier(comm));

		if (rr == r)
		{
			printf("rank %d: start: %zd -> pos %zd, res %zd\n", r, start, pos, res);
			fflush(stdout);
		}
		MPI_CHECK(MPI_Barrier(comm));
	}
	ptrdiff_t finalpos[rc + 1];
	MPI_CHECK(MPI_Allgather(&pos, 1, MPI_INT64_T, finalpos, 1, MPI_INT64_T, comm));
	finalpos[rc] = ~0;

	ptrdiff_t finalidx[rc + 1];
		for (int rr = 0; rr <= rc; ++rr)
		{
			const ptrdiff_t idx = lb_u32(keys, count, finalpos[rr]);
			
			if (idx != count)
				finalidx[rr] = MAX(0, idx - 1);
			else
				finalidx[rr] = count;
		}

		ptrdiff_t histo[rc+1];
		for (int i = 0; i < rc; ++i)
			histo[i] = finalidx[i + 1] - finalidx[i];
		histo[rc] = 0;

		const int keyrange_count = rc + 1;

		ptrdiff_t * global_bas = calloc(sizeof(*global_bas), keyrange_count);

			/* global_bas : "vertical" exclusive scan */
			{
				MPI_CHECK(MPI_Exscan(histo, global_bas, keyrange_count, MPI_INT64_T, MPI_SUM, comm));

				ptrdiff_t * tmp = NULL;
				POSIX_CHECK(tmp = malloc(keyrange_count * sizeof(*tmp)));

				/* TODO: optimize as bcast global_bas + histo from rank rc - 1 */
				MPI_CHECK(MPI_Allreduce(histo, tmp, keyrange_count, MPI_INT64_T, MPI_SUM, comm));

				exscan_inplace(keyrange_count, tmp);

				for (ptrdiff_t i = 0; i < keyrange_count; ++i)
					global_bas[i] += tmp[i];

				free(tmp);
			}

			for (int rr = 0; rr < rc; ++rr)
			{
				MPI_CHECK(MPI_Barrier(comm));

				if (rr == r)
				{
					printf("rank %d:", r);
					for (int i = 0; i < rc; ++i)
						printf(" %03zd", global_bas[i]);
					
					printf("\n");
					fflush(stdout);
				}
				MPI_CHECK(MPI_Barrier(comm));
				
			}

			

	//int32_t * tmp = NULL;
	//POSIX_CHECK(tmp = malloc(n * sizeof(*tmp)));
	//memcpy(tmp, in, n * sizeof(*tmp));

	//qsort(tmp, n, sizeof(*tmp), compar);


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
	const ptrdiff_t bsz = ((ic + rc - 1) / rc);

	/* local element range */
	ptrdiff_t rangelo = (r + 0) * bsz;
	ptrdiff_t rangehi = (r + 1) * bsz;

	rangehi = MIN(rangehi, ic - 0);
	rangelo = MIN(rangelo, ic - 0);

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
		dsort_uint32(rangec, keys, comm);
	
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
