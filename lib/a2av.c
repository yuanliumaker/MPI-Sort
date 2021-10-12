#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>

#include <math.h>
#include <string.h>

#include "macros.h"
#include "a2av.h"

static uint16_t hash16 (const uint64_t data)
{
	const uint16_t t0 = (data >>  0) & 0xffff;
	const uint16_t t1 = (data >>  8) & 0xffff;
	const uint16_t t2 = (data >> 16) & 0xffff;
	const uint16_t t3 = (data >> 24) & 0xffff;

	return t0 ^ t1 ^ t2 ^ t3;
}

void a2av (
	const void * in,
	const ptrdiff_t * sendcounts,
	const ptrdiff_t * sdispls,
	MPI_Datatype type,
	void * out,
	const ptrdiff_t * recvcounts,
	const ptrdiff_t * rdispls,
	MPI_Comm comm )
{
	int r, rc;
	MPI_CHECK(MPI_Comm_rank(comm, &r));
	MPI_CHECK(MPI_Comm_size(comm, &rc));

	ptrdiff_t msglen_homo;

	/* compute msglen_homo:
	   average 95-percentiles of the message sizes first */
	{
		ptrdiff_t s[rc];
		memcpy(s, sendcounts, sizeof(s));

		__extension__ int compar(const void * a, const void * b) { return *(ptrdiff_t *)a - *(ptrdiff_t *)b; }
		qsort(s, rc, sizeof(ptrdiff_t), compar);

		msglen_homo = s[MAX(0, MIN(rc - 1, (int)round(0.95 * rc)))];

		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &msglen_homo, 1, MPI_INT64_T, MPI_SUM, comm));
		msglen_homo /= rc;
	}

	/* element size */
	ptrdiff_t esz;

	{
		int t;
		MPI_CHECK(MPI_Type_size(type, &t));
		esz = t;
	}

	/* send around keys via A2A */
	{
		const ptrdiff_t maxcount = MAX(1, 128 / esz);

		void *sendbuf, *recvbuf;
		DIE_UNLESS(sendbuf = malloc(esz * maxcount * rc));
		DIE_UNLESS(recvbuf = malloc(esz * maxcount * rc));

		for (ptrdiff_t base = 0; base < msglen_homo; base += maxcount)
		{
			const ptrdiff_t n = MIN(maxcount, msglen_homo - base);

			/* pack sendbuf */
			for (int rr = 0; rr < rc; ++rr)
				if (base < sendcounts[rr])
					memcpy(esz * n * rr + (int8_t *)sendbuf,
						   esz * (base + sdispls[rr]) + (int8_t *)in,
						   esz * MIN(n, sendcounts[rr] - base));

			/* relaxing irregularities -- sending around some invalid entries */
			MPI_CHECK(MPI_Alltoall(sendbuf, n, type, recvbuf, n, type, comm));

			/* unpack recvbuf */
			for (ptrdiff_t rr = 0; rr < rc; ++rr)
				if (base < recvcounts[rr])
					memcpy(esz * (base + rdispls[rr]) + (int8_t *)out,
						   esz * n * rr + (int8_t *)recvbuf,
						   esz * MIN(n, recvcounts[rr] - base));
		}

		free(recvbuf);
		free(sendbuf);
	}

	/* recv/send the remaining keys via P2P */
	{
		int reqc = 0;
		MPI_Request reqs[rc];

		for (int rr = 0; rr < rc; ++rr)
		{
			const ptrdiff_t rem = recvcounts[rr] - msglen_homo;

			if (rem > 0)
				MPI_CHECK(MPI_Irecv(esz * (rdispls[rr] + msglen_homo) + (int8_t *)out,
									rem, type, rr, hash16(rr + (ptrdiff_t)rc * r), comm, reqs + reqc++));
		}

		for (int rr = 0; rr < rc; ++rr)
		{
			const ptrdiff_t rem = sendcounts[rr] - msglen_homo;

			if (rem > 0)
				MPI_CHECK(MPI_Send(esz * (sdispls[rr] + msglen_homo) + (int8_t *)in,
								   rem, type, rr, hash16(r + (ptrdiff_t)rc * rr), comm));
		}

		/* wait now for receiving all messages */
		MPI_CHECK(MPI_Waitall(reqc, reqs, MPI_STATUSES_IGNORE));
	}
}
