#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <math.h>
#include <assert.h>
#include <string.h>

#include <mpi.h>

#include "macros.h"
#include "common.h"
#include "static.h"

#define CAT(x, y) x ## y
#define NAME(x) CAT(dsort_, x)

int NAME(KEY_T) (
	const int stable,
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
	
	const double t0 = MPI_Wtime();

	DIE_UNLESS(sendcount >= 0 && recvcount >= 0);

	int r, rc;
	MPI_CHECK(MPI_Comm_rank(comm, &r));
	MPI_CHECK(MPI_Comm_size(comm, &rc));
	MPI_CHECK(MPI_Barrier(comm));
	if (!r)
	{
		printf("hello dsort\n");
		fflush(stdout);
	}
	MPI_CHECK(MPI_Barrier(comm));

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

	const double t1 = MPI_Wtime();

	/* local sort */
	const ptrdiff_t local_count =
		counting_sort(keyrange.begin, keyrange.end, sendcount, sendkeys, histo, start, order);

	const double t2 = MPI_Wtime();

	/* exclusive scan of global histogram */
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

	const double t3 = MPI_Wtime();

	MPI_CHECK(MPI_Barrier(comm));
	if (!r)
	{
		printf("so far local sort in %g s, global histo in + global recv scan in %g s\n",
			   t2 - t1, t3 - t2);
		fflush(stdout);
	}
	MPI_CHECK(MPI_Barrier(comm));

	double t4 = t3;

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
			const ptrdiff_t t0 = recvstart_rank[r + 0];
			const ptrdiff_t t1 = recvstart_rank[r + 1];

			const ptrdiff_t k0 =
				lowerbound(global_start, global_start + keyrange_count, t0);

			const ptrdiff_t k1 =
				lowerbound(global_start, global_start + keyrange_count, t1);

			const ptrdiff_t first = k0 - 1;
			const ptrdiff_t last = k1 - 1;

			if (first >= 0)
				recv_start[first] = global_start[k0] - t0;

			for(ptrdiff_t i = first + 1; i < last; ++i)
				recv_start[i] = global_start[i + 1] - global_start[i];

			if (last >= 0)
				recv_start[last] += t1 - global_start[MAX(k0, k1 - 1)];

#ifndef NDEBUG
			{
				memcpy(recv_histo, recv_start, sizeof(*recv_histo) * keyrange_count);

				assert(recvcount == recvstart_rank[r + 1] - recvstart_rank[r]);
				ASSERT_SUM(recvcount, keyrange_count, recv_histo);
			}
#endif
			exscan_inplace(keyrange_count, recv_start);
		}

		ptrdiff_t send_msgstart[rc + 1], send_msglen[rc], send_headlen[rc], recv_msglen[rc], recv_headlen[rc];
		/* compute msgstarts, msglens, and headlens */
		{
			ptrdiff_t * global_bas = calloc(sizeof(*global_bas), keyrange_count);

			/* global_bas : "vertical" exclusive scan */
			{
				MPI_CHECK(MPI_Exscan(histo, global_bas, keyrange_count, MPI_INT64_T, MPI_SUM, comm));

				ptrdiff_t * tmp = malloc(keyrange_count * sizeof(*tmp));

				/* TODO: optimize as bcast global_bas + histo from rank rc - 1 */
				MPI_CHECK(MPI_Allreduce(histo, tmp, keyrange_count, MPI_INT64_T, MPI_SUM, comm));

				exscan_inplace(keyrange_count, tmp);

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

			ASSERT_SUM(sendcount, rc, send_msglen);
#endif

			/* TODO: mux2 send_headlen and send_msglen into a single call */
			MPI_CHECK(MPI_Alltoall(send_headlen, 1, MPI_INT64_T, recv_headlen, 1, MPI_INT64_T, comm));
			MPI_CHECK(MPI_Alltoall(send_msglen, 1, MPI_INT64_T, recv_msglen, 1, MPI_INT64_T, comm));

			ASSERT_SUM(recvcount, rc, recv_msglen);

			free(global_bas);
		}

		t4 = MPI_Wtime();
		if (!r)
			printf("sending message around now... have passed %g s\n",
				   t4 - t0);

		
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

			__extension__ void post_and_send (
				const int d,
				MPI_Request * reqs)
			{
				/* post recv */
				{
					const int rsrc = stable ? d : (r + d) % rc;

					const ptrdiff_t mlen = recv_msglen[rsrc];

					if (mlen)
					{
						message_t * m = recv_msg + d;

						const ptrdiff_t hlen = recv_headlen[rsrc];
						m->keys = malloc(sizeof(KEY_T) * hlen);
						MPI_CHECK(MPI_Irecv(m->keys, hlen, MPI_KEY_T, rsrc, 0, comm, m->requests + 0));

						m->lengths = malloc(sizeof(ptrdiff_t) * hlen);
						MPI_CHECK(MPI_Irecv(m->lengths, hlen, MPI_INT64_T, rsrc, 1, comm, m->requests + 1));

						const ptrdiff_t mlen = recv_msglen[rsrc];
						m->values = malloc(esz01 * mlen);
						MPI_CHECK(MPI_Irecv(m->values, mlen, VALUE01, rsrc, 2, comm, m->requests + 2));
					}
				}

				/* send */
				{
					const int rdst = stable ? d : (r - d + rc) % rc;

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
			
						/* we RLE-compress the message */
						const ptrdiff_t runcount = rle(sortedkeys + mstart, mlen, keys, lengths);
#ifndef NDEBUG
						{
							const KEY_T * in = sortedkeys + mstart;

							for(ptrdiff_t l = 0; l < runcount; ++l)
							{
								ASSERT_CONSTANT(keys[l], lengths[l], in);
								in += lengths[l];
							}

							assert(runcount == hlen);

							for (ptrdiff_t i = 0; i < runcount; ++i)
								assert(keys[i] >= keyrange.begin && keys[i] < keyrange.end - 1);

							ASSERT_SUM(mlen, runcount, lengths);
						}
#endif
						if (reqs)
						{
							MPI_CHECK(MPI_Isend(keys, hlen, MPI_KEY_T, rdst, 0, comm, reqs + 0));
							MPI_CHECK(MPI_Isend(lengths, hlen, MPI_INT64_T, rdst, 1, comm, reqs + 1));
						}
						else
						{
							MPI_CHECK(MPI_Send(keys, hlen, MPI_KEY_T, rdst, 0, comm));
							MPI_CHECK(MPI_Send(lengths, hlen, MPI_INT64_T, rdst, 1, comm));
						}

						void * values = malloc(esz01 * mlen);

						if (recvvals0)
							gather(esz0, mlen, sendvals0, order + mstart, values);

						if (recvvals1)
							gather(esz1, mlen, sendvals1, order + mstart, esz0 * mlen + values);

#ifndef NDEBUG
						if (sendkeys == sendvals0 && !sendvals1 ||
							sendkeys == sendvals1 && !sendvals0 )
						{
							const KEY_T * const in0 = values;
							const KEY_T * in = in0;

							for(ptrdiff_t l = 0; l < runcount; ++l)
							{
								ASSERT_CONSTANT(keys[l], lengths[l], in);
								in += lengths[l];
							}

							assert(in - in0 == mlen);
						}
#endif
						if (reqs)
							MPI_CHECK(MPI_Isend(values, mlen, VALUE01, rdst, 2, comm, reqs + 2));
						else
							MPI_CHECK(MPI_Send(values, mlen, VALUE01, rdst, 2, comm));
						
						free(lengths);
						free(keys);
						free(values);
					}
				}
			}

			__extension__ void wait_and_update (const int d)
			{
				const int rsrc = stable ? d : (r + d) % rc;

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

#ifndef NDEBUG
					assert(c > 0);
					assert(recv_histo[k]);
					recv_histo[k] -= c;
					assert(recv_histo[k] >= 0);
#endif
					v0 += esz0 * c;
					v1 += esz1 * c;
					recv_start[k] += c;
				}

				ASSERT_SUM(mlen, hlen, m->lengths);

				free(m->values);
				free(m->lengths);
				free(m->keys);

				memset(m, 0, sizeof(*m));
			}		

			if (stable)
#if 0
				for (int d = 0; d < rc; ++d)
				{
					MPI_CHECK(MPI_Barrier(comm));
					if (!r)
					{
						printf("pass with d %d\n", d);
					fflush(stdout);
					}
					MPI_CHECK(MPI_Barrier(comm));

					const double t0 = MPI_Wtime();
					MPI_CHECK(MPI_Barrier(comm));
											
					/* we do not compress any message here, just call scatterv */
					int sendcounts[rc], displs[rc];
					
					if (d == r)
					{
						for (int i = 0; i < rc; ++i)
							sendcounts[i] = send_msglen[i];
					
						for (int i = 0; i < rc; ++i)
							displs[i] = send_msgstart[i];
					}
					
					const ptrdiff_t msglen = recv_msglen[d];

					KEY_T * rkeys = malloc(sizeof(*rkeys) * msglen);

					const double t1 = MPI_Wtime();
					MPI_CHECK(MPI_Barrier(comm));

					MPI_CHECK(MPI_Scatterv(sortedkeys, sendcounts, displs, MPI_KEY_T,
										   rkeys, msglen, MPI_KEY_T, d, comm));

					MPI_CHECK(MPI_Barrier(comm));
					const double t2 = MPI_Wtime();

					ptrdiff_t * positions = malloc(sizeof(*positions) * msglen);

					for (ptrdiff_t i = 0; i < msglen; ++i)
						positions[i] = recv_start[rkeys[i] - keyrange.begin]++;

#ifndef NDEBUG
					for (ptrdiff_t i = 0; i < msglen; ++i)
						--recv_histo[rkeys[i] - keyrange.begin];
#endif
					free(rkeys);

					const double t3 = MPI_Wtime();
					MPI_CHECK(MPI_Barrier(comm));

					if (!r)
					{
						printf("stable sort kernel timings: "
							   "init %g s, mpi-scatterv %g s, update %g s\n",
							   t1 - t0, t2 - t1, t3 - t2);
						fflush(stdout);
					}
					MPI_CHECK(MPI_Barrier(comm));
					
					/* gather and send values */
					for (int c = 0; c < 2; ++c)
					{
					const double t0B = MPI_Wtime();
					MPI_CHECK(MPI_Barrier(comm));

					const void * const srcbuf[] = { sendvals0, sendvals1 };
						const MPI_Datatype type[] = { valtype0, valtype1 };
						const ptrdiff_t esz[] = { esz0, esz1 };

						void * const dstbuf[] = { recvvals0, recvvals1 };

						if (!srcbuf[c] || !dstbuf[c])
							continue;

						void * sendbuf = NULL;

						if (d == r)
						{
							sendbuf = malloc(esz[c] * sendcount);

							gather(esz[c], sendcount, srcbuf[c], order, sendbuf);
						}

						void * recvbuf = malloc(esz[c] * msglen);

										const double t1B = MPI_Wtime();
					MPI_CHECK(MPI_Barrier(comm));
					MPI_CHECK(MPI_Scatterv(sendbuf, sendcounts, displs, type[c],
											   recvbuf, msglen, type[c], d, comm));
					MPI_CHECK(MPI_Barrier(comm));
					const double t2B = MPI_Wtime();
					MPI_CHECK(MPI_Barrier(comm));

						void * const dst = dstbuf[c];

						for (ptrdiff_t i = 0; i < msglen; ++i)
							memcpy(esz[c] * positions[i] + (int8_t *)dst,
								   esz[c] * i + (int8_t *)recvbuf,
								   esz[c]);
					
						free(recvbuf);
					
						if (sendbuf)
							free(sendbuf);
					MPI_CHECK(MPI_Barrier(comm));
					const double t3B = MPI_Wtime();
					MPI_CHECK(MPI_Barrier(comm));

					if (!r)
					{
						printf("DATA:::CHANNEL %d stable sort kernel timings: "
							   "init %g s, mpi-scatterv %g s, update %g s\n",
							   c, t1B - t0B, t2B - t1B, t3B - t2B);
						fflush(stdout);
					}
					MPI_CHECK(MPI_Barrier(comm));
					}

					free(positions);
				}
#elif 1 /* Allgather + large messages attempt */
			{
				int MPI_SORT_VERBOSE = 0;
				READENV(MPI_SORT_VERBOSE, atoi);

				MPI_SORT_VERBOSE &= !r;
				
				ptrdiff_t msglen_homo;
				
				/* compute average 95-percentiles of the message sizes first */
				{
					ptrdiff_t sizes[rc];
					memcpy(sizes, send_msglen, sizeof(sizes));

					__extension__ int compar(const void * a, const void * b) { return *(ptrdiff_t *)a - *(ptrdiff_t *)b; }
					qsort(sizes, rc, sizeof(*sizes), compar);

					msglen_homo = sizes[MAX(0, MIN(rc - 1, (int)round(0.95 * rc)))];

					MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &msglen_homo, 1, MPI_INT64_T, MPI_SUM, comm));
					msglen_homo /= rc;
					
					if (MPI_SORT_VERBOSE)
						printf("msglen_homo = %zd\n", msglen_homo);
				}

				KEY_T * dst = recvkeys;

				if (!dst)
					dst = malloc(sizeof(*dst) * recvcount);

				KEY_T * pdst = dst;
					
				/* send around keys via A2A */
				{
					const double t0 = MPI_Wtime();

					if (!r)
						printf("all to all begins...\n");
					
					enum { SENDCOUNT = 128 / sizeof(KEY_T) };

					KEY_T * sendbuf = malloc(sizeof(KEY_T) * SENDCOUNT * rc);
					KEY_T * recvbuf = malloc(sizeof(KEY_T) * SENDCOUNT * rc);
					
					for (ptrdiff_t base = 0; base < msglen_homo; base += SENDCOUNT)
					{
						if (base / SENDCOUNT % 10 == 0)
							if (!r)
								printf("exchanging data... base %zd of msglen_homo %zd (t %g s)\n",
									   base, msglen_homo, MPI_Wtime() - t0);
						
						/* real send count, at last can be less than SENDCOUNT */
						const ptrdiff_t n = MIN(SENDCOUNT, msglen_homo - base);

						/* populate send buffer */
						for (ptrdiff_t rr = 0; rr < rc; ++rr)
							if (base < send_msglen[rr])
								memcpy(sendbuf + n * rr,
									   sortedkeys + send_msgstart[rr] + base,
									   sizeof(KEY_T) * MIN(n, send_msglen[rr] - base));
						
						MPI_CHECK(MPI_Alltoall(sendbuf, n, MPI_KEY_T, recvbuf, n, MPI_KEY_T, comm));

						/* unpack recv buffer -- it may contain unused/invalid entries */
						for (ptrdiff_t rr = 0; rr < rc; ++rr)
							if (base < recv_msglen[rr])
							{
								const ptrdiff_t c = MIN(n, recv_msglen[rr] - base);
								memcpy(pdst, recvbuf + n * rr, sizeof(KEY_T) * c);
								pdst += c;
								}
					}
										
					free(recvbuf);
					free(sendbuf);

					const double t1 = MPI_Wtime();
					MPI_CHECK(MPI_Barrier(comm));
					if (!r)
						printf("all to all done in %g s\n", t1 - t0);

					fflush(stdout);
					MPI_CHECK(MPI_Barrier(comm));
				}

				/* recv/send the remaining keys via P2P */
				{
					if (!r)
						printf("sending now remaining messages...\n");
					MPI_CHECK(MPI_Barrier(comm));
					const double t0 = MPI_Wtime();					
					int rrc = 0, src = 0;
					MPI_Request recvreq[rc],sendreq[rc];

					/* receive request count and send request count */
					
					/* recv first */
					for (int rr = 0; rr < rc; ++rr)
					{
						const ptrdiff_t rem = recv_msglen[rr] - msglen_homo;

						if (rem > 0)
						{
							MPI_CHECK(MPI_Irecv(pdst, rem, MPI_KEY_T, rr, rr + rc * r, comm, recvreq + rrc++));
							pdst += rem;
						}
					}

					/* now send */
					for (int rr = 0; rr < rc; ++rr)
					{
						const ptrdiff_t rem = send_msglen[rr] - msglen_homo;

						if (rem > 0)
							MPI_CHECK(MPI_Send(sortedkeys + send_msgstart[rr] + msglen_homo, rem, MPI_KEY_T,
											   rr, r + rc * rr, comm));
					}

					/* wait now for receiving all messages */
					MPI_CHECK(MPI_Waitall(rrc, recvreq, MPI_STATUSES_IGNORE));

					
										MPI_CHECK(MPI_Barrier(comm));
										const double t1 = MPI_Wtime();`
					if (!r)
						printf("all to all done in %g s\n", t1 - t0);

					fflush(stdout);
					MPI_CHECK(MPI_Barrier(comm));

				}
				
				if (dst != recvkeys)
				{
					free(dst);
					dst = NULL;
				}
				
				MPI_CHECK(MPI_Barrier(comm));
				fflush(stdout);
				MPI_CHECK(MPI_Barrier(comm));
				exit(0);
			}
			
#elif 0 /* Allgatherv attempt */
			{
				int sendcounts[rc], sdispls[rc], recvcounts[rc], rdispls[rc + 1];

				__extension__ void _int (
					int * const out,
					const ptrdiff_t * const in )
				{
					for (int i = 0; i < rc; ++i)
						out[i] = in[i];
				}

				__extension__ ptrdiff_t _max (
					const ptrdiff_t * const in)
				{
					ptrdiff_t retval = in[0];

					for (int i = 1; i < rc; ++i)
						retval = MAX(retval, in[i]);

					MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &retval, 1, MPI_INT64_T, MPI_MAX, comm));
					
					return retval;
				}
				
				_int(sendcounts, send_msglen);
				_int(sdispls, send_msgstart);
				_int(recvcounts, recv_msglen);
					{
		MPI_File f;
		MPI_CHECK(MPI_File_open
				  (comm, "sendsizes-int32.raw", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f));

		MPI_CHECK(MPI_File_set_size(f, rc * rc * sizeof(int)));

		MPI_CHECK(MPI_File_write_at_all
				  (f, sizeof(int) * rc * r, sendcounts, rc, MPI_INT, MPI_STATUS_IGNORE));

		MPI_CHECK(MPI_File_close(&f));
	}

				//printf("my recv pointer is 0x%p\n", recvkeys);
				//_int(rdispls, recv_msgstart);
				rdispls[0];
				for (int i = 1; i <= rc; ++i)
					rdispls[i] = rdispls[i - 1] + recv_msglen[i - 1];

//				KEY_T * dst = recvkeys;

//				if (!dst)
//					dst = malloc(sizeof(KEY_T) * rdispls[rc]);

					enum { CHUNK = 2 };


					
					const ptrdiff_t msgmaxlen = _max(send_msglen);
					if (!r)
						printf("max msglen: %zd\n", msgmaxlen);
					
					const ptrdiff_t pc = (msgmaxlen + CHUNK - 1) / CHUNK;
					void * sendbuf = malloc(CHUNK * sizeof(KEY_T) * rc);
					memset(sendbuf, rc, sizeof(KEY_T) * CHUNK * rc);
					
					void * recvbuf = malloc(CHUNK * sizeof(KEY_T) * rc);


					
					const double t0 = MPI_Wtime();
					for (ptrdiff_t p = 0; p < pc; ++p)
					{
						if (!r && p % 10 == 0)
							printf("p %zd in %g s, %g MB/s\n", p, MPI_Wtime() - t0,
								   p * CHUNK * rc * sizeof(KEY_T) / (MPI_Wtime() - t0) * 1e-6);
						MPI_CHECK( MPI_Alltoall(sendbuf, CHUNK, MPI_KEY_T,
											recvbuf, CHUNK, MPI_KEY_T, comm));
					}
					const double t1 = MPI_Wtime();
/*											const void *sendbuf, int sendcount, MPI_Datatype sendtype,
       void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
*/				
/*				MPI_CHECK(MPI_Alltoallv(sortedkeys, sendcounts, sdispls, MPI_KEY_T,
										dst, recvcounts, rdispls, MPI_KEY_T, comm));

				if (dst != recvkeys)
					free(dst);
*/				
				MPI_Barrier(comm);
				if (!r)
					printf("all to all v done in %g s\n", t1 - t0);
				MPI_Barrier(comm);
				exit(0);
				for (int c = 0; c < 2; ++c)
				{
					
				}

				/* int MPI_Alltoallv(const void *sendbuf, const int *sendcounts,
       const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
       const int *recvcounts, const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) */
			}
#else /* P2P-based attempt*/
			{
				MPI_Barrier(comm);
				if (!r)
					printf("alternate\n");
				MPI_Barrier(comm);
				
				MPI_Request sendreqs[3 * rc];
				for (int d = 0; d < rc; ++d)
					sendreqs[d] = MPI_REQUEST_NULL;

				
				for(int d = 0; d < rc; ++d)
					post_and_send(d, sendreqs + 3 * d);

				MPI_Barrier(comm);
				if (!r)
					printf("post and send done\n");
				MPI_Barrier(comm);
				MPI_CHECK(MPI_Waitall(3 * rc, sendreqs, MPI_STATUSES_IGNORE));

				for (int d = 0; d < rc; ++d)
					wait_and_update(d);
			}
#endif
			else
			{
				/* communication-computation overlap */
				int MPI_SORT_CCO = 0;
				READENV(MPI_SORT_CCO, atoi);
				MPI_SORT_CCO = MIN(MPI_SORT_CCO, rc);

				if (0 > MPI_SORT_CCO)
					MPI_SORT_CCO = rc;

				for (int d = 0; d < rc; ++d)
				{
					post_and_send(d, NULL);

					if (MPI_SORT_CCO - 1 < d)
						wait_and_update(d - MPI_SORT_CCO);

					if (rc - 1 == d)
						for (int i = MPI_SORT_CCO - 1; i >= 0; --i)
							wait_and_update(d - i);
				}	
			}

			MPI_CHECK(MPI_Type_free(&MPI_KEY_T));
			MPI_CHECK(MPI_Type_free(&VALUE01));

#ifndef NDEBUG
			for (ptrdiff_t i = 0; i < keyrange_count; ++i)
				assert(!recv_histo[i]);
				
			free(recv_histo);
#endif
		}

		free(recv_start);
		free(sortedkeys);
	}

	const double t5 = MPI_Wtime();

	if (recvkeys)
	{
		const ptrdiff_t t0 = recvstart_rank[r + 0];
		const ptrdiff_t t1 = recvstart_rank[r + 1];

		const ptrdiff_t k0 =
			lowerbound(global_start, global_start + keyrange_count, t0);

		const ptrdiff_t k1 =
			lowerbound(global_start, global_start + keyrange_count, t1);

		const ptrdiff_t first = k0 - 1;
		const ptrdiff_t last = k1 - 1;

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

		if (first >= 0)
		{
			assert(dst - recvkeys + first_count <= recvcount);
			dst += fill(keyrange.begin + first, first_count, dst);
		}

		for(ptrdiff_t i = first + 1; i < last; ++i)
		{
			assert(dst - recvkeys + global_start[i + 1] - global_start[i] <= recvcount);
			dst += fill(keyrange.begin + i, global_start[i + 1] - global_start[i], dst);
		}

		if (last >= 0)
		{
			assert(dst - recvkeys + last_count <= recvcount);
			dst += fill(keyrange.begin + last, last_count, dst);
		}

		assert(dst - recvkeys == recvcount);
	}

	free(global_start);
	free(order);
	free(start);
	free(histo);
		
	const double t6 = MPI_Wtime();

	{
		int MPI_SORT_PROFILE = 0;
		READENV(MPI_SORT_PROFILE, atoi);

		if (MPI_SORT_PROFILE)
		{
			__extension__ double tts_ms (
				double tbegin,
				double tend )
			{
				MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &tbegin, 1, MPI_DOUBLE, MPI_MIN, comm));
				MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &tend  , 1, MPI_DOUBLE, MPI_MAX, comm));

				return 1e3 * (tend - tbegin);
			}

			const double tinit = tts_ms(t0, t1);
			const double tlocal = tts_ms(t1, t2);
			const double tremote = tts_ms(t2, t6);
			const double texscan = tts_ms(t2, t3);
			const double tcomminit = tts_ms(t3, t4);
			const double tcomm = tts_ms(t4, t5);
			const double tfill = tts_ms(t5, t6);

			if (!r)
			{
				printf("%s: INIT %g ms LOCALSORT %g ms REMOTE-COMM %g ms\n",
					   __FILE__, tinit, tlocal, tremote);

				if (recvvals0 || recvvals1)
					printf("%s: EXSCAN %g ms COMM-INIT %g ms COMM-DATA %g ms\n",
						   __FILE__, texscan, tcomminit, tcomm);

				if (recvkeys)
					printf("%s: FILL %g ms\n",
						   __FILE__, tfill);
			}
		}
	}

	return MPI_SUCCESS;
}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         