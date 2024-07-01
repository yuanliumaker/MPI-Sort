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

static float MPI_SORT_A2AV_TUNE = 0.95;
static ptrdiff_t MPI_SORT_A2AV_SIZE = 4096, MPI_SORT_A2AV_HOMO = -1;
static int MPI_SORT_P2P_OVERLAP = 1;//表明将在执行全局通信操作的同时，启动点对点通信操作 通过二者之间的重叠，提高整体性能

static void __attribute__((constructor)) init ()
{
	READENV(MPI_SORT_A2AV_TUNE, atof);
	READENV(MPI_SORT_A2AV_SIZE, atoll);
	READENV(MPI_SORT_A2AV_HOMO, atoll);
	READENV(MPI_SORT_P2P_OVERLAP, atoi);
}
/*in 发送缓冲区指针 即sendkeys
sendcounts  每个进程发送的元素数量 数组 scount
 sdispls 每个进程发送数据的偏移量数组 即sstart scount的前缀和
 out 接收缓冲区指针 recvkeys
 recvounts 每个进程接收的数据元素数量 数组 rcount
 rdispls 每个进程接收数据的偏移量数组 rstart  rcount 的前缀和*/
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
	const double t0 = MPI_Wtime();

	int r, rc;
	MPI_CHECK(MPI_Comm_rank(comm, &r));
	MPI_CHECK(MPI_Comm_size(comm, &rc));

	ptrdiff_t msglen_homo = MPI_SORT_A2AV_HOMO;

	/* compute msglen_homo:
	   max 95-percentiles of the message sizes first
	   TODO: refine balance between A2A and P2P,
	   it makes sense to spend few milliseconds for that */
	//    计算同构消息长度
	// 同构化消息长度，是一种标准化的消息长度，用于衡量各进程之间的消息大小，其作用是避免各进程之间的消息大小的不均衡
	// 导致通信瓶颈 提高通信效率
	if (msglen_homo < 0)
	{
		ptrdiff_t s[2 * rc];
		memcpy(s, sendcounts, sizeof(ptrdiff_t) * rc);
		memcpy(s + rc, recvcounts, sizeof(ptrdiff_t) * rc);

		__extension__ int compar(const void * a, const void * b) { return *(ptrdiff_t *)a - *(ptrdiff_t *)b; }
		qsort(s, 2 * rc, sizeof(ptrdiff_t), compar);
		// 各进程取排序后的s数组中的某个百分位数值
		msglen_homo = s[MAX(0, MIN(2 * rc - 1, (int)round(MPI_SORT_A2AV_TUNE * 2 * rc)))];
	//使用MPI_Allreduce将用于找到所有进程之间消息长度的最大值并存储在msglen_homo 中
		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &msglen_homo, 1, MPI_INT64_T, MPI_MAX, comm));
	}

	const double t1 = MPI_Wtime();

	/* element size */
	ptrdiff_t esz;

	{
		int t;
		MPI_CHECK(MPI_Type_size(type, &t));
		esz = t;
	}

	int reqc = 0;
	MPI_Request reqs[2 * rc];

	/* P2P reqs */
	__extension__ void post_p2p()
	{
		for (int rr = 0; rr < rc; ++rr)
		{
			// rem 用来判断该进程是否需要继续接收数据，rem>0 表明该进程需要接收的消息长度超过标准化的消息长度，还剩rem 个元素需要接收
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
	}

	if (1 == MPI_SORT_P2P_OVERLAP)
		post_p2p();
	else
		if (-1 == MPI_SORT_P2P_OVERLAP)
			MPI_CHECK(MPI_Barrier(comm));

	const double t2 = MPI_Wtime();

	// alltoall和p2p 通过h 来控制
	/* send around keys via A2A */
	{
		const ptrdiff_t msgsz = MAX(1, MPI_SORT_A2AV_SIZE);

		void *sendbuf, *recvbuf;
		DIE_UNLESS(sendbuf = malloc(msgsz * rc));
		DIE_UNLESS(recvbuf = malloc(msgsz * rc));

		/* byte count per rank */
		const ptrdiff_t basehi = esz * msglen_homo;

		for (ptrdiff_t base = 0; base < basehi; base += msgsz)
		{
			/* last message is likely to be truncated */
			/*最后一个的消息长度可能比预定的长度药效，因此需要截断*/
			const ptrdiff_t msgsz_trunc = MIN(msgsz, basehi - base);

			/* pack sendbuf */
			for (int rr = 0; rr < rc; ++rr)
				if (base < esz * sendcounts[rr])
					memcpy(msgsz_trunc * rr + (int8_t *)sendbuf,
						   base + esz * sdispls[rr] + (int8_t *)in,
						   MIN(msgsz_trunc, esz * sendcounts[rr] - base));

			/* relaxing irregularities via A2A */
			MPI_CHECK(MPI_Alltoall(sendbuf, msgsz_trunc, MPI_BYTE,
								   recvbuf, msgsz_trunc, MPI_BYTE, comm));

			/* unpack recvbuf */
			for (ptrdiff_t rr = 0; rr < rc; ++rr)
				if (base < esz * recvcounts[rr])
					memcpy(base + esz * rdispls[rr] + (int8_t *)out,
						   msgsz_trunc * rr + (int8_t *)recvbuf,
						   MIN(msgsz_trunc, esz * recvcounts[rr] - base));
		}

		free(recvbuf);
		free(sendbuf);
	}

	if (-1 == MPI_SORT_P2P_OVERLAP)
		MPI_CHECK(MPI_Barrier(comm));

	const double t3 = MPI_Wtime();

	if (1 != MPI_SORT_P2P_OVERLAP)
		post_p2p();

	/* recv/send the remaining keys via P2P */
	{
		/* wait now for receiving all messages */
		MPI_CHECK(MPI_Waitall(reqc, reqs, MPI_STATUSES_IGNORE));
	}

	const double t4 = MPI_Wtime();

	{
		int MPI_SORT_PROFILE = 0;
		READENV(MPI_SORT_PROFILE, atoi);

		if (MPI_SORT_PROFILE)
		{
			__extension__ double tts_ms (
				double tbegin,
				double tend )
			{
				MPI_CHECK(MPI_Reduce(r ? &tbegin : MPI_IN_PLACE, &tbegin, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
				MPI_CHECK(MPI_Reduce(r ? &tend : MPI_IN_PLACE, &tend, 1, MPI_DOUBLE, MPI_MAX, 0, comm));

				return tend - tbegin;
			}

			const double thomo = tts_ms(t0, t1);
			const double tsend = tts_ms(t1, t2);
			const double ta2a = tts_ms(t2, t3);
			const double trecv = tts_ms(t3, t4);
			const double ttotal = tts_ms(t0, t4);

			if (!r)
			{
				printf("%s: msglen_homo %zd (blocking send)\n", __FILE__, msglen_homo);
				printf("%s: MPI_SORT_P2P_OVERLAP=%d\n", __FILE__, MPI_SORT_P2P_OVERLAP);
				printf("%s: HOMO %g s SEND %g s A2A %g s RECV %g s (OVERALL %g s)\n",
					   __FILE__, thomo, tsend, ta2a, trecv, ttotal);
			}
		}
	}
}
