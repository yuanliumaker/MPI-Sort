#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <limits.h>
#include <assert.h>
#include <string.h>

#include "macros.h"
#include "a2av.h"
// ## 连接运算符，因此 以下拼接 如cat(uint 16)-> uint16 cat(uint16 _t)-> uint16_t
#define _CAT(a, b) a ## b
#define CAT(a, b) _CAT(a, b)

#define KEY_T CAT(CAT(uint, _KEYBITS_), _t)
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
// 给定有序数组 first 以及给定值，返回第一个大于给定值的索引
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

int CAT(CAT(sparse_uint, _KEYBITS_), _t) (
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
	const double t0 = MPI_Wtime();

	int rank, rankcount;
	MPI_CHECK(MPI_Comm_rank(comm, &rank));
	MPI_CHECK(MPI_Comm_size(comm, &rankcount));

	const ptrdiff_t rankcountp1 = rankcount + 1;

	int vsz = 0;

	if (sendvals)
		MPI_CHECK(MPI_Type_size(valtype, &vsz));
	// 在所有进程中进行归约操作获取vsz 最大值，得到各进程的数据类型的大小(byte)(考虑鲁棒性，可能各个进程的数据类型不一样，如此？)
	MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &vsz, 1, MPI_INT, MPI_MAX, comm));

	const double t1 = MPI_Wtime();
	// t1-t0: 计算初始化时间
	// 各进程local sort
	lsort(stable, sizeof(KEY_T), vsz, sendcount, sendkeys, sendvals);
	// t2-t1 :计算std::stable_sort or std::sort 运行时间 
	const double t2 = MPI_Wtime();

	KEY_T krmin = 0, krmax = (KEY_T)ULONG_MAX;

	if (sendcount)
	{	
		// 初始化keyrangemin 和keyrangemax
		krmin = sendkeys[0];
		krmax = sendkeys[sendcount - 1];
	}

	/* find key ranges */

	{
		// 所有进程中进行归约操作找出sendkeys 中的最大值与最小值存在krmin 和krmax中
		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &krmin, 1, MPI_KEY_T, MPI_MIN, comm));
		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &krmax, 1, MPI_KEY_T, MPI_MAX, comm));
	}
	// t3-t2 计算归约操作找最值的运行时间
	const double t3 = MPI_Wtime();

	ptrdiff_t recvstart_rank[rankcountp1];
	// 计算每个进程的接收数据的起始位置 
	/*假设有 4 个进程，recvcount 为每个进程接收的数据个数，分别为 10、20、30、40，则：

tmp 为 10、20、30、40。
MPI_Scan 计算前缀和，结果为 10、30、60、100，分别存储在 myend 中。
recvstart_rank 最终为 {0, 10, 30, 60, 100}。
这样，每个进程的接收数据起始位置和结束位置就确定了：

第 0 个进程：起始位置 0，结束位置 10。
第 1 个进程：起始位置 10，结束位置 30。
第 2 个进程：起始位置 30，结束位置 60。
第 3 个进程：起始位置 60，结束位置 100*/ 
	/* separators are defined by recvcounts */
	// 此处则为recvstart_rank[0,recvcount,recvcount+recvcount,...]
	{
		ptrdiff_t tmp = recvcount, myend = 0;
		// MPI_Scan 计算前缀和
		// tmp:当前进程的接收个数，输入数据
		// myend : 当前进程接收数据的结束位置 ，输出数据
		MPI_CHECK(MPI_Scan(&tmp, &myend, 1, MPI_INT64_T, MPI_SUM, comm));

		recvstart_rank[0] = 0;
		// MPI_Allgather 将每个进程的接收数据的结束位置myend收集到recvstart_rank数组中，从第二个位置开始
		MPI_CHECK(MPI_Allgather(&myend, 1, MPI_INT64_T, recvstart_rank + 1, 1, MPI_INT64_T, comm));
	}

	const double t4 = MPI_Wtime();

	KEY_T global_startkey[rankcount];
	ptrdiff_t global_count[rankcount];

	/* find approximately separators by exploring the key space */
	{
		size_t curkey = krmin, qcount = 0;

		for (KEY_T b = _KEYBITS_ - 1; b < _KEYBITS_; --b)
		{
			const KEY_T delta = ((KEY_T)1) << b;

			if (krmax - krmin < delta)
				continue;

			const KEY_T newkey = MIN(krmax, curkey + delta);

			KEY_T query[rankcount];
			// 每个进程会收到数组query[newkey,newkey,newkey...]
			MPI_CHECK(MPI_Allgather(&newkey, 1, MPI_KEY_T, query, 1, MPI_KEY_T, comm));

			ptrdiff_t partials[rankcount];
			//计算并返回小于等于query[r]的键的数量
			for (int r = 0; r < rankcount; ++r)
				partials[r] = lb(sendkeys, sendcount, query[r]);

			ptrdiff_t answer = 0;
			// 结合了MPI_Reduce和MPI_Scatter 的功能，作用是将所有进程的数据进行归约操作，然后将归约后的结果等分散发给所有进程
			// 因此每个进程中的answer 存储了小于等于query 的键的总数
			MPI_CHECK(MPI_Reduce_scatter_block
					  (partials, &answer, 1, MPI_INT64_T, MPI_SUM, comm));

			if (answer <= recvstart_rank[rank])
			{
				curkey = newkey;
				qcount = answer;
			}
		}
		// 每个进程的global_startkey 和global_count 数组会分别存储全局起始键和处理的数据量
		// 从而每个进程获得自己要处理的数据

		MPI_CHECK(MPI_Allgather(&curkey, 1, MPI_KEY_T, global_startkey, 1, MPI_KEY_T, comm));
		MPI_CHECK(MPI_Allgather(&qcount, 1, MPI_INT64_T, global_count, 1, MPI_INT64_T, comm));

		for (int r = 0; r < rankcount; ++r)
			assert(global_count[r] <= recvstart_rank[r]);
	}

	const double t5 = MPI_Wtime();

	ptrdiff_t sstart[rankcountp1];
	// 更精确的确定每个进程负责的数据范围，对于global_startkey有重复数的情况
	/* refine separator in index space */
	{
		sstart[0] = 0;

		for (int r = 1; r < rankcount; ++r)
			sstart[r] = lb(sendkeys, sendcount, global_startkey[r]);
		// 最后一个进程的结束位置
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
			// sstart 为global_count 进一步精确的结果 相当于local splitter
				sstart[r] += MAX(0, MIN(q[r], recvstart_rank[r] - global_count[r] - qstart[r]));
	}

	const double t6 = MPI_Wtime();


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
		// alltoall  之后各个进程的rcount 记录了从各进程接收的数据块大小 rcount[0] 表示从进程0 接收的数据量大小
		MPI_CHECK(MPI_Alltoall(scount, 1, MPI_INT64_T, rcount, 1, MPI_INT64_T, comm));


		ptrdiff_t rstart[rankcount];
		// rcount 计算前缀和 输出为rstart
		const ptrdiff_t __attribute__((unused)) check = exscan(rankcount, rcount, rstart);
		assert(check == recvcount);
		// 重新分配数据
		/* keys */
		a2av(sendkeys, scount, sstart, MPI_KEY_T, recvkeys, rcount, rstart, comm);

		/* values */
		if (vsz)
			a2av(sendvals, scount, sstart, valtype, recvvals, rcount, rstart, comm);
	}

	const double t7 = MPI_Wtime();

	/* sort once more */
	lsort(stable, sizeof(KEY_T), vsz, recvcount, recvkeys, recvvals);

	const double t8 = MPI_Wtime();

#ifndef NDEBUG
	for(ptrdiff_t i = 1; i < recvcount; ++i)
		assert(recvkeys[i - 1] <= recvkeys[i]);
#endif

	{
		int MPI_SORT_PROFILE = 0;
		READENV(MPI_SORT_PROFILE, atoi);

		if (MPI_SORT_PROFILE)
		{
			// 用于计算在分布式系统中一个操作的最小开始时间和最大结束时间之差，从而得到该操作的整体运行时间
			__extension__ double tts_ms (
				double tbegin,
				double tend )
			{
				MPI_CHECK(MPI_Reduce(rank ? &tbegin : MPI_IN_PLACE, &tbegin, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
				MPI_CHECK(MPI_Reduce(rank ? &tend : MPI_IN_PLACE, &tend, 1, MPI_DOUBLE, MPI_MAX, 0, comm));

				return tend - tbegin;
			}

			const double tinit = tts_ms(t0, t1);
			const double tlocal = tts_ms(t1, t2);
			const double trange = tts_ms(t2, t3);
			const double tsep = tts_ms(t3, t4);
			const double tquery = tts_ms(t4, t5);
			const double trefine = tts_ms(t5, t6);
			const double ta2a = tts_ms(t6, t7);
			const double tlocal2 = tts_ms(t7, t8);
			const double ttotal = tts_ms(t0, t8);

			if (!rank)
			{
				printf("%s: INIT %g s LOCALSORT %g s RANGE %g s SEPARATORS %g s QUERIES %g s REFINE %g s A2AV %g s LOCALSORT2 %g s (OVERALL %g s)\n",
					   __FILE__, tinit, tlocal, trange, tsep, tquery, trefine, ta2a, tlocal2, ttotal);
			}

			if (2 == MPI_SORT_PROFILE)
			{
				/* gather and print the local timings too */
				double * tlocals = NULL;

				if (!rank)
					DIE_UNLESS(tlocals = malloc(sizeof(*tlocals) * rankcount));

				MPI_CHECK(MPI_Gather(&tlocal, 1, MPI_DOUBLE, tlocals, 1, MPI_DOUBLE, 0, comm));

				if (!rank)
				{
					for (int rr = 0; rr < rankcount; ++rr)
						printf("LOCALSORT rank %d took %g s\n", rr, tlocals[rr]);

					free(tlocals);
				}
			}
		}
	}

	return MPI_SUCCESS;
}
