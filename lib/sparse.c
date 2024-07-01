#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <limits.h>
#include <assert.h>
#include <string.h>

#include "macros.h"
#include "a2av.h"
// ## �������������� ����ƴ�� ��cat(uint 16)-> uint16 cat(uint16 _t)-> uint16_t
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
// ������������ first �Լ�����ֵ�����ص�һ�����ڸ���ֵ������
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
	// �����н����н��й�Լ������ȡvsz ���ֵ���õ������̵��������͵Ĵ�С(byte)(����³���ԣ����ܸ������̵��������Ͳ�һ������ˣ�)
	MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &vsz, 1, MPI_INT, MPI_MAX, comm));

	const double t1 = MPI_Wtime();
	// t1-t0: �����ʼ��ʱ��
	// ������local sort
	lsort(stable, sizeof(KEY_T), vsz, sendcount, sendkeys, sendvals);
	// t2-t1 :����std::stable_sort or std::sort ����ʱ�� 
	const double t2 = MPI_Wtime();

	KEY_T krmin = 0, krmax = (KEY_T)ULONG_MAX;

	if (sendcount)
	{	
		// ��ʼ��keyrangemin ��keyrangemax
		krmin = sendkeys[0];
		krmax = sendkeys[sendcount - 1];
	}

	/* find key ranges */

	{
		// ���н����н��й�Լ�����ҳ�sendkeys �е����ֵ����Сֵ����krmin ��krmax��
		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &krmin, 1, MPI_KEY_T, MPI_MIN, comm));
		MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &krmax, 1, MPI_KEY_T, MPI_MAX, comm));
	}
	// t3-t2 �����Լ��������ֵ������ʱ��
	const double t3 = MPI_Wtime();

	ptrdiff_t recvstart_rank[rankcountp1];
	// ����ÿ�����̵Ľ������ݵ���ʼλ�� 
	/*������ 4 �����̣�recvcount Ϊÿ�����̽��յ����ݸ������ֱ�Ϊ 10��20��30��40����

tmp Ϊ 10��20��30��40��
MPI_Scan ����ǰ׺�ͣ����Ϊ 10��30��60��100���ֱ�洢�� myend �С�
recvstart_rank ����Ϊ {0, 10, 30, 60, 100}��
������ÿ�����̵Ľ���������ʼλ�úͽ���λ�þ�ȷ���ˣ�

�� 0 �����̣���ʼλ�� 0������λ�� 10��
�� 1 �����̣���ʼλ�� 10������λ�� 30��
�� 2 �����̣���ʼλ�� 30������λ�� 60��
�� 3 �����̣���ʼλ�� 60������λ�� 100*/ 
	/* separators are defined by recvcounts */
	// �˴���Ϊrecvstart_rank[0,recvcount,recvcount+recvcount,...]
	{
		ptrdiff_t tmp = recvcount, myend = 0;
		// MPI_Scan ����ǰ׺��
		// tmp:��ǰ���̵Ľ��ո�������������
		// myend : ��ǰ���̽������ݵĽ���λ�� ���������
		MPI_CHECK(MPI_Scan(&tmp, &myend, 1, MPI_INT64_T, MPI_SUM, comm));

		recvstart_rank[0] = 0;
		// MPI_Allgather ��ÿ�����̵Ľ������ݵĽ���λ��myend�ռ���recvstart_rank�����У��ӵڶ���λ�ÿ�ʼ
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
			// ÿ�����̻��յ�����query[newkey,newkey,newkey...]
			MPI_CHECK(MPI_Allgather(&newkey, 1, MPI_KEY_T, query, 1, MPI_KEY_T, comm));

			ptrdiff_t partials[rankcount];
			//���㲢����С�ڵ���query[r]�ļ�������
			for (int r = 0; r < rankcount; ++r)
				partials[r] = lb(sendkeys, sendcount, query[r]);

			ptrdiff_t answer = 0;
			// �����MPI_Reduce��MPI_Scatter �Ĺ��ܣ������ǽ����н��̵����ݽ��й�Լ������Ȼ�󽫹�Լ��Ľ���ȷ�ɢ�������н���
			// ���ÿ�������е�answer �洢��С�ڵ���query �ļ�������
			MPI_CHECK(MPI_Reduce_scatter_block
					  (partials, &answer, 1, MPI_INT64_T, MPI_SUM, comm));

			if (answer <= recvstart_rank[rank])
			{
				curkey = newkey;
				qcount = answer;
			}
		}
		// ÿ�����̵�global_startkey ��global_count �����ֱ�洢ȫ����ʼ���ʹ����������
		// �Ӷ�ÿ�����̻���Լ�Ҫ���������

		MPI_CHECK(MPI_Allgather(&curkey, 1, MPI_KEY_T, global_startkey, 1, MPI_KEY_T, comm));
		MPI_CHECK(MPI_Allgather(&qcount, 1, MPI_INT64_T, global_count, 1, MPI_INT64_T, comm));

		for (int r = 0; r < rankcount; ++r)
			assert(global_count[r] <= recvstart_rank[r]);
	}

	const double t5 = MPI_Wtime();

	ptrdiff_t sstart[rankcountp1];
	// ����ȷ��ȷ��ÿ�����̸�������ݷ�Χ������global_startkey���ظ��������
	/* refine separator in index space */
	{
		sstart[0] = 0;

		for (int r = 1; r < rankcount; ++r)
			sstart[r] = lb(sendkeys, sendcount, global_startkey[r]);
		// ���һ�����̵Ľ���λ��
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
			// sstart Ϊglobal_count ��һ����ȷ�Ľ�� �൱��local splitter
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
		// alltoall  ֮��������̵�rcount ��¼�˴Ӹ����̽��յ����ݿ��С rcount[0] ��ʾ�ӽ���0 ���յ���������С
		MPI_CHECK(MPI_Alltoall(scount, 1, MPI_INT64_T, rcount, 1, MPI_INT64_T, comm));


		ptrdiff_t rstart[rankcount];
		// rcount ����ǰ׺�� ���Ϊrstart
		const ptrdiff_t __attribute__((unused)) check = exscan(rankcount, rcount, rstart);
		assert(check == recvcount);
		// ���·�������
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
			// ���ڼ����ڷֲ�ʽϵͳ��һ����������С��ʼʱ���������ʱ��֮��Ӷ��õ��ò�������������ʱ��
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
