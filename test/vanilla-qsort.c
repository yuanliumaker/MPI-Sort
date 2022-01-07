#define _XOPEN_SOURCE 700

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include <sys/time.h>

#include <mpi.h>
#include "util.h"

double rdtss()
{
	struct timeval tv;
	POSIX_CHECK(0 == gettimeofday(&tv, NULL));

	return tv.tv_sec + 1e-6 * tv.tv_usec;
}

int compar (
    const void * element1,
    const void * element2 )
{
    const float a = *(float *)element1;
    const float b = *(float *)element2;

    return (a > b) - (a < b);
}

int main (
    const int argc,
    const char * argv[])
{
    if (argc != 1)
    {
		fprintf(stderr,
				"usage: %s < infp32.raw\n",
				argv[0]);

		return EXIT_FAILURE;
    }

	float * es;
	size_t ec = 0;

	{
		FILE * ftmp = NULL;
		POSIX_CHECK(ftmp = open_memstream((char **)&es, &ec));

		enum { BUNCH = 1 << 18 };
		float buf[BUNCH];

		while (1)
		{
			const size_t n = fread(buf, sizeof(float), BUNCH, stdin);

			POSIX_CHECK(n == fwrite(buf, sizeof(float), n, ftmp));

			if (BUNCH != n)
				break;
		}

		POSIX_CHECK(0 == fclose(ftmp));

		ec /= sizeof(*es);
	}

	fprintf(stderr, "I have read  %zd entries\n", ec);

	const double t0 = rdtss();

	qsort(es, ec, sizeof(*es), compar);

	const double t1 = rdtss();

	fprintf(stderr,
			"TTS: %g s. bye.\n",
			t1 - t0);

    return EXIT_SUCCESS;
}
