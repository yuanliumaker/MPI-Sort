#ifndef MPI_CHECK
#define MPI_CHECK(stmt)						\
    do								\
    {								\
	const int code = stmt;					\
								\
	if (code != MPI_SUCCESS)				\
	{							\
	    char msg[2048];					\
	    int len = sizeof(msg);				\
	    MPI_Error_string(code, msg, &len);			\
								\
	    fprintf(stderr,					\
		    "ERROR\n" #stmt "%s (%s:%d)\n",		\
			msg, __FILE__, __LINE__);		\
								\
	    fflush(stderr);					\
								\
	    MPI_Abort(MPI_COMM_WORLD, code);			\
	}							\
    }								\
    while(0)
#endif

#define DIE_UNLESS(stmt)					\
    do								\
    {								\
	if (!(stmt))						\
	{							\
	    fprintf(stderr,					\
		    "DIED AT" #stmt				\
		    "(%s:%d)\n", __FILE__, __LINE__);		\
								\
	    exit(EXIT_FAILURE);					\
	}							\
    }								\
    while(0)

#define READENV(x, op)				\
    do						\
    {						\
	if (getenv(#x))				\
	    x = op(getenv(#x));			\
    }						\
    while(0)
