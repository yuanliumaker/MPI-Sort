CFLAGS += -std=c99
CFLAGS += -I../include
CFLAGS += -DKEY_T=int32_t
CFLAGS += -DSIZE_T=ptrdiff_t

libmpisort.a : mpisort.o kernels.o
	ar rcs $@ $^

%.o : %.c
	$(CC) $(CFLAGS) -c $<

clean :
	rm -f libmpisort.a *.o

.PHONY : clean
