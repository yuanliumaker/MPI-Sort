CFLAGS += -std=c99
CFLAGS += -I../include
CFLAGS += -DKEY_T=uint8_t
CFLAGS += -DSIZE_T=ptrdiff_t

libmpi-sort.a : dispatcher.o mpi-sort.o kernels.o
	ar rcs $@ $^

%.o : %.c
	$(CC) $(CFLAGS) -c $<

clean :
	rm -f libmpisort.a *.o

.PHONY : clean
