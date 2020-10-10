CFLAGS += -std=c99
CFLAGS += -I../include

libmpi-sort.a : dispatch.o dsort-uint8.o dsort-int8.o common.o
	ar rcs $@ $^

dsort-uint8.o : dsort.c
	$(CC) $(CFLAGS) -DKEY_T=uint8_t -o $@ -c $<

dsort-int8.o : dsort.c
	$(CC) $(CFLAGS) -DKEY_T=int8_t -o $@ -c $<

%.o : %.c
	$(CC) $(CFLAGS) -c $<

clean :
	rm -f libmpi-sort.a *.o

.PHONY : clean
