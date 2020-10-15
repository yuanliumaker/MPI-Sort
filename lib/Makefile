CFLAGS += -std=c99
CFLAGS += -I../include

libmpi-sort.a : \
	dispatch.o \
	common.o \
	rmanip.o \
	xtract.o \
	dsort-uint8.o \
	dsort-uint16.o \
	dsort-uint32.o
	ar rcs $@ $^

dsort-uint8.o : dsort.c
	$(CC) $(CFLAGS) -DKEY_T=uint8_t -o $@ -c $<

dsort-uint16.o : dsort.c
	$(CC) $(CFLAGS) -DKEY_T=uint16_t -o $@ -c $<

dsort-uint32.o : dsort.c
	$(CC) $(CFLAGS) -DKEY_T=uint32_t -o $@ -c $<

rmanip.c : rmanip.m4
	m4 $< | indent > $@

xtract.c : xtract.m4
	m4 $< | indent > $@

%.o : %.c
	$(CC) $(CFLAGS) -c $<

clean :
	rm -f libmpi-sort.a *.o xtract.c rmanip.c

.PHONY : clean
