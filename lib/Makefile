CFLAGS += -std=c99
CFLAGS += -I../include
CFLAGS += -fvisibility=hidden

libmpi-sort.a : \
	dispatch.o \
	common.o \
	drange.o \
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

drange.c : drange.m4
	m4 $< | indent > $@

xtract.c : xtract.m4
	m4 $< | indent > $@

%.o : %.c
	$(CC) $(CFLAGS) -c $<

clean :
	rm -f libmpi-sort.a *.o xtract.c drange.c

.PHONY : clean
