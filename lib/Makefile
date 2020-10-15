CFLAGS += -std=c99
CFLAGS += -I../include
CFLAGS += -fvisibility=hidden -fPIC

OBJS = \
	dispatch.o \
	common.o \
	drange.o \
	xtract.o \
	dsort-uint8.o \
	dsort-uint16.o \
	dsort-uint32.o

GOALS = libmpi-sort.a libmpi-sort.so

all : $(GOALS)

libmpi-sort.a : $(OBJS)
	ar rcs $@ $^

libmpi-sort.so : $(OBJS)
	$(CC) $(CFLAGS) -shared -o $@ $^

dsort-uint8.o : dsort.c
	$(CC) $(CFLAGS) -DKEY_T=uint8_t -o $@ -c $<

dsort-uint16.o : dsort.c
	$(CC) $(CFLAGS) -DKEY_T=uint16_t -o $@ -c $<

dsort-uint32.o : dsort.c
	$(CC) $(CFLAGS) -DKEY_T=uint32_t -o $@ -c $<

drange.c : drange.m4
	m4 $< > $@

xtract.c : xtract.m4
	m4 $< > $@

%.o : %.c
	$(CC) $(CFLAGS) -c $<

clean :
	rm -f $(GOALS) *.o xtract.c drange.c

.PHONY : clean
