
CFLAGS += -std=c99
CFLAGS += -I../include
CFLAGS += -fvisibility=hidden -fPIC

tuned ?= 1

ifeq "$(tuned)" "1"
	CFLAGS += -D_TUNED_
endif

OBJS = \
	dispatch.o \
	common.o \
	drange.o \
	xtract.o \
	dsort-uint8.o \
	dsort-uint16.o \
	dsort-uint32.o \
	a2av.o

GOALS = libmpi-sort.a libmpi-sort.so

all : $(GOALS)

libmpi-sort.a : $(OBJS)
	ar rcs $@ $^

libmpi-sort.so : $(OBJS)
	$(CC) $(CFLAGS) -shared -o $@ $^

dsort-uint8.o : dsort.c csort-tuned-u8.h
	$(CC) $(CFLAGS) -D_TUNED_ -DKEY_T=uint8_t -D_BITCOUNT_=8 -o $@ -c $<

dsort-uint16.o : dsort.c csort-tuned-u16.h
	$(CC) $(CFLAGS) -D_TUNED_ -DKEY_T=uint16_t -D_BITCOUNT_=16 -o $@ -c $<

dsort-uint32.o : dsort.c
	$(CC) $(CFLAGS) -DKEY_T=uint32_t -o $@ -c $<

drange.c : drange.m4
	m4 $< > $@

xtract.c : xtract.m4
	m4 $< > $@

csort-tuned-u8.h : csort.m4
	m4 -D BITCOUNT=8 -D N=4 $< > $@

csort-tuned-u16.h : csort.m4
	m4 -D BITCOUNT=16 -D N=8 $< > $@

%.o : %.c
	$(CC) $(CFLAGS) -c $<

clean :
	rm -f $(GOALS) *.o xtract.c drange.c csort-tuned-u*.h

.PHONY : clean
