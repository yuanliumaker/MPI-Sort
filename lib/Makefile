CFLAGS += -std=c99
CFLAGS += -I../include
CFLAGS += -fvisibility=hidden
CFLAGS += -fPIC
CFLAGS += -Wno-unused-variable
CFLAGS += -Wno-unknown-pragmas

CXXFLAGS += -std=c++11
CXXFLAGS += -fno-exceptions -fno-rtti -static
CXXFLAGS += -fvisibility=hidden
CXXFLAGS += -fPIC

threads ?= 0

ifeq ($(threads),1)
	CXXFLAGS += -D_ENABLE_THREADS_
	CXXFLAGS += -fopenmp
endif

OBJS = \
	dispatch.o \
	common.o \
	drange.o \
	xtract.o \
	radix-uint8.o \
	radix-uint16.o \
	sparse-uint16.o \
	sparse-uint32.o \
	sparse-uint64.o \
	lsort.o \
	a2av.o

GOALS = libmpi-sort.a libmpi-sort.so

all : $(GOALS)

libmpi-sort.a : $(OBJS)
	ar rcs $@ $^

libmpi-sort.so : $(OBJS)
	$(CC) $(CFLAGS) -shared -o $@ $^

radix-uint8.o : radix.c csort-tuned-u8.h
	$(CC) $(CFLAGS) -D_TUNED_ -DKEY_T=uint8_t -D_BITCOUNT_=8 -o $@ -c $<

radix-uint16.o : radix.c csort-tuned-u16.h
	$(CC) $(CFLAGS) -D_TUNED_ -DKEY_T=uint16_t -D_BITCOUNT_=16 -o $@ -c $<

sparse-uint16.o : sparse.c
	$(CC) $(CFLAGS) -D_KEYBITS_=16 -o $@ -c $<

sparse-uint32.o : sparse.c
	$(CC) $(CFLAGS) -D_KEYBITS_=32 -o $@ -c $<

sparse-uint64.o : sparse.c
	$(CC) $(CFLAGS) -D_KEYBITS_=64 -o $@ -c $<

lsort.o : lsort.cxx
	$(CXX) $(CXXFLAGS) -c $<

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
