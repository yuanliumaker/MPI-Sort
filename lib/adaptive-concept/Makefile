CFLAGS += -I../
CFLAGS += -I../include -I../util

CFLAGS += -std=c99
CXXFLAGS += -std=c++11

#LDFLAGS += -L../lib
LDFLAGS := -lstdc++ -lm $(LDFLAGS)

VPATH = . ..

GOALS = test-64 test-32 test-16 test-8

all : $(GOALS)

test-64 : main.c ../xtract.o lsort.o a2av.o
	$(CC) $(CFLAGS) -D_KEYBITS_=64 -o $@ $^ $(LDFLAGS)

test-32 : main.c ../xtract.o lsort.o a2av.o
	$(CC) $(CFLAGS) -D_KEYBITS_=32 -o $@ $^ $(LDFLAGS)

test-16 : main.c ../xtract.o lsort.o a2av.o
	$(CC) $(CFLAGS) -D_KEYBITS_=16 -o $@ $^ $(LDFLAGS)

test-8 : main.c ../xtract.o lsort.o a2av.o
	$(CC) $(CFLAGS) -D_KEYBITS_=8 -o $@ $^ $(LDFLAGS)

lsort.o : lsort.cxx
	$(CXX) $(CXXFLAGS) -c $<

%.o : %.c
	$(CC) $(CFLAGS) -c $<

clean :
	rm -f test *.o $(GOALS)

.PHONY : clean
