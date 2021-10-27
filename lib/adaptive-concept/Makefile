CFLAGS += -I../include -I../util

CFLAGS += -std=c99
CXXFLAGS += -std=c++11

LDFLAGS += -L../lib
LDFLAGS := -l:libmpi-sort.a -lm $(LDFLAGS)

test : main.o ../xtract.o lsort.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

lsort.o : lsort.cxx
	$(CXX) $(CXXFLAGS) -c $<

%.o : %.c
	$(CC) $(CFLAGS) -c $<

clean :
	rm -f test *.o

.PHONY : clean
