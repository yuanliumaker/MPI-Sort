CFLAGS += -I../include -I../util

CFLAGS += -std=c99
CXXFLAGS += -std=c++11

LDFLAGS += -L../lib
LDFLAGS := -l:libmpi-sort.a -lstdc++ -lm $(LDFLAGS)

GOALS = mpi-sort vanilla-sort  percentiles
#vanilla-qsort
all : $(GOALS)

vanilla-sort : vanilla-sort.cxx
	$(CXX) $(CXXFLAGS) -o $@ $<

vanilla-qsort : vanilla-qsort.c
	$(CC) $(CFLAGS) -o $@ $<

% : %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean :
	rm -f $(GOALS)

.PHONY : clean
