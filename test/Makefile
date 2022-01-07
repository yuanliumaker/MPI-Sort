CFLAGS += -I../include -I../util

CFLAGS += -std=c99
CXXFLAGS += -std=c++11

LDFLAGS += -L../lib

threads ?= 0
ifeq ($(threads),1)
	LDFLAGS := -fopenmp $(LDFLAGS)
endif

LDFLAGS := -l:libmpi-sort.a -lstdc++ -lm $(LDFLAGS)

GOALS = mpi-sort vanilla-sort vanilla-qsort percentiles cpd

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
