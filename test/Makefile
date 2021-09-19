CFLAGS += -I../include

CXXFLAGS += -std=c++11

LDFLAGS += -L../lib
LDFLAGS := -l:libmpi-sort.a $(LDFLAGS)

GOALS = mpi-sort vanilla-sort

all : $(GOALS)

vanilla-sort : vanilla-sort.cxx
	$(CXX) $(CXXFLAGS) -o $@ $<

mpi-sort : mpi-sort.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean :
	rm -f $(GOALS)

.PHONY : clean
