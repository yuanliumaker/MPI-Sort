CFLAGS += -I../include -I../util

CFLAGS += -std=c99
CXXFLAGS += -std=c++11

LDFLAGS += -L../lib
LDFLAGS := -l:libmpi-sort.a -lm $(LDFLAGS)

GOALS = mpi-sort vanilla-sort percentiles

all : $(GOALS)

vanilla-sort : vanilla-sort.cxx
	$(CXX) $(CXXFLAGS) -o $@ $<

% : %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean :
	rm -f $(GOALS)

.PHONY : clean
