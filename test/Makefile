CFLAGS += -I../include

CXXFLAGS += -std=c++11

LDFLAGS += -L../lib
LDFLAGS := -lmpi-sort $(LDFLAGS)

GOALS = mpi-sort vanilla-sort

all : $(GOALS)

vanilla-sort : vanilla.cxx
	$(CXX) $(CXXFLAGS) -o $@ $<

mpi-sort : main.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean :
	rm -f $(GOALS)

.PHONY : clean
