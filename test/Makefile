CFLAGS += -I../include

CXXFLAGS += -std=c++11

LDFLAGS += -L../lib
LDFLAGS := -lmpi-sort $(LDFLAGS)

GOALS = sort-uchar sort-vanilla

all : $(GOALS)

sort-vanilla : vanilla.cxx
	$(CXX) $(CXXFLAGS) -o $@ $<

sort-uchar : main.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean :
	rm -f $(GOALS)

.PHONY : clean
