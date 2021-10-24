CFLAGS += -I../include -I../util

CFLAGS += -std=c99

LDFLAGS += -L../lib
LDFLAGS := -l:libmpi-sort.a -lm $(LDFLAGS)

test : main.o ../xtract.o
	$(CC) $(CFLAGS) -o $@ $^

%.o : %.c
	$(CC) $(CFLAGS) -c $<

clean :
	rm -f test *.o

.PHONY : clean
