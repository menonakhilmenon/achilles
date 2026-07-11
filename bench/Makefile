CFLAGS := -O3 -march=native -Wall -Wextra -pthread

all: membw ssdread

membw: membw.c
	$(CC) $(CFLAGS) -o $@ $<

ssdread: ssdread.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f membw ssdread
