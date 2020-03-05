CC=gcc
CFLAGS=-O3 -Wall -g -fopenmp

.c.o:
	$(CC) -c $(CFLAGS) $<

all: cfd bin2ppm

clean:
	rm -f bin2ppm cfd *.o output.bin output.ppm

cfd: utilities.o simulation.o constants.h cfd.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

bin2ppm: bin2ppm.o utilities.o constants.h
	$(CC) $(CFLAGS) -o $@ $^ -lm
