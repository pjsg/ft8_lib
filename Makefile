# Use -DUSE_KISS if you want the kiss fft library
# By default, this builds both a decode_ft8 (using FFTW3) and decode_ft8_kiss (using the KISS library)
# It is only the decode_ft8.c file that depends on the USE_KISS define.

SANITIZE = 
BSD = -lbsd

CFLAGS_NOLTO = -march=native -ffast-math $(SANITIZE) -O3
CFLAGS = $(CFLAGS_NOLTO) -flto -g # -fprofile-use
CPPFLAGS = -std=c11 -I. $(shell pkg-config --cflags fftw3f)
LDFLAGS = -lm $(shell pkg-config --libs fftw3f) $(BSD) $(CFLAGS)
LDFLAGS_KISS = -lm -flto $(BSD)
CC = gcc

TARGETS = gen_ft8 decode_ft8 test test_refine decode_ft8_kiss correlate

.PHONY: run_tests all clean

all: $(TARGETS)

run_tests: test
	@./test

%.s: %.c
	$(CC) $(CFLAGS_NOLTO) -S $< -o $@

gen_ft8: gen_ft8.o ft8/constants.o ft8/text.o ft8/pack.o ft8/encode.o ft8/crc.o common/wave.o
	$(CC) -o $@ $^ $(LDFLAGS)

test:  test.o ft8/pack.o ft8/encode.o ft8/crc.o ft8/text.o ft8/constants.o
	$(CC) -o $@ $^ $(LDFLAGS)

decode_ft8: main.c decode_ft8.o refine.o ft8/decode.o ft8/encode.o ft8/crc.o ft8/ldpc.o ft8/unpack.o ft8/text.o ft8/constants.o common/wave.o brent.o
	$(CC) -g -o $@ $^ $(LDFLAGS) 

test_refine: test_refine.o refine.o ft8/pack.o ft8/encode.o ft8/crc.o ft8/text.o ft8/constants.o common/wave.o fft/kiss_fft.o
	$(CC) -o $@ $^ $(LDFLAGS)

decode_ft8_kiss: main.o decode_ft8_kiss.o ft8/decode.o ft8/encode.o ft8/crc.o ft8/ldpc.o ft8/unpack.o ft8/text.o ft8/constants.o common/wave.o fft/kiss_fft.o fft/kiss_fftr.o
	$(CC) -o $@ $^ $(LDFLAGS_KISS) 

decode_ft8_kiss.o: decode_ft8.c
	$(CC) -c -o $@ $(CFLAGS) $(CCFLAGS) -DUSE_KISS $^ 

correlate: correlate.c ft8/pack.o refine.o ft8/text.o ft8/encode.o ft8/crc.o ft8/constants.o common/wave.o ft8/ldpc.o ft8/decode.c brent.o
	$(CC) -g -O3 -march=native -ffast-math -I. correlate.c ft8/pack.o refine.o brent.o ft8/text.o ft8/encode.o ft8/crc.o ft8/constants.o common/wave.o ft8/decode.o ft8/ldpc.o -lm -flto $(shell pkg-config --libs fftw3f) -o correlate $(BSD)

clean:
	rm -f *.o *.a ft8/*.o common/*.o fft/*.o $(TARGETS)
install:
	$(AR) rc libft8.a ft8/constants.o ft8/encode.o ft8/pack.o ft8/text.o common/wave.o
#	install libft8.a /usr/local/lib/libft8.a
	install decode_ft8 /usr/local/bin
