CFLAGS = -I./alpgrad
LIB_SRCS = alpgrad/src/Layer.c alpgrad/src/MLP.c alpgrad/src/Neuron.c alpgrad/src/Value.c

all: builds builds/example_1 builds/example_2 builds/test

builds:
	mkdir -p builds

builds/example_1: examples/example_1.c
	gcc $(CFLAGS) examples/example_1.c $(LIB_SRCS) -o builds/example_1 -lm

builds/example_2: examples/example_2.c
	gcc $(CFLAGS) examples/example_2.c $(LIB_SRCS) -o builds/example_2 -lm

builds/test: tests/test.c
	gcc $(CFLAGS) tests/test.c $(LIB_SRCS) -o builds/test -lm

clean:
	rm -rf builds

.PHONY: all clean