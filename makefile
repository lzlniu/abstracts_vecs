CXX    = g++
CFLAGS = -fpic -Wall -O3 -std=c++11

all: find_abstracts

clean:
	rm -f find_abstracts *.o

find_abstracts: find_abstracts.cpp
	$(CXX) $(CFLAGS) -pthread -o $@ $< -lboost_regex -lboost_iostreams -lm
