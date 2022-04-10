CXX    = g++
CFLAGS = -fpic -Wall -O3 -std=c++11

all: find_abstracts find_abstracts_multithread

clean:
	rm -f find_abstracts find_abstracts_multithread *.o

find_abstracts: find_abstracts.cpp
	$(CXX) $(CFLAGS) -pthread -o $@ $< -lboost_regex -lboost_iostreams -lm

find_abstracts_multithread: find_abstracts_multithread.cpp
	$(CXX) $(CFLAGS) -pthread -o $@ $< -lboost_regex -lboost_iostreams -lm
