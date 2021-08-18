appname := grasp_tw

CXX := g++
CXXFLAGS := -g -rdynamic -O3 -fopenmp -std=c++17 -Wall -Iinclude -fpic

srcfiles := $(shell find . -maxdepth 7 -name "*.cpp")
objects  := $(patsubst %.cpp, %.o, $(srcfiles))

all: $(appname)

$(appname): $(objects)
	$(CXX) $(CXXFLAGS) -o $(appname) $(objects)

clean:
	rm -f *.o $(objects)

