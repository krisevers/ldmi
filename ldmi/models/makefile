CXX=g++
CXXFLAGS = -std=c++11 -fopenmp -O2 -fPIC -c
SWIGFLAGS = -c++ -python -shadow

# location of Python header files
PYTHON_VERSION = 3.8
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

all: _Lorenz.so

Lorenz : _Lorenz.so

OBJS = Lorenz_wrap.o

_Lorenz.so: Lorenz_wrap.o
	$(CXX) -shared -fopenmp Lorenz_wrap.o -o _Lorenz.so

Lorenz_wrap.o : Lorenz_wrap.cpp Lorenz.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

Lorenz_wrap.cpp: Lorenz.i
	swig $(SWIGFLAGS) -o Lorenz_wrap.cpp Lorenz.i


.PHONEY: clean
clean :
	rm -rf *.o *.so *.pyc
	rm -rf Lorenz_wrap.cpp Lorenz.py
	rm -rf __pycache__

.PHONEY : clean all Lorenz
