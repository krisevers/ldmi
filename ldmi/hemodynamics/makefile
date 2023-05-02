CXX=g++
CXXFLAGS = -std=c++11 -fopenmp -O2 -fPIC -c
SWIGFLAGS = -c++ -python -shadow -w305

# location of Python header files
PYTHON_VERSION = 3.8
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

all: _LBR.so

LBR : _LBR.so

OBJS = LBR_wrap.o

# ------------------------------------------------------
_LBR.so: LBR_wrap.o
	$(CXX) -shared -fopenmp LBR_wrap.o -o _LBR.so

LBR_wrap.o : LBR_wrap.cpp LBR.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

LBR_wrap.cpp: LBR.i
	swig $(SWIGFLAGS) -o LBR_wrap.cpp LBR.i
# ------------------------------------------------------

.PHONEY: clean
clean :
	rm -rf *.o *.so *.pyc
	rm -rf LBR_wrap.cpp LBR.py
	rm -rf __pycache__

.PHONEY : clean all LBR

