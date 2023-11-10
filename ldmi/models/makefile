CXX=g++
CXXFLAGS = -std=c++11 -fopenmp -O2 -fPIC -c
SWIGFLAGS = -c++ -python -shadow

# location of Python header files
PYTHON_VERSION = 3.10
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

all: _Lorenz.so _DMF.so _DMF_OU.so

Lorenz : _Lorenz.so

DMF : _DMF.so

DMF_OU : _DMF_OU.so

OBJS = Lorenz_wrap.o _DMF.so _DMF_OU.so

# ------------------------------------------------------
_Lorenz.so: Lorenz_wrap.o
	$(CXX) -shared -fopenmp Lorenz_wrap.o -o _Lorenz.so

Lorenz_wrap.o : Lorenz_wrap.cpp Lorenz.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

Lorenz_wrap.cpp: Lorenz.i
	swig $(SWIGFLAGS) -o Lorenz_wrap.cpp Lorenz.i
# ------------------------------------------------------
_DMF.so: DMF_wrap.o
	$(CXX) -shared -fopenmp DMF_wrap.o -o _DMF.so

DMF_wrap.o : DMF_wrap.cpp DMF.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

DMF_wrap.cpp: DMF.i
	swig $(SWIGFLAGS) -o DMF_wrap.cpp DMF.i
# ------------------------------------------------------
_DMF_OU.so: DMF_OU_wrap.o
	$(CXX) -shared -fopenmp DMF_OU_wrap.o -o _DMF_OU.so

DMF_OU_wrap.o : DMF_OU_wrap.cpp DMF_OU.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

DMF_OU_wrap.cpp: DMF_OU.i
	swig $(SWIGFLAGS) -o DMF_OU_wrap.cpp DMF_OU.i
# ------------------------------------------------------

.PHONEY: clean
clean :
	rm -rf *.o *.so *.pyc
	rm -rf Lorenz_wrap.cpp Lorenz.py
	rm -rf DMF_wrap.cpp DMF.py
	rm -rf DMF_OU_wrap.cpp DMF_OU.py
	rm -rf __pycache__

.PHONEY : clean all Lorenz DMF DMF_OU

