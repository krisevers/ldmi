# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _Lorenz
else:
    import _Lorenz

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _Lorenz.delete_SwigPyIterator

    def value(self):
        return _Lorenz.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _Lorenz.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _Lorenz.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _Lorenz.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _Lorenz.SwigPyIterator_equal(self, x)

    def copy(self):
        return _Lorenz.SwigPyIterator_copy(self)

    def next(self):
        return _Lorenz.SwigPyIterator_next(self)

    def __next__(self):
        return _Lorenz.SwigPyIterator___next__(self)

    def previous(self):
        return _Lorenz.SwigPyIterator_previous(self)

    def advance(self, n):
        return _Lorenz.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _Lorenz.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _Lorenz.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _Lorenz.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _Lorenz.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _Lorenz.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _Lorenz.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _Lorenz:
_Lorenz.SwigPyIterator_swigregister(SwigPyIterator)

class IntVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _Lorenz.IntVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _Lorenz.IntVector___nonzero__(self)

    def __bool__(self):
        return _Lorenz.IntVector___bool__(self)

    def __len__(self):
        return _Lorenz.IntVector___len__(self)

    def __getslice__(self, i, j):
        return _Lorenz.IntVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _Lorenz.IntVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _Lorenz.IntVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _Lorenz.IntVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _Lorenz.IntVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _Lorenz.IntVector___setitem__(self, *args)

    def pop(self):
        return _Lorenz.IntVector_pop(self)

    def append(self, x):
        return _Lorenz.IntVector_append(self, x)

    def empty(self):
        return _Lorenz.IntVector_empty(self)

    def size(self):
        return _Lorenz.IntVector_size(self)

    def swap(self, v):
        return _Lorenz.IntVector_swap(self, v)

    def begin(self):
        return _Lorenz.IntVector_begin(self)

    def end(self):
        return _Lorenz.IntVector_end(self)

    def rbegin(self):
        return _Lorenz.IntVector_rbegin(self)

    def rend(self):
        return _Lorenz.IntVector_rend(self)

    def clear(self):
        return _Lorenz.IntVector_clear(self)

    def get_allocator(self):
        return _Lorenz.IntVector_get_allocator(self)

    def pop_back(self):
        return _Lorenz.IntVector_pop_back(self)

    def erase(self, *args):
        return _Lorenz.IntVector_erase(self, *args)

    def __init__(self, *args):
        _Lorenz.IntVector_swiginit(self, _Lorenz.new_IntVector(*args))

    def push_back(self, x):
        return _Lorenz.IntVector_push_back(self, x)

    def front(self):
        return _Lorenz.IntVector_front(self)

    def back(self):
        return _Lorenz.IntVector_back(self)

    def assign(self, n, x):
        return _Lorenz.IntVector_assign(self, n, x)

    def resize(self, *args):
        return _Lorenz.IntVector_resize(self, *args)

    def insert(self, *args):
        return _Lorenz.IntVector_insert(self, *args)

    def reserve(self, n):
        return _Lorenz.IntVector_reserve(self, n)

    def capacity(self):
        return _Lorenz.IntVector_capacity(self)
    __swig_destroy__ = _Lorenz.delete_IntVector

# Register IntVector in _Lorenz:
_Lorenz.IntVector_swigregister(IntVector)

class DoubleVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _Lorenz.DoubleVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _Lorenz.DoubleVector___nonzero__(self)

    def __bool__(self):
        return _Lorenz.DoubleVector___bool__(self)

    def __len__(self):
        return _Lorenz.DoubleVector___len__(self)

    def __getslice__(self, i, j):
        return _Lorenz.DoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _Lorenz.DoubleVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _Lorenz.DoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _Lorenz.DoubleVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _Lorenz.DoubleVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _Lorenz.DoubleVector___setitem__(self, *args)

    def pop(self):
        return _Lorenz.DoubleVector_pop(self)

    def append(self, x):
        return _Lorenz.DoubleVector_append(self, x)

    def empty(self):
        return _Lorenz.DoubleVector_empty(self)

    def size(self):
        return _Lorenz.DoubleVector_size(self)

    def swap(self, v):
        return _Lorenz.DoubleVector_swap(self, v)

    def begin(self):
        return _Lorenz.DoubleVector_begin(self)

    def end(self):
        return _Lorenz.DoubleVector_end(self)

    def rbegin(self):
        return _Lorenz.DoubleVector_rbegin(self)

    def rend(self):
        return _Lorenz.DoubleVector_rend(self)

    def clear(self):
        return _Lorenz.DoubleVector_clear(self)

    def get_allocator(self):
        return _Lorenz.DoubleVector_get_allocator(self)

    def pop_back(self):
        return _Lorenz.DoubleVector_pop_back(self)

    def erase(self, *args):
        return _Lorenz.DoubleVector_erase(self, *args)

    def __init__(self, *args):
        _Lorenz.DoubleVector_swiginit(self, _Lorenz.new_DoubleVector(*args))

    def push_back(self, x):
        return _Lorenz.DoubleVector_push_back(self, x)

    def front(self):
        return _Lorenz.DoubleVector_front(self)

    def back(self):
        return _Lorenz.DoubleVector_back(self)

    def assign(self, n, x):
        return _Lorenz.DoubleVector_assign(self, n, x)

    def resize(self, *args):
        return _Lorenz.DoubleVector_resize(self, *args)

    def insert(self, *args):
        return _Lorenz.DoubleVector_insert(self, *args)

    def reserve(self, n):
        return _Lorenz.DoubleVector_reserve(self, n)

    def capacity(self):
        return _Lorenz.DoubleVector_capacity(self)
    __swig_destroy__ = _Lorenz.delete_DoubleVector

# Register DoubleVector in _Lorenz:
_Lorenz.DoubleVector_swigregister(DoubleVector)

class DoubleVector2(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _Lorenz.DoubleVector2_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _Lorenz.DoubleVector2___nonzero__(self)

    def __bool__(self):
        return _Lorenz.DoubleVector2___bool__(self)

    def __len__(self):
        return _Lorenz.DoubleVector2___len__(self)

    def __getslice__(self, i, j):
        return _Lorenz.DoubleVector2___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _Lorenz.DoubleVector2___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _Lorenz.DoubleVector2___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _Lorenz.DoubleVector2___delitem__(self, *args)

    def __getitem__(self, *args):
        return _Lorenz.DoubleVector2___getitem__(self, *args)

    def __setitem__(self, *args):
        return _Lorenz.DoubleVector2___setitem__(self, *args)

    def pop(self):
        return _Lorenz.DoubleVector2_pop(self)

    def append(self, x):
        return _Lorenz.DoubleVector2_append(self, x)

    def empty(self):
        return _Lorenz.DoubleVector2_empty(self)

    def size(self):
        return _Lorenz.DoubleVector2_size(self)

    def swap(self, v):
        return _Lorenz.DoubleVector2_swap(self, v)

    def begin(self):
        return _Lorenz.DoubleVector2_begin(self)

    def end(self):
        return _Lorenz.DoubleVector2_end(self)

    def rbegin(self):
        return _Lorenz.DoubleVector2_rbegin(self)

    def rend(self):
        return _Lorenz.DoubleVector2_rend(self)

    def clear(self):
        return _Lorenz.DoubleVector2_clear(self)

    def get_allocator(self):
        return _Lorenz.DoubleVector2_get_allocator(self)

    def pop_back(self):
        return _Lorenz.DoubleVector2_pop_back(self)

    def erase(self, *args):
        return _Lorenz.DoubleVector2_erase(self, *args)

    def __init__(self, *args):
        _Lorenz.DoubleVector2_swiginit(self, _Lorenz.new_DoubleVector2(*args))

    def push_back(self, x):
        return _Lorenz.DoubleVector2_push_back(self, x)

    def front(self):
        return _Lorenz.DoubleVector2_front(self)

    def back(self):
        return _Lorenz.DoubleVector2_back(self)

    def assign(self, n, x):
        return _Lorenz.DoubleVector2_assign(self, n, x)

    def resize(self, *args):
        return _Lorenz.DoubleVector2_resize(self, *args)

    def insert(self, *args):
        return _Lorenz.DoubleVector2_insert(self, *args)

    def reserve(self, n):
        return _Lorenz.DoubleVector2_reserve(self, n)

    def capacity(self):
        return _Lorenz.DoubleVector2_capacity(self)
    __swig_destroy__ = _Lorenz.delete_DoubleVector2

# Register DoubleVector2 in _Lorenz:
_Lorenz.DoubleVector2_swigregister(DoubleVector2)

class Sim(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, dt, rho, sigma, beta, t_sim, y, progress=True):
        _Lorenz.Sim_swiginit(self, _Lorenz.new_Sim(dt, rho, sigma, beta, t_sim, y, progress))

    def derivative(self, y, dydt, t):
        return _Lorenz.Sim_derivative(self, y, dydt, t)

    def integrate(self, method):
        return _Lorenz.Sim_integrate(self, method)

    def eulerIntegrate(self):
        return _Lorenz.Sim_eulerIntegrate(self)

    def euler(self, y, t):
        return _Lorenz.Sim_euler(self, y, t)

    def rk4Integrate(self):
        return _Lorenz.Sim_rk4Integrate(self)

    def rk4(self, y, t):
        return _Lorenz.Sim_rk4(self, y, t)

    def get_states(self):
        return _Lorenz.Sim_get_states(self)

    def get_times(self):
        return _Lorenz.Sim_get_times(self)
    __swig_destroy__ = _Lorenz.delete_Sim

# Register Sim in _Lorenz:
_Lorenz.Sim_swigregister(Sim)



