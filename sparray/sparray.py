"""
Module for sparse arrays using dictionaries. Inspired in part
by ndsparse (https://launchpad.net/ndsparse) by Pim Schellart

Jan Erik Solem, Feb 9 2010.
solem@maths.lth.se (bug reports and feedback welcome)
"""

import numpy

class sparray(object):
    """ Class for n-dimensional sparse array objects using
        Python's dictionary structure.
    """
    def __init__(self, shape, origin=0, default=0, dtype=complex):

        self.__default = default #default value of non-assigned elements
        self.shape = tuple(shape)
        if isinstance(origin,int):
            self.origin = tuple([origin]*len(shape))
        else:
            self.origin = origin
        self.ndim = len(shape)
        self.dtype = dtype
        self.__data = {}

    def __setitem__(self, index, value):
        """ set value to position given in index, where index is a tuple. """
        self.__data[index] = value

    def __getitem__(self, index):
        """ get value at position given in index, where index is a tuple. """
        return self.__data.get(index,self.__default)

    def __delitem__(self, index):
        """ index is tuples of element to be deleted. """
        if index in self.__data:
            del(self.__data[index])

    def __abs__(self):
        """ Absolute value (element wise). """
        if self.dtype == complex:
            dtype = float
        else:
            dtype = self.dtype
        out = self.__class__(self.shape, origin=self.origin, dtype=dtype)
        for k in self.__data.keys():
            out.__data[k] = numpy.abs(self.__data[k])
        return out

    def __add__(self, other):
        """ Add two arrays or add a scalar to all elements of an array. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.__data = self.__data.copy()
            for k in self.__data.keys():
                out.__data[k] = self.__data[k] + other
            # out.__default = self.__default + other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.__data = self.__data.copy()
                for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                    out.__data[k] = out.__data[k] + other.__default
                out.__default = self.__default + other.__default
                for k in other.__data.keys():
                    old_val = out.__data.setdefault(k,self.__default)
                    out.__data[k] = old_val + other.__data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """ Subtract two arrays or substract a scalar to all elements of an array. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.__data = self.__data.copy()
            for k in self.__data.keys():
                out.__data[k] = self.__data[k] - other
            # out.__default = self.__default - other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.__data = self.__data.copy()
                for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                    out.__data[k] = out.__data[k] - other.__default
                out.__default = self.__default - other.__default
                for k in other.__data.keys():
                    old_val = out.__data.setdefault(k,self.__default)
                    out.__data[k] = old_val - other.__data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __mul__(self, other):
        """ Multiply two arrays (element wise) or multiply a scalar to all elements of an array. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.__data = self.__data.copy()
            for k in self.__data.keys():
                out.__data[k] = self.__data[k] * other
            # out.__default = self.__default * other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.__data = self.__data.copy()
                for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                    out.__data[k] = out.__data[k] * other.__default
                out.__default = self.__default * other.__default
                for k in other.__data.keys():
                    old_val = out.__data.setdefault(k,self.__default)
                    out.__data[k] = old_val * other.__data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        """ Divide two arrays (element wise).
            Type of division is determined by dtype.
            Or divide by a scalar all elements of an array. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.__data = self.__data.copy()
            for k in self.__data.keys():
                out.__data[k] = self.__data[k] / other
            # out.__default = self.__default / other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.__data = self.__data.copy()
                for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                    out.__data[k] = out.__data[k] / other.__default
                out.__default = self.__default / other.__default
                for k in other.__data.keys():
                    old_val = out.__data.setdefault(k,self.__default)
                    out.__data[k] = old_val / other.__data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __truediv__(self, other):
        """ Divide two arrays (element wise).
            Type of division is determined by dtype.
            Or divide by a scalar all elements of an array. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.__data = self.__data.copy()
            for k in self.__data.keys():
                out.__data[k] = self.__data[k] / other
            # out.__default = self.__default / other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.__data = self.__data.copy()
                for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                    out.__data[k] = out.__data[k] / other.__default
                out.__default = self.__default / other.__default
                for k in other.__data.keys():
                    old_val = out.__data.setdefault(k,self.__default)
                    out.__data[k] = old_val / other.__data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __floordiv__(self, other):
        """ Floor divide ( // ) two arrays (element wise)
        or floor divide by a scalar all elements of an array. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.__data = self.__data.copy()
            for k in self.__data.keys():
                out.__data[k] = self.__data[k] // other
            # out.__default = self.__default // other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.__data = self.__data.copy()
                for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                    out.__data[k] = out.__data[k] // other.__default
                out.__default = self.__default // other.__default
                for k in other.__data.keys():
                    old_val = out.__data.setdefault(k,self.__default)
                    out.__data[k] = old_val // other.__data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __mod__(self, other):
        """ mod of two arrays (element wise)
        or mod of all elements of an array and a scalar. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.__data = self.__data.copy()
            for k in self.__data.keys():
                out.__data[k] = self.__data[k] % other
            # out.__default = self.__default % other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.__data = self.__data.copy()
                for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                    out.__data[k] = out.__data[k] % other.__default
                out.__default = self.__default % other.__default
                for k in other.__data.keys():
                    old_val = out.__data.setdefault(k,self.__default)
                    out.__data[k] = old_val % other.__data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __pow__(self, other):
        """ power (**) of two arrays (element wise)
        or power of all elements of an array with a scalar. """
        if numpy.isscalar(other):
            out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
            out.__data = self.__data.copy()
            for k in self.__data.keys():
                out.__data[k] = self.__data[k] ** other
            # out.__default = self.__default ** other
            return out
        else:
            if self.shape == other.shape:
                out = self.__class__(self.shape, origin=self.origin, dtype=self.dtype)
                out.__data = self.__data.copy()
                for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                    out.__data[k] = out.__data[k] ** other.__default
                out.__default = self.__default ** other.__default
                for k in other.__data.keys():
                    old_val = out.__data.setdefault(k,self.__default)
                    out.__data[k] = old_val ** other.__data[k]
                return out
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __iadd__(self, other):
        if numpy.isscalar(other):
            for k in self.__data.keys():
                self.__data[k] = self.__data[k] + other
            # self.__default = self.__default + other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                    self.__data[k] = self.__data[k] + other.__default
                self.__default = self.__default + other.__default
                for k in other.__data.keys():
                    old_val = self.__data.setdefault(k,self.__default)
                    self.__data[k] = old_val + other.__data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __isub__(self, other):
        if numpy.isscalar(other):
            for k in self.__data.keys():
                self.__data[k] = self.__data[k] - other
            # self.__default = self.__default - other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                    self.__data[k] = self.__data[k] - other.__default
                self.__default = self.__default - other.__default
                for k in other.__data.keys():
                    old_val = self.__data.setdefault(k,self.__default)
                    self.__data[k] = old_val - other.__data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __imul__(self, other):
        if numpy.isscalar(other):
            for k in self.__data.keys():
                self.__data[k] = self.__data[k] * other
            # self.__default = self.__default * other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                    self.__data[k] = self.__data[k] * other.__default
                self.__default = self.__default * other.__default
                for k in other.__data.keys():
                    old_val = self.__data.setdefault(k,self.__default)
                    self.__data[k] = old_val * other.__data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __idiv__(self, other):
        if numpy.isscalar(other):
            for k in self.__data.keys():
                self.__data[k] = self.__data[k] / other
            # self.__default = self.__default / other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                    self.__data[k] = self.__data[k] / other.__default
                self.__default = self.__default / other.__default
                for k in other.__data.keys():
                    old_val = self.__data.setdefault(k,self.__default)
                    self.__data[k] = old_val / other.__data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __itruediv__(self, other):
        if numpy.isscalar(other):
            for k in self.__data.keys():
                self.__data[k] = self.__data[k] / other
            # self.__default = self.__default / other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                    self.__data[k] = self.__data[k] / other.__default
                self.__default = self.__default / other.__default
                for k in other.__data.keys():
                    old_val = self.__data.setdefault(k,self.__default)
                    self.__data[k] = old_val / other.__data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __ifloordiv__(self, other):
        if numpy.isscalar(other):
            for k in self.__data.keys():
                self.__data[k] = self.__data[k] // other
            # self.__default = self.__default // other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                    self.__data[k] = self.__data[k] // other.__default
                self.__default = self.__default // other.__default
                for k in other.__data.keys():
                    old_val = self.__data.setdefault(k,self.__default)
                    self.__data[k] = old_val // other.__data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __imod__(self, other):
        if numpy.isscalar(other):
            for k in self.__data.keys():
                self.__data[k] = self.__data[k] % other
            # self.__default = self.__default % other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                    self.__data[k] = self.__data[k] % other.__default
                self.__default = self.__default % other.__default
                for k in other.__data.keys():
                    old_val = self.__data.setdefault(k,self.__default)
                    self.__data[k] = old_val % other.__data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __ipow__(self, other):
        if numpy.isscalar(other):
            for k in self.__data.keys():
                self.__data[k] = self.__data[k] ** other
            # self.__default = self.__default ** other
            return self
        else:
            if self.shape == other.shape:
                for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                    self.__data[k] = self.__data[k] ** other.__default
                self.__default = self.__default ** other.__default
                for k in other.__data.keys():
                    old_val = self.__data.setdefault(k,self.__default)
                    self.__data[k] = old_val ** other.__data[k]
                return self
            else:
                raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __str__(self):
        return str(self.dense())

    def dense(self):
        """ Convert to dense NumPy array. """
        out = self.__default * numpy.ones(self.shape, dtype=self.dtype)
        for ind in self.__data:
            shift = tuple(numpy.asarray(ind)-numpy.asarray(self.origin))
            out[shift] = self.__data[ind]
        return out

    def sum(self):
        """ Sum of elements."""
        s = self.__default * numpy.array(self.shape).prod()
        for ind in self.__data:
            s += (self.__data[ind] - self.__default)
        return s

    def get_items(self):
        """ Get multi_indices list with their values. Default values of
        non-assigned elements are not included"""
        return list(self.__data.items())

    def get_values(self):
        """ Get all values. Default values of
        non-assigned elements are not included"""
        return list(self.__data.values())

    def get_multi_indices(self):
        """ Get all multi_indices. Default values of
        non-assigned elements are not included"""
        return list(self.__data.keys())

    def sort(self):
        """ Sort multi_index and values. """
        items = self.get_items()
        items.sort()
        self.__data = {}
        for item in items:
            self.__data[item[0]] = item[1]
        return self

    def conj(self):
        """ Conjugate value (element wise). """
        out = self.__class__(self.shape, origin=self.origin,\
            default=self.__default, dtype=self.dtype)
        for k in self.__data.keys():
            out.__data[k] = numpy.conj(self.__data[k])
        return out

    def hierarchy_augmentation(self, copy=False):
        """ Hierarchy augmentation includes the boundary of
        the multi index set. """
        multi_index = self.get_multi_indices()
        new_multi_index = []
        m = numpy.asarray([3]*self.ndim)
        surrounding = []
        it = numpy.nditer(numpy.ones(tuple(m)),flags=['multi_index'])
        while not it.finished:
            surrounding.append(numpy.asarray(it.multi_index)\
                -numpy.ones(self.ndim,dtype=int))
            it.iternext()
        for index in multi_index:
            for surr_j in surrounding:
                new_multi_index.append(tuple(numpy.asarray(index)+surr_j))
        new_multi_index = list(dict.fromkeys(new_multi_index))
        augm_factor = numpy.asarray([2]*self.ndim)
        augm_shape = tuple(numpy.asarray(self.shape)+augm_factor)
        augm_origin = tuple(numpy.asarray(self.origin)-numpy.ones(self.ndim))
        augm = self.__class__(augm_shape, origin=augm_origin,\
            default=0, dtype=self.dtype)
        for index in new_multi_index:
            augm[index] = 0.0
        if copy:
            for index in multi_index:
                augm[index] = self[index]
        augm.sort()
        return augm
