# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

"""
Implementation of MinMaxHeap for streaming median.
"""
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf

cimport cython


# =======================
# List
# =======================


cdef class _List:
    """
    List data structure.
    """

    # Public Python API
    property size:
        def __get__(self):
            return self.size

    def __cinit__(self, float[:] input_vals):
        """
        Constructor.
        """
        cdef SIZE_t n_samples = input_vals.shape[0]

        cdef DTYPE_t* vals = <DTYPE_t *>malloc(n_samples * sizeof(DTYPE_t))
        cdef SIZE_t i = 0

        # copy data into C pointer array
        for i in range(n_samples):
            vals[i] = input_vals[i]

        self.arr = vals
        self.size = n_samples
        self.capacity = n_samples

    def __str__(self):
        """
        Return string representation of List.
        """
        out_str = '['
        cdef SIZE_t i = 0
        for i in range(self.size):
            out_str += '%.5f' % self.arr[i]
            if i != self.size - 1:
                out_str += ', '
        out_str += ']'
        return out_str

    cpdef void append(self, float x):
        """
        Add item to internal list.
        """
        self._append(<DTYPE_t>x)

    cpdef float pop(self):
        """
        Pop and return the last item.
        """
        return <float> self._pop()

    # Public C API
    cdef void _append(self, DTYPE_t x) nogil:
        """
        Add item to the end of the array,
        double memory capacity if necessary.
        """
        if self.capacity == self.size:
            self._increase_capacity()

        self.arr[self.size] = x
        self.size += 1

    cdef DTYPE_t _pop(self) nogil:
        """
        Pop the last element off of the list and return it.
        """
        if self.size == 0:
            raise ValueError('List is empty!')

        cdef DTYPE_t result = self.arr[self.size - 1]
        self.size -= 1
        return result

    # private
    cdef void _increase_capacity(self) nogil:
        """
        Double array size.
        """
        self.capacity *= 2
        self.arr = <DTYPE_t *>realloc(self.arr, self.capacity * sizeof(DTYPE_t))
