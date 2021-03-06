import numpy as np
cimport numpy as np

from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t


cdef class _MinMaxHeap:
    """
    Streaming median implementation using a min and a max heap.
    """

    # Inner structures
    cdef _MinHeap min_heap
    cdef _MaxHeap max_heap
    cdef SIZE_t   size
    cdef DTYPE_t  total

    # Python API
    cpdef void insert_list(self, float[:] input_vals)
    cpdef void insert(self, float item)
    cpdef void remove(self, float item)

    # C API
    cdef void _insert_list(self, DTYPE_t* values, SIZE_t n) nogil
    cdef void _insert(self, DTYPE_t x) nogil
    cdef void _remove(self, DTYPE_t x) nogil
    cdef DTYPE_t _median(self) nogil
    cdef DTYPE_t _mean(self) nogil


cdef class _BaseHeap:
    """
    Abstract Heap class.
    """

    # Inner structures
    cdef _List heap

    # Python API
    cpdef void insert(self, float item)
    cpdef float pop(self)
    cpdef float insertpop(self, float item)
    cpdef void remove(self, float item)

    # C API
    cdef void _insert(self, DTYPE_t item) nogil
    cdef DTYPE_t _pop(self) nogil
    cdef DTYPE_t _insertpop(self, DTYPE_t item) nogil  # ABSTRACT
    cdef void _remove(self, DTYPE_t item) nogil
    cdef SIZE_t _size(self) nogil
    cdef DTYPE_t _root(self) nogil

    # private
    cdef SIZE_t _parent_pos(self, SIZE_t pos) nogil
    cdef SIZE_t _left_child_pos(self, SIZE_t pos) nogil
    cdef SIZE_t _right_child_pos(self, SIZE_t pos) nogil
    cdef void _swap(self, SIZE_t i, SIZE_t j) nogil
    cdef void _siftdown(self, SIZE_t start_pos, SIZE_t pos) nogil  # ABSTRACT
    cdef void _siftup(self, SIZE_t pos) nogil  # ABSTRACT
    cdef str _str_helper(self, SIZE_t pos, str indent, bint last)


cdef class _MinHeap(_BaseHeap):
    """
    Min heap implementation.
    """

    # C API
    cdef DTYPE_t _insertpop(self, DTYPE_t item) nogil

    # private
    cdef void _siftdown(self, SIZE_t start_pos, SIZE_t pos) nogil
    cdef void _siftup(self, SIZE_t pos) nogil


cdef class _MaxHeap(_BaseHeap):
    """
    Max heap implementation.
    """

    # C API
    cdef DTYPE_t _insertpop(self, DTYPE_t item) nogil

    # private
    cdef void _siftdown(self, SIZE_t start_pos, SIZE_t pos) nogil
    cdef void _siftup(self, SIZE_t pos) nogil


cdef class _List:
    """
    List data structure with similar functionality
    to Python 3's List type.
    """

    # Inner structures
    cdef DTYPE_t* arr
    cdef SIZE_t   size
    cdef SIZE_t   capacity

    # Python API
    cpdef void append(self, float x)
    cpdef float pop(self)

    # C API
    cdef void _append(self, DTYPE_t x) nogil
    cdef DTYPE_t _pop(self) nogil
    cdef SIZE_t _index(self, DTYPE_t item) nogil

    # private
    cdef void _increase_capacity(self) nogil
