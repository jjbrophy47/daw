import numpy as np
cimport numpy as np

from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t

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
    cdef void _increase_capacity(self) nogil
