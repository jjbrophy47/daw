import numpy as np
cimport numpy as np

from ._tree cimport IntList
from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t

cdef class _DataManager:
    """
    Manages the database.
    """

    # Internal structure
    cdef SIZE_t    n_samples       # Number of samples
    cdef SIZE_t    n_features      # Number of features
    cdef DTYPE_t** X               # Sample data
    cdef INT32_t*  y               # Label data

    # Python API
    cpdef void clear_data(self)

    # C API
    cdef void get_data(self, DTYPE_t*** X_ptr, INT32_t** y_ptr) nogil
