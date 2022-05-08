# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Module that handles all manipulations to the database.
"""
from cpython cimport Py_INCREF
from cpython.ref cimport PyObject

cimport cython

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf
from libc.time cimport time

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport copy_indices

cdef INT32_t UNDEF = -1

# =====================================
# Manager
# =====================================

cdef class _DataManager:
    """
    Database manager.
    """

    property n_samples:
        def __get__(self):
            return self.n_samples

    property n_features:
        def __get__(self):
            return self.n_features

    def __cinit__(self, float[:, :] X_in, int[:] y_in):
        """
        Constructor.
        """
        cdef SIZE_t n_samples = X_in.shape[0]
        cdef SIZE_t n_features = X_in.shape[1]

        cdef DTYPE_t** X = <DTYPE_t **>malloc(n_samples * sizeof(DTYPE_t *))
        cdef INT32_t*  y = <INT32_t *>malloc(n_samples * sizeof(INT32_t))

        cdef SIZE_t i
        cdef SIZE_t j

        # copy data into C pointer arrays
        for i in range(n_samples):
            X[i] = <DTYPE_t *>malloc(n_features * sizeof(DTYPE_t))
            for j in range(n_features):
                X[i][j] = X_in[i][j]
            y[i] = y_in[i]

        self.X = X
        self.y = y
        self.n_samples = n_samples
        self.n_features = n_features

    def __dealloc__(self):
        """
        Destructor.
        """
        # printf('[M] dealloc\n')
        if self.X:
            for i in range(self.n_samples):
                if self.X[i]:
                    free(self.X[i])
            free(self.X)
        if self.y:
            free(self.y)

    cpdef void clear_data(self):
        """
        Deallaocate data.
        """
        # printf('[M] clear data\n')
        if self.X:
            for i in range(self.n_samples):
                if self.X[i]:
                    free(self.X[i])
            free(self.X)
        if self.y:
            free(self.y)

        self.X = NULL
        self.y = NULL

    cdef void get_data(self, DTYPE_t*** X_ptr, INT32_t** y_ptr) nogil:
        """
        Receive pointers to the data.
        """
        X_ptr[0] = self.X
        y_ptr[0] = self.y
