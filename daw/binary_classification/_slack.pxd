import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_intp    SIZE_t           # Type for indices and counters
ctypedef np.npy_int32   INT32_t          # Signed 32 bit integer
ctypedef np.npy_uint32  UINT32_t         # Unsigned 32 bit integer

from ._tree cimport Threshold

cdef SIZE_t compute_slack(Threshold* best_threshold,
                          Threshold* second_threshold,
                          SIZE_t     n,
                          bint       use_gini) nogil

cdef DTYPE_t compute_score_gap(Threshold* split1,
                               Threshold* split2,
                               SIZE_t     n,
                               bint       use_gini) nogil

cdef DTYPE_t reduce_score_gap(Threshold* split1,
                              Threshold* split2,
                              SIZE_t     n,
                              bint       use_gini) nogil
