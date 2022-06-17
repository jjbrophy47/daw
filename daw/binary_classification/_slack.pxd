import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_intp    SIZE_t           # Type for indices and counters
ctypedef np.npy_int32   INT32_t          # Signed 32 bit integer
ctypedef np.npy_uint32  UINT32_t         # Unsigned 32 bit integer

from ._tree cimport Threshold

# public
cdef SIZE_t compute_leaf_slack(
    SIZE_t n,
    SIZE_t n_pos
) nogil

cdef SIZE_t compute_split_slack_deletion(
    Threshold* best_threshold,
    Threshold* second_threshold,
    SIZE_t     n,
    bint       use_gini
) nogil

cdef SIZE_t compute_split_slack_addition(
    Threshold* best_threshold,
    Threshold* second_threshold,
    SIZE_t     n,
    bint       use_gini
) nogil

# private
cdef DTYPE_t _compute_score_gap(
    Threshold* split1,
    Threshold* split2,
    SIZE_t     n,
    bint       use_gini
) nogil

cdef DTYPE_t _reduce_gap_deletion(
    Threshold* split1,
    Threshold* split2,
    SIZE_t     n,
    bint       use_gini
) nogil

cdef DTYPE_t _reduce_gap_addition(
    Threshold* split1,
    Threshold* split2,
    SIZE_t     n,
    bint       use_gini
) nogil
