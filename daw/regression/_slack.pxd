import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_intp    SIZE_t           # Type for indices and counters
ctypedef np.npy_int32   INT32_t          # Signed 32 bit integer
ctypedef np.npy_uint32  UINT32_t         # Unsigned 32 bit integer

from ._tree cimport Threshold
from ._tree cimport IntList
from ._heap cimport MinMaxHeap

# constants
cdef DTYPE_t MAX_FLOAT32 = 3.4028235e+38

cdef SIZE_t compute_slack(DTYPE_t**  X,
                          DTYPE_t*   y,
                          IntList*   samples,
                          Threshold* best_threshold,
                          Threshold* second_threshold,
                          SIZE_t     criterion) nogil

# private
cdef DTYPE_t compute_score_gap(MinMaxHeap* s1_left,
                               MinMaxHeap* s1_right,
                               MinMaxHeap* s2_left,
                               MinMaxHeap* s2_right,
                               SIZE_t      criterion) nogil

cdef DTYPE_t reduce_score_gap(DTYPE_t     score_gap,
                              MinMaxHeap* s1_left,
                              MinMaxHeap* s1_right,
                              MinMaxHeap* s2_left,
                              MinMaxHeap* s2_right,
                              Threshold*  split1,
                              Threshold*  split2,
                              SIZE_t      criterion) nogil

cdef void print_split(MinMaxHeap* s_left,
                      MinMaxHeap* s_right) nogil
