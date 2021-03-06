import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_intp    SIZE_t           # Type for indices and counters
ctypedef np.npy_int32   INT32_t          # Signed 32 bit integer
ctypedef np.npy_uint32  UINT32_t         # Unsigned 32 bit integer

from ._tree cimport Node
from ._tree cimport Threshold
from ._tree cimport Feature
from ._tree cimport IntList
from ._splitter cimport SplitRecord
from ._heap cimport MinMaxHeap

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


# Sampling methods
cdef UINT32_t our_rand_r(UINT32_t* seed) nogil
cdef SIZE_t rand_int(SIZE_t low, SIZE_t high,
                     UINT32_t* random_state) nogil
cdef double rand_uniform(double    low,
                         double    high,
                         UINT32_t* random_state) nogil

# Split-scoring methods
cdef DTYPE_t compute_split_score(DTYPE_t**  X,
                                 DTYPE_t*   y,
                                 IntList*   samples,
                                 Threshold* threshold,
                                 SIZE_t     criterion) nogil
cdef DTYPE_t _compute_split_score(MinMaxHeap* left,
                                  MinMaxHeap* right,
                                  SIZE_t      criterion) nogil
cdef DTYPE_t compute_leaf_value(IntList* samples,
                                DTYPE_t* y,
                                SIZE_t   criterion) nogil

# Feature / threshold methods
cdef Feature* create_feature(SIZE_t feature_index) nogil
cdef Threshold* create_threshold(SIZE_t feature_index,
                                 DTYPE_t value) nogil
cdef Feature** copy_features(Feature** features,
                             SIZE_t n_features) nogil
cdef Feature* copy_feature(Feature* feature) nogil
cdef Threshold** copy_thresholds(Threshold** thresholds,
                                 SIZE_t n_thresholds) nogil
cdef Threshold* copy_threshold(Threshold* threshold) nogil
cdef void free_features(Feature** features,
                        SIZE_t n_features) nogil
cdef void free_feature(Feature* feature) nogil
cdef void free_thresholds(Threshold** thresholds,
                          SIZE_t n_thresholds) nogil

# IntList methods
cdef IntList* create_intlist(SIZE_t n_elem, bint initialize) nogil
cdef IntList* copy_intlist(IntList* obj, SIZE_t n_elem) nogil
cdef void free_intlist(IntList* obj) nogil

# Array methods
cdef SIZE_t* convert_int_ndarray(np.ndarray arr)
cdef INT32_t* copy_int_array(INT32_t* arr,
                             SIZE_t n_elem) nogil
cdef SIZE_t* copy_indices(SIZE_t* arr,
                          SIZE_t n_elem) nogil

# Utility methods
cdef DTYPE_t* extract_labels(IntList* samples, DTYPE_t* y) nogil
cdef DTYPE_t compute_mean(DTYPE_t* vals, SIZE_t n) nogil
cdef DTYPE_t compute_median(DTYPE_t* vals, SIZE_t n) nogil

# Node methods
cdef SIZE_t split_labels(DTYPE_t**  X,
                         DTYPE_t*   y,
                         IntList*   samples,
                         Threshold* threshold,
                         DTYPE_t**  y_left,
                         DTYPE_t**  y_right) nogil
cdef void split_samples(Node*        node,
                        DTYPE_t**    X,
                        DTYPE_t*     y,
                        IntList*     samples,
                        SplitRecord* split,
                        bint         copy_constant_features) nogil
cdef void dealloc(Node *node) nogil
