from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t

cdef class _Config:
    """
    Class to hold training parameters.
    """
    cdef SIZE_t       min_samples_split    # Minimum number of samples in an internal node
    cdef SIZE_t       min_samples_leaf     # Minimum number of samples in a leafs
    cdef SIZE_t       max_depth            # Maximal tree depth
    cdef SIZE_t       topd                 # Number of top semi-random layers
    cdef SIZE_t       k                    # Number of candidate thresholds to consider for each feature
    cdef SIZE_t       max_features         # Maximum number of features to consider at each split
    cdef SIZE_t       criterion            # Split-score criterion (0 - absolute error, 1 - squared error)
    cdef UINT32_t     rand_r_state         # sklearn_rand_r random number state
