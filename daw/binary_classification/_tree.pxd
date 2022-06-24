import numpy as np
cimport numpy as np

from ._manager cimport _DataManager
from ._splitter cimport SplitRecord
from ._splitter cimport _Splitter
from ._config cimport _Config
from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t

# constants
cdef INT32_t UNDEF = -1
cdef DTYPE_t UNDEF_LEAF_VAL = 0.5

"""
Struct that can be three different types of nodes.


Greedy node:
-Maximizes a split criterion.

Random node:
-Extremely Randomized Node (k=1): https://link.springer.com/article/10.1007/s10994-006-6226-1

Leaf node:
-Stores average label values for prediction.
"""
cdef struct Node:

    # Mandatory node properties
    SIZE_t   node_id                   # Node identifier
    SIZE_t   n_samples                 # Number of samples in the node
    SIZE_t   n_pos_samples             # Number of pos. samples in the node
    SIZE_t   depth                     # Depth of node
    bint     is_left                   # Whether this node is a left child
    Node*    left                      # Left child node
    Node*    right                     # Right child node

    # Greedy node properties
    Feature**  features                # Array of valid feature pointers
    SIZE_t     n_features              # Number of valid feature pointers
    SIZE_t     del_slack               # Minimum number of examples DELETED before structure changes
    SIZE_t     add_slack               # Minimum number of examples ADDED before structure changes

    # Greedy / Random node properties
    IntList*   constant_features       # Array of constant feature indices
    Feature*   chosen_feature          # Chosen feature, if decision node
    Threshold* chosen_threshold        # Chosen threshold, if decision node

    # Leaf-specific properties
    SIZE_t   leaf_id                   # Leaf identifier
    bint     is_leaf                   # Whether this node is a leaf
    DTYPE_t  value                     # Value, if leaf node
    SIZE_t   leaf_slack                # Min. no. deletions/additions to flip leaf prediction
    SIZE_t*  leaf_samples              # Array of sample indices if leaf

"""
Struct to hold feature information: index ID, candidate split array, etc.
"""
cdef struct Feature:

    # Greedy / Random node properties
    SIZE_t      index                 # Feature index pertaining to the original database

    # Greedy node properties
    Threshold** thresholds            # Array of candidate threshold pointers
    SIZE_t      n_thresholds          # Number of candidate thresholds for this feature

"""
Struct to hold metadata pertaining to a particular feature threshold.
"""
cdef struct Threshold:
    SIZE_t  feature               # Feature index
    DTYPE_t value                 # Greedy node: midpoint of v1 and v2, Random node: value
    SIZE_t  n_left_samples        # Number of samples for the left branch
    SIZE_t  n_right_samples       # Number of samples for the right branch
    SIZE_t  n_left_pos_samples    # Number of positive samples for the left branch
    SIZE_t  n_right_pos_samples   # Number of positive samples for the right branch

"""
Structure to hold a SIZE_t pointer and the no. elements.
"""
cdef struct IntList:
    SIZE_t* arr
    SIZE_t  n

cdef class _Tree:
    """
    The Tree object is a binary tree structure constructed by the
    TreeBuilder. The tree structure is used for predictions.
    """

    # Inner structures
    cdef Node*   root                    # Root node
    cdef SIZE_t  node_count_             # Number of nodes in the tree
    cdef SIZE_t  leaf_count_             # Number of leaves in the tree

    # Python API
    cpdef np.ndarray predict(self, float[:, :] X)
    cpdef np.ndarray apply(self, float[:, :] X)
    cpdef np.ndarray get_leaf_slack(self, float[:, :] X)
    cpdef np.ndarray deletion_slack(self, float[:, :] X)
    cpdef np.ndarray addition_slack(self, float[:, :] X)
    cpdef np.ndarray leaf_path(self, DTYPE_t[:, :] X, bint output, bint weighted)
    cpdef SIZE_t get_structure_memory(self)
    cpdef SIZE_t get_decision_stats_memory(self)
    cpdef SIZE_t get_leaf_stats_memory(self)
    cpdef SIZE_t get_node_count(self)
    cpdef SIZE_t get_random_node_count(self, SIZE_t topd)
    cpdef SIZE_t get_greedy_node_count(self, SIZE_t topd)
    cpdef str print_tree(self)

    # C API
    cdef SIZE_t _get_structure_memory(self, Node* node) nogil
    cdef SIZE_t _get_decision_stats_memory(self, Node* node) nogil
    cdef SIZE_t _get_leaf_stats_memory(self, Node* node) nogil
    cdef SIZE_t _get_node_count(self, Node* node) nogil
    cdef SIZE_t _get_random_node_count(self, Node* node, SIZE_t topd) nogil
    cdef SIZE_t _get_greedy_node_count(self, Node* node, SIZE_t topd) nogil
    cdef str _print_tree(self, Node* node, int depth, str indent, bint last)
    cdef str _print_node(self, Node* node)

cdef class _TreeBuilder:
    """
    The TreeBuilder recursively builds a Tree object from training samples,
    using a Splitter object for splitting internal nodes and assigning values to leaves.

    This class controls the various stopping criteria and the node splitting
    evaluation order, e.g. depth-first.
    """

    cdef _DataManager manager              # Database manager
    cdef _Splitter    splitter             # Splitter object that chooses the attribute to split on
    cdef _Config      config               # Configuration object holding training parameters
    cdef SIZE_t       node_count_          # Tracks total number of nodes
    cdef SIZE_t       leaf_count_          # Tracks total number of leaves

    # Python API
    cpdef void build(self, _Tree tree)

    # C API
    cdef Node* _build(self,
                      DTYPE_t** X,
                      INT32_t*  y,
                      IntList*  samples,
                      IntList*  constant_features,
                      SIZE_t    depth,
                      bint      is_left) nogil

    cdef void set_leaf_node(self,
                            Node*   node,
                            IntList* samples) nogil

    cdef Node* initialize_node(self,
                               SIZE_t   depth,
                               bint     is_left,
                               INT32_t* y,
                               IntList* samples,
                               IntList* constant_features) nogil
