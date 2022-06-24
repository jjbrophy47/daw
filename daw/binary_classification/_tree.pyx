# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Tree and tree builder objects.
"""
cimport cython

from cpython cimport Py_INCREF
from cpython.ref cimport PyObject

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf
from libc.math cimport pow

import numpy as np
cimport numpy as np
np.import_array()

from ._slack cimport compute_leaf_slack
from ._utils cimport split_samples
from ._utils cimport copy_indices
from ._utils cimport create_intlist
from ._utils cimport free_intlist
from ._utils cimport dealloc

# =====================================
# TreeBuilder
# =====================================

cdef class _TreeBuilder:
    """
    Build a decision tree in depth-first fashion.
    """

    def __cinit__(self,
                  _DataManager manager,
                  _Splitter    splitter,
                  _Config      config):

        self.manager = manager
        self.splitter = splitter
        self.config = config

    cpdef void build(self, _Tree tree):
        """
        Build a decision tree from the training set (X, y).
        """

        # Data containers
        cdef DTYPE_t** X = NULL
        cdef INT32_t*  y = NULL
        self.manager.get_data(&X, &y)

        # create list of sample indices
        cdef IntList* samples = create_intlist(self.manager.n_samples, 1)
        for i in range(samples.n):
            samples.arr[i] = i

        # initialize container for constant features
        cdef IntList* constant_features = create_intlist(self.manager.n_features, 0)

        self.node_count_ = 0
        self.leaf_count_ = 0
        tree.root = self._build(X, y, samples, constant_features, 0, 0)
        tree.node_count_ = self.node_count_
        tree.leaf_count_ = self.leaf_count_

    cdef Node* _build(self,
                      DTYPE_t** X,
                      INT32_t*  y,
                      IntList*  samples,
                      IntList*  constant_features,
                      SIZE_t    depth,
                      bint      is_left) nogil:
        """
        Build a subtree given a partition of samples.
        """

        # create node
        # printf('[B] initializing node\n')
        cdef Node *node = self.initialize_node(depth, is_left, y, samples, constant_features)
        # printf('[B] done initializing node\n')

        # data variables
        cdef SIZE_t n_total_features = self.manager.n_features

        # boolean variables
        cdef bint is_bottom_leaf = (depth >= self.config.max_depth)
        cdef bint is_middle_leaf = (samples.n < self.config.min_samples_split or
                                    samples.n < 2 * self.config.min_samples_leaf or
                                    node.n_pos_samples == 0 or
                                    node.n_pos_samples == node.n_samples)

        # result containers
        cdef SplitRecord split
        cdef SIZE_t      n_usable_thresholds = 0

        # printf('\n[B] samples.n: %ld, depth: %ld, is_left: %d\n', samples.n, depth, is_left)

        # leaf node
        if is_bottom_leaf or is_middle_leaf:
            # printf('[B] bottom / middle leaf\n')
            self.set_leaf_node(node, samples)
            # printf('[B] leaf.value: %.2f\n', node.value)

        # leaf or decision node
        else:

            # select a threshold to to split the samples
            # printf('[B] select threshold\n')
            n_usable_thresholds = self.splitter.select_threshold(node, X, y, samples, n_total_features)
            # printf('[B] no_usable_thresholds: %ld\n', n_usable_thresholds)

            # no usable thresholds, create leaf
            if n_usable_thresholds == 0:
                # printf('no usable thresholds\n')
                dealloc(node)  # free allocated memory
                self.set_leaf_node(node, samples)
                # printf('[B] leaf.value: %.2f\n', node.value)

            # decision node
            else:
                # printf('[B] split samples\n')
                split_samples(node, X, y, samples, &split, 1)
                # printf('[B] depth: %ld, chosen_feature.index: %ld, chosen_threshold.value: %.2f\n',
                #       node.depth, node.chosen_feature.index, node.chosen_threshold.value)

                # printf('[B] split.left_samples.n: %ld, split.right_samples.n: %ld\n',
                #        split.left_samples.n, split.right_samples.n)

                # traverse to left and right branches
                node.left = self._build(X, y, split.left_samples, split.left_constant_features, depth + 1, 1)
                node.right = self._build(X, y, split.right_samples, split.right_constant_features, depth + 1, 0)

        return node

    cdef void set_leaf_node(self,
                            Node*    node,
                            IntList* samples) nogil:
        """
        Compute leaf value and set all other attributes.
        """
        # set leaf node properties
        node.leaf_id = self.leaf_count_
        node.is_leaf = True
        node.leaf_samples = copy_indices(samples.arr, samples.n)
        node.leaf_slack = compute_leaf_slack(node.n_samples, node.n_pos_samples)
        node.value = node.n_pos_samples / <double> node.n_samples

        # set greedy node properties
        node.features = NULL
        node.n_features = 0
        node.del_slack = -2
        node.add_slack = -2

        # set greedy / random node properties
        if node.constant_features != NULL:
            free_intlist(node.constant_features)
        node.constant_features = NULL
        node.chosen_feature = NULL
        node.chosen_threshold = NULL

        # no children
        node.left = NULL
        node.right = NULL

        # free samples
        free_intlist(samples)

        self.leaf_count_ += 1

    cdef Node* initialize_node(self,
                               SIZE_t   depth,
                               bint     is_left,
                               INT32_t* y,
                               IntList* samples,
                               IntList* constant_features) nogil:
        """
        Create and initialize a new node.
        """

        # compute number of positive samples
        cdef SIZE_t n_pos_samples = 0
        # printf('[B - IN] computing n_pos_samples\n')
        # printf('[B - IN] samples.n: %lu\n', samples.n)
        for i in range(samples.n):
            n_pos_samples += y[samples.arr[i]]

        # printf('[B - IN] n_pos_samples: %lu\n', n_pos_samples)

        # create node
        cdef Node *node = <Node *>malloc(sizeof(Node))

        # initialize mandatory properties
        node.node_id = self.node_count_
        node.n_samples = samples.n
        node.n_pos_samples = n_pos_samples
        node.depth = depth
        node.is_left = is_left
        node.left = NULL
        node.right = NULL

        # initialize greedy node properties
        node.features = NULL
        node.n_features = 0
        node.del_slack = -1
        node.add_slack = -1

        # initialize greedy / random node properties
        node.constant_features = constant_features
        node.chosen_feature = NULL
        node.chosen_threshold = NULL

        # initialize leaf-specific properties
        node.leaf_id = -1
        node.is_leaf = False
        node.value = UNDEF_LEAF_VAL
        node.leaf_slack = -1
        node.leaf_samples = NULL

        self.node_count_ += 1

        return node


# =====================================
# Tree
# =====================================

cdef class _Tree:

    property node_count_:
        def __get__(self):
            return self.node_count_

    property leaf_count_:
        def __get__(self):
            return self.leaf_count_

    def __cinit__(self):
        """
        Constructor.
        """
        self.root = NULL
        self.node_count_ = 0
        self.leaf_count_ = 0

    def __dealloc__(self):
        """
        Destructor.
        """
        # printf('deallocing tree\n')
        if self.root:
            dealloc(self.root)
            free(self.root)
        self.node_count_ = 0
        self.leaf_count_ = 0

    cpdef np.ndarray predict(self, float[:,:] X):
        """
        Predict probability of positive label for X.
        """

        # In / out
        cdef SIZE_t n_samples = X.shape[0]
        cdef np.ndarray[float] out = np.zeros((n_samples,), dtype=np.float32)

        # Incrementers
        cdef SIZE_t i = 0
        cdef Node* node

        with nogil:

            for i in range(n_samples):
                node = self.root

                while not node.is_leaf:
                    if X[i, node.chosen_feature.index] <= node.chosen_threshold.value:
                        node = node.left
                    else:
                        node = node.right

                out[i] = node.value

        return out

    cpdef np.ndarray apply(self, float[:, :] X):
        """
        Predict leaf index for x in X.
        """

        # In / out
        cdef SIZE_t n_samples = X.shape[0]
        cdef np.ndarray[int] out = np.zeros((n_samples,), dtype=np.int32)

        # Incrementers
        cdef SIZE_t i = 0
        cdef Node*  node = NULL

        with nogil:

            for i in range(n_samples):
                node = self.root

                while not node.is_leaf:
                    if X[i, node.chosen_feature.index] <= node.chosen_threshold.value:
                        node = node.left
                    else:
                        node = node.right

                out[i] = node.leaf_id

        return out

    cpdef np.ndarray get_leaf_slack(self, float[:, :] X):
        """
        Predict leaf index for x in X.
        """

        # In / out
        cdef SIZE_t n_samples = X.shape[0]
        cdef np.ndarray[int] out = np.zeros((n_samples,), dtype=np.int32)

        # Incrementers
        cdef SIZE_t i = 0
        cdef SIZE_t j = 0
        cdef Node*  node = NULL

        with nogil:

            for i in range(n_samples):
                node = self.root

                while not node.is_leaf:
                    if X[i, node.chosen_feature.index] <= node.chosen_threshold.value:
                        node = node.left
                    else:
                        node = node.right

                out[i] = node.leaf_slack

        return out

    cpdef np.ndarray deletion_slack(self, float[:, :] X):
        """
        Predict leaf index for x in X.
        """

        # In / out
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_nodes = self._get_node_count(self.root)
        cdef np.ndarray[int, ndim=2] out = np.zeros((n_samples, n_nodes), dtype=np.int32)

        # Incrementers
        cdef SIZE_t i = 0
        cdef SIZE_t j = 0
        cdef Node*  node = NULL

        with nogil:

            for i in range(n_samples):
                node = self.root
                j = 0
                out[i, j] = node.del_slack
                j += 1

                while not node.is_leaf:
                    if X[i, node.chosen_feature.index] <= node.chosen_threshold.value:
                        node = node.left
                    else:
                        node = node.right

                    out[i, j] = node.del_slack
                    j += 1

        return out

    cpdef np.ndarray addition_slack(self, float[:, :] X):
        """
        Predict leaf index for x in X.
        """

        # In / out
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_nodes = self._get_node_count(self.root)
        cdef np.ndarray[int, ndim=2] out = np.zeros((n_samples, n_nodes), dtype=np.int32)

        # Incrementers
        cdef SIZE_t i = 0
        cdef SIZE_t j = 0
        cdef Node*  node = NULL

        with nogil:

            for i in range(n_samples):
                node = self.root
                j = 0
                out[i, j] = node.add_slack
                j += 1

                while not node.is_leaf:
                    if X[i, node.chosen_feature.index] <= node.chosen_threshold.value:
                        node = node.left
                    else:
                        node = node.right

                    out[i, j] = node.add_slack
                    j += 1

        return out

    cpdef np.ndarray leaf_path(self, DTYPE_t[:, :] X, bint output, bint weighted):
        """
        Return 2d vector of leaf one-hot encodings, shape=(X.shape[0], no. leaves).
        """

        # In / out
        cdef SIZE_t        n_samples = X.shape[0]
        cdef SIZE_t        n_leaves = self.leaf_count_
        cdef DTYPE_t[:, :] out = np.zeros((n_samples, n_leaves), dtype=np.float32)
        cdef DTYPE_t       val = 1.0

        # Incrementers
        cdef SIZE_t i = 0
        cdef Node*  node = NULL

        with nogil:

            for i in range(n_samples):
                node = self.root

                while not node.is_leaf:
                    if X[i, node.chosen_feature.index] <= node.chosen_threshold.value:
                        node = node.left
                    else:
                        node = node.right

                val = 1.0

                if output:
                    val = node.value

                if weighted:
                    val /= <DTYPE_t> node.n_samples

                out[i][node.leaf_id] = val

        return np.asarray(out)

    # tree information
    cpdef SIZE_t get_structure_memory(self):
        """
        Return total memory (in bytes) of the tree that
        is ONLY used for generating predictions.
        """
        return self._get_structure_memory(self.root)

    cpdef SIZE_t get_decision_stats_memory(self):
        """
        Return total memory (in bytes) of the tree that
        is used for storing additional decision-node statistics.
        """
        return self._get_decision_stats_memory(self.root)

    cpdef SIZE_t get_leaf_stats_memory(self):
        """
        Return total memory (in bytes) of the tree that
        is used for storing additional leaf-node statistics.
        """
        return self._get_leaf_stats_memory(self.root)

    cpdef SIZE_t get_node_count(self):
        """
        Get number of nodes total.
        """
        return self._get_node_count(self.root)

    cpdef SIZE_t get_random_node_count(self, SIZE_t topd):
        """
        Count number of exact nodes in the top d layers.
        """
        return self._get_random_node_count(self.root, topd)

    cpdef SIZE_t get_greedy_node_count(self, SIZE_t topd):
        """
        Count number of greedy nodes.
        """
        return self._get_greedy_node_count(self.root, topd)

    cpdef str print_tree(self):
        """
        Return a string representation of the tree structure.
        """
        return self._print_tree(self.root, 0, '', False)

    # private
    cdef SIZE_t _get_structure_memory(self, Node* node) nogil:
        """
        Return total memory (in bytes) of the tree that is
        ONLY used for generating predictions.
        """
        cdef SIZE_t result = 0

        # end of traversal
        if not node:
            return result

        # current node
        else:

            # structual information
            result += sizeof(node.node_id)
            result += sizeof(node.left)
            result += sizeof(node.right)

            # leaf info
            result += sizeof(node.leaf_id)
            result += sizeof(node.is_leaf)
            result += sizeof(node.value)

            # not strictly necessary, but useful information
            result += sizeof(node.depth)
            result += sizeof(node.is_left)

            # decision node
            if not node.is_leaf:
                result += sizeof(node.chosen_feature.index)
                result += sizeof(node.chosen_threshold.value)

            return result + self._get_structure_memory(node.left) + self._get_structure_memory(node.right)

    cdef SIZE_t _get_decision_stats_memory(self, Node* node) nogil:
        """
        Return total memory (in bytes) of the tree.
        """
        cdef SIZE_t result = 0

        # end of traversal
        if not node:
            return result

        # decision node
        if not node.is_leaf:

            # add instance counts
            result += sizeof(node.n_samples)
            result += sizeof(node.n_pos_samples)

            # add constant features memory usage
            result += sizeof(node.constant_features)
            result += sizeof(node.constant_features[0])
            result += sizeof(node.constant_features.arr[0]) * node.constant_features.n

            # add chosen feature memory usage
            result += sizeof(node.chosen_feature)
            result += sizeof(node.chosen_feature[0]) - sizeof(node.chosen_feature.index)
            result += sizeof(node.chosen_feature.thresholds[0][0]) * node.chosen_feature.n_thresholds

            # add chosen threshold memory usage
            result += sizeof(node.chosen_threshold[0]) - sizeof(node.chosen_threshold.value)

            # greedy node
            if node.features != NULL:

                # add attribute-threshold-pairs memory-usage
                result += sizeof(node.features)
                result += sizeof(node.n_features)

                for i in range(node.n_features):
                    result += sizeof(node.features[i][0])
                    result += sizeof(node.features[i].thresholds[0][0]) * node.features[i].n_thresholds

        return result + self._get_decision_stats_memory(node.left) + self._get_decision_stats_memory(node.right)

    cdef SIZE_t _get_leaf_stats_memory(self, Node* node) nogil:
        """
        Return total memory (in bytes) of the tree.
        """
        cdef SIZE_t result = 0

        # end of traversal
        if not node:
            return result

        # leaf node
        if node.is_leaf:

            # add instance counts
            result += sizeof(node.n_samples)
            result += sizeof(node.n_pos_samples)

            # add instance pointers
            result += sizeof(node.leaf_samples)
            result += sizeof(node.leaf_samples[0]) * node.n_samples

        return result + self._get_leaf_stats_memory(node.left) + self._get_leaf_stats_memory(node.right)

    cdef SIZE_t _get_node_count(self, Node* node) nogil:
        """
        Count total no. of nodes in the tree.
        """
        if not node:
            return 0
        else:
            return 1 + self._get_node_count(node.left) + self._get_node_count(node.right)

    cdef SIZE_t _get_random_node_count(self, Node* node, SIZE_t topd) nogil:
        """
        Count no. random nodes in the tree.
        """
        cdef SIZE_t result = 0

        if not node or node.is_leaf:
            result = 0

        else:
            if node.depth < topd:
                result = 1

            result += self._get_random_node_count(node.left, topd)
            result += self._get_random_node_count(node.right, topd)

        return result

    cdef SIZE_t _get_greedy_node_count(self, Node* node, SIZE_t topd) nogil:
        """
        Count no. greedy nodes in the tree.
        """
        cdef SIZE_t result = 0

        if not node or node.is_leaf:
            result = 0

        else:
            if node.depth >= topd:
                result = 1

            result += self._get_greedy_node_count(node.left, topd)
            result += self._get_greedy_node_count(node.right, topd)

        return result

    cdef str _print_tree(self, Node* node, int depth, str indent, bint last):
        """
        Recursively print the tree using pre-order traversal.
        """
        out_str = ''
        if not node:
            return out_str

        if depth == 0:
            out_str += f'{self._print_node(node)}'
        else:
            out_str += f'\n{indent}'
            if last:
                out_str += f'R----{self._print_node(node)}'
                indent += "     "
            else:
                out_str += f'L----{self._print_node(node)}'
                indent += "|    "
        out_str += self._print_tree(node.left, depth + 1, indent, False)
        out_str += self._print_tree(node.right, depth + 1, indent, True)
        return out_str

    cdef str _print_node(self, Node* node):
        """
        Return string representation of the given node.
        """
        res = ''

        if node.is_leaf:
            res += f'[Leaf n={node.n_samples}'
            res += f', n+={node.n_pos_samples}]'
            res += f' val: {node.value:.5f}'

        else:
            res += f'[Split n={node.n_samples}'
            res += f', n+={node.n_pos_samples}]'
            res += f' feat: {node.chosen_threshold.feature}'
            res += f', val: {node.chosen_threshold.value:.5f}'

        return res
