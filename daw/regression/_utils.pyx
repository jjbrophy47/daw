# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Utility methods.
"""
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdlib cimport free
from libc.stdlib cimport rand
from libc.stdlib cimport srand
from libc.stdlib cimport RAND_MAX
from libc.stdio cimport printf
from libc.math cimport ceil
from libc.math cimport pow
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport log2
from libc.math cimport fabs

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from ._heap cimport mmh_free
from ._heap cimport mmh_create
from ._heap cimport mmh_median
from ._heap cimport mmh_mean
from ._argsort cimport sort
from ._tree cimport UNDEF
from ._tree cimport UNDEF_LEAF_VAL

# constants
cdef inline UINT32_t DEFAULT_SEED = 1
cdef double MAX_DBL = 1.79768e+308

# SAMPLING METHODS

# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    """
    Generate a pseudo-random np.uint32 from a np.uint32 seed.
    """
    # seed shouldn't ever be 0.
    if (seed[0] == 0): seed[0] = DEFAULT_SEED

    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    # Note: we must be careful with the final line cast to np.uint32 so that
    # the function behaves consistently across platforms.
    #
    # The following cast might yield different results on different platforms:
    # wrong_cast = <UINT32_t> RAND_R_MAX + 1
    #
    # We can use:
    # good_cast = <UINT32_t>(RAND_R_MAX + 1)
    # or:
    # cdef np.uint32_t another_good_cast = <UINT32_t>RAND_R_MAX + 1
    return seed[0] % <UINT32_t>(RAND_R_MAX + 1)


cdef inline SIZE_t rand_int(SIZE_t low, SIZE_t high,
                            UINT32_t* random_state) nogil:
    """
    Generate a random integer in [low; end).
    """
    return low + our_rand_r(random_state) % (high - low)


cdef inline double rand_uniform(double low, double high,
                                UINT32_t* random_state) nogil:
    """
    Generate a random double in [low; high).
    """
    return ((high - low) * <double> our_rand_r(random_state) /
            <double> RAND_R_MAX) + low


# SCORING METHODS


cdef DTYPE_t compute_split_score(DTYPE_t**  X,
                                 DTYPE_t*   y,
                                 IntList*   samples,
                                 Threshold* threshold,
                                 SIZE_t     criterion) nogil:
    """
    Compute total error reduction for the given split.
    """
    cdef DTYPE_t* y_left = <DTYPE_t *>malloc(samples.n * sizeof(DTYPE_t))  # must free
    cdef DTYPE_t* y_right = <DTYPE_t *>malloc(samples.n * sizeof(DTYPE_t))  # must free
    cdef SIZE_t n_left = split_labels(X, y, samples, threshold, &y_left, &y_right)
    cdef SIZE_t n_right = samples.n - n_left

    cdef MinMaxHeap* left = mmh_create(y_left, n_left)  # must free
    cdef MinMaxHeap* right = mmh_create(y_right, n_right)  # must free

    cdef DTYPE_t result = _compute_split_score(left, right, criterion)

    # clean up
    free(y_left)
    free(y_right)

    mmh_free(left)
    mmh_free(right)

    return result


# private
cdef DTYPE_t _compute_split_score(MinMaxHeap* left,
                                  MinMaxHeap* right,
                                  SIZE_t      criterion) nogil:
    """
    Computes purity of the leaf via mean-squared error or median absolute error.

    Return
        weighted median absolute error:
            https://scikit-learn.org/stable/modules/model_evaluation.html#median-absolute-error
        OR
        weighted mean squared error:
            https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error
    """
    cdef DTYPE_t  result = 0
    cdef DTYPE_t* left_errors = <DTYPE_t *>malloc(left.size * sizeof(DTYPE_t))  # must free
    cdef DTYPE_t* right_errors = <DTYPE_t *>malloc(right.size * sizeof(DTYPE_t))  # must free

    cdef DTYPE_t total = left.size + right.size
    cdef DTYPE_t left_weight = <DTYPE_t> left.size / total
    cdef DTYPE_t right_weight = <DTYPE_t> right.size / total

    cdef DTYPE_t left_val
    cdef DTYPE_t right_val

    cdef DTYPE_t left_score
    cdef DTYPE_t right_score

    cdef SIZE_t i = 0
    cdef SIZE_t k = 0

    # median absolute error
    if criterion == 0:
        left_val = mmh_median(left)
        right_val = mmh_median(right)

        # compute left branch errors
        k = 0
        for i in range(left.min_heap.size):
            left_errors[k] = fabs(left.min_heap.arr[i] - left_val)
            # printf('%.5f, %.5f, %.5f\n', left_val, left.min_heap.arr[i], left_errors[k])
            k += 1
        for i in range(left.max_heap.size):
            left_errors[k] = fabs(left.max_heap.arr[i] - left_val)
            # printf('%.5f, %.5f, %.5f\n', left_val, left.min_heap.arr[i], left_errors[k])
            k += 1

        # compute right branch errors
        k = 0
        for i in range(right.min_heap.size):
            right_errors[k] = fabs(right.min_heap.arr[i] - right_val)
            # printf('%.5f, %.5f, %.5f\n', right_val, right.min_heap.arr[i], right_errors[k])
            k += 1
        for i in range(right.max_heap.size):
            right_errors[k] = fabs(right.max_heap.arr[i] - right_val)
            # printf('%.5f, %.5f, %.5f\n', right_val, right.max_heap.arr[i], right_errors[k])
            k += 1

        # printf('%ld, %ld\n', left.size, right.size)
        left_score = compute_median(left_errors, left.size)
        right_score = compute_median(right_errors, right.size)

    # mean squared error
    elif criterion == 1:
        left_val = mmh_mean(left)
        right_val = mmh_mean(right)

        # compute left branch errors
        k = 0

        for i in range(left.min_heap.size):
            left_errors[k] = pow(left.min_heap.arr[i] - left_val, 2)
            k += 1
        for i in range(left.max_heap.size):
            left_errors[k] = pow(left.max_heap.arr[i] - left_val, 2)
            k += 1

        # compute right branch errors
        k = 0

        for i in range(right.min_heap.size):
            right_errors[k] = pow(right.min_heap.arr[i] - right_val, 2)
            k += 1
        for i in range(right.max_heap.size):
            right_errors[k] = pow(right.max_heap.arr[i] - right_val, 2)
            k += 1

        left_score = compute_mean(left_errors, left.size)
        right_score = compute_mean(right_errors, right.size)
    else:
        raise ValueError('Unknown criterion: %ld', criterion)

    # clean up
    free(left_errors)
    free(right_errors)

    # printf('[U - CSS] left_weight: %.5f, left_score: %.5f, right_weight: %.5f, right_score: %.5f\n',
    #     left_weight, left_score, right_weight, right_score)

    return (left_weight * left_score) + (right_weight * right_score)


cdef DTYPE_t compute_leaf_value(IntList* samples,
                                DTYPE_t* y,
                                SIZE_t   criterion) nogil:
    """
    Compute leaf value depending on the criterion.
    """
    cdef DTYPE_t* values = <DTYPE_t *>malloc(samples.n * sizeof(DTYPE_t))  # must free

    cdef DTYPE_t result
    cdef SIZE_t i = 0

    for i in range(samples.n):
        values[i] = y[samples.arr[i]]

    if criterion == 0:  # absolute error
        result = compute_median(values, samples.n)
    elif criterion == 1:
        result = compute_mean(values, samples.n)
    else:
        raise ValueError('Unknown criterion %ld\n', criterion)

    # clean up
    free(values)

    return result


# FEATURE / THRESHOLD METHODS


cdef Feature* create_feature(SIZE_t feature_index) nogil:
    """
    Allocate memory for a feature object.
    """
    cdef Feature* feature = <Feature *>malloc(sizeof(Feature))
    feature.index = feature_index
    feature.thresholds = NULL
    feature.n_thresholds = 0
    return feature


cdef Threshold* create_threshold(SIZE_t feature_index, DTYPE_t value) nogil:
    """
    Allocate memory for a threshold object.
    """
    cdef Threshold* threshold = <Threshold *>malloc(sizeof(Threshold))
    threshold.feature = feature_index
    threshold.value = value
    return threshold


cdef Feature** copy_features(Feature** features, SIZE_t n_features) nogil:
    """
    Copies the contents of a features array to a new array.
    """
    new_features = <Feature **>malloc(n_features * sizeof(Feature *))
    for j in range(n_features):
        new_features[j] = copy_feature(features[j])
    return new_features


cdef Feature* copy_feature(Feature* feature) nogil:
    """
    Copies the contents of a feature to a new feature.
    """
    cdef Feature* new_feature = <Feature *>malloc(sizeof(Feature))
    new_feature.index = feature.index
    new_feature.n_thresholds = feature.n_thresholds
    new_feature.thresholds = copy_thresholds(feature.thresholds, feature.n_thresholds)
    return new_feature


cdef Threshold** copy_thresholds(Threshold** thresholds, SIZE_t n_thresholds) nogil:
    """
    Copies the contents of a thresholds array to a new array.
    """
    new_thresholds = <Threshold **>malloc(n_thresholds * sizeof(Threshold *))
    for j in range(n_thresholds):
        new_thresholds[j] = copy_threshold(thresholds[j])
    return new_thresholds


cdef Threshold* copy_threshold(Threshold* threshold) nogil:
    """
    Copies the contents of a threshold to a new threshold.
    """
    cdef Threshold* new_threshold = <Threshold *>malloc(sizeof(Threshold))
    new_threshold.feature = threshold.feature
    new_threshold.value = threshold.value
    return new_threshold


cdef void free_features(Feature** features,
                           SIZE_t n_features) nogil:
    """
    Deallocate a features array and all thresholds.
    """
    cdef SIZE_t j = 0

    # free each feature and then the array
    if features != NULL:
        for j in range(n_features):
            free_feature(features[j])
        free(features)


cdef void free_feature(Feature* feature) nogil:
    """
    Frees all properties of this feature, and then the feature.
    """
    if feature != NULL:
        if feature.thresholds != NULL:
            free_thresholds(feature.thresholds, feature.n_thresholds)
        free(feature)


cdef void free_thresholds(Threshold** thresholds,
                          SIZE_t n_thresholds) nogil:
    """
    Deallocate a thresholds array and its contents
    """
    cdef SIZE_t k = 0

    # free each threshold and then the array
    if thresholds != NULL:
        for k in range(n_thresholds):
            free(thresholds[k])
        free(thresholds)


# INTLIST METHODS


cdef IntList* create_intlist(SIZE_t n_elem, bint initialize) nogil:
    """
    Allocate memory for:
    -IntList object.
    -IntList.arr object with size n_elem.
    If `initialize` is True, Set IntList.n = n, IntList.n = 0.
    """
    cdef IntList* obj = <IntList *>malloc(sizeof(IntList))
    obj.arr = <SIZE_t *>malloc(n_elem * sizeof(SIZE_t))

    # set n
    if initialize:
        obj.n = n_elem
    else:
        obj.n = 0

    return obj


cdef IntList* copy_intlist(IntList* obj, SIZE_t n_elem) nogil:
    """
    -Creates a new IntList object.
    -Allocates the `arr` with size `n_elem`.
    -Copies values from `obj.arr` up to `obj.n`.
    -`n` is set to `obj.n`.

    NOTE: n_elem >= obj.n.
    """
    cdef IntList* new_obj = create_intlist(n_elem, 0)

    # copy array values
    for i in range(obj.n):
        new_obj.arr[i] = obj.arr[i]

    # set n
    new_obj.n = obj.n

    return new_obj


cdef void free_intlist(IntList* obj) nogil:
    """
    Deallocate IntList object.
    """
    free(obj.arr)
    free(obj)
    obj = NULL


# ARRAY METHODS


cdef SIZE_t* convert_int_ndarray(np.ndarray arr):
    """
    Converts a numpy array into a C int array.
    """
    cdef SIZE_t  n_elem = arr.shape[0]
    cdef SIZE_t* new_arr = <SIZE_t *>malloc(n_elem * sizeof(SIZE_t))

    for i in range(n_elem):
        new_arr[i] = arr[i]

    return new_arr


cdef INT32_t* copy_int_array(INT32_t* arr, SIZE_t n_elem) nogil:
    """
    Copies a C int array into a new C int array.
    """
    cdef INT32_t* new_arr = <INT32_t *>malloc(n_elem * sizeof(INT32_t))

    for i in range(n_elem):
        new_arr[i] = arr[i]

    return new_arr


cdef SIZE_t* copy_indices(SIZE_t* arr, SIZE_t n_elem) nogil:
    """
    Copies a C int array into a new C int array.
    """
    cdef SIZE_t* new_arr = <SIZE_t *>malloc(n_elem * sizeof(SIZE_t))

    for i in range(n_elem):
        new_arr[i] = arr[i]

    return new_arr


# Utility methods

cdef DTYPE_t* extract_labels(IntList* samples, DTYPE_t* y) nogil:
    """
    Extracts label data for a given set of samples.

    NOTE: Returns an allocated array, caller must free this array!
    """
    cdef DTYPE_t* vals = <DTYPE_t *>malloc(samples.n * sizeof(DTYPE_t))
    for i in range(samples.n):
        vals[i] = y[samples.arr[i]]
    return vals

cdef DTYPE_t compute_mean(DTYPE_t* vals, SIZE_t n) nogil:
    """
    Compute the mean value of the given array.
    """
    cdef SIZE_t i = 0
    cdef DTYPE_t val_sum = 0

    for i in range(n):
        val_sum += vals[i]

    return val_sum / n


cdef DTYPE_t compute_median(DTYPE_t* vals, SIZE_t n) nogil:
    """
    Compute the median value of the given array.
    """

    # catch base cases
    if n == 1:
        return vals[0]
    elif n == 2:
        return (vals[0] + vals[1]) / 2.0

    cdef SIZE_t  i = 0
    cdef DTYPE_t result = 0

    # copy label values and indices
    cdef DTYPE_t* sorted_vals = <DTYPE_t *>malloc(n * sizeof(DTYPE_t))  # MUST FREE
    for i in range(n):
        sorted_vals[i] = vals[i]

    # sort label values and indices based on the label values
    sort(sorted_vals, NULL, n)

    # extract median
    if n % 2 == 1:  # odd
        i = n // 2
        result = sorted_vals[i]
    else:  # even
        i = n / 2
        result = (sorted_vals[i-1] + sorted_vals[i]) / 2.0

    # clean up
    free(sorted_vals)

    return result


# NODE METHODS


cdef SIZE_t split_labels(DTYPE_t**  X,
                         DTYPE_t*   y,
                         IntList*   samples,
                         Threshold* threshold,
                         DTYPE_t**  y_left,
                         DTYPE_t**  y_right) nogil:
    """
    Splits label values into left and right branches.

    NOTE:
        - y_left and y_right should be NULL.
        - y_left and y_right must be freed by caller.
    """
    cdef SIZE_t left_count = 0
    cdef SIZE_t right_count = 0

    if y_left[0] is NULL or y_right[0] is NULL:
        raise ValueError('y_left or y_right is NULL!')

    # loop through the deleted samples
    for i in range(samples.n):

        # add sample to the left branch
        if X[samples.arr[i]][threshold.feature] <= threshold.value:
            y_left[0][left_count] = y[samples.arr[i]]
            left_count += 1

        # add sample to the right branch
        else:
            y_right[0][right_count] = y[samples.arr[i]]
            right_count += 1

    # resize left branch
    if left_count > 0:
        y_left[0] = <DTYPE_t *>realloc(y_left[0], left_count * sizeof(DTYPE_t))
    else:
        free(y_left)
        y_left = NULL

    # resize right branch
    if right_count > 0:
        y_right[0] = <DTYPE_t *>realloc(y_right[0], right_count * sizeof(DTYPE_t))
    else:
        free(y_right)
        y_right = NULL

    return left_count


cdef void split_samples(Node*        node,
                        DTYPE_t**    X,
                        DTYPE_t*     y,
                        IntList*     samples,
                        SplitRecord* split,
                        bint         copy_constant_features) nogil:
    """
    Split samples based on the chosen feature / threshold.

    NOTE: frees `samples.arr` and `samples` object.
    """

    # split samples based on the chosen feature / threshold
    split.left_samples = create_intlist(samples.n, 0)
    split.right_samples = create_intlist(samples.n, 0)

    # printf('[U - SS] node.chosen_feature.index: %ld, node.chosen_threshold.value: %.5f\n',
    #        node.chosen_feature.index, node.chosen_threshold.value)
    # printf('[U - SS] node.chosen_threshold.n_left_samples: %ld, node.chosen_threshold.n_right_samples: %ld\n',
    #        node.chosen_threshold.n_left_samples, node.chosen_threshold.n_right_samples)

    # loop through the deleted samples
    for i in range(samples.n):

        # printf('[U - SS] X[samples.arr[%ld]][%ld]: %.32f, node.chosen_threshold.value: %.32f\n',
        #        i, node.chosen_feature.index, X[samples.arr[i]][node.chosen_feature.index], node.chosen_threshold.value)

        # add sample to the left branch
        if X[samples.arr[i]][node.chosen_feature.index] <= node.chosen_threshold.value:
            split.left_samples.arr[split.left_samples.n] = samples.arr[i]
            split.left_samples.n += 1

        # add sample to the right branch
        else:
            split.right_samples.arr[split.right_samples.n] = samples.arr[i]
            split.right_samples.n += 1

    # printf('[U - SS] split.left_samples.n: %ld, split.right_samples.n: %ld\n',
    #        split.left_samples.n, split.right_samples.n)

    # assign left branch deleted samples
    if split.left_samples.n > 0:
        split.left_samples.arr = <SIZE_t *>realloc(split.left_samples.arr,
                                                   split.left_samples.n * sizeof(SIZE_t))
    else:
        # printf('[U - SS] NO LEFT SAMPLES\n')
        free_intlist(split.left_samples)
        split.left_samples = NULL

    # assign right branch deleted samples
    if split.right_samples.n > 0:
        split.right_samples.arr = <SIZE_t *>realloc(split.right_samples.arr,
                                                    split.right_samples.n * sizeof(SIZE_t))
    else:
        # printf('[U - SS] NO RIGHT SAMPLES\n')
        free_intlist(split.right_samples)
        split.right_samples = NULL

    # printf('[U - SS] copy constant features\n')

    # copy constant features array for both branches
    if copy_constant_features:
        split.left_constant_features = copy_intlist(node.constant_features, node.constant_features.n)
        split.right_constant_features = copy_intlist(node.constant_features, node.constant_features.n)

    # printf('[U - SS] freeing samples\n')

    # clean up, no more use for the original samples array
    free_intlist(samples)

    # printf('[U - SS] done freeing samples\n')


cdef void dealloc(Node *node) nogil:
    """
    Recursively free all nodes in the subtree.

    NOTE: Does not deallocate "root" node, that must
          be done by the caller!
    """
    if not node:
        return

    # traverse to the bottom nodes first
    dealloc(node.left)
    dealloc(node.right)

    # leaf node
    if node.is_leaf:
        free(node.leaf_samples)

    # decision node
    else:

        # clear chosen feature
        if node.chosen_feature != NULL:
            if node.chosen_feature.thresholds != NULL:
                for k in range(node.chosen_feature.n_thresholds):
                    free(node.chosen_feature.thresholds[k])
                free(node.chosen_feature.thresholds)
            free(node.chosen_feature)

        # clear chosen threshold
        if node.chosen_threshold != NULL:
            free(node.chosen_threshold)

        # clear constant features
        if node.constant_features != NULL:
            free_intlist(node.constant_features)

        # clear features array
        if node.features != NULL:
            for j in range(node.n_features):

                if node.features[j] != NULL:
                    for k in range(node.features[j].n_thresholds):
                        free(node.features[j].thresholds[k])

                    free(node.features[j].thresholds)
                    free(node.features[j])

            free(node.features)

        # free children
        free(node.left)
        free(node.right)

    # reset general node properties
    node.node_id = -1
    node.left = NULL
    node.right = NULL

    # reset leaf properties
    node.leaf_id = -1
    node.is_leaf = False
    node.value = UNDEF_LEAF_VAL
    node.leaf_samples = NULL

    # reset decision node properties
    node.features = NULL
    node.n_features = 0
    node.constant_features = NULL
    node.chosen_feature = NULL
    node.chosen_threshold = NULL
