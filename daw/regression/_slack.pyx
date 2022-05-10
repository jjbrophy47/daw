# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Utility methods.
"""
from libc.stdlib cimport free
from libc.stdlib cimport realloc
from libc.stdio cimport printf

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport split_labels
from ._utils cimport _compute_split_score

# constants
cdef double MAX_FLOAT32 = 3.4028235e+38


# public
cdef SIZE_t compute_slack(DTYPE_t**  X,
                          DTYPE_t*   y,
                          IntList*   samples,
                          Threshold* best_threshold,
                          Threshold* second_threshold,
                          SIZE_t     criterion) nogil:
    """
    Compute ADD slack between the first and second best splits.

    Note: It is assumed `best_threshold` is a "better" split than `second_threshold`.
    """

    # TODO: what should this return if only 1 valid threshold?
    if second_threshold == NULL:
        return 1

    cdef DTYPE_t* y_s1_left = NULL  # must free
    cdef DTYPE_t* y_s1_right = NULL  # must free
    split_labels(X, y, samples, best_threshold, y_s1_left, y_s1_right)

    cdef DTYPE_t* y_s2_left = NULL  # must free
    cdef DTYPE_t* y_s2_right = NULL  # must free
    split_labels(X, y, samples, second_threshold, y_s2_left, y_s2_right)

    cdef _MinMaxHeap s1_left = _MinMaxHeap()  # TODO: free?
    cdef _MinMaxHeap s1_right = _MinMaxHeap()
    cdef _MinMaxHeap s2_left = _MinMaxHeap()
    cdef _MinMaxHeap s2_right = _MinMaxHeap()

    s1_left.insert_list(y_s1_left)
    s1_right.insert_list(y_s1_right)
    s2_left.insert_list(y_s2_left)
    s2_right.insert_list(y_s2_right)

    free(y_s1_left)
    free(y_s1_right)
    free(y_s2_left)
    free(y_s2_right)

    # initial gap
    cdef DTYPE_t score_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right,
        criterion)
    if score_gap < 0:
        printf('[WARNING] split2 is better than split1!')

    # compute slack
    cdef SIZE_t slack = 0
    while score_gap >= 0:
        score_gap = reduce_score_gap(s1_left, s1_right, s2_left, s2_right,
            best_threshold, second_threshold)
        slack += 1
        n += 1

    return slack

# private
cdef DTYPE_t compute_score_gap(_MinMaxHeap s1_left,
                               _MinMaxHeap s1_right,
                               _MinMaxHeap s2_left,
                               _MinMaxHeap s2_right,
                               SIZE_t      criterion) nogil:
    """
    Computes the score gap between the given splits.

    Return
        DTYPE_t, score gap.
    """
    cdef DTYPE_t g1 = _compute_split_score(s1_left, s1_right, criterion)
    cdef DTYPE_t g2 = _compute_split_score(s2_left, s2_right, criterion)
    cdef DTYPE_t score_gap = g2 - g1
    return score_gap


cdef DTYPE_t reduce_score_gap(_MinMaxHeap s1_left,
                              _MinMaxHeap s1_right,
                              _MinMaxHeap s2_left,
                              _MinMaxHeap s2_right
                              Threshold*  split1,
                              Threshold*  split2) nogil:
    """
    Finds the ADD operation that reduces the score gap the most.

    Return
        DTYPE_t, new score gap.
    """
    cdef DTYPE_t temp_gap
    cdef DTYPE_t best_gap = MAX_FLOAT32
    cdef SIZE_t  best_case = -1

    n += 1

    # Case 0: add -inf to left branches
    s1_left.insert(-MAX_FLOAT32)
    s2_left.insert(-MAX_FLOAT32)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_rightm, criterion)
    if temp_gap < best_gap:
        best_gap = temp_gap
        best_case = 0
    s1_left.remove(-MAX_FLOAT32)
    s2_left.remove(-MAX_FLOAT32)

    # Case 1: add +inf to left branches
    s1_left.insert(MAX_FLOAT32)
    s2_left.insert(MAX_FLOAT32)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    if temp_gap < best_gap:
        best_gap = temp_gap
        best_case = 1
    s1_left.remove(MAX_FLOAT32)
    s2_left.remove(MAX_FLOAT32)

    # Case 2: add -inf to right branches
    s1_right.insert(-MAX_FLOAT32)
    s2_right.insert(-MAX_FLOAT32)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    if temp_gap < best_gap:
        best_gap = temp_gap
        best_case = 2
    s1_right.remove(-MAX_FLOAT32)
    s2_right.remove(-MAX_FLOAT32)

    # Case 3: add +inf to right branches
    s1_right.insert(MAX_FLOAT32)
    s2_right.insert(MAX_FLOAT32)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    if temp_gap < best_gap:
        best_gap = temp_gap
        best_case = 3
    s1_right.remove(MAX_FLOAT32)
    s2_right.remove(MAX_FLOAT32)

    if (split1.feature != split2.feature) or (split1.value > split2.value):

        # Case 4: add -inf to split1 left branch and split2 right branch
        s1_left.insert(-MAX_FLOAT32)
        s2_right.insert(-MAX_FLOAT32)
        temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
        if temp_gap < best_gap:
            best_gap = temp_gap
            best_case = 4
        s1_left.remove(-MAX_FLOAT32)
        s2_right.remove(-MAX_FLOAT32)

        # Case 5: add +inf to split1 left branch and split2 right branch
        s1_left.insert(MAX_FLOAT32)
        s2_right.insert(MAX_FLOAT32)
        temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
        if temp_gap < best_gap:
            best_gap = temp_gap
            best_case = 5
        s1_left.remove(MAX_FLOAT32)
        s2_right.remove(MAX_FLOAT32)

    if (split1.feature != split2.feature) or (split1.value < split2.value):

        # Case 6: add -inf to split1 right branch and split2 left branch
        s1_right.insert(-MAX_FLOAT32)
        s2_left.insert(-MAX_FLOAT32)
        temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
        if temp_gap < best_gap:
            best_gap = temp_gap
            best_case = 6
        s1_right.remove(-MAX_FLOAT32)
        s2_left.remove(-MAX_FLOAT32)

        # Case 7: add +inf to split1 right branch and split2 left branch
        s1_right.insert(MAX_FLOAT32)
        s2_left.insert(MAX_FLOAT32)
        temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
        if temp_gap < best_gap:
            best_gap = temp_gap
            best_case = 7
        s1_right.remove(MAX_FLOAT32)
        s2_left.remove(MAX_FLOAT32)

    # update splits based on the chosen operation
    if best_case == -1:
        printf('\n[WARNING] No operation reduced score gap!')
        printf('\nArbitrarily adding -inf to left branch of both splits...')
        best_case = 0

    if best_case == 0:
        s1_left.insert(-MAX_FLOAT32)
        s2_left.insert(-MAX_FLOAT32)
    elif best_case == 1:
        s1_left.insert(MAX_FLOAT32)
        s2_left.insert(MAX_FLOAT32)
    elif best_case == 2:
        s1_right.insert(-MAX_FLOAT32)
        s2_right.insert(-MAX_FLOAT32)
    elif best_case == 3:
        s1_right.insert(MAX_FLOAT32)
        s2_right.insert(MAX_FLOAT32)
    elif best_case == 4:
        s1_left.insert(-MAX_FLOAT32)
        s2_right.insert(-MAX_FLOAT32)
    elif best_case == 5:
        s1_left.insert(MAX_FLOAT32)
        s2_right.insert(MAX_FLOAT32)
    elif best_case == 6:
        s1_right.insert(-MAX_FLOAT32)
        s2_left.insert(-MAX_FLOAT32)
    elif best_case == 7:
        s1_right.insert(MAX_FLOAT32)
        s2_left.insert(MAX_FLOAT32)

    return score_gap
