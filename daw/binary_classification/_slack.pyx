# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Utility methods.
"""
from libc.stdlib cimport free
from libc.stdio cimport printf

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport compute_split_score
from ._utils cimport copy_threshold


cdef SIZE_t compute_slack(Threshold* best_threshold,
                          Threshold* second_threshold,
                          SIZE_t     n,
                          bint       use_gini) nogil:
    """
    Compute ADD slack between the first and second best splits.

    Note: It is assumed `best_threshold` is a "better" split than `second_threshold`.
    """

    # TODO: what should this return if only 1 valid threshold?
    if second_threshold == NULL:
        return 1

    # initial conditions
    cdef Threshold* split1 = copy_threshold(best_threshold)
    cdef Threshold* split2 = copy_threshold(second_threshold)
    cdef DTYPE_t score_gap = compute_score_gap(split1, split2, n, use_gini)

    if score_gap < 0:
        printf('[WARNING] split2 is better than split1!')

    # result variable
    cdef SIZE_t slack = 0

    # compute slack
    while score_gap >= 0:
        score_gap = reduce_score_gap(split1, split2, n, use_gini)
        slack += 1
        n += 1

    free(split1)
    free(split2)

    return slack


cdef DTYPE_t compute_score_gap(Threshold* split1,
                               Threshold* split2,
                               SIZE_t     n,
                               bint       use_gini) nogil:
    """
    Computes the score gap between the given splits.

    Return
        DTYPE_t, score gap.
    """
    cdef DTYPE_t g1 = compute_split_score(use_gini, n, split1)
    cdef DTYPE_t g2 = compute_split_score(use_gini, n, split2)
    cdef DTYPE_t score_gap = g2 - g1
    return score_gap


cdef DTYPE_t reduce_score_gap(Threshold* split1,
                              Threshold* split2,
                              SIZE_t     n,
                              bint       use_gini) nogil:
    """
    Finds the ADD operation that reduces the score gap the most.

    Return
        DTYPE_t, new score gap.
    """
    cdef DTYPE_t score_gap = 1000000
    cdef DTYPE_t temp_gap = -1
    cdef SIZE_t  best_case = -1

    n += 1

    # Case 0: add 0 to left branches
    split1.n_left_samples += 1
    split2.n_left_samples += 1
    temp_gap = compute_score_gap(split1, split2, n, use_gini)
    if temp_gap < score_gap:
        score_gap = temp_gap
        best_case = 0
    split1.n_left_samples -= 1
    split2.n_left_samples -= 1

    # Case 1: add 1 to left branches
    split1.n_left_samples += 1
    split1.n_left_pos_samples += 1
    split2.n_left_samples += 1
    split2.n_left_pos_samples += 1
    temp_gap = compute_score_gap(split1, split2, n, use_gini)
    if temp_gap < score_gap:
        score_gap = temp_gap
        best_case = 1
    split1.n_left_samples -= 1
    split1.n_left_pos_samples -= 1
    split2.n_left_samples -= 1
    split2.n_left_pos_samples -= 1

    # Case 2: add 0 to right branches
    split1.n_right_samples += 1
    split2.n_right_samples += 1
    temp_gap = compute_score_gap(split1, split2, n, use_gini)
    if temp_gap < score_gap:
        score_gap = temp_gap
        best_case = 2
    split1.n_right_samples -= 1
    split2.n_right_samples -= 1

    # Case 3: add 1 to right branches
    split1.n_right_samples += 1
    split1.n_right_pos_samples += 1
    split2.n_right_samples += 1
    split2.n_right_pos_samples += 1
    temp_gap = compute_score_gap(split1, split2, n, use_gini)
    if temp_gap < score_gap:
        score_gap = temp_gap
        best_case = 3
    split1.n_right_samples -= 1
    split1.n_right_pos_samples -= 1
    split2.n_right_samples -= 1
    split2.n_right_pos_samples -= 1

    if (split1.feature != split2.feature) or (split1.value > split2.value):

        # Case 4: add 0 to split1 left branch and split2 right branch
        split1.n_left_samples += 1
        split2.n_right_samples += 1
        temp_gap = compute_score_gap(split1, split2, n, use_gini)
        if temp_gap < score_gap:
            score_gap = temp_gap
            best_case = 4
        split1.n_left_samples -= 1
        split2.n_right_samples -= 1

        # Case 5: add 1 to split1 left branch and split2 right branch
        split1.n_left_samples += 1
        split1.n_left_pos_samples += 1
        split2.n_right_samples += 1
        split2.n_right_pos_samples += 1
        temp_gap = compute_score_gap(split1, split2, n, use_gini)
        if temp_gap < score_gap:
            score_gap = temp_gap
            best_case = 5
        split1.n_left_samples -= 1
        split1.n_left_pos_samples -= 1
        split2.n_right_samples -= 1
        split2.n_right_pos_samples -= 1

    if (split1.feature != split2.feature) or (split1.value < split2.value):

        # Case 6: add 0 to split1 right branch and split2 left branch
        split1.n_right_samples += 1
        split2.n_left_samples += 1
        temp_gap = compute_score_gap(split1, split2, n, use_gini)
        if temp_gap < score_gap:
            score_gap = temp_gap
            best_case = 6
        split1.n_right_samples -= 1
        split2.n_left_samples -= 1

        # Case 7: add 1 to split1 right branch and split2 left branch
        split1.n_right_samples += 1
        split1.n_right_pos_samples += 1
        split2.n_left_samples += 1
        split2.n_left_pos_samples += 1
        temp_gap = compute_score_gap(split1, split2, n, use_gini)
        if temp_gap < score_gap:
            score_gap = temp_gap
            best_case = 7
        split1.n_right_samples -= 1
        split1.n_right_pos_samples -= 1
        split2.n_left_samples -= 1
        split2.n_left_pos_samples -= 1

    # update splits based on the chosen operation
    if best_case == 0:
        split1.n_left_samples += 1
        split2.n_left_samples += 1
    elif best_case == 1:
        split1.n_left_samples += 1
        split1.n_left_pos_samples += 1
        split2.n_left_samples += 1
        split2.n_left_pos_samples += 1
    elif best_case == 2:
        split1.n_right_samples += 1
        split2.n_right_samples += 1
    elif best_case == 3:
        split1.n_right_samples += 1
        split1.n_right_pos_samples += 1
        split2.n_right_samples += 1
        split2.n_right_pos_samples += 1
    elif best_case == 4:
        split1.n_left_samples += 1
        split2.n_right_samples += 1
    elif best_case == 5:
        split1.n_left_samples += 1
        split1.n_left_pos_samples += 1
        split2.n_right_samples += 1
        split2.n_right_pos_samples += 1
    elif best_case == 6:
        split1.n_right_samples += 1
        split2.n_left_samples += 1
    elif best_case == 7:
        split1.n_right_samples += 1
        split1.n_right_pos_samples += 1
        split2.n_left_samples += 1
        split2.n_left_pos_samples += 1
    else:
        printf('\n[WARNING] No operation reduced score gap!')

    return score_gap
