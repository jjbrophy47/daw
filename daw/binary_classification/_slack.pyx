# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Utility methods.
"""
from libc.stdlib cimport free
from libc.stdio cimport printf
from libc.math cimport fabs

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport compute_split_score
from ._utils cimport copy_threshold

# constants
cdef DTYPE_t UNDEF_SCORE_GAP = 100000


# public
cdef SIZE_t compute_leaf_slack(
    SIZE_t n,
    SIZE_t n_pos) nogil:
    """
    Compute minimum number of deletions/additions
    needed to FLIP the leaf value prediction.
    """
    cdef SIZE_t slack = 0
    cdef SIZE_t n_neg = n - n_pos

    # even split
    if n_pos == n_neg:
        slack = 1
    else:
        slack = <SIZE_t> fabs(n_pos - n_neg) + 1

    return slack


cdef SIZE_t compute_split_slack_deletion(
    Threshold* best_threshold,
    Threshold* second_threshold,
    SIZE_t     n,
    bint       use_gini) nogil:
    """
    Compute DELETION slack between the first and second best splits for a DECISION node.

    Note: It is assumed `best_threshold` is a "better" split than `second_threshold`.
    """

    # TODO: what should this return if only 1 valid threshold?
    if second_threshold == NULL:
        return 1

    # initial conditions
    cdef Threshold* split1 = copy_threshold(best_threshold)
    cdef Threshold* split2 = copy_threshold(second_threshold)
    cdef DTYPE_t score_gap = _compute_score_gap(split1, split2, n, use_gini)

    # printf('score_gap: %.5f\n', score_gap)

    if score_gap < 0:
        printf('[WARNING] split2 is better than split1!')

    # result variable
    cdef SIZE_t slack = 0

    # compute slack
    while score_gap >= 0:
        score_gap = _reduce_gap_deletion(split1, split2, n, use_gini)
        slack += 1
        n += 1
        # printf('score_gap: %.5f\n', score_gap)

        if score_gap == UNDEF_SCORE_GAP:
            slack = -100
            break

    free(split1)
    free(split2)

    return slack


cdef SIZE_t compute_split_slack_addition(
    Threshold* best_threshold,
    Threshold* second_threshold,
    SIZE_t     n,
    bint       use_gini) nogil:
    """
    Compute ADD slack between the first and second best splits for a DECISION node.

    Note: It is assumed `best_threshold` is a "better" split than `second_threshold`.
    """

    # TODO: what should this return if only 1 valid threshold?
    if second_threshold == NULL:
        return 1

    # initial conditions
    cdef Threshold* split1 = copy_threshold(best_threshold)
    cdef Threshold* split2 = copy_threshold(second_threshold)
    cdef DTYPE_t score_gap = _compute_score_gap(split1, split2, n, use_gini)

    # printf('score_gap: %.5f\n', score_gap)

    if score_gap < 0:
        printf('[WARNING] split2 is better than split1!')

    # result variable
    cdef SIZE_t slack = 0

    # compute slack
    while score_gap >= 0:
        score_gap = _reduce_gap_addition(split1, split2, n, use_gini)
        slack += 1
        n += 1

    free(split1)
    free(split2)

    return slack


# private
cdef DTYPE_t _compute_score_gap(Threshold* split1,
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

    # printf('g1: %.5f, g2: %.5f\n', g1, g2)
    # printf('[g1] n: %ld, left: %ld, left_pos: %ld, right: %ld, right_pos: %ld\n',
    #     n, split1.n_left_samples, split1.n_left_pos_samples,
    #     split1.n_right_samples, split1.n_right_pos_samples)
    # printf('[g2] n: %ld, left: %ld, left_pos: %ld, right: %ld, right_pos: %ld\n',
    #     n, split2.n_left_samples, split2.n_left_pos_samples,
    #     split2.n_right_samples, split2.n_right_pos_samples)

    return score_gap


cdef DTYPE_t _reduce_gap_deletion(
    Threshold* split1,
    Threshold* split2,
    SIZE_t     n,
    bint       use_gini) nogil:
    """
    Finds the DELETION operation that reduces the score gap the most.

    Return
        DTYPE_t, new score gap.
    """
    cdef DTYPE_t score_gap = UNDEF_SCORE_GAP
    cdef DTYPE_t temp_gap = -1
    cdef SIZE_t  best_case = -1

    n -= 1

    # Cases 0 & 1: left branches
    if split1.n_left_samples > 1 and split2.n_left_samples > 1:

        # Case 0: remove 0 from left branches
        if split1.n_left_samples - split1.n_left_pos_samples > 0\
            and split2.n_left_samples - split2.n_left_pos_samples > 0:
            split1.n_left_samples -= 1
            split2.n_left_samples -= 1
            temp_gap = _compute_score_gap(split1, split2, n, use_gini)
            if temp_gap < score_gap:
                score_gap = temp_gap
                best_case = 0
            split1.n_left_samples += 1
            split2.n_left_samples += 1

        # Case 1: remove 1 from left branches
        if split1.n_left_pos_samples > 0 and split2.n_left_pos_samples > 0:
            split1.n_left_samples -= 1
            split1.n_left_pos_samples -= 1
            split2.n_left_samples -= 1
            split2.n_left_pos_samples -= 1
            temp_gap = _compute_score_gap(split1, split2, n, use_gini)
            if temp_gap < score_gap:
                score_gap = temp_gap
                best_case = 1
            split1.n_left_samples += 1
            split1.n_left_pos_samples += 1
            split2.n_left_samples += 1
            split2.n_left_pos_samples += 1

    # Cases 2 & 3: right branches
    if split1.n_right_samples > 1 and split2.n_right_samples > 1:

        # Case 2: remove 0 from right branches
        if split1.n_right_samples - split1.n_right_pos_samples > 0\
            and split2.n_right_samples - split2.n_right_pos_samples > 0:
            split1.n_right_samples -= 1
            split2.n_right_samples -= 1
            temp_gap = _compute_score_gap(split1, split2, n, use_gini)
            if temp_gap < score_gap:
                score_gap = temp_gap
                best_case = 2
            split1.n_right_samples += 1
            split2.n_right_samples += 1

        # Case 3: remove 1 from right branches
        if split1.n_right_pos_samples > 0 and split2.n_right_pos_samples > 0:
            split1.n_right_samples -= 1
            split1.n_right_pos_samples -= 1
            split2.n_right_samples -= 1
            split2.n_right_pos_samples -= 1
            temp_gap = _compute_score_gap(split1, split2, n, use_gini)
            if temp_gap < score_gap:
                score_gap = temp_gap
                best_case = 3
            split1.n_right_samples += 1
            split1.n_right_pos_samples += 1
            split2.n_right_samples += 1
            split2.n_right_pos_samples += 1

    if (split1.feature != split2.feature) or (split1.value > split2.value):

        # Cases 4 & 5: left and right branches
        if split1.n_left_samples > 1 and split2.n_right_samples > 1:

            # Case 4: remove 0 from split1 left branch and split2 right branch
            if split1.n_left_samples - split1.n_left_pos_samples > 0\
                and split2.n_right_samples - split2.n_right_pos_samples > 0:
                split1.n_left_samples -= 1
                split2.n_right_samples -= 1
                temp_gap = _compute_score_gap(split1, split2, n, use_gini)
                if temp_gap < score_gap:
                    score_gap = temp_gap
                    best_case = 4
                split1.n_left_samples += 1
                split2.n_right_samples += 1

            # Case 5: remove 1 from split1 left branch and split2 right branch
            if split1.n_left_pos_samples > 0 and split2.n_right_pos_samples > 0:
                split1.n_left_samples += 1
                split1.n_left_pos_samples += 1
                split2.n_right_samples += 1
                split2.n_right_pos_samples += 1
                temp_gap = _compute_score_gap(split1, split2, n, use_gini)
                if temp_gap < score_gap:
                    score_gap = temp_gap
                    best_case = 5
                split1.n_left_samples -= 1
                split1.n_left_pos_samples -= 1
                split2.n_right_samples -= 1
                split2.n_right_pos_samples -= 1

    if (split1.feature != split2.feature) or (split1.value < split2.value):

        # Cases 5 & 6: right and left branches
        if split1.n_right_samples > 1 and split2.n_left_samples > 1:

            # Case 6: remove 0 from split1 right branch and split2 left branch
            if split1.n_right_samples - split1.n_right_pos_samples > 0\
                and split2.n_left_samples - split2.n_left_pos_samples > 0:
                split1.n_right_samples -= 1
                split2.n_left_samples -= 1
                temp_gap = _compute_score_gap(split1, split2, n, use_gini)
                if temp_gap < score_gap:
                    score_gap = temp_gap
                    best_case = 6
                split1.n_right_samples += 1
                split2.n_left_samples += 1

            # Case 7: remove 1 from split1 right branch and split2 left branch
            if split1.n_right_pos_samples > 0 and split2.n_left_pos_samples > 0:
                split1.n_right_samples -= 1
                split1.n_right_pos_samples -= 1
                split2.n_left_samples -= 1
                split2.n_left_pos_samples -= 1
                temp_gap = _compute_score_gap(split1, split2, n, use_gini)
                if temp_gap < score_gap:
                    score_gap = temp_gap
                    best_case = 7
                split1.n_right_samples += 1
                split1.n_right_pos_samples += 1
                split2.n_left_samples += 1
                split2.n_left_pos_samples += 1

    # update splits based on the chosen operation
    if best_case == 0:
        split1.n_left_samples -= 1
        split2.n_left_samples -= 1
    elif best_case == 1:
        split1.n_left_samples -= 1
        split1.n_left_pos_samples -= 1
        split2.n_left_samples -= 1
        split2.n_left_pos_samples -= 1
    elif best_case == 2:
        split1.n_right_samples -= 1
        split2.n_right_samples -= 1
    elif best_case == 3:
        split1.n_right_samples -= 1
        split1.n_right_pos_samples -= 1
        split2.n_right_samples -= 1
        split2.n_right_pos_samples -= 1
    elif best_case == 4:
        split1.n_left_samples -= 1
        split2.n_right_samples -= 1
    elif best_case == 5:
        split1.n_left_samples -= 1
        split1.n_left_pos_samples -= 1
        split2.n_right_samples -= 1
        split2.n_right_pos_samples -= 1
    elif best_case == 6:
        split1.n_right_samples -= 1
        split2.n_left_samples -= 1
    elif best_case == 7:
        split1.n_right_samples -= 1
        split1.n_right_pos_samples -= 1
        split2.n_left_samples -= 1
        split2.n_left_pos_samples -= 1
    else:
        printf('\n[WARNING] No operation reduced score gap!')

    # printf('best_case: %ld\n', best_case)

    return score_gap


cdef DTYPE_t _reduce_gap_addition(
    Threshold* split1,
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
    temp_gap = _compute_score_gap(split1, split2, n, use_gini)
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
    temp_gap = _compute_score_gap(split1, split2, n, use_gini)
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
    temp_gap = _compute_score_gap(split1, split2, n, use_gini)
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
    temp_gap = _compute_score_gap(split1, split2, n, use_gini)
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
        temp_gap = _compute_score_gap(split1, split2, n, use_gini)
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
        temp_gap = _compute_score_gap(split1, split2, n, use_gini)
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
        temp_gap = _compute_score_gap(split1, split2, n, use_gini)
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
        temp_gap = _compute_score_gap(split1, split2, n, use_gini)
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

    # printf('best_case: %ld\n', best_case)

    return score_gap
