# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Utility methods.
"""
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdio cimport printf

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from ._heap cimport list_print
from ._heap cimport mmh_create
from ._heap cimport mmh_free
from ._heap cimport mmh_insert
from ._heap cimport mmh_remove
from ._heap cimport mmh_median
from ._heap cimport mmh_min
from ._heap cimport mmh_max
from ._utils cimport split_labels
from ._utils cimport _compute_split_score

# constants
cdef DTYPE_t MAX_FLOAT32 = 3.4028235e+38


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

    # create left and right branches for split 1
    cdef DTYPE_t* y_s1_left = <DTYPE_t *>malloc(samples.n * sizeof(DTYPE_t))  # must free
    cdef DTYPE_t* y_s1_right = <DTYPE_t *>malloc(samples.n * sizeof(DTYPE_t))  # must free
    cdef SIZE_t s1_left_count = split_labels(X, y, samples, best_threshold, &y_s1_left, &y_s1_right)
    cdef SIZE_t s1_right_count = samples.n - s1_left_count

    cdef MinMaxHeap* s1_left = mmh_create(y_s1_left, s1_left_count)  # must free
    cdef MinMaxHeap* s1_right = mmh_create(y_s1_right, s1_right_count)  # must free

    free(y_s1_left)
    free(y_s1_right)

    # printf('\n[S - CS] s1 left/right count: %ld/%ld\n', s1_left.size, s1_right.size)

    # create left and right branches for split 2
    cdef DTYPE_t* y_s2_left = <DTYPE_t *>malloc(samples.n * sizeof(DTYPE_t))  # must free
    cdef DTYPE_t* y_s2_right = <DTYPE_t *>malloc(samples.n * sizeof(DTYPE_t))  # must free
    cdef SIZE_t s2_left_count = split_labels(X, y, samples, second_threshold, &y_s2_left, &y_s2_right)
    cdef SIZE_t s2_right_count = samples.n - s2_left_count

    cdef MinMaxHeap* s2_left = mmh_create(y_s2_left, s2_left_count)  # must free
    cdef MinMaxHeap* s2_right = mmh_create(y_s2_right, s2_right_count)  # must free

    free(y_s2_left)
    free(y_s2_right)

    # printf('[S - CS] s1 left/right count: %ld/%ld\n', s1_left.size, s1_right.size)
    # printf('[S - CS] s2 left/right count: %ld/%ld\n', s2_left.size, s2_right.size)

    # print_split(s1_left, s1_right)
    # print_split(s2_left, s2_right)

    # initial gap
    cdef DTYPE_t score_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right,
        criterion)
    if score_gap < 0:
        printf('[WARNING] split2 is better than split1!')

    # printf('\n[S - CS] initial gap: %.5f\n', score_gap)

    # compute slack
    cdef SIZE_t slack = 0
    while score_gap >= 0:
        # print_split(s1_left, s1_right)
        # print_split(s2_left, s2_right)
        score_gap = reduce_score_gap(score_gap, s1_left, s1_right, s2_left, s2_right,
            best_threshold, second_threshold, criterion)
        slack += 1

    # clean up
    mmh_free(s1_left)
    mmh_free(s1_right)
    mmh_free(s2_left)
    mmh_free(s2_right)

    return slack

# private
cdef DTYPE_t compute_score_gap(MinMaxHeap* s1_left,
                               MinMaxHeap* s1_right,
                               MinMaxHeap* s2_left,
                               MinMaxHeap* s2_right,
                               SIZE_t      criterion) nogil:
    """
    Computes the score gap between the given splits.

    Return
        DTYPE_t, score gap.
    """
    cdef DTYPE_t g1 = _compute_split_score(s1_left, s1_right, criterion)
    cdef DTYPE_t g2 = _compute_split_score(s2_left, s2_right, criterion)
    # printf('\n[S - CSG] g1: %.5f, g2: %.5f\n', g1, g2)
    cdef DTYPE_t score_gap = g2 - g1
    return score_gap


cdef DTYPE_t reduce_score_gap(DTYPE_t     best_gap,
                              MinMaxHeap* s1_left,
                              MinMaxHeap* s1_right,
                              MinMaxHeap* s2_left,
                              MinMaxHeap* s2_right,
                              Threshold*  split1,
                              Threshold*  split2,
                              SIZE_t      criterion) nogil:
    """
    Finds the ADD operation that reduces the score gap the most.

    Return
        DTYPE_t, new score gap.
    """
    cdef DTYPE_t temp_gap
    cdef SIZE_t  best_case = -1

    cdef DTYPE_t median_s2_left
    cdef DTYPE_t median_s2_right

    # cdef DTYPE_t min_s2_left
    # cdef DTYPE_t min_s2_right

    # cdef DTYPE_t max_s2_left
    # cdef DTYPE_t max_s2_right

    # printf('MAX_FLOAT32: %.5f, best_gap: %.5f\n', MAX_FLOAT32, best_gap)

    # Case 0: add -inf to left branches
    mmh_insert(s1_left, -MAX_FLOAT32)
    mmh_insert(s2_left, -MAX_FLOAT32)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 0] temp_gap: %.5f\n', temp_gap)
    if temp_gap <= best_gap:
        best_gap = temp_gap
        best_case = 0
    mmh_remove(s1_left, -MAX_FLOAT32)
    mmh_remove(s2_left, -MAX_FLOAT32)

    # Case 1: add +inf to left branches
    mmh_insert(s1_left, MAX_FLOAT32)
    mmh_insert(s2_left, MAX_FLOAT32)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 1] temp_gap: %.5f\n', temp_gap)
    if temp_gap <= best_gap:
        best_gap = temp_gap
        best_case = 1
    mmh_remove(s1_left, MAX_FLOAT32)
    mmh_remove(s2_left, MAX_FLOAT32)

    # Case 2: add -inf to right branches
    mmh_insert(s1_right, -MAX_FLOAT32)
    mmh_insert(s2_right, -MAX_FLOAT32)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 2] temp_gap: %.5f\n', temp_gap)
    if temp_gap <= best_gap:
        best_gap = temp_gap
        best_case = 2
    mmh_remove(s1_right, -MAX_FLOAT32)
    mmh_remove(s2_right, -MAX_FLOAT32)

    # Case 3: add +inf to right branches
    mmh_insert(s1_right, MAX_FLOAT32)
    mmh_insert(s2_right, MAX_FLOAT32)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 3] temp_gap: %.5f\n', temp_gap)
    if temp_gap <= best_gap:
        best_gap = temp_gap
        best_case = 3
    mmh_remove(s1_right, MAX_FLOAT32)
    mmh_remove(s2_right, MAX_FLOAT32)

    # if (split1.feature != split2.feature) or (split1.value > split2.value):

    # Case 4: add -inf to split1 left branch and split2 right branch
    mmh_insert(s1_left, -MAX_FLOAT32)
    mmh_insert(s2_right, -MAX_FLOAT32)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 4] temp_gap: %.5f\n', temp_gap)
    if temp_gap <= best_gap:
        best_gap = temp_gap
        best_case = 4
    mmh_remove(s1_left, -MAX_FLOAT32)
    mmh_remove(s2_right, -MAX_FLOAT32)

    # Case 5: add +inf to split1 left branch and split2 right branch
    mmh_insert(s1_left, MAX_FLOAT32)
    mmh_insert(s2_right, MAX_FLOAT32)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 5] temp_gap: %.5f\n', temp_gap)
    if temp_gap <= best_gap:
        best_gap = temp_gap
        best_case = 5
    mmh_remove(s1_left, MAX_FLOAT32)
    mmh_remove(s2_right, MAX_FLOAT32)

    # if (split1.feature != split2.feature) or (split1.value < split2.value):

    # Case 6: add -inf to split1 right branch and split2 left branch
    mmh_insert(s1_right, -MAX_FLOAT32)
    mmh_insert(s2_left, -MAX_FLOAT32)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 6] temp_gap: %.5f\n', temp_gap)
    if temp_gap <= best_gap:
        best_gap = temp_gap
        best_case = 6
    mmh_remove(s1_right, -MAX_FLOAT32)
    mmh_remove(s2_left, -MAX_FLOAT32)

    # Case 7: add +inf to split1 right branch and split2 left branch
    mmh_insert(s1_right, MAX_FLOAT32)
    mmh_insert(s2_left, MAX_FLOAT32)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 7] temp_gap: %.5f\n', temp_gap)
    if temp_gap <= best_gap:
        best_gap = temp_gap
        best_case = 7
    mmh_remove(s1_right, MAX_FLOAT32)
    mmh_remove(s2_left, MAX_FLOAT32)

    # MEDIANS
    median_s2_left = mmh_median(s2_left)
    median_s2_right = mmh_median(s2_right)

    # Case 8: add split2 left median to BOTH LEFT branches
    mmh_insert(s1_left, median_s2_left)
    mmh_insert(s2_left, median_s2_left)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 8] temp_gap: %.5f\n', temp_gap)
    if temp_gap <= best_gap:
        best_gap = temp_gap
        best_case = 8
    mmh_remove(s1_left, median_s2_left)
    mmh_remove(s2_left, median_s2_left)

    # Case 9: add split2 right median to BOTH RIGHT branches
    mmh_insert(s1_right, median_s2_right)
    mmh_insert(s2_right, median_s2_right)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 9] temp_gap: %.5f\n', temp_gap)
    if temp_gap <= best_gap:
        best_gap = temp_gap
        best_case = 9
    mmh_remove(s1_right, median_s2_right)
    mmh_remove(s2_right, median_s2_right)

    # if (split1.feature != split2.feature) or (split1.value < split2.value):

    # Case 10: add split2 left median to split1 RIGHT branch and split2 LEFT branch
    mmh_insert(s1_right, median_s2_left)
    mmh_insert(s2_left, median_s2_left)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 10] temp_gap: %.5f\n', temp_gap)
    if temp_gap <= best_gap:
        best_gap = temp_gap
        best_case = 10
    mmh_remove(s1_right, median_s2_left)
    mmh_remove(s2_left, median_s2_left)

    # if (split1.feature != split2.feature) or (split1.value > split2.value):

    # Case 11: add split2 right median to split1 LEFT branch and split2 RIGHT branch
    mmh_insert(s1_left, median_s2_right)
    mmh_insert(s2_right, median_s2_right)
    temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 11] temp_gap: %.5f\n', temp_gap)
    if temp_gap <= best_gap:
        best_gap = temp_gap
        best_case = 11
    mmh_remove(s1_left, median_s2_right)
    mmh_remove(s2_right, median_s2_right)

    # # MINS
    # min_s2_left = mmh_min(s2_left)
    # min_s2_right = mmh_min(s2_right)

    # # Case 12: add split2 left min to BOTH LEFT branches
    # mmh_insert(s1_left, min_s2_left)
    # mmh_insert(s2_left, min_s2_left)
    # temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 12] temp_gap: %.5f\n', temp_gap)
    # if temp_gap < best_gap:
    #     best_gap = temp_gap
    #     best_case = 12
    # mmh_remove(s1_left, min_s2_left)
    # mmh_remove(s2_left, min_s2_left)

    # # Case 13: add split2 right min to BOTH RIGHT branches
    # mmh_insert(s1_right, min_s2_right)
    # mmh_insert(s2_right, min_s2_right)
    # temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 13] temp_gap: %.5f\n', temp_gap)
    # if temp_gap < best_gap:
    #     best_gap = temp_gap
    #     best_case = 13
    # mmh_remove(s1_right, min_s2_right)
    # mmh_remove(s2_right, min_s2_right)

    # if (split1.feature != split2.feature) or (split1.value < split2.value):

    #     # Case 14: add split2 left min to split1 RIGHT branch and split2 LEFT branch
    #     mmh_insert(s1_right, min_s2_left)
    #     mmh_insert(s2_left, min_s2_left)
    #     temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    #     printf('[CASE 14] temp_gap: %.5f\n', temp_gap)
    #     if temp_gap < best_gap:
    #         best_gap = temp_gap
    #         best_case = 14
    #     mmh_remove(s1_right, min_s2_left)
    #     mmh_remove(s2_left, min_s2_left)

    # if (split1.feature != split2.feature) or (split1.value > split2.value):

    #     # Case 15: add split2 right min to split1 LEFT branch and split2 RIGHT branch
    #     mmh_insert(s1_left, min_s2_right)
    #     mmh_insert(s2_right, min_s2_right)
    #     temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    #     printf('[CASE 15] temp_gap: %.5f\n', temp_gap)
    #     if temp_gap < best_gap:
    #         best_gap = temp_gap
    #         best_case = 15
    #     mmh_remove(s1_left, min_s2_right)
    #     mmh_remove(s2_right, min_s2_right)

    # # MAX
    # max_s2_left = mmh_max(s2_left)
    # max_s2_right = mmh_max(s2_right)

    # # Case 16: add split2 left max to BOTH LEFT branches
    # mmh_insert(s1_left, max_s2_left)
    # mmh_insert(s2_left, max_s2_left)
    # temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 16] temp_gap: %.5f\n', temp_gap)
    # if temp_gap < best_gap:
    #     best_gap = temp_gap
    #     best_case = 16
    # mmh_remove(s1_left, max_s2_left)
    # mmh_remove(s2_left, max_s2_left)

    # # Case 17: add split2 right max to BOTH RIGHT branches
    # mmh_insert(s1_right, max_s2_right)
    # mmh_insert(s2_right, max_s2_right)
    # temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    # printf('[CASE 17] temp_gap: %.5f\n', temp_gap)
    # if temp_gap < best_gap:
    #     best_gap = temp_gap
    #     best_case = 17
    # mmh_remove(s1_right, max_s2_right)
    # mmh_remove(s2_right, max_s2_right)

    # if (split1.feature != split2.feature) or (split1.value < split2.value):

    #     # Case 18: add split2 left max to split1 RIGHT branch and split2 LEFT branch
    #     mmh_insert(s1_right, max_s2_left)
    #     mmh_insert(s2_left, max_s2_left)
    #     temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    #     printf('[CASE 18] temp_gap: %.5f\n', temp_gap)
    #     if temp_gap < best_gap:
    #         best_gap = temp_gap
    #         best_case = 18
    #     mmh_remove(s1_right, max_s2_left)
    #     mmh_remove(s2_left, max_s2_left)

    # if (split1.feature != split2.feature) or (split1.value > split2.value):

    #     # Case 19: add split2 right max to split1 LEFT branch and split2 RIGHT branch
    #     mmh_insert(s1_left, max_s2_right)
    #     mmh_insert(s2_right, max_s2_right)
    #     temp_gap = compute_score_gap(s1_left, s1_right, s2_left, s2_right, criterion)
    #     printf('[CASE 19] temp_gap: %.5f\n', temp_gap)
    #     if temp_gap < best_gap:
    #         best_gap = temp_gap
    #         best_case = 19
    #     mmh_remove(s1_left, max_s2_right)
    #     mmh_remove(s2_right, max_s2_right)

    # printf('best_gap: %.5f, best_case: %ld\n', best_gap, best_case)

    # update splits based on the chosen operation
    if best_case == -1:
        printf('\n[WARNING] No operation reduced score gap!\n')
        return -1
        # raise ValueError('\n[WARNING] No operation reduced score gap!')

    if best_case == 0:
        mmh_insert(s1_left, -MAX_FLOAT32)
        mmh_insert(s2_left, -MAX_FLOAT32)
    elif best_case == 1:
        mmh_insert(s1_left, MAX_FLOAT32)
        mmh_insert(s2_left, MAX_FLOAT32)
    elif best_case == 2:
        mmh_insert(s1_right, -MAX_FLOAT32)
        mmh_insert(s2_right, -MAX_FLOAT32)
    elif best_case == 3:
        mmh_insert(s1_right, MAX_FLOAT32)
        mmh_insert(s2_right, MAX_FLOAT32)
    elif best_case == 4:
        mmh_insert(s1_left, -MAX_FLOAT32)
        mmh_insert(s2_right, -MAX_FLOAT32)
    elif best_case == 5:
        mmh_insert(s1_left, MAX_FLOAT32)
        mmh_insert(s2_right, MAX_FLOAT32)
    elif best_case == 6:
        mmh_insert(s1_right, -MAX_FLOAT32)
        mmh_insert(s2_left, -MAX_FLOAT32)
    elif best_case == 7:
        mmh_insert(s1_right, MAX_FLOAT32)
        mmh_insert(s2_left, MAX_FLOAT32)
    # medians
    elif best_case == 8:
        mmh_insert(s1_left, median_s2_left)
        mmh_insert(s2_left, median_s2_left)
    elif best_case == 9:
        mmh_insert(s1_right, median_s2_right)
        mmh_insert(s2_right, median_s2_right)
    elif best_case == 10:
        mmh_insert(s1_right, median_s2_left)
        mmh_insert(s2_left, median_s2_left)
    elif best_case == 11:
        mmh_insert(s1_left, median_s2_right)
        mmh_insert(s2_right, median_s2_right)
    # # mins
    # elif best_case == 12:
    #     mmh_insert(s1_left, min_s2_left)
    #     mmh_insert(s2_left, min_s2_left)
    # elif best_case == 13:
    #     mmh_insert(s1_right, min_s2_right)
    #     mmh_insert(s2_right, min_s2_right)
    # elif best_case == 14:
    #     mmh_insert(s1_right, min_s2_left)
    #     mmh_insert(s2_left, min_s2_left)
    # elif best_case == 15:
    #     mmh_insert(s1_left, min_s2_right)
    #     mmh_insert(s2_right, min_s2_right)
    # # maxs
    # elif best_case == 16:
    #     mmh_insert(s1_left, max_s2_left)
    #     mmh_insert(s2_left, max_s2_left)
    # elif best_case == 17:
    #     mmh_insert(s1_right, max_s2_right)
    #     mmh_insert(s2_right, max_s2_right)
    # elif best_case == 18:
    #     mmh_insert(s1_right, max_s2_left)
    #     mmh_insert(s2_left, max_s2_left)
    # elif best_case == 19:
    #     mmh_insert(s1_left, max_s2_right)
    #     mmh_insert(s2_right, max_s2_right)

    # printf('[S - CS] best gap: %.5f, best case: %ld\n', best_gap, best_case)

    return best_gap


cdef void print_split(MinMaxHeap* s_left, MinMaxHeap* s_right) nogil:
    """
    Print the target values of the left and right branches.
    """
    cdef SIZE_t i = 0

    for i in range(s_left.max_heap.size):
        printf('%.5f ', s_left.max_heap.arr[i])

    for i in range(s_left.min_heap.size):
        printf('%.5f ', s_left.min_heap.arr[i])

    printf('| ')

    for i in range(s_right.max_heap.size):
        printf('%.5f ', s_right.max_heap.arr[i])

    for i in range(s_right.min_heap.size):
        printf('%.5f ', s_right.min_heap.arr[i])

    printf('\n')
