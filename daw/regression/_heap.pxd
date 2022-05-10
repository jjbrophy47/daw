import numpy as np
cimport numpy as np

from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t

# Streaming median data structure
cdef struct MinMaxHeap:
    List* min_heap
    List* max_heap
    SIZE_t  size
    DTYPE_t total

# List structure
cdef struct List:
    DTYPE_t* arr
    SIZE_t   size
    SIZE_t   capacity

# =======================
# Streaming median
# =======================

# Python API
cdef class _MinMaxHeap:
    cdef MinMaxHeap* mmh
    cpdef void insert(self, float item)
    cpdef void remove(self, float item)

# C API
cdef MinMaxHeap* mmh_create(DTYPE_t* input_vals, SIZE_t n) nogil
cdef void mmh_free(MinMaxHeap* mmh) nogil
cdef void mmh_insert(MinMaxHeap* mmh, DTYPE_t x) nogil
cdef void mmh_remove(MinMaxHeap* mmh, DTYPE_t x) nogil
cdef DTYPE_t mmh_median(MinMaxHeap* mmh) nogil
cdef DTYPE_t mmh_mean(MinMaxHeap* mmh) nogil

# =======================
# Heap methods
# =======================

# Python API
cdef class _MinHeap:
    cdef List* heap
    cpdef void insert(self, float item)
    cpdef float pop(self)
    cpdef float insertpop(self, float item)
    cpdef void remove(self, float item)

cdef class _MaxHeap:
    cdef List* heap
    cpdef void insert(self, float item)
    cpdef float pop(self)
    cpdef float insertpop(self, float item)
    cpdef void remove(self, float item)

cdef str heap_str(List* heap, SIZE_t pos, str indent, bint last)

# C API
cdef DTYPE_t heap_root(List* heap) nogil
cdef void min_heap_insert(List* heap, DTYPE_t item) nogil
cdef void max_heap_insert(List* heap, DTYPE_t item) nogil
cdef DTYPE_t min_heap_pop(List* heap) nogil
cdef DTYPE_t max_heap_pop(List* heap) nogil
cdef DTYPE_t min_heap_insertpop(List* heap, DTYPE_t item) nogil
cdef DTYPE_t max_heap_insertpop(List* heap, DTYPE_t item) nogil
cdef void min_heap_remove(List* heap, DTYPE_t item) nogil
cdef void max_heap_remove(List* heap, DTYPE_t item) nogil

# private C API
cdef SIZE_t _heap_parent_pos(SIZE_t pos) nogil
cdef SIZE_t _heap_left_child_pos(SIZE_t pos) nogil
cdef SIZE_t _heap_right_child_pos(SIZE_t pos) nogil
cdef void _heap_swap(List* heap, SIZE_t i, SIZE_t j) nogil
cdef void _min_heap_siftdown(List* heap, SIZE_t start_pos, SIZE_t pos) nogil
cdef void _max_heap_siftdown(List* heap, SIZE_t start_pos, SIZE_t pos) nogil
cdef void _min_heap_siftup(List* heap, SIZE_t pos) nogil
cdef void _max_heap_siftup(List* heap, SIZE_t pos) nogil

# =======================
# List methods
# =======================

# Python API
cdef str list_str(List* my_list)

# C API
cdef List* list_create() nogil
cdef void list_free(List* my_list) nogil
cdef void list_append(List* my_list, DTYPE_t x) nogil
cdef DTYPE_t list_pop(List* my_list) nogil
cdef SIZE_t list_index(List* my_list, DTYPE_t item) nogil

# private C API
cdef void _list_increase_capacity(List* my_list) nogil
