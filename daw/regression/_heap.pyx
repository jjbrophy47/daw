# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

"""
Implementation of MinMaxHeap for streaming median.
"""
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf

cimport cython

from ._slack cimport MAX_FLOAT32


# =======================
# Streaming median
# =======================


# Python API
cdef class _MinMaxHeap:
    """
    Streaming median data structure consisting of
    a _MinHeap and a _MaxHeap.
    """
    property median:
        def __get__(self):
            return mmh_median(self.mmh)

    property mean:
        def __get__(self):
            return mmh_mean(self.mmh)

    property size:
        def __get__(self):
            return self.mmh.size

    def __cinit__(self, float[:] input_vals):
        """
        Constructor.
        """
        cdef SIZE_t n = input_vals.shape[0]
        cdef DTYPE_t* values = <DTYPE_t *>malloc(n * sizeof(DTYPE_t))
        cdef SIZE_t i = 0
        for i in range(n):
            values[i] = input_vals[i]
        self.mmh = mmh_create(values, n)
        free(values)

    def __dealloc__(self):
        """
        Destructor.
        """
        mmh_free(self.mmh)

    def __str__(self):
        out_str = f'MinHeap:\n{heap_str(self.mmh.min_heap, 0, "", False)}'
        out_str += f'\nMaxHeap:\n{heap_str(self.mmh.max_heap, 0, "", False)}'
        return out_str

    cpdef void insert(self, float item):
        mmh_insert(self.mmh, <DTYPE_t>item)

    cpdef void remove(self, float item):
        mmh_remove(self.mmh, <DTYPE_t>item)


# C API
cdef MinMaxHeap* mmh_create(DTYPE_t* input_vals, SIZE_t n) nogil:
    """
    Initialize a MinMaxHeap object.

    NOTE: caller must free.
    """
    cdef SIZE_t i = 0

    cdef MinMaxHeap* mmh = <MinMaxHeap*>malloc(sizeof(MinMaxHeap))
    mmh.min_heap = list_create()
    mmh.max_heap = list_create()
    mmh.size = 0
    mmh.total = 0

    for i in range(n):
        mmh_insert(mmh, input_vals[i])

    return mmh


cdef void mmh_free(MinMaxHeap* mmh) nogil:
    """
    Free contents of MinMaxHeap and MinMaxHeap.
    """
    list_free(mmh.min_heap)
    list_free(mmh.max_heap)
    free(mmh)
    mmh = NULL


cdef void mmh_insert(MinMaxHeap* mmh, DTYPE_t x) nogil:
    """
    Insert x into either the min or max heap.
    """
    cdef DTYPE_t item

    if mmh.size == 0:
        min_heap_insert(mmh.min_heap, x)
        mmh.size += 1
        mmh.total += x
        return

    if mmh.min_heap.size > mmh.max_heap.size:
        if x < heap_root(mmh.min_heap):
            max_heap_insert(mmh.max_heap, x)
        else:
            item = min_heap_insertpop(mmh.min_heap, x)
            max_heap_insert(mmh.max_heap, item)

    else:
        if x > heap_root(mmh.min_heap):
            min_heap_insert(mmh.min_heap, x)
        else:
            item = max_heap_insertpop(mmh.max_heap, x)
            min_heap_insert(mmh.min_heap, item)

    mmh.size += 1
    mmh.total += x


cdef void mmh_remove(MinMaxHeap* mmh, DTYPE_t x) nogil:
    """
    Remove x from either the min or max heap.
    """
    cdef DTYPE_t item

    if mmh.size == 1:
        if mmh.min_heap.size == 1:
            min_heap_remove(mmh.min_heap, x)
        else:
            max_heap_remove(mmh.max_heap, x)
        mmh.size -= 1
        mmh.total -= x
        return

    # remove from smaller half of list, ensure heaps are balanced
    if x < mmh_median(mmh):
        max_heap_remove(mmh.max_heap, x)  # raises IndexError if not in heap
        if mmh.min_heap.size - mmh.max_heap.size > 1:
            item = min_heap_pop(mmh.min_heap)
            max_heap_insert(mmh.max_heap, item)

    # remove from larger half of list, ensure min heap is equal to or
    # larger than max heap by 1
    else:
        min_heap_remove(mmh.min_heap, x)  # raises IndexError if not in heap
        if mmh.max_heap.size - mmh.min_heap.size > 0:
            item = max_heap_pop(mmh.max_heap)
            min_heap_insert(mmh.min_heap, item)

    mmh.size -= 1
    mmh.total -= x


cdef DTYPE_t mmh_median(MinMaxHeap* mmh) nogil:
    """
    Return median.
    """
    cdef DTYPE_t result

    if mmh.min_heap.size > mmh.max_heap.size:
        result = heap_root(mmh.min_heap)
    elif mmh.min_heap.size < mmh.max_heap.size:
        result = heap_root(mmh.max_heap)
    else:
        result = (heap_root(mmh.min_heap) + heap_root(mmh.max_heap)) / 2
    return result


cdef DTYPE_t mmh_mean(MinMaxHeap* mmh) nogil:
    """
    Return mean.
    """
    return mmh.total / mmh.size


cdef DTYPE_t mmh_min(MinMaxHeap* mmh) nogil:
    """
    Return min value of the MinMaxHeap.
    """
    cdef DTYPE_t result = MAX_FLOAT32
    cdef SIZE_t i

    if mmh.size == 1:
        return mmh.min_heap.arr[0]

    for i in range(mmh.max_heap.size):
        if mmh.max_heap.arr[i] < result:
            result = mmh.max_heap.arr[i]

    return result


cdef DTYPE_t mmh_max(MinMaxHeap* mmh) nogil:
    """
    Return min value of the MinMaxHeap.
    """
    cdef DTYPE_t result = -MAX_FLOAT32
    cdef SIZE_t i

    if mmh.size == 1:
        return mmh.min_heap.arr[0]

    for i in range(mmh.min_heap.size):
        if mmh.min_heap.arr[i] > result:
            result = mmh.min_heap.arr[i]

    return result


# =======================
# Heap methods
# =======================

cdef class _MinHeap:
    property size:
        def __get__(self):
            return self.heap.size

    property root:
        def __get__(self):
            return heap_root(self.heap)

    def __cinit__(self):
        self.heap = list_create()

    def __dealloc__(self):
        list_free(self.heap)

    def __str__(self):
        return heap_str(self.heap, 0, '', False)

    cpdef void insert(self, float item):
        min_heap_insert(self.heap, <DTYPE_t>item)

    cpdef float pop(self):
        return <float> min_heap_pop(self.heap)

    cpdef float insertpop(self, float item):
        return <float> min_heap_insertpop(self.heap, <DTYPE_t>item)

    cpdef void remove(self, float item):
        min_heap_remove(self.heap, <DTYPE_t>item)


cdef class _MaxHeap:
    property size:
        def __get__(self):
            return self.heap.size

    property root:
        def __get__(self):
            return heap_root(self.heap)

    def __cinit__(self):
        self.heap = list_create()

    def __dealloc__(self):
        list_free(self.heap)

    def __str__(self):
        return heap_str(self.heap, 0, '', False)

    cpdef void insert(self, float item):
        max_heap_insert(self.heap, <DTYPE_t>item)

    cpdef float pop(self):
        return <float> max_heap_pop(self.heap)

    cpdef float insertpop(self, float item):
        return <float> max_heap_insertpop(self.heap, <DTYPE_t>item)

    cpdef void remove(self, float item):
        max_heap_remove(self.heap, <DTYPE_t>item)


# Python API
cdef str heap_str(List* heap, SIZE_t pos, str indent, bint last):
    """
    Recursively print the tree using pre-prder traversal.
    """
    out_str = ''
    if pos < heap.size:
        if pos == 0:
            out_str += f'{heap.arr[pos]}'
        else:
            out_str += f'\n{indent}'
            if last:
                out_str += f'R----{heap.arr[pos]}'
                indent += "     "
            else:
                out_str += f'L----{heap.arr[pos]}'
                indent += "|    "
        out_str += heap_str(heap, _heap_left_child_pos(pos), indent, False)
        out_str += heap_str(heap, _heap_right_child_pos(pos), indent, True)
    return out_str


# C API
cdef DTYPE_t heap_root(List* heap) nogil:
    """
    Return root value of heap.
    """
    if heap.size == 0:
        raise IndexError('Empty heap!')
    return heap.arr[0]


cdef void min_heap_insert(List* heap, DTYPE_t item) nogil:
    """
    Add item to the heap, then ensure heap invariance.
    """
    list_append(heap, item)
    _min_heap_siftdown(heap, 0, heap.size - 1)


cdef void max_heap_insert(List* heap, DTYPE_t item) nogil:
    """
    Add item to the heap, then ensure heap invariance.
    """
    list_append(heap, item)
    _max_heap_siftdown(heap, 0, heap.size - 1)


cdef DTYPE_t min_heap_pop(List* heap) nogil:
    """
    Pop smallest item from the heap, then ensure heap invariance.
    """
    cdef DTYPE_t last_item = list_pop(heap)  # raises IndexError if heap is empty
    cdef DTYPE_t return_item

    if heap.size > 0:
        return_item = heap.arr[0]
        heap.arr[0] = last_item
        _min_heap_siftup(heap, 0)  # move to leaf, then to its right place
        return return_item

    return last_item


cdef DTYPE_t max_heap_pop(List* heap) nogil:
    """
    Pop largest item from the heap, then ensure heap invariance.
    """
    cdef DTYPE_t last_item = list_pop(heap)  # raises IndexError if heap is empty
    cdef DTYPE_t return_item

    if heap.size > 0:
        return_item = heap.arr[0]
        heap.arr[0] = last_item
        _max_heap_siftup(heap, 0)  # move to leaf, then to its right place
        return return_item

    return last_item


cdef DTYPE_t min_heap_insertpop(List* heap, DTYPE_t item) nogil:
    """
    Fast version of insert followed by pop.
    """
    cdef DTYPE_t temp

    if heap.size == 0:
        raise IndexError('Empty heap!')

    if item > heap.arr[0]:
        temp = heap.arr[0]
        heap.arr[0] = item
        item = temp
        _min_heap_siftup(heap, 0)
    return item


cdef DTYPE_t max_heap_insertpop(List* heap, DTYPE_t item) nogil:
    """
    Fast version of insert followed by pop.
    """
    cdef DTYPE_t temp

    if heap.size == 0:
        raise IndexError('Empty heap!')

    if item < heap.arr[0]:
        temp = heap.arr[0]
        heap.arr[0] = item
        item = temp
        _max_heap_siftup(heap, 0)
    return item


cdef void min_heap_remove(List* heap, DTYPE_t item) nogil:
    """
    Remove first found occurrence of item in heap.
    """
    cdef SIZE_t pos = list_index(heap, item)  # raises ValueError if item not in heap
    heap.arr[pos] = heap.arr[heap.size - 1]
    list_pop(heap)  # removes last element
    if pos < heap.size:
        _min_heap_siftup(heap, pos)
        _min_heap_siftdown(heap, 0, pos)


cdef void max_heap_remove(List* heap, DTYPE_t item) nogil:
    """
    Remove first found occurrence of item in heap.
    """
    cdef SIZE_t pos = list_index(heap, item)  # raises ValueError if item not in heap
    heap.arr[pos] = heap.arr[heap.size - 1]
    list_pop(heap)  # removes last element
    if pos < heap.size:
        _max_heap_siftup(heap, pos)
        _max_heap_siftdown(heap, 0, pos)


# private C API
cdef SIZE_t _heap_parent_pos(SIZE_t pos) nogil:
    """
    Return parent pos given pos.
    """
    return (pos - 1) >> 1


cdef SIZE_t _heap_left_child_pos(SIZE_t pos) nogil:
    """
    Return left child pos given pos.
    """
    return (2 * pos) + 1


cdef SIZE_t _heap_right_child_pos(SIZE_t pos) nogil:
    """
    Return right child pos given pos.
    """
    return (2 * pos) + 2


cdef void _heap_swap(List* heap, SIZE_t i, SIZE_t j) nogil:
    """
    Swap elements at pos i and j.
    """
    cdef DTYPE_t temp = heap.arr[i]
    heap.arr[i] = heap.arr[j]
    heap.arr[j] = temp


cdef void _min_heap_siftdown(List* heap, SIZE_t start_pos, SIZE_t pos) nogil:
    """
    Move node at pos up, moving parents down until start_pos.
    """
    cdef DTYPE_t new_item = heap.arr[pos]
    cdef SIZE_t parent_pos
    cdef DTYPE_t parent

    # bubble new_item up
    while pos > start_pos:
        parent_pos = _heap_parent_pos(pos)
        parent = heap.arr[parent_pos]

        # move parent down
        if new_item < parent:
            heap.arr[pos] = parent
            pos = parent_pos
            continue

        break

    heap.arr[pos] = new_item


cdef void _max_heap_siftdown(List* heap, SIZE_t start_pos, SIZE_t pos) nogil:
    """
    Move node at pos up, moving parents down until start_pos.
    """
    cdef DTYPE_t new_item = heap.arr[pos]
    cdef SIZE_t parent_pos
    cdef DTYPE_t parent

    # bubble new_item up
    while pos > start_pos:
        parent_pos = _heap_parent_pos(pos)
        parent = heap.arr[parent_pos]

        # move parent down
        if new_item > parent:
            heap.arr[pos] = parent
            pos = parent_pos
            continue

        break

    heap.arr[pos] = new_item


cdef void _min_heap_siftup(List* heap, SIZE_t pos) nogil:
    """
    Move node at pos down to a leaf, moving child nodes up.
    """
    cdef SIZE_t start_pos = pos
    cdef SIZE_t end_pos = heap.size
    cdef DTYPE_t new_item = heap.arr[pos]
    cdef SIZE_t child_pos = _heap_left_child_pos(pos)
    cdef SIZE_t right_pos

    # move new_item down
    while child_pos < end_pos:
        right_pos = child_pos + 1
        if right_pos < end_pos and not heap.arr[child_pos] < heap.arr[right_pos]:
            child_pos = right_pos
        heap.arr[pos] = heap.arr[child_pos]
        pos = child_pos
        child_pos = _heap_left_child_pos(pos)

    heap.arr[pos] = new_item
    _min_heap_siftdown(heap, start_pos, pos)


cdef void _max_heap_siftup(List* heap, SIZE_t pos) nogil:
    """
    Move node at pos down to a leaf, moving child nodes up.
    """
    cdef SIZE_t start_pos = pos
    cdef SIZE_t end_pos = heap.size
    cdef DTYPE_t new_item = heap.arr[pos]
    cdef SIZE_t child_pos = _heap_left_child_pos(pos)
    cdef SIZE_t right_pos

    # move new_item down
    while child_pos < end_pos:
        right_pos = child_pos + 1
        if right_pos < end_pos and not heap.arr[child_pos] > heap.arr[right_pos]:
            child_pos = right_pos
        heap.arr[pos] = heap.arr[child_pos]
        pos = child_pos
        child_pos = _heap_left_child_pos(pos)

    heap.arr[pos] = new_item
    _max_heap_siftdown(heap, start_pos, pos)


# =======================
# List methods
# =======================


# Python API
cdef str list_str(List* my_list):
    """
    Return string representation of List.
    """
    out_str = '['
    cdef SIZE_t i = 0
    for i in range(my_list.size):
        out_str += '%.5f' % my_list.arr[i]
        if i != my_list.size - 1:
            out_str += ', '
    out_str += ']'
    return out_str


# C API
cdef List* list_create() nogil:
    """
    Initialize a list object.

    Note: Caller must free.
    """
    cdef List* my_list = <List *>malloc(sizeof(List))
    my_list.size = 0
    my_list.capacity = 10
    my_list.arr = <DTYPE_t *>malloc(my_list.capacity * sizeof(DTYPE_t))
    return my_list

cdef void list_free(List* my_list) nogil:
    """
    Free contents of list and free list pointer.
    """
    free(my_list.arr)
    free(my_list)
    my_list = NULL

cdef void list_append(List* my_list, DTYPE_t x) nogil:
    """
    Add item to the end of the array,
    double memory capacity if necessary.
    """
    if my_list.capacity == my_list.size:
        _list_increase_capacity(my_list)

    my_list.arr[my_list.size] = x
    my_list.size += 1

cdef DTYPE_t list_pop(List* my_list) nogil:
    """
    Pop the last element off of the list and return it.
    """
    if my_list.size == 0:
        raise ValueError('List is empty!')

    cdef DTYPE_t result = my_list.arr[my_list.size - 1]
    my_list.size -= 1
    return result

cdef SIZE_t list_index(List* my_list, DTYPE_t item) nogil:
    """
    Find index of item, throw exception if not found.
    """
    cdef SIZE_t i = 0

    if my_list.size == 0:
        raise ValueError('List is empty!')

    for i in range(my_list.size):
        if my_list.arr[i] == item:
            return i

    raise IndexError('Item not in list!')

cdef void list_print(List* my_list) nogil:
    """
    Print string representation of List.
    """
    cdef SIZE_t i = 0
    printf('[')
    for i in range(my_list.size):
        printf('%.5f', my_list.arr[i])
        if i != my_list.size - 1:
            printf(', ')
    printf(']\n')


# private C API
cdef void _list_increase_capacity(List* my_list) nogil:
    """
    Double array size.
    """
    my_list.capacity *= 2
    my_list.arr = <DTYPE_t *>realloc(my_list.arr, my_list.capacity * sizeof(DTYPE_t))
