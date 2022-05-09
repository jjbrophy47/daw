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


# =======================
# Heap
# =======================


cdef class _BaseHeap:
    """
    Base Heap class, do not use on its own!
    """

    # Public Python API
    property size:
        def __get__(self):
            return self.heap.size

    property root:
        def __get__(self):
            if self.heap.size == 0:
                raise IndexError('Empty heap!')
            return self.heap.arr[0]

    # Public C API
    def __cinit__(self):
        """
        Constructor.
        """
        self.heap = _List()

    def __str__(self):
        """
        Function to print the contents of the heap.
        """
        return self._str_helper(0, '', False)

    # Public Python API
    cpdef void insert(self, float item):
        self._insert(<DTYPE_t>item)

    cpdef float pop(self):
        return <float> self._pop()

    cpdef float insertpop(self, float item):
        return <float> self._insertpop(item)

    cpdef void remove(self, float item):
        self._remove(<DTYPE_t>item)

    # Public C API
    cdef void _insert(self, DTYPE_t item) nogil:
        """
        Add item to the heap, then ensure heap invariance.
        """
        self.heap._append(item)
        self._siftdown(0, self.heap.size - 1)

    cdef DTYPE_t _pop(self) nogil:
        """
        Pop smallest item from the heap, then ensure heap invariance.
        """
        cdef DTYPE_t last_item = self.heap._pop()  # raises IndexError if heap is empty
        cdef DTYPE_t return_item

        if self.heap.size > 0:
            return_item = self.heap.arr[0]
            self.heap.arr[0] = last_item
            self._siftup(0)  # move to leaf, then to its right place
            return return_item

        return last_item

    cdef DTYPE_t _insertpop(self, DTYPE_t item) nogil:
        """
        ABSTRACT METHOD

        Fast version of insert followed by pop.
        """
        pass

    cdef void _remove(self, DTYPE_t item) nogil:
        """
        Remove first found occurrence of item in heap.
        """
        cdef SIZE_t pos = self.heap._index(item)  # raises ValueError if item not in heap
        self.heap.arr[pos] = self.heap.arr[self.heap.size - 1]
        self.heap._pop()  # removes last element
        if pos < self.heap.size:
            self._siftup(pos)
            self._siftdown(0, pos)

    # private
    cdef SIZE_t _parent_pos(self, SIZE_t pos) nogil:
        """
        Return parent pos given pos.
        """
        return (pos - 1) >> 1

    cdef SIZE_t _left_child_pos(self, SIZE_t pos) nogil:
        """
        Return left child pos given pos.
        """
        return (2 * pos) + 1

    cdef SIZE_t _right_child_pos(self, SIZE_t pos) nogil:
        """
        Return right child pos given pos.
        """
        return (2 * pos) + 2

    cdef void _swap(self, SIZE_t i, SIZE_t j) nogil:
        """
        Swap elements at pos i and j.
        """
        cdef DTYPE_t temp = self.heap.arr[i]
        self.heap.arr[i] = self.heap.arr[j]
        self.heap.arr[j] = temp

    cdef void _siftdown(self, SIZE_t start_pos, SIZE_t pos) nogil:
        """
        ABSTRACT METHOD

        Move node at pos up, moving parents down until start_pos.
        """
        pass

    cdef void _siftup(self, SIZE_t pos) nogil:
        """
        ABSTRACT METHOD

        Move node at pos down to a leaf, moving child nodes up.
        """
        pass

    cdef str _str_helper(self, SIZE_t pos, str indent, bint last):
        """
        Recursively print the tree using pre-prder traversal.
        """
        out_str = ''
        if pos < self.heap.size:
            if pos == 0:
                out_str += f'{self.heap.arr[pos]}'
            else:
                out_str += f'\n{indent}'
                if last:
                    out_str += f'R----{self.heap.arr[pos]}'
                    indent += "     "
                else:
                    out_str += f'L----{self.heap.arr[pos]}'
                    indent += "|    "
            out_str += self._str_helper(self._left_child_pos(pos), indent, False)
            out_str += self._str_helper(self._right_child_pos(pos), indent, True)
        return out_str


cdef class _MinHeap(_BaseHeap):
    """
    Minimum heap, parent node values should always be less
    than child node values.
    """

    # Public C API
    cdef DTYPE_t _insertpop(self, DTYPE_t item) nogil:
        """
        Fast version of insert followed by pop.
        """
        cdef DTYPE_t temp

        if self.heap.size == 0:
            raise IndexError('Empty heap!')

        if self.heap.arr[0] < item:
            temp = self.heap.arr[0]
            self.heap.arr[0] = item
            item = temp
            self._siftup(0)
        return item

    # private
    cdef void _siftdown(self, SIZE_t start_pos, SIZE_t pos) nogil:
        """
        Move node at pos up, moving parents down until start_pos.
        """
        cdef DTYPE_t new_item = self.heap.arr[pos]
        cdef SIZE_t parent_pos
        cdef DTYPE_t parent

        # bubble new_item up
        while pos > start_pos:
            parent_pos = self._parent_pos(pos)
            parent = self.heap.arr[parent_pos]

            # move parent down
            if new_item < parent:
                self.heap.arr[pos] = parent
                pos = parent_pos
                continue

            break

        self.heap.arr[pos] = new_item

    cdef void _siftup(self, SIZE_t pos) nogil:
        """
        Move node at pos down to a leaf, moving child nodes up.
        """
        cdef SIZE_t start_pos = pos
        cdef SIZE_t end_pos = self.heap.size
        cdef DTYPE_t new_item = self.heap.arr[pos]
        cdef SIZE_t child_pos = self._left_child_pos(pos)
        cdef SIZE_t right_pos

        # move new_item down
        while child_pos < end_pos:
            right_pos = child_pos + 1
            if right_pos < end_pos and not self.heap.arr[child_pos] < self.heap.arr[right_pos]:
                child_pos = right_pos
            self.heap.arr[pos] = self.heap.arr[child_pos]
            pos = child_pos
            child_pos = self._left_child_pos(pos)

        self.heap.arr[pos] = new_item
        self._siftdown(start_pos, pos)


cdef class _MaxHeap(_BaseHeap):
    """
    Maximum heap, parent node values should always be greater
    than child node values.
    """

    # Public C API
    cdef DTYPE_t _insertpop(self, DTYPE_t item) nogil:
        """
        Fast version of insert followed by pop.
        """
        cdef DTYPE_t temp

        if self.heap.size == 0:
            raise IndexError('Empty heap!')

        if item < self.heap.arr[0]:
            temp = self.heap.arr[0]
            self.heap.arr[0] = item
            item = temp
            self._siftup(0)
        return item

    # private
    cdef void _siftdown(self, SIZE_t start_pos, SIZE_t pos) nogil:
        """
        Move node at pos up, moving parents down until start_pos.
        """
        cdef DTYPE_t new_item = self.heap.arr[pos]
        cdef SIZE_t parent_pos
        cdef DTYPE_t parent

        # bubble new_item up
        while pos > start_pos:
            parent_pos = self._parent_pos(pos)
            parent = self.heap.arr[parent_pos]

            # move parent down
            if new_item > parent:
                self.heap.arr[pos] = parent
                pos = parent_pos
                continue

            break

        self.heap.arr[pos] = new_item

    cdef void _siftup(self, SIZE_t pos) nogil:
        """
        Move node at pos down to a leaf, moving child nodes up.
        """
        cdef SIZE_t start_pos = pos
        cdef SIZE_t end_pos = self.heap.size
        cdef DTYPE_t new_item = self.heap.arr[pos]
        cdef SIZE_t child_pos = self._left_child_pos(pos)
        cdef SIZE_t right_pos

        # move new_item down
        while child_pos < end_pos:
            right_pos = child_pos + 1
            if right_pos < end_pos and not self.heap.arr[child_pos] > self.heap.arr[right_pos]:
                child_pos = right_pos
            self.heap.arr[pos] = self.heap.arr[child_pos]
            pos = child_pos
            child_pos = self._left_child_pos(pos)

        self.heap.arr[pos] = new_item
        self._siftdown(start_pos, pos)


# =======================
# List
# =======================


cdef class _List:
    """
    List data structure for floats.
    """

    # Public Python API
    property size:
        def __get__(self):
            return self.size

    def __cinit__(self):
        """
        Constructor.
        """
        self.size = 0
        self.capacity = 10
        self.arr = <DTYPE_t *>malloc(self.capacity * sizeof(DTYPE_t))

    def __str__(self):
        """
        Return string representation of List.
        """
        out_str = '['
        cdef SIZE_t i = 0
        for i in range(self.size):
            out_str += '%.5f' % self.arr[i]
            if i != self.size - 1:
                out_str += ', '
        out_str += ']'
        return out_str

    cpdef void append(self, float x):
        """
        Add item to internal list.
        """
        self._append(<DTYPE_t>x)

    cpdef float pop(self):
        """
        Pop and return the last item.
        """
        return <float> self._pop()

    # Public C API
    cdef void _append(self, DTYPE_t x) nogil:
        """
        Add item to the end of the array,
        double memory capacity if necessary.
        """
        if self.capacity == self.size:
            self._increase_capacity()

        self.arr[self.size] = x
        self.size += 1

    cdef DTYPE_t _pop(self) nogil:
        """
        Pop the last element off of the list and return it.
        """
        if self.size == 0:
            raise ValueError('List is empty!')

        cdef DTYPE_t result = self.arr[self.size - 1]
        self.size -= 1
        return result

    cdef SIZE_t _index(self, DTYPE_t item) nogil:
        """
        Find index of item, throw exception if not found.
        """
        cdef SIZE_t i = 0

        if self.size == 0:
            raise ValueError('List is empty!')

        for i in range(self.size):
            if self.arr[i] == item:
                return i

        raise IndexError('Item not in list!')

    # private
    cdef void _increase_capacity(self) nogil:
        """
        Double array size.
        """
        self.capacity *= 2
        self.arr = <DTYPE_t *>realloc(self.arr, self.capacity * sizeof(DTYPE_t))
