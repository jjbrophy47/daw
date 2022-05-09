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
# Streaming median
# =======================


cdef class _MinMaxHeap:
    """
    Streaming median data structure consisting of
    a _MinHeap and a _MaxHeap.
    """

    # Public Python API
    property median:
        def __get__(self):
            return self._median()

    property mean:
        def __get__(self):
            return self._mean()

    property size:
        def __get__(self):
            return self.size

    def __cinit__(self, float[:] input_vals):
        """
        Constructor.
        """
        self.min_heap = _MinHeap()
        self.max_heap = _MaxHeap()
        self.size = 0
        self.total = 0

        cdef SIZE_t n_samples = input_vals.shape[0]
        cdef SIZE_t i = 0

        for i in range(n_samples):
            self._insert(input_vals[i])

    def __str__(self):
        out_str = f'MinHeap:\n{self.min_heap.__str__()}'
        out_str += f'\nMaxHeap:\n{self.max_heap.__str__()}'
        return out_str

    cpdef void insert(self, float item):
        self._insert(<DTYPE_t>item)

    cpdef void remove(self, float item):
        self._remove(<DTYPE_t>item)

    # Public C API
    cdef void _insert(self, DTYPE_t x) nogil:
        """
        Insert x into either the min or max heap.
        """
        cdef DTYPE_t item

        if self.size == 0:
            self.min_heap._insert(x)
            self.size += 1
            self.total += x
            return

        if self.min_heap._size() > self.max_heap._size():
            if x < self.min_heap._root():
                self.max_heap._insert(x)
            else:
                item = self.min_heap._insertpop(x)
                self.max_heap._insert(item)

        else:
            if x > self.min_heap._root():
                self.min_heap._insert(x)
            else:
                item = self.max_heap._insertpop(x)
                self.min_heap._insert(item)

        self.size += 1
        self.total += x

    cdef void _remove(self, DTYPE_t x) nogil:
        """
        Remove x from either the min or max heap.
        """
        cdef DTYPE_t item

        if self.size == 1:
            if self.min_heap._size() == 1:
                self.min_heap._remove(x)
            else:
                self.max_heap._remove(x)
            self.size -= 1
            self.total -= x
            return

        if x < self._median():
            self.max_heap._remove(x)  # raises IndexError if not in heap
            if self.min_heap._size() - self.max_heap._size() > 1:
                item = self.min_heap._pop()
                self.max_heap._insert(item)
        else:
            self.min_heap._remove(x)  # raises IndexError if not in heap
            if self.max_heap._size() - self.min_heap._size() > 1:
                item = self.max_heap._pop()
                self.min_heap._insert(item)

        self.size -= 1
        self.total -= x

    cdef DTYPE_t _median(self) nogil:
        """
        Return median.
        """
        cdef DTYPE_t result

        if self.min_heap._size() > self.max_heap._size():
            result = self.min_heap._root()
        elif self.min_heap._size() < self.max_heap._size():
            result = self.max_heap._root()
        else:
            result = (self.min_heap._root() + self.max_heap._root()) / 2
        return result

    cdef DTYPE_t _mean(self) nogil:
        """
        Return mean.
        """
        return self.total / self.size


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
            return self._size()

    property root:
        def __get__(self):
            return self._root()

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

    cdef SIZE_t _size(self) nogil:
        """
        Get heap size.
        """
        return self.heap.size

    cdef DTYPE_t _root(self) nogil:
        """
        Get root value.
        """
        if self.heap.size == 0:
            raise IndexError('Empty heap!')
        return self.heap.arr[0]

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

    def __dealloc__(self):
        """
        Destructor.
        """
        free(self.arr)

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
