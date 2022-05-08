"""
MinHeap, MaxHeap, and MinMaxHeap for streaming median.
"""
import sys
from abc import ABC, abstractmethod

class BaseHeap(object):
    """
    Base Heap class, do not use on its own!
    """

    # public
    def __init__(self, input_list=[]):
        self.heap = input_list.copy()
        if self.heap:
            self._heapify()

    def __str__(self):
        """
        Function to print the contents of the heap.
        """
        return self._str_helper(0, '', False)

    def insert(self, item):
        """
        Add item to the heap, then ensure heap invariance.
        """
        self.heap.append(item)
        self._siftdown(0, len(self.heap) - 1)

    def pop(self):
        """
        Pop smallest item from the heap, then ensure heap invariance.
        """
        last_item = self.heap.pop()  # raises IndexError if heap is empty
        if self.heap:
            return_item = self.heap[0]
            self.heap[0] = last_item
            self._siftup(0)  # move to leaf, then to its right place
            return return_item
        return last_item

    @abstractmethod
    def insertpop(self, item):
        """
        Fast version of insert followed by pop.
        """
        pass

    def remove(self, item):
        """
        Remove first found occurrence of item in heap.
        """
        pos = self.heap.index(item)  # raises ValuesError if item not in heap
        self.heap[pos] = self.heap[-1]
        self.heap.pop()  # removes last element
        if pos < len(self.heap):
            self._siftup(pos)
            self._siftdown(0, pos)

    @property
    def root(self):
        """
        Return root value.
        """
        if len(self.heap) == 0:
            raise IndexError('Empty heap!')
        return self.heap[0]

    @property
    def size(self):
        """
        Return number of nodes in heap.
        """
        return len(self.heap)

    # private
    def _parent_pos(self, pos):
        """
        Return parent pos given pos.
        """
        return (pos - 1) >> 1

    def _left_child_pos(self, pos):
        """
        Return left child pos given pos.
        """
        return (2 * pos) + 1

    def _right_child_pos(self, pos):
        """
        Return right child pos given pos.
        """
        return (2 * pos) + 2

    def _swap(self, i, j):
        """
        Swap elements at i and j pos.
        """
        self.heap[i], self.heap[j] = (self.heap[j], self.heap[i])

    # private
    def _heapify(self):
        """
        Ensure heap invariance.
        """
        n = len(self.heap)
        for pos in reversed(range(n // 2)):
            self._siftup(pos)

    def _str_helper(self, pos, indent, last):
        """
        Recursively print the tree using pre-prder traversal.
        """
        out_str = ''
        if pos < len(self.heap):
            if pos == 0:
                out_str += f'{self.heap[pos]}'
            else:
                out_str += f'\n{indent}'
                if last:
                    out_str += f'R----{self.heap[pos]}'
                    indent += "     "
                else:
                    out_str += f'L----{self.heap[pos]}'
                    indent += "|    "
            out_str += self._str_helper(self._left_child_pos(pos), indent, False)
            out_str += self._str_helper(self._right_child_pos(pos), indent, True)
        return out_str


class MinHeap(BaseHeap):
    """
    Minimum heap, parent node values should always be greater
    than child node values.
    """

    # public
    def insertpop(self, item):
        """
        Fast version of insert followed by pop.
        """
        if self.root < item:
            item, self.heap[0] = self.heap[0], item
            self._siftup(0)
        return item

    # private
    def _siftdown(self, start_pos, pos):
        """
        Move node at pos up, moving parents down
        until start_pos.
        """
        new_item = self.heap[pos]

        # bubble new_item up
        while pos > start_pos:
            parent_pos = self._parent_pos(pos)
            parent = self.heap[parent_pos]

            # move parent down
            if new_item < parent:
                self.heap[pos] = parent
                pos = parent_pos
                continue

            break

        self.heap[pos] = new_item

    def _siftup(self, pos):
        """
        Move node at pos down to a leaf, moving child nodes up.
        """
        start_pos = pos
        end_pos = len(self.heap)
        new_item = self.heap[pos]
        child_pos = self._left_child_pos(pos)

        # move new_item down
        while child_pos < end_pos:
            right_pos = child_pos + 1
            if right_pos < end_pos and not self.heap[child_pos] < self.heap[right_pos]:
                child_pos = right_pos
            self.heap[pos] = self.heap[child_pos]
            pos = child_pos
            child_pos = self._left_child_pos(pos)

        self.heap[pos] = new_item
        self._siftdown(start_pos, pos)


class MaxHeap(BaseHeap):
    """
    Minimum heap, parent node values should always be greater
    than child node values.
    """

    def insertpop(self, item):
        """
        Fast version of insert followed by pop.
        """
        if item < self.root:
            item, self.heap[0] = self.heap[0], item
            self._siftup(0)
        return item

    # private
    def _siftdown(self, start_pos, pos):
        """
        Move node at pos up, moving parents down
        until start_pos.
        """
        new_item = self.heap[pos]

        # bubble new_item up
        while pos > start_pos:
            parent_pos = self._parent_pos(pos)
            parent = self.heap[parent_pos]

            # move parent down
            if new_item > parent:
                self.heap[pos] = parent
                pos = parent_pos
                continue

            break

        self.heap[pos] = new_item

    def _siftup(self, pos):
        """
        Move node at pos down to a leaf, moving child nodes up.
        """
        start_pos = pos
        end_pos = len(self.heap)
        new_item = self.heap[pos]
        child_pos = self._left_child_pos(pos)

        # move new_item down
        while child_pos < end_pos:
            right_pos = child_pos + 1
            if right_pos < end_pos and not self.heap[child_pos] > self.heap[right_pos]:
                child_pos = right_pos
            self.heap[pos] = self.heap[child_pos]
            pos = child_pos
            child_pos = self._left_child_pos(pos)

        self.heap[pos] = new_item
        self._siftdown(start_pos, pos)


class MinMaxHeap(object):
    """
    Streaming median data structure consisting of
    a MinHeap and a MaxHeap.
    """

    # public
    def __init__(self, input_list=[]):
        self.min_heap = MinHeap()
        self.max_heap = MaxHeap()
        self.size = 0

        if input_list:
            for x in input_list:
                self.insert(x)

    def __str__(self):
        out_str = f'MinHeap:\n{self.min_heap.__str__()}'
        out_str += f'\nMaxHeap:\n{self.max_heap.__str__()}'
        return out_str

    def insert(self, x):
        """
        Insert x into either the min or max heap.
        """
        if self.size == 0:
            self.min_heap.insert(x)
            self.size += 1
            return

        if self.min_heap.size > self.max_heap.size:
            if x < self.min_heap.root:
                self.max_heap.insert(x)
            else:
                item = self.min_heap.insertpop(x)
                self.max_heap.insert(item)

        else:
            if x > self.min_heap.root:
                self.min_heap.insert(x)
            else:
                item = self.max_heap.insertpop(x)
                self.min_heap.insert(item)

        self.size += 1


    def remove(self, x):
        """
        Remove x from either the min or max heap.
        """
        if self.size == 1:
            if self.min_heap.size == 1:
                self.min_heap.remove(x)
            else:
                self.max_heap.remove(x)
            self.size -= 1
            return

        if x < self.median:
            self.max_heap.remove(x)  # raises IndexError if not in heap
            if self.min_heap.size - self.max_heap.size > 1:
                item = self.min_heap.pop()
                self.max_heap.insert(item)
        else:
            self.min_heap.remove(x)  # raises IndexError if not in heap
            if self.max_heap.size - self.min_heap.size > 1:
                item = self.max_heap.pop()
                self.min_heap.insert(item)

        self.size -= 1

    @property
    def median(self):
        """
        Return current median value.
        """
        if self.min_heap.size > self.max_heap.size:
            result = self.min_heap.root
        elif self.min_heap.size < self.max_heap.size:
            result = self.max_heap.root
        else:
            result = (self.min_heap.root + self.max_heap.root) / 2
        return result


# Driver Code
if __name__ == "__main__":

    data_list = [5, 84, 17, 10, 3, -19, 6, 22, 9, 0, 5]

    print(data_list)

    min_heap = MinHeap(input_list=data_list)
    max_heap = MaxHeap(input_list=data_list)

    print(f'The MinHeap is:\n{min_heap}')
    print(f'\nsize: {min_heap.size}, root value: {min_heap.root}')

    print(f'The MaxHeap is:\n{max_heap}')
    print(f'\nsize: {max_heap.size}, root value: {max_heap.root}')

    min_max_heap = MinMaxHeap(input_list=data_list)
    print(f'MinMaxHeap is:\n{min_max_heap}')
