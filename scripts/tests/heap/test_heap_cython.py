"""
Tests for the heap module.
"""
import os
import sys
from statistics import median

import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')
from daw.regression._heap import _MinHeap, _MaxHeap


def get_data():
    """
    Return list of test data.
    """
    return np.array([5, 84, 17, 10, -19, 3, 6, 22, -19, 9, 0, 5,], dtype=np.float32)


def get_empty_data():
    """
    Return list of test data.
    """
    return np.array([], dtype=np.float32)


def test_min_heap_insert():
    data_list = get_data()

    heap = _MinHeap()
    temp_list = []

    for i, x in enumerate(data_list):
        heap.insert(x)
        temp_list.append(x)
        assert heap.size == i + 1
        assert heap.root == min(temp_list)

    print('test_min_heap_insert: passed')


def test_min_heap_pop():
    data_list = get_data()

    heap = _MinHeap()
    for x in data_list:  # insert values
        heap.insert(x)
    temp_list = list(data_list)

    for i in range(len(data_list) - 1):
        item = heap.pop()
        temp_list.remove(item)
        assert heap.size == len(temp_list)
        assert heap.root == min(temp_list)

    print('test_min_heap_pop: passed')


def test_min_heap_insertpop():
    data_list = get_data()

    heap = _MinHeap()
    for x in data_list:  # insert values
        heap.insert(x)
    temp_list = list(data_list)

    for i, item in enumerate(data_list):
        temp_list.append(item)
        min_item = min(temp_list)
        temp_list.remove(min_item)
        popped_item = heap.insertpop(item)
        assert heap.size == len(temp_list)
        assert popped_item == min_item
        assert heap.root == min(temp_list)

    print('test_min_heap_insertpop: passed')


def test_min_heap_remove():
    data_list = get_data()

    heap = _MinHeap()
    for x in data_list:  # insert values
        heap.insert(x)
    temp_list = list(data_list)

    for i in range(len(data_list) - 1):
        item = data_list[i]
        heap.remove(item)
        temp_list.remove(item)
        assert heap.size == len(temp_list)
        assert heap.root == min(temp_list)

    print('test_min_heap_remove: passed')


def test_max_heap_insert():
    data_list = get_data()

    heap = _MaxHeap()
    temp_list = []

    for i, x in enumerate(data_list):
        heap.insert(x)
        temp_list.append(x)
        assert heap.size == i + 1
        assert heap.root == max(temp_list)

    print('test_max_heap_insert: passed')


def test_max_heap_pop():
    data_list = get_data()

    heap = _MaxHeap()
    for x in data_list:  # insert values
        heap.insert(x)
    temp_list = list(data_list)

    for i in range(len(data_list) - 1):
        item = heap.pop()
        temp_list.remove(item)
        assert heap.size == len(temp_list)
        assert heap.root == max(temp_list)

    print('test_max_heap_pop: passed')


def test_max_heap_insertpop():
    data_list = get_data()

    heap = _MaxHeap()
    for x in data_list:  # insert values
        heap.insert(x)
    temp_list = list(data_list)

    for i, item in enumerate(data_list):
        temp_list.append(item)
        max_item = max(temp_list)
        temp_list.remove(max_item)
        popped_item = heap.insertpop(item)
        assert heap.size == len(temp_list)
        assert popped_item == max_item
        assert heap.root == max(temp_list)

    print('test_max_heap_insertpop: passed')


def test_max_heap_remove():
    data_list = get_data()

    heap = _MaxHeap()
    for x in data_list:  # insert values
        heap.insert(x)
    temp_list = list(data_list)

    for i in range(len(data_list) - 1):
        item = data_list[i]
        heap.remove(item)
        temp_list.remove(item)
        assert heap.size == len(temp_list)
        assert heap.root == max(temp_list)

    print('test_max_heap_remove: passed')


def test_min_max_heap_initialize():
    data_list = get_data()

    heap = MinMaxHeap(input_list=data_list)
    assert heap.median == median(data_list)

    print('test_min_max_heap_initialize: passed')


def test_min_max_heap_insert():
    data_list = get_data()

    heap = MinMaxHeap(input_list=[])
    temp_list = []

    for i, item in enumerate(data_list):
        heap.insert(item)
        temp_list.append(item)
        assert heap.size == len(temp_list)
        assert heap.median == median(temp_list)

    print('test_max_heap_insert: passed')


def test_min_max_heap_remove():
    data_list = get_data()

    heap = MinMaxHeap(input_list=data_list)
    temp_list = data_list.copy()

    for i in range(len(data_list) - 1):
        item = data_list[i]
        heap.remove(item)
        temp_list.remove(item)
        assert heap.size == len(temp_list)
        assert heap.median == median(temp_list)

    print('test_min_max_heap_remove: passed')


def test_min_max_heap_insert_remove():
    data_list = get_data()

    heap = MinMaxHeap(input_list=data_list)
    temp_list = data_list.copy()

    for i in range(len(data_list)):
        remove_item = data_list[i]
        insert_item = data_list[-i]
        heap.remove(remove_item)
        heap.insert(insert_item)
        temp_list.remove(remove_item)
        temp_list.append(insert_item)
        assert heap.size == len(temp_list)
        assert heap.median == median(temp_list)

    print('test_min_max_heap_insert_remove: passed')


# Driver Code
if __name__ == "__main__":
    test_min_heap_insert()
    test_min_heap_pop()
    test_min_heap_insertpop()
    test_min_heap_remove()
    test_max_heap_insert()
    test_max_heap_pop()
    test_max_heap_insertpop()
    test_max_heap_remove()
    # test_min_max_heap_initialize()
    # test_min_max_heap_insert()
    # test_min_max_heap_remove()
    # test_min_max_heap_insert_remove()
