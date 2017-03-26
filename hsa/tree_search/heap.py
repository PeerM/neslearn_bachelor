import heapq
from itertools import count


class Heap(object):
    def __init__(self, initial=None, key=lambda x: x):
        self.key = key
        self.tiebreaker = count()
        if initial:
            self._data = [(key(item), next(self.tiebreaker), item) for item in initial]
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, (self.key(item), next(self.tiebreaker), item))

    def pop(self):
        return heapq.heappop(self._data)[2]
