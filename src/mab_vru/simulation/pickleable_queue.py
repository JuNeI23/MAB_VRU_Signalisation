"""
A pickleable queue implementation for use with multiprocessing.
"""
from typing import List, TypeVar, Generic
import heapq

T = TypeVar('T')

class PickleableQueue(Generic[T]):
    """A pickleable priority queue implementation."""
    
    def __init__(self):
        """Initialize an empty queue."""
        self._queue: List[T] = []
        self._index = 0
    
    def put(self, item: T) -> None:
        """Add an item to the queue."""
        heapq.heappush(self._queue, item)
    
    def get(self) -> T:
        """Remove and return the highest priority item."""
        if not self._queue:
            raise IndexError("Queue is empty")
        return heapq.heappop(self._queue)
    
    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return len(self._queue) == 0
    
    def __getstate__(self):
        """Get the state for pickling."""
        return {'_queue': self._queue, '_index': self._index}
    
    def __setstate__(self, state):
        """Restore the state after unpickling."""
        self._queue = state['_queue']
        self._index = state['_index']
