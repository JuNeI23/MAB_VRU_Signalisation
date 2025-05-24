"""
Spatial indexing utility for efficient nearest neighbor searches.
"""
from typing import List, Tuple, Optional, TypeVar, Generic
import math

T = TypeVar('T')

class SpatialIndex(Generic[T]):
    """Simple spatial index for efficient nearest neighbor search."""
    
    def __init__(self):
        self.nodes = []
        
    def add_node(self, node: T, x: float, y: float):
        """Add a node to the spatial index."""
        self.nodes.append((node, x, y))
        
    def clear(self):
        """Clear all nodes from the index."""
        self.nodes = []
        
    def nearest_neighbor(self, x: float, y: float, exclude: Optional[T] = None) -> Tuple[Optional[T], float]:
        """Find the nearest neighbor to the given coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            exclude: Optional node to exclude from the search
            
        Returns:
            Tuple of (nearest node, distance) or (None, inf) if no nodes found
        """
        nearest = None
        min_distance = float('inf')
        
        for node, node_x, node_y in self.nodes:
            if node == exclude:
                continue
                
            distance = math.sqrt((x - node_x)**2 + (y - node_y)**2)
            if distance < min_distance:
                min_distance = distance
                nearest = node
                
        return nearest, min_distance
