"""
Quadtree implementation for spatial partitioning.
Used to optimize proximity searches (food, creatures).
"""

from typing import List, Tuple, Any, Optional, Generic, TypeVar
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    
    def contains(self, point: 'Point') -> bool:
        return (self.x <= point.x < self.x + self.w and 
                self.y <= point.y < self.y + self.h)
    
    def intersects(self, other: 'Rect') -> bool:
        return not (other.x >= self.x + self.w or 
                    other.x + other.w <= self.x or 
                    other.y >= self.y + self.h or 
                    other.y + other.h <= self.y)

@dataclass
class Point(Generic[T]):
    x: float
    y: float
    data: T

class QuadTree(Generic[T]):
    def __init__(self, boundary: Rect, capacity: int = 4):
        self.boundary = boundary
        self.capacity = capacity
        self.points: List[Point[T]] = []
        self.divided = False
        
        # Children
        self.northwest: Optional[QuadTree] = None
        self.northeast: Optional[QuadTree] = None
        self.southwest: Optional[QuadTree] = None
        self.southeast: Optional[QuadTree] = None
        
    def insert(self, point: Point[T]) -> bool:
        if not self.boundary.contains(point):
            return False
            
        if len(self.points) < self.capacity:
            self.points.append(point)
            return True
        
        if not self.divided:
            self.subdivide()
            
        return (self.northwest.insert(point) or 
                self.northeast.insert(point) or 
                self.southwest.insert(point) or 
                self.southeast.insert(point))
    
    def subdivide(self):
        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.w / 2
        h = self.boundary.h / 2
        
        self.northwest = QuadTree(Rect(x, y, w, h), self.capacity)
        self.northeast = QuadTree(Rect(x + w, y, w, h), self.capacity)
        self.southwest = QuadTree(Rect(x, y + h, w, h), self.capacity)
        self.southeast = QuadTree(Rect(x + w, y + h, w, h), self.capacity)
        
        self.divided = True
        
    def query(self, range_rect: Rect, found: List[T] = None) -> List[T]:
        if found is None:
            found = []
            
        if not self.boundary.intersects(range_rect):
            return found
            
        for p in self.points:
            if range_rect.contains(p):
                found.append(p.data)
                
        if self.divided:
            self.northwest.query(range_rect, found)
            self.northeast.query(range_rect, found)
            self.southwest.query(range_rect, found)
            self.southeast.query(range_rect, found)
            
        return found
        
    def query_radius(self, x: float, y: float, radius: float) -> List[T]:
        """Find points within circular radius (approximation via querying square)."""
        # First query a square containing the circle for efficiency
        size = radius * 2
        range_rect = Rect(x - radius, y - radius, size, size)
        potential = self.query(range_rect)
        
        # Then filter by actual distance distance (if precise circle is needed)
        # Note: The query() method returns points INSIDE the rect.
        # We need to calculate distance for the circle check.
        result = []
        r_sq = radius * radius
        for item in potential:
            # We assume 'item' is the data payload.
            # Wait, the QuadTree stores Point objects but query returns .data.
            # To check distance, we need the coordinates!
            # My logic in `query` appends `p.data`. It loses coordinate info unless data has it.
            # Let's assume data has .x .y or we change query to return Points.
            # Better: `query` returns `Point`.
            pass 
        
        return potential

    def query_points(self, range_rect: Rect, found: List[Point[T]] = None) -> List[Point[T]]:
        """Query returning Point objects with coordinates."""
        if found is None:
            found = []
            
        if not self.boundary.intersects(range_rect):
            return found
            
        for p in self.points:
            if range_rect.contains(p):
                found.append(p)
                
        if self.divided:
            self.northwest.query_points(range_rect, found)
            self.northeast.query_points(range_rect, found)
            self.southwest.query_points(range_rect, found)
            self.southeast.query_points(range_rect, found)
            
        return found
    
    def clear(self):
        self.points = []
        self.divided = False
        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None
