"""
Mental Map System (TIER 3: Environmental Intelligence)
=======================================================

Creatures build internal spatial maps of their environment.

Key concepts:
1. Spatial Memory: Remember locations of resources and dangers
2. Personal Experience: Each creature's own discoveries
3. Inheritance: Pass spatial knowledge to offspring
4. Decay: Old memories fade if not reinforced

This enables creatures to navigate efficiently to known
resources instead of random wandering.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import deque


@dataclass
class MemoryLocation:
    """A remembered location in the world."""
    x: float
    y: float
    location_type: str  # 'food', 'water', 'danger', 'shelter', 'other'
    value: float  # How valuable/dangerous (positive=good, negative=bad)
    confidence: float  # How certain (0-1), decays over time
    last_visited: float  # Timestamp of last visit
    visit_count: int  # Number of times visited
    
    def decay(self, dt: float, decay_rate: float = 0.001):
        """Reduce confidence over time if not reinforced."""
        self.confidence = max(0.0, self.confidence - decay_rate * dt)
    
    def reinforce(self, new_value: Optional[float] = None):
        """Strengthen memory when location is visited again."""
        self.visit_count += 1
        self.confidence = min(1.0, self.confidence + 0.2)
        if new_value is not None:
            # Blend old and new value estimates
            self.value = 0.7 * self.value + 0.3 * new_value


class MentalMap:
    """
    A creature's internal spatial representation of the world.
    
    Stores remembered locations of resources and dangers.
    Can be partially inherited by offspring.
    """
    
    # Memory parameters
    MAX_LOCATIONS = 50  # Maximum stored locations
    MERGE_DISTANCE = 60.0  # Distance within which locations are merged
    MIN_CONFIDENCE = 0.1  # Forget locations below this confidence
    
    def __init__(self, creature_id: int):
        self.creature_id = creature_id
        
        # Stored locations by type for fast lookup
        self.food_locations: List[MemoryLocation] = []
        self.water_locations: List[MemoryLocation] = []
        self.danger_locations: List[MemoryLocation] = []
        self.other_locations: List[MemoryLocation] = []
        
        # Current position (updated each frame)
        self.current_x: float = 0.0
        self.current_y: float = 0.0
        
        # Navigation state
        self.target_location: Optional[MemoryLocation] = None
        self.exploration_bias: float = 0.3  # Tendency to explore vs exploit
        
        # Statistics
        self.total_discoveries = 0
        self.successful_returns = 0
    
    def update_position(self, x: float, y: float):
        """Update current position."""
        self.current_x = x
        self.current_y = y
    
    def remember_food(self, x: float, y: float, quality: float = 1.0, timestamp: float = 0):
        """Remember a food location."""
        self._add_location(self.food_locations, x, y, 'food', quality, timestamp)
    
    def remember_water(self, x: float, y: float, quality: float = 1.0, timestamp: float = 0):
        """Remember a water location."""
        self._add_location(self.water_locations, x, y, 'water', quality, timestamp)
    
    def remember_danger(self, x: float, y: float, severity: float = 1.0, timestamp: float = 0):
        """Remember a dangerous location."""
        # Negative value for dangers
        self._add_location(self.danger_locations, x, y, 'danger', -severity, timestamp)
    
    def _add_location(
        self, 
        location_list: List[MemoryLocation], 
        x: float, y: float, 
        loc_type: str, 
        value: float, 
        timestamp: float
    ):
        """Add or update a location in memory."""
        # Check for existing nearby location
        for loc in location_list:
            dist = np.sqrt((loc.x - x)**2 + (loc.y - y)**2)
            if dist < self.MERGE_DISTANCE:
                # Reinforce existing memory
                loc.reinforce(value)
                loc.last_visited = timestamp
                return
        
        # Create new memory
        new_loc = MemoryLocation(
            x=x, y=y,
            location_type=loc_type,
            value=value,
            confidence=0.8,
            last_visited=timestamp,
            visit_count=1
        )
        location_list.append(new_loc)
        self.total_discoveries += 1
        
        # Prune if too many
        if len(location_list) > self.MAX_LOCATIONS // 4:
            # Remove lowest confidence
            location_list.sort(key=lambda l: l.confidence, reverse=True)
            del location_list[self.MAX_LOCATIONS // 4:]
    
    def decay_memories(self, dt: float):
        """Apply memory decay to all locations."""
        for loc_list in [self.food_locations, self.water_locations, 
                         self.danger_locations, self.other_locations]:
            # Decay all
            for loc in loc_list:
                loc.decay(dt)
            # Remove forgotten memories
            loc_list[:] = [l for l in loc_list if l.confidence >= self.MIN_CONFIDENCE]
    
    def get_nearest_food(self, max_dist: float = 1000) -> Optional[Tuple[float, float, float]]:
        """Get nearest remembered food location."""
        return self._get_nearest(self.food_locations, max_dist, positive_only=True)
    
    def get_nearest_water(self, max_dist: float = 1000) -> Optional[Tuple[float, float, float]]:
        """Get nearest remembered water location."""
        return self._get_nearest(self.water_locations, max_dist, positive_only=True)
    
    def get_nearest_danger(self, max_dist: float = 200) -> Optional[Tuple[float, float, float]]:
        """Get nearest remembered danger location."""
        return self._get_nearest(self.danger_locations, max_dist, positive_only=False)
    
    def _get_nearest(
        self, 
        location_list: List[MemoryLocation], 
        max_dist: float,
        positive_only: bool = True
    ) -> Optional[Tuple[float, float, float]]:
        """Get nearest location from list, weighted by value and confidence."""
        if not location_list:
            return None
        
        best = None
        best_score = -np.inf
        
        for loc in location_list:
            if positive_only and loc.value < 0:
                continue
            
            dist = np.sqrt((loc.x - self.current_x)**2 + (loc.y - self.current_y)**2)
            if dist > max_dist:
                continue
            
            # Score = value * confidence / (1 + distance/100)
            # High value, high confidence, close = high score
            score = loc.value * loc.confidence / (1 + dist / 100)
            
            if score > best_score:
                best_score = score
                best = loc
        
        if best is None:
            return None
        
        return (best.x, best.y, best.value)
    
    def is_known_danger(self, x: float, y: float, radius: float = 80) -> bool:
        """Check if location is remembered as dangerous."""
        for loc in self.danger_locations:
            dist = np.sqrt((loc.x - x)**2 + (loc.y - y)**2)
            if dist < radius and loc.confidence > 0.3:
                return True
        return False
    
    def get_exploration_direction(self) -> Optional[Tuple[float, float]]:
        """
        Get direction to explore (away from known areas).
        
        Returns a direction vector pointing to unexplored territory.
        """
        if not (self.food_locations or self.water_locations):
            # No knowledge yet - explore randomly
            angle = np.random.random() * 2 * np.pi
            return (np.cos(angle) * 100, np.sin(angle) * 100)
        
        # Combine all known locations
        all_locs = self.food_locations + self.water_locations + self.danger_locations
        
        if not all_locs:
            return None
        
        # Calculate centroid of known locations
        cx = np.mean([l.x for l in all_locs])
        cy = np.mean([l.y for l in all_locs])
        
        # Explore away from centroid (toward unknown areas)
        dx = self.current_x - cx
        dy = self.current_y - cy
        
        # Normalize and extend
        mag = max(1.0, float(np.sqrt(dx**2 + dy**2)))
        return (float(self.current_x + dx/mag * 200), float(self.current_y + dy/mag * 200))
    
    def get_inheritable_map(self, strength: float = 0.5) -> 'MentalMap':
        """
        Create a copy of this map for offspring inheritance.
        
        Args:
            strength: How much of the parent's knowledge to transfer (0-1)
            
        Returns:
            New MentalMap with inherited locations at reduced confidence
        """
        child_map = MentalMap(creature_id=-1)  # ID will be set later
        
        # Copy high-confidence locations with reduced confidence
        for loc in self.food_locations:
            if loc.confidence > 0.5 and np.random.random() < strength:
                child_loc = MemoryLocation(
                    x=loc.x, y=loc.y,
                    location_type=loc.location_type,
                    value=loc.value,
                    confidence=loc.confidence * strength * 0.6,  # Reduced
                    last_visited=0,
                    visit_count=0  # Not personally visited
                )
                child_map.food_locations.append(child_loc)
        
        for loc in self.water_locations:
            if loc.confidence > 0.5 and np.random.random() < strength:
                child_loc = MemoryLocation(
                    x=loc.x, y=loc.y,
                    location_type=loc.location_type,
                    value=loc.value,
                    confidence=loc.confidence * strength * 0.6,
                    last_visited=0,
                    visit_count=0
                )
                child_map.water_locations.append(child_loc)
        
        for loc in self.danger_locations:
            if loc.confidence > 0.6 and np.random.random() < strength * 1.2:
                # Danger knowledge is more important to pass on
                child_loc = MemoryLocation(
                    x=loc.x, y=loc.y,
                    location_type=loc.location_type,
                    value=loc.value,
                    confidence=loc.confidence * strength * 0.8,
                    last_visited=0,
                    visit_count=0
                )
                child_map.danger_locations.append(child_loc)
        
        return child_map
    
    def merge_from(self, other: 'MentalMap', trust_level: float = 0.5):
        """
        Merge knowledge from another creature's map.
        
        Used for:
        - Inheritance (from parent)
        - Social learning (from observed successful creature)
        
        Args:
            other: Another creature's mental map
            trust_level: How much to trust the other's knowledge (0-1)
        """
        for loc in other.food_locations:
            self._add_location(
                self.food_locations,
                loc.x, loc.y, 'food',
                loc.value * trust_level,
                0
            )
            # Reduce confidence for non-personal knowledge
            if self.food_locations:
                self.food_locations[-1].confidence *= trust_level
        
        for loc in other.water_locations:
            self._add_location(
                self.water_locations,
                loc.x, loc.y, 'water',
                loc.value * trust_level,
                0
            )
            if self.water_locations:
                self.water_locations[-1].confidence *= trust_level
        
        for loc in other.danger_locations:
            self._add_location(
                self.danger_locations,
                loc.x, loc.y, 'danger',
                loc.value,  # Don't reduce danger severity
                0
            )
            if self.danger_locations:
                self.danger_locations[-1].confidence *= trust_level * 1.2
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mental map statistics."""
        return {
            'food_locations': len(self.food_locations),
            'water_locations': len(self.water_locations),
            'danger_locations': len(self.danger_locations),
            'total_discoveries': self.total_discoveries,
            'successful_returns': self.successful_returns,
            'avg_food_confidence': np.mean([l.confidence for l in self.food_locations]) if self.food_locations else 0,
            'avg_water_confidence': np.mean([l.confidence for l in self.water_locations]) if self.water_locations else 0,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for saving."""
        def loc_to_dict(loc: MemoryLocation) -> Dict:
            return {
                'x': loc.x, 'y': loc.y,
                'type': loc.location_type,
                'value': loc.value,
                'confidence': loc.confidence,
                'last_visited': loc.last_visited,
                'visit_count': loc.visit_count
            }
        
        return {
            'creature_id': self.creature_id,
            'food': [loc_to_dict(l) for l in self.food_locations],
            'water': [loc_to_dict(l) for l in self.water_locations],
            'danger': [loc_to_dict(l) for l in self.danger_locations],
            'stats': {
                'total_discoveries': self.total_discoveries,
                'successful_returns': self.successful_returns
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MentalMap':
        """Deserialize from saved data."""
        mm = cls(data.get('creature_id', -1))
        
        def dict_to_loc(d: Dict) -> MemoryLocation:
            return MemoryLocation(
                x=d['x'], y=d['y'],
                location_type=d['type'],
                value=d['value'],
                confidence=d['confidence'],
                last_visited=d['last_visited'],
                visit_count=d['visit_count']
            )
        
        mm.food_locations = [dict_to_loc(d) for d in data.get('food', [])]
        mm.water_locations = [dict_to_loc(d) for d in data.get('water', [])]
        mm.danger_locations = [dict_to_loc(d) for d in data.get('danger', [])]
        
        stats = data.get('stats', {})
        mm.total_discoveries = stats.get('total_discoveries', 0)
        mm.successful_returns = stats.get('successful_returns', 0)
        
        return mm
