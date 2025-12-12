"""
Social Learning System (TIER 3: Cultural Evolution)
====================================================

Creatures learn from observing others' successful behaviors.

Key concepts:
1. Observation: Track what nearby creatures are doing
2. Success Detection: Notice when others eat, drink, or avoid pain
3. Imitation: Copy successful movement patterns and goals

This enables cultural transmission of learned behaviors across
individuals within a generation (not just genetic inheritance).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import deque


@dataclass
class ObservedBehavior:
    """Record of an observed behavior from another creature."""
    observer_id: int
    observed_id: int
    behavior_type: str  # 'eating', 'drinking', 'fleeing', 'approaching'
    target_position: Tuple[float, float]  # Where the observed creature was going
    observer_position: Tuple[float, float]  # Where observer was
    outcome: str  # 'success', 'failure', 'unknown'
    reward_magnitude: float  # How successful (0-1)
    timestamp: float
    cortical_pattern: Optional[np.ndarray] = None  # Brain state during observation


class SocialLearningSystem:
    """
    Manages social learning for a creature.
    
    Watches nearby creatures and learns from their successes/failures.
    """
    
    # Observation parameters
    OBSERVATION_RANGE = 200.0  # How far can creature see others
    ATTENTION_CAPACITY = 3     # Max creatures to track at once
    MEMORY_SIZE = 50           # How many observations to remember
    
    # Learning parameters
    IMITATION_THRESHOLD = 0.6  # How successful must behavior be to imitate
    SOCIAL_LEARNING_RATE = 0.3 # How much to weight observed vs personal experience
    
    @property
    def observation_range(self) -> float:
        """Instance access to observation range."""
        return self.OBSERVATION_RANGE
    
    def __init__(self, creature_id: int):
        self.creature_id = creature_id
        
        # Currently watched creatures
        self.watched_creatures: Dict[int, Dict[str, Any]] = {}
        
        # Recent observations (for learning)
        self.observations: deque = deque(maxlen=self.MEMORY_SIZE)
        
        # Successful behaviors seen (for imitation)
        self.successful_behaviors: List[ObservedBehavior] = []
        
        # Statistics
        self.total_observations = 0
        self.successful_imitations = 0
        
        # Current timestamp for observations
        self._current_time = 0.0
    
    def update_watched_creatures(
        self,
        observer_x: Optional[float] = None,
        observer_y: Optional[float] = None,
        visible_creatures: Optional[List[Dict[str, Any]]] = None,
        # Legacy signature support
        own_position: Optional[Tuple[float, float]] = None,
        other_creatures: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Update the set of creatures being watched.
        
        Args:
            observer_x, observer_y: Position of this creature (new API)
            visible_creatures: List of dicts with creature data (new API)
            own_position: (x, y) tuple (legacy API)
            other_creatures: List of dicts (legacy API)
        """
        # Handle both old and new API
        if own_position is not None:
            ox, oy = own_position
        else:
            ox, oy = observer_x or 0, observer_y or 0
            
        creatures = visible_creatures or other_creatures or []
        
        # Find creatures in observation range
        in_range = []
        for c in creatures:
            if c.get('id') == self.creature_id:
                continue  # Don't watch self
            
            cx, cy = c.get('x', 0), c.get('y', 0)
            dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
            
            if dist <= self.OBSERVATION_RANGE:
                in_range.append((dist, c))
        
        # Sort by distance, keep closest
        in_range.sort(key=lambda x: x[0])
        in_range = in_range[:self.ATTENTION_CAPACITY]
        
        # Update watched set
        new_watched = {}
        for dist, c in in_range:
            cid = c.get('id', id(c))
            
            # Carry over previous state if exists
            prev = self.watched_creatures.get(cid, {})
            
            new_watched[cid] = {
                'id': cid,
                'x': c.get('x', 0),
                'y': c.get('y', 0),
                'prev_x': prev.get('x', c.get('x', 0)),
                'prev_y': prev.get('y', c.get('y', 0)),
                'behavior': c.get('behavior', 'idle'),
                'prev_behavior': prev.get('behavior', 'idle'),
                'health': c.get('health', 1.0),
                'prev_health': prev.get('health', 1.0),
                'food_eaten': c.get('food_eaten', 0),
                'prev_food_eaten': prev.get('food_eaten', 0),
                'water_consumed': c.get('water_consumed', 0),
                'prev_water_consumed': prev.get('water_consumed', 0),
                'distance': dist,
            }
        
        self.watched_creatures = new_watched
    
    def detect_success_events(
        self,
        visible_creatures: Optional[List[Dict[str, Any]]] = None,
        cultural_knowledge: Optional['CulturalKnowledge'] = None,
        # Legacy signature support
        own_position: Optional[Tuple[float, float]] = None,
        timestamp: Optional[float] = None
    ) -> List[ObservedBehavior]:
        """
        Check watched creatures for success events.
        
        Args:
            visible_creatures: List of creature dicts (new API, updates watched first)
            cultural_knowledge: CulturalKnowledge to update (new API)
            own_position: (x, y) tuple (legacy API)
            timestamp: Current time (legacy API)
        
        Returns list of successful behaviors observed.
        """
        # Use internal time if not provided
        if timestamp is None:
            self._current_time += 0.033  # ~30 FPS
            timestamp = self._current_time
        
        # Default observer position
        observer_pos = own_position or (0.0, 0.0)
        
        successes = []
        
        for cid, info in self.watched_creatures.items():
            # Eating success: food_eaten increased
            if info.get('food_eaten', 0) > info.get('prev_food_eaten', 0):
                obs = ObservedBehavior(
                    observer_id=self.creature_id,
                    observed_id=cid,
                    behavior_type='eating',
                    target_position=(info['x'], info['y']),
                    observer_position=observer_pos,
                    outcome='success',
                    reward_magnitude=0.8,
                    timestamp=timestamp
                )
                successes.append(obs)
                self.observations.append(obs)
                self.successful_behaviors.append(obs)
                self.total_observations += 1
                
                # Update cultural knowledge if provided
                if cultural_knowledge is not None:
                    cultural_knowledge.add_food_location(info['x'], info['y'], 0.8)
            
            # Drinking success: water_consumed increased
            if info.get('water_consumed', 0) > info.get('prev_water_consumed', 0):
                obs = ObservedBehavior(
                    observer_id=self.creature_id,
                    observed_id=cid,
                    behavior_type='drinking',
                    target_position=(info['x'], info['y']),
                    observer_position=observer_pos,
                    outcome='success',
                    reward_magnitude=0.6,
                    timestamp=timestamp
                )
                successes.append(obs)
                self.observations.append(obs)
                self.successful_behaviors.append(obs)
                self.total_observations += 1
                
                # Update cultural knowledge if provided
                if cultural_knowledge is not None:
                    cultural_knowledge.add_water_location(info['x'], info['y'])
            
            # Pain avoidance: health stopped dropping / improved
            if info.get('health', 1) > info.get('prev_health', 0) + 0.01:
                # They were hurt but now recovering
                if info.get('prev_behavior') == 'fleeing':
                    obs = ObservedBehavior(
                        observer_id=self.creature_id,
                        observed_id=cid,
                        behavior_type='fleeing',
                        target_position=(info['x'], info['y']),
                        observer_position=observer_pos,
                        outcome='success',
                        reward_magnitude=0.5,
                        timestamp=timestamp
                    )
                    successes.append(obs)
                    self.observations.append(obs)
                    self.successful_behaviors.append(obs)
                    self.total_observations += 1
        
        return successes
    
    def get_imitation_target(
        self,
        own_position: Optional[Tuple[float, float]] = None,
        own_need: Optional[str] = None  # 'hunger', 'thirst', 'safety'
    ) -> Optional[Dict[str, Any]]:
        """
        Get a target to imitate based on observed successes.
        
        If another creature recently succeeded at meeting this need,
        suggest going to where they were.
        
        Returns:
            Dict with 'x', 'y', 'behavior' or None if no relevant observations
        """
        # If no recent observations, nothing to imitate
        if not self.successful_behaviors:
            return None
        
        # Map needs to behavior types
        need_to_behavior = {
            'hunger': 'eating',
            'thirst': 'drinking',
            'safety': 'fleeing',
        }
        
        # Filter by need if specified
        if own_need:
            target_behavior = need_to_behavior.get(own_need, 'eating')
            relevant = [
                obs for obs in self.successful_behaviors[-20:]  # Recent only
                if obs.behavior_type == target_behavior
                and obs.reward_magnitude >= self.IMITATION_THRESHOLD
            ]
        else:
            # Get any successful behavior
            relevant = [
                obs for obs in self.successful_behaviors[-20:]
                if obs.reward_magnitude >= self.IMITATION_THRESHOLD
            ]
        
        if not relevant:
            return None
        
        # Pick the most recent one
        best = relevant[-1]
        
        # Return the target position where success happened
        return {
            'x': best.target_position[0],
            'y': best.target_position[1],
            'behavior': best.behavior_type,
            'reward': best.reward_magnitude
        }
    
    def should_imitate(self, own_confidence: Optional[float] = None) -> bool:
        """
        Decide whether to rely on imitation vs personal exploration.
        
        Low confidence in own decisions → more likely to imitate.
        If no confidence provided, uses default probability.
        """
        # If we've seen successful behaviors and we're uncertain, imitate
        if not self.successful_behaviors:
            return False
        
        if own_confidence is None:
            # Default: 30% chance to imitate if we have observations
            return np.random.random() < self.SOCIAL_LEARNING_RATE
        
        # Lower confidence = more likely to imitate
        imitation_probability = self.SOCIAL_LEARNING_RATE * (1.0 - own_confidence)
        return np.random.random() < imitation_probability
    
    def get_stats(self) -> Dict[str, Any]:
        """Get social learning statistics."""
        return {
            'total_observations': self.total_observations,
            'successful_imitations': self.successful_imitations,
            'watched_creatures': len(self.watched_creatures),
            'remembered_successes': len(self.successful_behaviors),
        }
    
    def clear_old_observations(self, current_time: float, max_age: float = 60.0):
        """Remove old observations (forgetting)."""
        self.successful_behaviors = [
            obs for obs in self.successful_behaviors
            if current_time - obs.timestamp < max_age
        ]


class CulturalKnowledge:
    """
    Shared knowledge that can be transmitted between creatures.
    
    This represents "cultural memory" - locations and patterns
    that have been discovered by any member of the group.
    """
    
    def __init__(self):
        # Known resource locations (food, water)
        self.food_locations: List[Tuple[float, float, float]] = []  # (x, y, quality)
        self.water_locations: List[Tuple[float, float, float]] = []
        
        # Known danger locations
        self.danger_locations: List[Tuple[float, float, str]] = []  # (x, y, type)
        
        # Successful movement patterns (sequences of positions)
        self.successful_routes: List[List[Tuple[float, float]]] = []
        
        # Update timestamps
        self.last_update = 0.0
    
    def add_food_location(self, x: float, y: float, quality: float = 1.0):
        """Record a food location discovered by any creature."""
        # Check for duplicates
        for fx, fy, fq in self.food_locations:
            if np.sqrt((fx - x)**2 + (fy - y)**2) < 50:
                return  # Already known
        
        self.food_locations.append((x, y, quality))
        
        # Keep only most recent/best locations
        if len(self.food_locations) > 20:
            self.food_locations.sort(key=lambda l: l[2], reverse=True)
            self.food_locations = self.food_locations[:20]
    
    def add_water_location(self, x: float, y: float, quality: float = 1.0):
        """Record a water location discovered by any creature."""
        for wx, wy, wq in self.water_locations:
            if np.sqrt((wx - x)**2 + (wy - y)**2) < 50:
                return
        
        self.water_locations.append((x, y, quality))
        
        if len(self.water_locations) > 20:
            self.water_locations.sort(key=lambda l: l[2], reverse=True)
            self.water_locations = self.water_locations[:20]
    
    def add_danger(self, x: float, y: float, danger_type: str = "hazard"):
        """Record a danger location discovered by any creature."""
        for dx, dy, dt in self.danger_locations:
            if np.sqrt((dx - x)**2 + (dy - y)**2) < 30:
                return
        
        self.danger_locations.append((x, y, danger_type))
        
        if len(self.danger_locations) > 30:
            self.danger_locations = self.danger_locations[-30:]
    
    def add_danger_location(self, x: float, y: float, intensity: float = 1.0, danger_type: str = "hazard"):
        """Alias for add_danger with intensity parameter for compatibility."""
        self.add_danger(x, y, f"{danger_type}_{intensity:.1f}")
    
    def get_nearest_food(self, x: float, y: float, max_dist: float = 500) -> Optional[Tuple[float, float]]:
        """Get nearest known food location."""
        if not self.food_locations:
            return None
        
        best = None
        best_dist = max_dist
        for fx, fy, fq in self.food_locations:
            d = np.sqrt((fx - x)**2 + (fy - y)**2)
            if d < best_dist:
                best_dist = d
                best = (fx, fy)
        
        return best
    
    def get_nearest_water(self, x: float, y: float, max_dist: float = 500) -> Optional[Tuple[float, float]]:
        """Get nearest known water location."""
        if not self.water_locations:
            return None
        
        best = None
        best_dist = max_dist
        for wx, wy, wq in self.water_locations:
            d = np.sqrt((wx - x)**2 + (wy - y)**2)
            if d < best_dist:
                best_dist = d
                best = (wx, wy)
        
        return best
    
    def is_dangerous(self, x: float, y: float, radius: float = 50) -> bool:
        """Check if location is known to be dangerous."""
        for dx, dy, dt in self.danger_locations:
            if np.sqrt((dx - x)**2 + (dy - y)**2) < radius:
                return True
        return False
