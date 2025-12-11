"""
Movement Controller - Game engine handles ALL movement

The brain/state machine only says WHAT to do:
- "go_to_food"
- "go_to_water" 
- "flee_from_danger"
- "wander"
- "stay"

This controller handles HOW:
- Pathfinding
- Smooth interpolation
- Jumping out of water
- Avoiding obstacles
- Animation

NO neural control of movement. Period.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from enum import Enum, auto


class MovementGoal(Enum):
    """High-level movement goals - brain picks these, not directions."""
    STAY = auto()           # Don't move
    WANDER = auto()         # Random exploration
    GO_TO_TARGET = auto()   # Move toward target_x, target_y
    FLEE_FROM = auto()      # Move away from threat_x, threat_y
    EXIT_WATER = auto()     # Get out of water (find nearest land)
    EXIT_HAZARD = auto()    # Get away from fire/danger


@dataclass
class MovementController:
    """
    Handles ALL movement logic. Brain only sets goals.
    
    This is how Creatures 1, Spore, and every working ALife does it.
    """
    
    # Current goal
    goal: MovementGoal = MovementGoal.STAY
    
    # Target position (for GO_TO_TARGET)
    target_x: float = 0.0
    target_y: float = 0.0
    
    # Threat position (for FLEE_FROM)
    threat_x: float = 0.0
    threat_y: float = 0.0
    
    # Known hazard positions (for avoidance)
    hazard_positions: list = field(default_factory=list)
    
    # Movement parameters
    walk_speed: float = 120.0      # Pixels per second
    run_speed: float = 200.0       # When fleeing
    jump_power: float = 250.0      # Jump velocity
    
    # State
    facing_right: bool = True
    is_jumping: bool = False
    stuck_timer: float = 0.0       # Detect if stuck
    blocked_timer: float = 0.0     # Can't reach goal
    last_x: float = 0.0
    last_goal_dist: float = 9999.0 # Track if making progress
    wander_direction: float = 1.0  # 1 = right, -1 = left
    wander_timer: float = 0.0
    detour_direction: float = 0.0  # When detouring around obstacle
    in_detour: bool = False
    
    # Reach threshold
    reach_distance: float = 30.0   # How close = "arrived"
    
    def set_hazards(self, hazards: list):
        """Update known hazard positions. Each is (x, y, radius)."""
        self.hazard_positions = hazards
    
    def set_goal_stay(self):
        """Stop moving."""
        self.goal = MovementGoal.STAY
        self.in_detour = False
        
    def set_goal_wander(self):
        """Wander randomly."""
        self.goal = MovementGoal.WANDER
        self.in_detour = False
        
    def set_goal_go_to(self, x: float, y: float):
        """Move toward a specific position."""
        # Only reset detour if target changed significantly
        if abs(x - self.target_x) > 50 or abs(y - self.target_y) > 50:
            self.in_detour = False
            self.blocked_timer = 0.0
        self.goal = MovementGoal.GO_TO_TARGET
        self.target_x = x
        self.target_y = y
        
    def set_goal_flee_from(self, x: float, y: float):
        """Run away from a position."""
        self.goal = MovementGoal.FLEE_FROM
        self.threat_x = x
        self.threat_y = y
        self.in_detour = False
        
    def set_goal_exit_water(self):
        """Get out of water."""
        self.goal = MovementGoal.EXIT_WATER
        self.in_detour = False
        
    def set_goal_exit_hazard(self, hazard_x: float, hazard_y: float):
        """Get away from hazard."""
        self.goal = MovementGoal.FLEE_FROM
        self.threat_x = hazard_x
        self.threat_y = hazard_y
        self.in_detour = False
    
    def _check_hazard_in_path(self, x: float, y: float, target_x: float, avoidance_dist: float = 100) -> Optional[Tuple[float, float]]:
        """Check if there's a hazard between creature and target. Returns hazard pos or None."""
        direction = 1.0 if target_x > x else -1.0
        
        for hx, hy, radius in self.hazard_positions:
            # Is hazard between us and target?
            if direction > 0:
                hazard_in_path = hx > x - 30 and hx < target_x + 30
            else:
                hazard_in_path = hx < x + 30 and hx > target_x - 30
            
            # Is hazard close enough vertically (on same level)?
            vertical_close = abs(hy - y) < 60
            
            # Is hazard close enough to matter?
            dist_to_hazard = np.sqrt((hx - x)**2 + (hy - y)**2)
            
            if hazard_in_path and vertical_close and dist_to_hazard < avoidance_dist + radius:
                return (hx, hy)
        
        return None
    
    def update(self, dt: float, creature_x: float, creature_y: float,
               on_ground: bool, in_water: bool, 
               world) -> Dict[str, float]:
        """
        Compute movement for this frame.
        
        Returns dict with vx, vy to set directly on creature.
        """
        result = {
            'vx': 0.0,
            'vy': None,  # None = don't change, let gravity work
            'jump': False,
        }
        
        # Stuck detection
        if abs(creature_x - self.last_x) < 1.0:
            self.stuck_timer += dt
        else:
            self.stuck_timer = 0.0
        self.last_x = creature_x
        
        # Handle each goal type
        if self.goal == MovementGoal.STAY:
            result['vx'] = 0.0
            
        elif self.goal == MovementGoal.WANDER:
            result = self._do_wander(dt, creature_x, creature_y, on_ground, in_water, world)
            
        elif self.goal == MovementGoal.GO_TO_TARGET:
            result = self._do_go_to(dt, creature_x, creature_y, on_ground, in_water, world)
            
        elif self.goal == MovementGoal.FLEE_FROM:
            result = self._do_flee(dt, creature_x, creature_y, on_ground, in_water, world)
            
        elif self.goal == MovementGoal.EXIT_WATER:
            result = self._do_exit_water(dt, creature_x, creature_y, in_water, world)
            
        # Update facing direction
        if result['vx'] > 5:
            self.facing_right = True
        elif result['vx'] < -5:
            self.facing_right = False
            
        return result
    
    def _do_wander(self, dt: float, x: float, y: float, 
                   on_ground: bool, in_water: bool, world) -> Dict:
        """Wander randomly, changing direction occasionally."""
        result = {'vx': 0.0, 'vy': None, 'jump': False}
        
        # Change direction occasionally
        self.wander_timer += dt
        if self.wander_timer > 3.0 or self.stuck_timer > 0.5:
            self.wander_timer = 0.0
            self.wander_direction *= -1
            self.stuck_timer = 0.0
        
        # Check for edges/walls
        check_ahead = x + self.wander_direction * 30
        
        # Don't walk off edges
        ground_ahead = world.is_solid(check_ahead, y + 20)
        if not ground_ahead and on_ground:
            self.wander_direction *= -1
            
        # Don't walk into walls
        wall_ahead = world.is_solid(check_ahead, y)
        if wall_ahead:
            if on_ground and self.stuck_timer > 0.2:
                result['jump'] = True
            else:
                self.wander_direction *= -1
        
        # HAZARD AVOIDANCE during wander
        for hx, hy, radius in self.hazard_positions:
            dist = np.sqrt((hx - x)**2 + (hy - y)**2)
            if dist < 100:
                # Turn away from hazard
                if hx > x:
                    self.wander_direction = -1.0
                else:
                    self.wander_direction = 1.0
        
        result['vx'] = self.wander_direction * self.walk_speed
        
        # Water is passable - no special handling needed
            
        return result
    
    def _do_go_to(self, dt: float, x: float, y: float,
                  on_ground: bool, in_water: bool, world) -> Dict:
        """Move toward target position with hazard avoidance."""
        result = {'vx': 0.0, 'vy': None, 'jump': False}
        
        dx = self.target_x - x
        dy = self.target_y - y
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Arrived?
        if dist < self.reach_distance:
            self.in_detour = False
            return result
        
        # Track if we're making progress
        if dist >= self.last_goal_dist - 5:
            self.blocked_timer += dt
        else:
            self.blocked_timer = max(0, self.blocked_timer - dt * 2)
        self.last_goal_dist = dist
        
        # Check for hazard in path
        hazard_in_way = self._check_hazard_in_path(x, y, self.target_x)
        
        # If blocked for too long or hazard in way, go into detour mode
        if self.blocked_timer > 1.0 or hazard_in_way:
            if not self.in_detour:
                self.in_detour = True
                # Pick a detour direction (perpendicular to goal)
                self.detour_direction = 1.0 if np.random.random() > 0.5 else -1.0
                self.blocked_timer = 0.0
        
        # IN DETOUR MODE: Move perpendicular to goal until clear
        if self.in_detour:
            # Check if path is now clear
            if not hazard_in_way and self.blocked_timer < 0.5:
                # Try going straight again
                self.in_detour = False
            else:
                # Move perpendicular (detour around obstacle)
                result['vx'] = self.detour_direction * self.walk_speed
                
                # Jump to get over/around
                if on_ground and self.stuck_timer > 0.2:
                    result['jump'] = True
                
                # No water swimming needed - water is passable
                
                return result
        
        # NORMAL MOVEMENT: Direction to target
        direction = 1.0 if dx > 0 else -1.0
        
        # Add avoidance vector from nearby hazards
        avoidance_vx = 0.0
        for hx, hy, radius in self.hazard_positions:
            hdist = np.sqrt((hx - x)**2 + (hy - y)**2)
            if hdist < 120:
                # Push away from hazard
                push_strength = (120 - hdist) / 120 * 0.8
                if hx > x:
                    avoidance_vx -= push_strength * self.walk_speed
                else:
                    avoidance_vx += push_strength * self.walk_speed
        
        # Combine goal direction with avoidance
        goal_vx = direction * self.walk_speed
        result['vx'] = goal_vx + avoidance_vx
        
        # Clamp speed
        result['vx'] = np.clip(result['vx'], -self.walk_speed, self.walk_speed)
        
        # Check for obstacles
        check_x = x + np.sign(result['vx']) * 20
        wall_ahead = world.is_solid(check_x, y)
        gap_ahead = not world.is_solid(check_x, y + 30) and on_ground
        
        # Jump over obstacles or gaps
        if wall_ahead and on_ground:
            result['jump'] = True
        elif gap_ahead and abs(dx) > 50:
            result['jump'] = True
            
        # Water is passable - no special handling needed
            
        # Stuck? Try jumping
        if self.stuck_timer > 0.3 and on_ground:
            result['jump'] = True
            self.stuck_timer = 0.0
            
        return result
    
    def _do_flee(self, dt: float, x: float, y: float,
                 on_ground: bool, in_water: bool, world) -> Dict:
        """Run away from threat at full speed."""
        result = {'vx': 0.0, 'vy': None, 'jump': False}
        
        dx = x - self.threat_x  # Opposite direction
        
        # Run AWAY from threat
        direction = 1.0 if dx > 0 else -1.0
        
        # Full speed fleeing
        result['vx'] = direction * self.run_speed
        
        # Always try to jump out of danger
        if on_ground:
            result['jump'] = True
            
        # Swim frantically in water
        if in_water:
            result['vy'] = -self.jump_power * 0.8
            
        return result
    
    def _do_exit_water(self, dt: float, x: float, y: float,
                       in_water: bool, world) -> Dict:
        """Get out of water - swim up and toward nearest land."""
        result = {'vx': 0.0, 'vy': None, 'jump': False}
        
        if not in_water:
            # Already out, switch to wander
            self.goal = MovementGoal.WANDER
            return result
        
        # Swim UP
        result['vy'] = -self.jump_power * 0.7
        
        # Find which direction has land
        # Check left and right for non-water
        left_is_water = world.is_water(x - 50, y)
        right_is_water = world.is_water(x + 50, y)
        
        if not right_is_water:
            result['vx'] = self.walk_speed
        elif not left_is_water:
            result['vx'] = -self.walk_speed
        else:
            # Both sides water, just keep swimming up and pick a direction
            result['vx'] = self.wander_direction * self.walk_speed * 0.5
            
        return result
    
    def has_reached_target(self, creature_x: float, creature_y: float) -> bool:
        """Check if we've reached the target."""
        if self.goal != MovementGoal.GO_TO_TARGET:
            return False
        dx = self.target_x - creature_x
        dy = self.target_y - creature_y
        return np.sqrt(dx*dx + dy*dy) < self.reach_distance
