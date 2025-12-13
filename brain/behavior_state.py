"""
Behavioral State Machine - The key to stopping oscillation

This module provides:
1. Persistent behavioral states (not reset every frame)
2. Movement damping (smooth, not jerky)
3. Symmetry breaking (lateral bias)
4. Sustained action selection (commit to actions)
5. Evidence accumulation (don't flip-flop every tick)

Without these, creatures will oscillate until death.
"""

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List


class BehaviorState(Enum):
    """Discrete behavioral modes - animals have modes, not continuous policies."""
    IDLE = auto()           # Standing still, low alertness
    EXPLORING = auto()      # Wandering, looking for resources
    SEEKING_FOOD = auto()   # Actively moving toward food
    SEEKING_WATER = auto()  # Actively moving toward water
    EATING = auto()         # Consuming food (stationary)
    DRINKING = auto()       # Consuming water (stationary)
    FLEEING = auto()        # Running from danger
    RESTING = auto()        # Recovering energy
    SLEEPING = auto()       # Deep rest, minimal awareness
    SOCIALIZING = auto()    # Interacting with others
    # Mating behaviors
    MATE_CALLING = auto()   # Courtship display: dancing, vocalizing, showing objects
    MATING = auto()         # Actively mating (brief)
    # TIER 4: Tool Use States
    SEEKING_TOOL = auto()   # Moving toward a tool
    PICKING_UP = auto()     # Grasping an object
    CARRYING = auto()       # Holding a tool, can do other things
    DROPPING = auto()       # Releasing held object
    THROWING = auto()       # Hurling held object
    USING_TOOL = auto()     # Using tool for a purpose (digging, reaching, breaking)
    

@dataclass
class MotorState:
    """Persistent motor state with damping."""
    # Target direction (-1 = left, 0 = still, 1 = right)
    target_direction: float = 0.0
    # Current smoothed direction (damped)
    current_direction: float = 0.0
    # Target speed (0-1)
    target_speed: float = 0.0
    # Current smoothed speed
    current_speed: float = 0.0
    # Jump intent
    wants_jump: bool = False
    # Damping factor (lower = faster response, was 0.7 which was too slow)
    damping: float = 0.3
    
    def update(self, dt: float = 0.033):
        """Apply damping to smooth motor output."""
        # Smooth direction changes (faster response now)
        self.current_direction = (
            self.current_direction * self.damping + 
            self.target_direction * (1 - self.damping)
        )
        # Smooth speed changes
        self.current_speed = (
            self.current_speed * self.damping + 
            self.target_speed * (1 - self.damping)
        )
        # Decay jump intent
        self.wants_jump = False
    
    def get_motor_output(self) -> Dict[str, float]:
        """Convert motor state to action signals."""
        output = {
            'move_left': 0.0,
            'move_right': 0.0,
            'jump': 0.0,
        }
        
        # Convert direction + speed to left/right
        # Use speed directly, direction just determines which way
        if self.current_direction < -0.05 and self.current_speed > 0.05:
            output['move_left'] = self.current_speed  # Full speed, not multiplied
        elif self.current_direction > 0.05 and self.current_speed > 0.05:
            output['move_right'] = self.current_speed  # Full speed, not multiplied
        
        if self.wants_jump:
            output['jump'] = 0.8
        
        return output


@dataclass
class BehaviorStateMachine:
    """
    Persistent behavioral state machine.
    
    Key features:
    - States persist until explicitly changed
    - Transitions require thresholds (hysteresis)
    - Evidence accumulates before state change
    - Lateral bias breaks symmetry
    """
    
    # Current behavioral state
    state: BehaviorState = BehaviorState.IDLE
    # How long in current state (seconds)
    state_duration: float = 0.0
    # Minimum time before state can change (reduced for faster reactions)
    min_state_duration: float = 0.2
    
    # Motor state (with damping)
    motor: MotorState = field(default_factory=MotorState)
    
    # Target position (for seeking behaviors)
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    
    # Lateral bias (-1 to 1, breaks symmetry)
    lateral_bias: float = field(default_factory=lambda: np.random.uniform(-0.3, 0.3))
    
    # Wander state
    wander_timer: float = 0.0
    wander_direction: float = 0.0
    
    # Evidence accumulators (require sustained signal to trigger)
    food_evidence: float = 0.0
    water_evidence: float = 0.0
    danger_evidence: float = 0.0
    rest_evidence: float = 0.0
    
    # Evidence thresholds (lowered for faster reactions)
    evidence_threshold: float = 0.3
    evidence_decay: float = 0.85  # Faster decay = quicker calming down
    
    # Flee timeout - max time to flee before forced rest
    flee_timer: float = 0.0
    max_flee_duration: float = 3.0  # Stop fleeing after 3 seconds max
    
    # Last successful action direction (reinforcement)
    last_successful_direction: float = 0.0
    success_memory: float = 0.0  # Decays over time
    
    # UPGRADE 9: Behavioral Persistence (Hysteresis + Switching Costs)
    last_state_change_time: float = 0.0      # When did we last switch states?
    state_switch_cooldown: float = 0.5       # Minimum seconds between switches
    task_switch_cost: float = 0.1            # Energy cost to change tasks
    state_commitment: float = 1.0            # How committed to current state (0-1)
    
    def __post_init__(self):
        """Initialize with random lateral bias."""
        if self.lateral_bias == 0:
            self.lateral_bias = np.random.uniform(-0.3, 0.3)
    
    def update(self, dt: float, sensory: Dict, drives: Dict, 
               creature_x: float, creature_y: float) -> Dict[str, float]:
        """
        Update state machine and return motor commands.
        
        Args:
            dt: Time delta
            sensory: Sensory input dict
            drives: Drive levels dict
            creature_x, creature_y: Creature position
            
        Returns:
            Motor output dict with move_left, move_right, jump, eat, drink
        """
        self.state_duration += dt
        self.wander_timer -= dt
        
        # Track how long we've been fleeing
        if self.state == BehaviorState.FLEEING:
            self.flee_timer += dt
            # Force stop fleeing after max duration (exhaustion)
            if self.flee_timer > self.max_flee_duration:
                self.danger_evidence = 0.0  # Reset danger
                self.flee_timer = 0.0
        
        # Decay evidence accumulators
        self.food_evidence *= self.evidence_decay
        self.water_evidence *= self.evidence_decay
        self.danger_evidence *= self.evidence_decay
        self.rest_evidence *= self.evidence_decay
        self.success_memory *= 0.99  # Slow decay for success
        
        # Accumulate evidence from sensory/drives
        self._accumulate_evidence(sensory, drives)
        
        # Check for state transitions
        self._check_transitions(sensory, drives, creature_x, creature_y)
        
        # Execute current state behavior
        output = self._execute_state(sensory, drives, creature_x, creature_y)
        
        # Update motor state (apply damping)
        self.motor.update(dt)
        
        # Merge motor output with state output
        motor_output = self.motor.get_motor_output()
        for key in ['move_left', 'move_right', 'jump']:
            if key in motor_output:
                output[key] = max(output.get(key, 0), motor_output[key])
        
        return output
    
    def _accumulate_evidence(self, sensory: Dict, drives: Dict):
        """Accumulate evidence for state transitions (prevents flip-flopping)."""
        # Food evidence
        food_dist = sensory.get('nearest_food_distance', 1000)
        hunger = drives.get('hunger', 0)
        if food_dist < 200 and hunger > 0.3:
            # More evidence if food is close and hungry
            self.food_evidence += (1 - food_dist / 200) * hunger * 0.1
        
        # Water evidence
        water_dist = sensory.get('nearest_water_distance', 1000)
        thirst = drives.get('thirst', 0)
        if water_dist < 200 and thirst > 0.3:
            self.water_evidence += (1 - water_dist / 200) * thirst * 0.1
        
        # Danger evidence - check HAZARDS (fire) and threats
        # Check visible hazards first
        visible_hazards = sensory.get('visible_hazards', [])
        for hazard in visible_hazards:
            dist = hazard.get('dist', 1000)
            if dist < 100:
                # Very close to hazard - IMMEDIATE danger
                self.danger_evidence += (1 - dist / 100) * 0.5
        
        # Also check nearest_hazard_distance from enriched data
        hazard_dist = sensory.get('nearest_hazard_distance', float('inf'))
        if hazard_dist < 100:
            # Standing on or very near hazard - high danger
            self.danger_evidence += (1 - hazard_dist / 100) * 0.6
        
        # Also check nearest_threat_distance for creatures
        threat_dist = sensory.get('nearest_threat_distance', 1000)
        if threat_dist < 150:
            self.danger_evidence += (1 - threat_dist / 150) * 0.2
        
        # CRITICAL: If taking EXTERNAL damage (not hunger/thirst pain), high danger
        # Only trigger flee for sudden/high pain spikes from hazards
        pain = sensory.get('pain', 0)
        damage_source = sensory.get('damage_source', 'none')
        if pain > 0.3 and damage_source in ['hazard', 'attack', 'poison']:
            self.danger_evidence = min(1.0, self.danger_evidence + 0.5)
        
        # Rest evidence
        fatigue = drives.get('rest', 0)
        if fatigue > 0.6:
            self.rest_evidence += fatigue * 0.05
        
        # Clamp evidence
        self.food_evidence = min(1.0, self.food_evidence)
        self.water_evidence = min(1.0, self.water_evidence)
        self.danger_evidence = min(1.0, self.danger_evidence)
        self.rest_evidence = min(1.0, self.rest_evidence)
    
    def _check_transitions(self, sensory: Dict, drives: Dict,
                           creature_x: float, creature_y: float):
        """Check for state transitions with hysteresis."""
        # UPGRADE 9: Enhanced Hysteresis - prevent rapid state flipping
        time_since_switch = self.state_duration
        if time_since_switch < self.state_switch_cooldown:
            # Still in cooldown, strengthen commitment to current state
            self.state_commitment = min(1.0, self.state_commitment + 0.1)
            # Exception: extreme danger can always interrupt
            if self.danger_evidence < 0.95:
                return
        
        # Don't transition too fast (prevents oscillation)
        if self.state_duration < self.min_state_duration:
            # Exception: danger can always interrupt
            if self.danger_evidence < 0.9:
                return
        
        old_state = self.state
        
        # Priority-based transitions (danger > needs > exploration)
        
        # 1. DANGER - highest priority, can always interrupt
        # Desperation Logic: If starving, ignore moderate danger (User Request: "rabbit eat bear")
        hunger = drives.get('hunger', 0)
        effective_threshold = self.evidence_threshold
        
        if hunger > 0.8:
            # STARVATION MODE: Only flee from immediate death (0.9), ignore discomfort
            effective_threshold = 0.9
        elif hunger > 0.6:
            # Hungry: Braver (0.6)
            effective_threshold = 0.6
            
        if self.danger_evidence > effective_threshold:
            self._transition_to(BehaviorState.FLEEING, sensory, creature_x, creature_y)
            return
        
        # 2. CRITICAL NEEDS
        hunger = drives.get('hunger', 0)
        thirst = drives.get('thirst', 0)
        
        # Already eating/drinking? Check if should continue
        if self.state == BehaviorState.EATING:
            if hunger < 0.1 or sensory.get('nearest_food_distance', 1000) > 40:
                # Done eating
                self._transition_to(BehaviorState.IDLE, sensory, creature_x, creature_y)
            return
        
        if self.state == BehaviorState.DRINKING:
            if thirst < 0.1 or not sensory.get('in_water', False):
                # Done drinking
                self._transition_to(BehaviorState.IDLE, sensory, creature_x, creature_y)
            return
        
        # Check if should start eating (very close to food and hungry)
        food_dist = sensory.get('nearest_food_distance', 1000)
        if food_dist < 30 and hunger > 0.2:
            self._transition_to(BehaviorState.EATING, sensory, creature_x, creature_y)
            return
        
        # Check if should start drinking
        if sensory.get('in_water', False) and thirst > 0.3:
            self._transition_to(BehaviorState.DRINKING, sensory, creature_x, creature_y)
            return
        
        # 3. SEEKING RESOURCES
        if self.food_evidence > self.evidence_threshold and hunger > 0.3:
            self._transition_to(BehaviorState.SEEKING_FOOD, sensory, creature_x, creature_y)
            return
        
        if self.water_evidence > self.evidence_threshold and thirst > 0.3:
            self._transition_to(BehaviorState.SEEKING_WATER, sensory, creature_x, creature_y)
            return
        
        # 4. REST
        if self.rest_evidence > self.evidence_threshold:
            sleep_need = drives.get('sleep', 0)
            if sleep_need > 0.7:
                self._transition_to(BehaviorState.SLEEPING, sensory, creature_x, creature_y)
            else:
                self._transition_to(BehaviorState.RESTING, sensory, creature_x, creature_y)
            return
        
        # 5. DEFAULT - explore or idle (faster transitions)
        if self.state == BehaviorState.IDLE and self.state_duration > 0.5:
            # Been idle too long, start exploring immediately
            self._transition_to(BehaviorState.EXPLORING, sensory, creature_x, creature_y)
        elif self.state == BehaviorState.EXPLORING and self.state_duration > 8.0:
            # Been exploring a while, take a short break
            self._transition_to(BehaviorState.IDLE, sensory, creature_x, creature_y)
        elif self.state in [BehaviorState.SEEKING_FOOD, BehaviorState.SEEKING_WATER]:
            # Lost track of target - go back to exploring
            if self.state_duration > 5.0:
                self._transition_to(BehaviorState.EXPLORING, sensory, creature_x, creature_y)
    
    def _transition_to(self, new_state: BehaviorState, sensory: Dict,
                       creature_x: float, creature_y: float):
        """Transition to a new state."""
        if new_state == self.state:
            return
        
        self.state = new_state
        self.state_duration = 0.0
        
        # Set up state-specific targets
        if new_state == BehaviorState.SEEKING_FOOD:
            food_x = sensory.get('nearest_food_x', None)
            food_y = sensory.get('nearest_food_y', None)
            if food_x is not None:
                self.target_x = food_x
                self.target_y = food_y
        elif new_state == BehaviorState.SEEKING_WATER:
            water_x = sensory.get('nearest_water_x', None)
            water_y = sensory.get('nearest_water_y', None)
            if water_x is not None:
                self.target_x = water_x
                self.target_y = water_y
        elif new_state == BehaviorState.FLEEING:
            self.flee_timer = 0.0  # Reset flee timer when starting to flee
            threat_x = sensory.get('nearest_threat_x', None)
            if threat_x is not None:
                self.target_x = threat_x  # We'll run AWAY from this
        elif new_state == BehaviorState.EXPLORING:
            # Pick a random direction with lateral bias
            self.motor.target_direction = 1.0 if np.random.random() + self.lateral_bias > 0.5 else -1.0
            self.target_x = None
    
    def _execute_state(self, sensory: Dict, drives: Dict,
                       creature_x: float, creature_y: float) -> Dict[str, float]:
        """Execute behavior for current state."""
        output = {
            'move_left': 0.0,
            'move_right': 0.0,
            'jump': 0.0,
            'eat': 0.0,
            'drink': 0.0,
            'rest': 0.0,
            'sleep': 0.0,
        }
        
        if self.state == BehaviorState.IDLE:
            # Stand still, low speed
            self.motor.target_speed = 0.0
            self.motor.target_direction = 0.0
        
        elif self.state == BehaviorState.EXPLORING:
            # Wander with long-term persistence
            self.motor.target_speed = 0.8 # Slightly slower than max run
            
            # Check world bounds
            bounds = sensory.get('world_bounds', {})
            min_x = bounds.get('min_x')
            max_x = bounds.get('max_x')
            
            # Wall avoidance (Higher priority than random wander)
            wall_override = False
            if min_x is not None and creature_x < min_x + 100:
                self.wander_direction = 1.0 # Force Right
                self.wander_timer = max(self.wander_timer, 1.0) # Ensure we stick to it
                wall_override = True
            elif max_x is not None and creature_x > max_x - 100:
                self.wander_direction = -1.0 # Force Left
                self.wander_timer = max(self.wander_timer, 1.0)
                wall_override = True
            
            if not wall_override and self.wander_timer <= 0:
                # Pick new direction and duration
                self.wander_timer = np.random.uniform(2.0, 6.0) # Wander for 2-6 seconds
                # Pick direction: Favor continuing, but sometimes flip
                if np.random.random() < 0.7:
                     # Continue or slight variation
                     self.wander_direction = 1.0 if self.lateral_bias > 0 else -1.0
                else:
                     # Flip
                     self.wander_direction = -1.0 if self.lateral_bias > 0 else 1.0
                
                # Add heavy randomness
                if np.random.random() < 0.5:
                    self.wander_direction *= -1

            self.motor.target_direction = self.wander_direction

            
            # Avoid walls
            if sensory.get('wall_left', False):
                 self.motor.target_direction = 1.0
                 self.wander_direction = 1.0
                 self.wander_timer = 2.0 # Reset timer to walk away from wall
            elif sensory.get('wall_right', False):
                 self.motor.target_direction = -1.0
                 self.wander_direction = -1.0
                 self.wander_timer = 2.0
        
        elif self.state == BehaviorState.SEEKING_FOOD:
            self._seek_target(sensory, creature_x, 'nearest_food_x', 'nearest_food_distance')
            self.motor.target_speed = 1.0  # FULL SPEED toward food
        
        elif self.state == BehaviorState.SEEKING_WATER:
            self._seek_target(sensory, creature_x, 'nearest_water_x', 'nearest_water_distance')
            self.motor.target_speed = 1.0  # FULL SPEED toward water
        
        elif self.state == BehaviorState.EATING:
            # Stop moving, eat
            self.motor.target_speed = 0.0
            self.motor.target_direction = 0.0
            output['eat'] = 1.0
            # Record successful direction for reinforcement
            self._record_success()
        
        elif self.state == BehaviorState.DRINKING:
            # Stop moving, drink
            self.motor.target_speed = 0.0
            self.motor.target_direction = 0.0
            output['drink'] = 1.0
            self._record_success()
        
        elif self.state == BehaviorState.FLEEING:
            # Run AWAY from threat or hazard
            # First check for visible hazards (fire)
            visible_hazards = sensory.get('visible_hazards', [])
            if visible_hazards:
                # Find nearest hazard
                nearest = min(visible_hazards, key=lambda h: h.get('dist', 1000))
                hazard_dx = nearest.get('dx', 0)
                if hazard_dx < 0:
                    # Hazard is left, run right
                    self.motor.target_direction = 1.0
                else:
                    # Hazard is right, run left
                    self.motor.target_direction = -1.0
            else:
                # Check for creature threats
                threat_x = sensory.get('nearest_threat_x', None)
                if threat_x is not None:
                    creature_x = sensory.get('creature_x', 0)
                    if threat_x < creature_x:
                        self.motor.target_direction = 1.0
                    else:
                        self.motor.target_direction = -1.0
                else:
                    # No visible threat, just run in bias direction
                    self.motor.target_direction = 1.0 if self.lateral_bias > 0 else -1.0
            
            self.motor.target_speed = 1.0  # Maximum speed
            # Jump to escape obstacles
            if sensory.get('wall_left', False) or sensory.get('wall_right', False):
                self.motor.wants_jump = True
        
        elif self.state == BehaviorState.RESTING:
            self.motor.target_speed = 0.0
            self.motor.target_direction = 0.0
            output['rest'] = 1.0
        
        elif self.state == BehaviorState.SLEEPING:
            self.motor.target_speed = 0.0
            self.motor.target_direction = 0.0
            output['sleep'] = 1.0
        
        return output
    
    def _seek_target(self, sensory: Dict, creature_x: float,
                     target_x_key: str, target_dist_key: str):
        """Move toward a target with proper direction computation."""
        target_x = sensory.get(target_x_key, None)
        target_dist = sensory.get(target_dist_key, 1000)
        
        if target_x is not None:
            # ACTUAL DIRECTION VECTOR (this is what was missing!)
            direction = target_x - creature_x
            
            # Normalize to -1 to 1
            if abs(direction) > 1:
                self.motor.target_direction = 1.0 if direction > 0 else -1.0
            else:
                self.motor.target_direction = direction
            
            # Add slight bias to break perfect symmetry
            self.motor.target_direction += self.lateral_bias * 0.05
            
            # Use success memory if we have it
            if self.success_memory > 0.3:
                self.motor.target_direction += self.last_successful_direction * 0.2
            
            # Clamp
            self.motor.target_direction = np.clip(self.motor.target_direction, -1, 1)
            
            # Jump over obstacles
            wall_ahead = (
                (self.motor.target_direction > 0 and sensory.get('wall_right', False)) or
                (self.motor.target_direction < 0 and sensory.get('wall_left', False))
            )
            if wall_ahead:
                self.motor.wants_jump = True
        else:
            # No target visible - use last successful direction or bias
            if self.success_memory > 0.2:
                self.motor.target_direction = self.last_successful_direction
            else:
                self.motor.target_direction = 1.0 if self.lateral_bias > 0 else -1.0
    
    def _record_success(self):
        """Record successful action for reinforcement."""
        self.last_successful_direction = self.motor.current_direction
        self.success_memory = 1.0
    
    def reinforce(self, reward: float):
        """Apply reward to strengthen current behavior."""
        if reward > 0:
            self.success_memory = min(1.0, self.success_memory + reward)
            # Strengthen lateral bias toward successful direction
            if abs(self.motor.current_direction) > 0.1:
                bias_delta = np.sign(self.motor.current_direction) * reward * 0.1
                self.lateral_bias = np.clip(self.lateral_bias + bias_delta, -0.5, 0.5)
    
    def get_state_name(self) -> str:
        """Get human-readable state name."""
        return self.state.name.lower().replace('_', ' ')
