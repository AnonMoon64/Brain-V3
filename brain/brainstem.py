"""
Brainstem: The survival foundation beneath cortical intelligence.

This module implements the non-negotiable survival layer that real animals have:
- Hardwired reflexes (no learning required)
- Drive-based motor biases
- Priority arbitration (pain > fear > hunger > curiosity)
- Dopamine reward prediction

The cortex can MODULATE behavior, but the brainstem keeps the creature alive.
Without this, creatures are "all thinking, no surviving."

Architecture:
    Reflexes (instant, hardwired)
        ↓
    Drive Nuclei (hunger, thirst, fear, pain)
        ↓
    Motivational Arbitration (priority selection)
        ↓
    Motor Pattern Generators (approach, flee, eat, rest)
        ↓
    Actual Movement

The cortex sits ON TOP of this, not instead of it.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum, auto


class ReflexType(Enum):
    """Hardwired reflexes - these fire WITHOUT thinking."""
    APPROACH_FOOD = auto()      # Move toward detected food
    EAT_ON_CONTACT = auto()     # Mouth opens when food touches
    DRINK_ON_CONTACT = auto()   # Drink when in water
    FLEE_PAIN = auto()          # Move away from pain source
    WITHDRAW = auto()           # Pull back from sudden stimulus
    STARTLE = auto()            # Freeze briefly on loud noise
    ORIENT = auto()             # Turn toward novel stimulus
    GRASP = auto()              # Hold food once obtained


class DriveType(Enum):
    """Motivational drives from hypothalamus-like nuclei."""
    HUNGER = auto()
    THIRST = auto()
    FEAR = auto()
    PAIN = auto()
    FATIGUE = auto()
    CURIOSITY = auto()
    SOCIAL = auto()
    TEMPERATURE = auto()


@dataclass
class Reflex:
    """A single hardwired reflex arc."""
    reflex_type: ReflexType
    threshold: float = 0.3          # Activation threshold
    strength: float = 1.0           # Output magnitude
    latency: float = 0.02           # Reaction time (seconds)
    habituation: float = 0.0        # Reduces with repeated activation
    habituation_rate: float = 0.1   # How fast it habituates
    recovery_rate: float = 0.05     # How fast habituation recovers
    
    # State
    activation: float = 0.0
    time_since_trigger: float = 0.0
    
    def trigger(self, stimulus: float) -> float:
        """Trigger reflex if stimulus exceeds threshold."""
        # Apply habituation
        effective_threshold = self.threshold * (1 + self.habituation)
        
        if stimulus > effective_threshold:
            self.activation = self.strength * (stimulus - effective_threshold)
            self.time_since_trigger = 0.0
            # Habituate to repeated stimuli
            self.habituation = min(2.0, self.habituation + self.habituation_rate)
            return self.activation
        return 0.0
    
    def update(self, dt: float):
        """Update reflex state."""
        self.time_since_trigger += dt
        # Recover from habituation
        self.habituation = max(0.0, self.habituation - self.recovery_rate * dt)
        # Decay activation
        self.activation *= 0.9


@dataclass
class DriveNucleus:
    """A hypothalamus-like nucleus that tracks a biological need."""
    drive_type: DriveType
    level: float = 0.0              # Current drive level (0-1)
    baseline: float = 0.0           # Resting level
    rise_rate: float = 0.01         # How fast it increases
    decay_rate: float = 0.1         # How fast it decreases when satisfied
    urgency_threshold: float = 0.5  # When it becomes priority
    critical_threshold: float = 0.8 # When it overrides everything
    
    # Priority weight (higher = more important when active)
    priority: float = 1.0
    
    def update(self, dt: float, satisfaction: float = 0.0):
        """Update drive level based on time and satisfaction."""
        # Natural rise (needs increase over time)
        self.level += self.rise_rate * dt
        
        # Satisfaction reduces the drive
        if satisfaction > 0:
            self.level -= self.decay_rate * satisfaction * dt
        
        # Clamp
        self.level = np.clip(self.level, 0.0, 1.0)
    
    def get_urgency(self) -> float:
        """Get urgency signal (0 if below threshold, scaled above)."""
        if self.level < self.urgency_threshold:
            return 0.0
        return (self.level - self.urgency_threshold) / (1.0 - self.urgency_threshold)
    
    def is_critical(self) -> bool:
        """Is this drive at critical level?"""
        return self.level >= self.critical_threshold


@dataclass 
class MotorPattern:
    """A motor pattern generator - produces coordinated movement."""
    name: str
    move_x: float = 0.0     # Horizontal bias (-1 to 1)
    move_y: float = 0.0     # Vertical bias (negative = up/jump)
    speed: float = 1.0      # Speed multiplier
    eat: bool = False       # Mouth action
    duration: float = 0.5   # How long pattern runs
    
    # State
    active: bool = False
    time_remaining: float = 0.0
    
    def activate(self, direction: float = 0.0):
        """Start this motor pattern."""
        self.active = True
        self.time_remaining = self.duration
        if direction != 0:
            self.move_x = np.sign(direction)
    
    def update(self, dt: float) -> Dict[str, float]:
        """Get motor outputs if active."""
        if not self.active:
            return {}
        
        self.time_remaining -= dt
        if self.time_remaining <= 0:
            self.active = False
            return {}
        
        outputs = {}
        if self.move_x > 0:
            outputs['move_right'] = abs(self.move_x) * self.speed
        elif self.move_x < 0:
            outputs['move_left'] = abs(self.move_x) * self.speed
        if self.move_y < 0:
            outputs['jump'] = abs(self.move_y) * self.speed
        if self.eat:
            outputs['eat'] = 1.0
        
        return outputs


class DopamineSystem:
    """
    Simple reward prediction system.
    
    Tracks: action -> outcome
    Updates: if outcome > expected, reinforce
    
    This is what makes creatures LEARN which actions lead to food.
    """
    
    def __init__(self):
        # Action-value estimates (simple TD learning)
        self.action_values: Dict[str, float] = {
            'approach_food': 0.5,
            'eat': 1.0,
            'explore': 0.1,
            'flee': 0.3,
            'rest': 0.2,
        }
        
        # Prediction of upcoming reward
        self.reward_prediction: float = 0.0
        
        # Recent actions for credit assignment
        self.action_trace: List[Tuple[str, float]] = []
        self.trace_decay: float = 0.9
        
        # Learning rate
        self.learning_rate: float = 0.1
        
        # Current dopamine level (phasic + tonic)
        self.tonic_level: float = 0.5
        self.phasic_level: float = 0.0
    
    def record_action(self, action: str, strength: float = 1.0):
        """Record an action for credit assignment."""
        self.action_trace.append((action, strength))
        # Keep trace manageable
        if len(self.action_trace) > 20:
            self.action_trace.pop(0)
    
    def receive_reward(self, reward: float):
        """
        Receive actual reward - compute prediction error.
        
        δ = reward - prediction
        If δ > 0: better than expected -> dopamine burst
        If δ < 0: worse than expected -> dopamine dip
        """
        prediction_error = reward - self.reward_prediction
        
        # Phasic dopamine signal
        self.phasic_level = np.clip(prediction_error, -1.0, 1.0)
        
        # Update action values based on trace
        for action, strength in self.action_trace:
            if action in self.action_values:
                # TD-style update
                self.action_values[action] += (
                    self.learning_rate * prediction_error * strength * self.trace_decay
                )
                self.action_values[action] = np.clip(
                    self.action_values[action], 0.0, 2.0
                )
        
        # Decay trace
        self.action_trace = [
            (a, s * self.trace_decay) for a, s in self.action_trace
            if s * self.trace_decay > 0.01
        ]
        
        # Update prediction
        self.reward_prediction = 0.8 * self.reward_prediction + 0.2 * reward
    
    def update(self, dt: float):
        """Update dopamine levels."""
        # Phasic decays quickly
        self.phasic_level *= 0.9
        
        # Tonic drifts toward baseline
        self.tonic_level += (0.5 - self.tonic_level) * 0.1 * dt
    
    def get_action_value(self, action: str) -> float:
        """Get learned value of an action."""
        return self.action_values.get(action, 0.1)
    
    def get_level(self) -> float:
        """Get total dopamine level."""
        return np.clip(self.tonic_level + self.phasic_level, 0.0, 1.5)


class Brainstem:
    """
    The survival foundation.
    
    This is NOT intelligent. This is ANCESTRAL.
    It keeps the creature alive while the cortex learns.
    
    Priority order (hardcoded, not learned):
        1. Pain/damage -> flee
        2. Critical hunger -> approach food desperately
        3. Fear -> avoid threats
        4. Hunger -> seek food
        5. Thirst -> seek water  
        6. Fatigue -> rest
        7. Curiosity -> explore
    """
    
    def __init__(self):
        # === REFLEXES (instant, no thinking) ===
        self.reflexes: Dict[ReflexType, Reflex] = {
            ReflexType.APPROACH_FOOD: Reflex(
                ReflexType.APPROACH_FOOD,
                threshold=0.1,  # Very low - almost always active when food detected
                strength=1.0,
            ),
            ReflexType.EAT_ON_CONTACT: Reflex(
                ReflexType.EAT_ON_CONTACT,
                threshold=0.5,  # Food must be close
                strength=1.0,
                habituation_rate=0.0,  # Never habituate to eating
            ),
            ReflexType.DRINK_ON_CONTACT: Reflex(
                ReflexType.DRINK_ON_CONTACT,
                threshold=0.3,  # Drink when thirsty and in water
                strength=1.0,
                habituation_rate=0.0,  # Never habituate to drinking
            ),
            ReflexType.FLEE_PAIN: Reflex(
                ReflexType.FLEE_PAIN,
                threshold=0.2,
                strength=1.5,  # Strong response
            ),
            ReflexType.WITHDRAW: Reflex(
                ReflexType.WITHDRAW,
                threshold=0.4,
                strength=0.8,
            ),
            ReflexType.STARTLE: Reflex(
                ReflexType.STARTLE,
                threshold=0.6,
                strength=0.5,
            ),
            ReflexType.ORIENT: Reflex(
                ReflexType.ORIENT,
                threshold=0.3,
                strength=0.6,
                habituation_rate=0.2,  # Quick habituation
            ),
        }
        
        # === DRIVE NUCLEI (motivational centers) ===
        self.drives: Dict[DriveType, DriveNucleus] = {
            DriveType.PAIN: DriveNucleus(
                DriveType.PAIN,
                priority=10.0,  # Highest priority
                rise_rate=0.0,  # Driven by damage, not time
                decay_rate=0.3,
                urgency_threshold=0.1,
                critical_threshold=0.3,
            ),
            DriveType.FEAR: DriveNucleus(
                DriveType.FEAR,
                priority=8.0,
                rise_rate=0.0,  # Driven by threats
                decay_rate=0.2,
                urgency_threshold=0.3,
                critical_threshold=0.6,
            ),
            DriveType.HUNGER: DriveNucleus(
                DriveType.HUNGER,
                priority=6.0,
                rise_rate=0.005,  # Slow rise over time
                decay_rate=0.8,   # Fast satisfaction when eating
                urgency_threshold=0.4,
                critical_threshold=0.7,
            ),
            DriveType.THIRST: DriveNucleus(
                DriveType.THIRST,
                priority=5.0,
                rise_rate=0.003,
                decay_rate=0.9,
                urgency_threshold=0.4,
                critical_threshold=0.7,
            ),
            DriveType.FATIGUE: DriveNucleus(
                DriveType.FATIGUE,
                priority=3.0,
                rise_rate=0.002,
                decay_rate=0.5,
                urgency_threshold=0.5,
                critical_threshold=0.8,
            ),
            DriveType.CURIOSITY: DriveNucleus(
                DriveType.CURIOSITY,
                priority=1.0,  # Lowest priority
                rise_rate=0.01,
                decay_rate=0.1,
                urgency_threshold=0.3,
                critical_threshold=0.9,  # Never critical
            ),
        }
        
        # === MOTOR PATTERNS ===
        self.motor_patterns: Dict[str, MotorPattern] = {
            'approach': MotorPattern('approach', move_x=1.0, speed=0.8),
            'flee': MotorPattern('flee', move_x=-1.0, speed=1.5),
            'eat': MotorPattern('eat', eat=True, duration=0.3),
            'rest': MotorPattern('rest', speed=0.0, duration=2.0),
            'explore': MotorPattern('explore', move_x=1.0, speed=0.5),
            'jump_escape': MotorPattern('jump_escape', move_y=-1.0, speed=1.0),
        }
        
        # === DOPAMINE SYSTEM ===
        self.dopamine = DopamineSystem()
        
        # State
        self.current_behavior: str = 'idle'
        self.behavior_direction: float = 0.0  # -1 left, +1 right
        self.last_food_direction: float = 0.0
        
    def process_sensory(self, sensory: Dict) -> Dict[str, float]:
        """
        Process sensory input through brainstem.
        
        Returns motor commands that BYPASS cortical processing.
        These are survival-critical and non-negotiable.
        """
        outputs = {
            'move_left': 0.0,
            'move_right': 0.0,
            'jump': 0.0,
            'eat': 0.0,
            'drink': 0.0,
        }
        
        # === EXTRACT SENSORY INFO ===
        food_distance = sensory.get('nearest_food_distance', 1000)
        food_direction = sensory.get('food_direction', 0)  # -1 left, +1 right
        food_touching = sensory.get('food_touching', False)
        pain = sensory.get('pain', 0)
        pain_direction = sensory.get('pain_direction', 0)
        threat_distance = sensory.get('threat_distance', 1000)
        threat_direction = sensory.get('threat_direction', 0)
        in_water = sensory.get('in_water', False)
        
        # Remember food direction
        if food_distance < 200:
            self.last_food_direction = food_direction
        
        # === REFLEX PROCESSING (instant) ===
        
        # 1. Eat reflex - HIGHEST PRIORITY when touching food
        if food_touching:
            eat_response = self.reflexes[ReflexType.EAT_ON_CONTACT].trigger(1.0)
            if eat_response > 0:
                outputs['eat'] = 1.0
                self.current_behavior = 'eating'
                self.dopamine.record_action('eat', 1.0)
        
        # 1b. Drink reflex - when in/near water and thirsty
        thirst = self.drives[DriveType.THIRST].level
        near_water = in_water or sensory.get('near_water', False)
        if near_water and thirst > 0.2:
            drink_response = self.reflexes[ReflexType.DRINK_ON_CONTACT].trigger(thirst)
            if drink_response > 0:
                outputs['drink'] = 1.0
                self.current_behavior = 'drinking'
                self.dopamine.record_action('drink', 0.5)
        
        # 2. Pain/flee reflex
        # Desperation Logic: If critically hungry, tolerate more pain before fleeing
        hunger = self.drives[DriveType.HUNGER].level
        flee_threshold = 0.1
        
        if hunger > 0.8:
            # Starving: Only flee if pain is severe (near death)
            flee_threshold = 0.5 
        elif hunger > 0.6:
            # Very hungry: Tolerant
            flee_threshold = 0.3
            
        if pain > flee_threshold:
            flee_response = self.reflexes[ReflexType.FLEE_PAIN].trigger(pain)
            if flee_response > 0:
                # Flee AWAY from pain source
                if pain_direction >= 0:
                    outputs['move_left'] = flee_response
                else:
                    outputs['move_right'] = flee_response
                outputs['jump'] = flee_response * 0.5
                self.current_behavior = 'fleeing'
                self.drives[DriveType.PAIN].level = pain
        
        # 3. Food approach reflex (when hungry)
        hunger = self.drives[DriveType.HUNGER].level
        if hunger > 0.2 and food_distance < 300:
            # Stronger approach when hungrier
            approach_strength = hunger * (1.0 - food_distance / 300)
            approach_response = self.reflexes[ReflexType.APPROACH_FOOD].trigger(approach_strength)
            
            if approach_response > 0 and self.current_behavior != 'fleeing':
                if food_direction > 0:
                    outputs['move_right'] = max(outputs['move_right'], approach_response)
                elif food_direction < 0:
                    outputs['move_left'] = max(outputs['move_left'], approach_response)
                else:
                    # Food below or above - random direction or last known
                    if self.last_food_direction != 0:
                        if self.last_food_direction > 0:
                            outputs['move_right'] = approach_response * 0.5
                        else:
                            outputs['move_left'] = approach_response * 0.5
                
                self.current_behavior = 'approaching_food'
                self.dopamine.record_action('approach_food', approach_response)
        
        return outputs
    
    def compute_drive_behavior(self, homeostasis) -> Dict[str, float]:
        """
        Compute behavior based on drive states.
        
        This is the ARBITRATION system - decides what behavior wins.
        """
        outputs = {
            'move_left': 0.0,
            'move_right': 0.0,
            'jump': 0.0,
            'eat': 0.0,
            'explore': 0.0,
        }
        
        # Update drives from homeostasis
        if homeostasis:
            self.drives[DriveType.HUNGER].level = 1.0 - homeostasis.energy
            self.drives[DriveType.THIRST].level = 1.0 - homeostasis.hydration
            self.drives[DriveType.FATIGUE].level = homeostasis.fatigue
            self.drives[DriveType.PAIN].level = homeostasis.pain
        
        # Find highest priority active drive
        active_drives = []
        for drive_type, nucleus in self.drives.items():
            urgency = nucleus.get_urgency()
            if urgency > 0:
                active_drives.append((drive_type, urgency * nucleus.priority))
        
        if not active_drives:
            # No urgent drives - default to exploration
            self.current_behavior = 'exploring'
            outputs['explore'] = 0.3
            return outputs
        
        # Sort by weighted urgency
        active_drives.sort(key=lambda x: -x[1])
        dominant_drive, urgency = active_drives[0]
        
        # Execute behavior for dominant drive
        if dominant_drive == DriveType.PAIN:
            self.current_behavior = 'fleeing'
            # Already handled by reflex, but add jump
            outputs['jump'] = min(1.0, urgency * 0.5)
            
        elif dominant_drive == DriveType.FEAR:
            self.current_behavior = 'hiding'
            # Move away from threats - direction set by sensory
            outputs['explore'] = -1.0  # Signal to flee
            
        elif dominant_drive == DriveType.HUNGER:
            self.current_behavior = 'seeking_food'
            # Approach food direction
            if self.last_food_direction > 0:
                outputs['move_right'] = urgency
            elif self.last_food_direction < 0:
                outputs['move_left'] = urgency
            else:
                # No known food direction - explore
                outputs['explore'] = urgency
            
        elif dominant_drive == DriveType.THIRST:
            self.current_behavior = 'seeking_water'
            outputs['explore'] = urgency * 0.8
            
        elif dominant_drive == DriveType.FATIGUE:
            self.current_behavior = 'resting'
            # Don't move
            
        elif dominant_drive == DriveType.CURIOSITY:
            self.current_behavior = 'exploring'
            outputs['explore'] = urgency * 0.5
        
        return outputs
    
    def update(self, dt: float, sensory: Dict, homeostasis) -> Dict[str, float]:
        """
        Main brainstem update loop.
        
        Returns motor outputs that should be combined with (or override) 
        cortical outputs depending on urgency.
        """
        # Update all reflexes
        for reflex in self.reflexes.values():
            reflex.update(dt)
        
        # Update all drives (natural rise)
        for drive in self.drives.values():
            drive.update(dt)
        
        # Update dopamine
        self.dopamine.update(dt)
        
        # Process sensory reflexes (instant responses)
        reflex_outputs = self.process_sensory(sensory)
        
        # Compute drive-based behavior
        drive_outputs = self.compute_drive_behavior(homeostasis)
        
        # Combine: reflexes have priority over drive behaviors
        final_outputs = {}
        for key in ['move_left', 'move_right', 'jump', 'eat']:
            reflex_val = reflex_outputs.get(key, 0)
            drive_val = drive_outputs.get(key, 0)
            # Reflexes override
            final_outputs[key] = max(reflex_val, drive_val)
        
        # Add explore signal for instincts to use
        final_outputs['explore'] = drive_outputs.get('explore', 0)
        
        # Track if we're in survival mode (high urgency)
        final_outputs['survival_mode'] = any(
            d.is_critical() for d in self.drives.values()
        )
        
        return final_outputs
    
    def receive_food_reward(self, amount: float):
        """Called when creature eats food."""
        # Satisfy hunger
        self.drives[DriveType.HUNGER].update(0, satisfaction=amount * 10)
        
        # Big dopamine reward
        self.dopamine.receive_reward(amount)
    
    def receive_water_reward(self, amount: float):
        """Called when creature drinks water."""
        # Satisfy thirst
        self.drives[DriveType.THIRST].update(0, satisfaction=amount * 10)
        
        # Moderate dopamine reward
        self.dopamine.receive_reward(amount * 0.5)
    
    def receive_damage(self, amount: float, direction: float = 0):
        """Called when creature takes damage."""
        self.drives[DriveType.PAIN].level = min(1.0, self.drives[DriveType.PAIN].level + amount)
        self.drives[DriveType.FEAR].level = min(1.0, self.drives[DriveType.FEAR].level + amount * 0.5)
        
        # Negative dopamine signal
        self.dopamine.receive_reward(-amount)
    
    def set_phenotype_traits(self, bravery: float):
        """
        Adjust brainstem parameters based on DNA.
        
        Args:
           bravery: 0.0 (Coward) to 1.0 (Brave).
                    Controls how fast PAIN/FEAR drives decay.
        """
        # Base decay rates
        base_pain_decay = 0.3
        base_fear_decay = 0.2
        
        # Modifier: Brave creatures recover 2x faster, Cowards 2x slower
        # bravery 0.5 = 1.0x (Normal)
        # bravery 1.0 = 2.0x
        # bravery 0.0 = 0.5x
        multiplier = 0.5 + (bravery * 1.5)
        
        self.drives[DriveType.PAIN].decay_rate = base_pain_decay * multiplier
        self.drives[DriveType.FEAR].decay_rate = base_fear_decay * multiplier
        
    def get_state(self) -> Dict:
        """Get brainstem state for debugging/display."""
        return {
            'behavior': self.current_behavior,
            'drives': {dt.name: dn.level for dt, dn in self.drives.items()},
            'urgencies': {dt.name: dn.get_urgency() for dt, dn in self.drives.items()},
            'dopamine': self.dopamine.get_level(),
            'food_direction': self.last_food_direction,
        }
