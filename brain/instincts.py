"""
Instinct Layer - Primitive reflexes and behaviors

This module provides the "instinct layer" between drives and actions:
- Reflexes that fire automatically (flee from pain, approach food)
- Behavioral primitives that the brain can modulate
- Reward signals for drive satisfaction

Instincts are NOT hardcoded behaviors - they're weighted tendencies
that the brain learns to enhance or suppress over time.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


# =============================================================================
# INSTINCT TYPES
# =============================================================================

class InstinctType(Enum):
    """Types of innate behavioral tendencies."""
    # Survival
    FLEE_THREAT = "flee_threat"
    AVOID_HAZARD = "avoid_hazard"
    SEEK_FOOD = "seek_food"
    SEEK_WATER = "seek_water"
    SEEK_SHELTER = "seek_shelter"
    SEEK_WARMTH = "seek_warmth"
    SEEK_COOLING = "seek_cooling"
    SURFACE_BREATHE = "surface_breathe"   # Swim up when drowning
    
    # Social
    APPROACH_MATE = "approach_mate"
    FOLLOW_SAME_SPECIES = "follow_same_species"
    AVOID_LARGER = "avoid_larger"
    SIGNAL_DISTRESS = "signal_distress"
    
    # Exploration
    EXPLORE_NOVEL = "explore_novel"
    RETURN_HOME = "return_home"
    
    # Rest
    SEEK_REST = "seek_rest"
    SEEK_SLEEP = "seek_sleep"             # Find safe spot to sleep
    FREEZE = "freeze"


@dataclass
class Instinct:
    """
    A single instinctual tendency.
    
    Instincts compute a behavioral "vote" based on sensory/drive state.
    Multiple instincts combine to suggest actions.
    """
    type: InstinctType
    base_strength: float = 1.0      # Innate strength (from DNA)
    learned_weight: float = 1.0     # Modified by experience
    threshold: float = 0.3          # Activation threshold
    cooldown: float = 0.0           # Current cooldown
    cooldown_rate: float = 0.5      # Time between activations
    
    def get_effective_strength(self) -> float:
        """Get current effective strength."""
        return self.base_strength * self.learned_weight
    
    def can_activate(self) -> bool:
        """Check if instinct can fire."""
        return self.cooldown <= 0
    
    def activate(self):
        """Fire the instinct, start cooldown."""
        self.cooldown = self.cooldown_rate
    
    def update(self, dt: float):
        """Update cooldown."""
        self.cooldown = max(0, self.cooldown - dt)
    
    def modulate(self, reward: float):
        """
        Modify learned weight based on reward.
        
        Positive reward strengthens, negative weakens.
        """
        # Simple Hebbian-like learning
        self.learned_weight = np.clip(
            self.learned_weight + 0.01 * reward,
            0.1, 3.0
        )


# =============================================================================
# INSTINCT INHIBITION MATRIX
# =============================================================================

# Which instincts inhibit which others
# Format: (inhibitor, inhibited, strength)
INHIBITION_RULES = [
    # Survival overrides everything
    (InstinctType.SURFACE_BREATHE, InstinctType.SEEK_FOOD, 0.9),
    (InstinctType.SURFACE_BREATHE, InstinctType.EXPLORE_NOVEL, 1.0),
    (InstinctType.SURFACE_BREATHE, InstinctType.APPROACH_MATE, 1.0),
    (InstinctType.SURFACE_BREATHE, InstinctType.SEEK_SLEEP, 1.0),
    
    # Fear/danger inhibits most things
    # TUNING: Reduced inhibition to allow "panic eating" and survival actions
    (InstinctType.FLEE_THREAT, InstinctType.SEEK_FOOD, 0.2), # Was 0.7 - Startled eating
    (InstinctType.FLEE_THREAT, InstinctType.EXPLORE_NOVEL, 0.9),
    (InstinctType.FLEE_THREAT, InstinctType.APPROACH_MATE, 0.6), # Was 0.8
    (InstinctType.FLEE_THREAT, InstinctType.SEEK_SLEEP, 0.5), # Was 1.0 - Pass out if exhausted
    (InstinctType.AVOID_HAZARD, InstinctType.SEEK_FOOD, 0.3), # Was 0.6
    (InstinctType.AVOID_HAZARD, InstinctType.EXPLORE_NOVEL, 0.8),
    
    # Hunger inhibits exploration (focus on food)
    (InstinctType.SEEK_FOOD, InstinctType.EXPLORE_NOVEL, 0.5),
    (InstinctType.SEEK_FOOD, InstinctType.APPROACH_MATE, 0.4),
    
    # Sleepiness inhibits everything except survival
    (InstinctType.SEEK_SLEEP, InstinctType.EXPLORE_NOVEL, 0.8),
    (InstinctType.SEEK_SLEEP, InstinctType.SEEK_FOOD, 0.3),
    (InstinctType.SEEK_SLEEP, InstinctType.APPROACH_MATE, 0.7),
    
    # Mating focus inhibits exploration
    (InstinctType.APPROACH_MATE, InstinctType.EXPLORE_NOVEL, 0.6),
]

# Instinct synergies - combinations that boost each other
SYNERGY_RULES = [
    # Fear + Pain = strong flee (not freeze)
    (InstinctType.FLEE_THREAT, InstinctType.AVOID_HAZARD, 1.3),
    # Cold + Shelter seeking
    (InstinctType.SEEK_WARMTH, InstinctType.SEEK_SHELTER, 1.2),
]


class InstinctArbitrator:
    """
    Arbitrates between competing instincts using:
    - Mutual inhibition
    - Winner-take-most (not winner-take-all)
    - Urgency weighting by drive severity
    - Synergy bonuses
    """
    
    def __init__(self):
        # Build inhibition lookup
        self.inhibition: Dict[InstinctType, List[Tuple[InstinctType, float]]] = {}
        for inhibitor, inhibited, strength in INHIBITION_RULES:
            if inhibitor not in self.inhibition:
                self.inhibition[inhibitor] = []
            self.inhibition[inhibitor].append((inhibited, strength))
        
        # Build synergy lookup
        self.synergies: Dict[Tuple[InstinctType, InstinctType], float] = {}
        for a, b, boost in SYNERGY_RULES:
            self.synergies[(a, b)] = boost
            self.synergies[(b, a)] = boost
    
    def arbitrate(self, 
                  instinct_activations: Dict[InstinctType, float],
                  drive_urgencies: Dict[str, float]) -> Dict[InstinctType, float]:
        """
        Arbitrate between instincts, returning modulated activations.
        
        Args:
            instinct_activations: Raw activation levels per instinct
            drive_urgencies: Urgency level per drive (0-1)
            
        Returns:
            Modulated activations after inhibition/synergy
        """
        # Start with raw activations
        modulated = dict(instinct_activations)
        
        # Apply urgency weighting
        # Map drives to instincts
        drive_to_instinct = {
            'hunger': InstinctType.SEEK_FOOD,
            'thirst': InstinctType.SEEK_WATER,
            'safety': InstinctType.FLEE_THREAT,
            'breathe': InstinctType.SURFACE_BREATHE,
            'sleep': InstinctType.SEEK_SLEEP,
            'rest': InstinctType.SEEK_REST,
            'reproduction': InstinctType.APPROACH_MATE,
            'exploration': InstinctType.EXPLORE_NOVEL,
        }
        
        for drive, instinct_type in drive_to_instinct.items():
            urgency = drive_urgencies.get(drive, 0)
            if instinct_type in modulated:
                # Urgency boosts activation
                modulated[instinct_type] *= (1 + urgency * 0.5)
        
        # Apply mutual inhibition
        for inhibitor, targets in self.inhibition.items():
            if inhibitor not in modulated:
                continue
            inhibitor_strength = modulated[inhibitor]
            if inhibitor_strength < 0.1:
                continue
            
            for inhibited, strength in targets:
                if inhibited in modulated:
                    # Stronger inhibitor = more suppression
                    suppression = inhibitor_strength * strength
                    modulated[inhibited] *= (1 - suppression)
        
        # Apply synergies
        active_instincts = [i for i, v in modulated.items() if v > 0.2]
        for i, inst_a in enumerate(active_instincts):
            for inst_b in active_instincts[i+1:]:
                if (inst_a, inst_b) in self.synergies:
                    boost = self.synergies[(inst_a, inst_b)]
                    modulated[inst_a] *= boost
                    modulated[inst_b] *= boost
        
        # Winner-take-most: boost the top instinct(s), suppress weaker ones
        if modulated:
            max_activation = max(modulated.values())
            if max_activation > 0.3:
                for inst in modulated:
                    ratio = modulated[inst] / max_activation
                    # Compress lower activations
                    modulated[inst] *= (0.3 + 0.7 * ratio)
        
        # Clip to valid range
        for inst in modulated:
            modulated[inst] = np.clip(modulated[inst], 0, 1)
        
        return modulated


# =============================================================================
# INSTINCT SYSTEM
# =============================================================================

class InstinctSystem:
    """
    Manages all instincts and computes behavioral suggestions.
    
    The brain receives instinct outputs as additional input signals
    and can learn to follow or override them.
    """
    
    def __init__(self, drive_params: Optional[Dict[str, float]] = None):
        """
        Initialize instinct system.
        
        Args:
            drive_params: Initial drive strengths from DNA
        """
        drive_params = drive_params or {}
        
        # Arbitrator for instinct competition
        self.arbitrator = InstinctArbitrator()
        
        # Create instincts with DNA-influenced strengths
        self.instincts: Dict[InstinctType, Instinct] = {
            InstinctType.FLEE_THREAT: Instinct(
                type=InstinctType.FLEE_THREAT,
                base_strength=1.0 + drive_params.get('fear', 0.5),
                threshold=0.2
            ),
            InstinctType.AVOID_HAZARD: Instinct(
                type=InstinctType.AVOID_HAZARD,
                base_strength=1.5,  # Strong innate response
                threshold=0.1
            ),
            InstinctType.SEEK_FOOD: Instinct(
                type=InstinctType.SEEK_FOOD,
                base_strength=1.0 + drive_params.get('hunger', 0.5),
                threshold=0.3
            ),
            InstinctType.SEEK_WATER: Instinct(
                type=InstinctType.SEEK_WATER,
                base_strength=1.0,
                threshold=0.3
            ),
            InstinctType.SEEK_SHELTER: Instinct(
                type=InstinctType.SEEK_SHELTER,
                base_strength=0.5 + drive_params.get('fear', 0.5) * 0.5,
                threshold=0.4
            ),
            InstinctType.SEEK_WARMTH: Instinct(
                type=InstinctType.SEEK_WARMTH,
                base_strength=1.0,
                threshold=0.3
            ),
            InstinctType.SEEK_COOLING: Instinct(
                type=InstinctType.SEEK_COOLING,
                base_strength=1.0,
                threshold=0.3
            ),
            InstinctType.APPROACH_MATE: Instinct(
                type=InstinctType.APPROACH_MATE,
                base_strength=0.8 + drive_params.get('social', 0.5) * 0.4,
                threshold=0.5
            ),
            InstinctType.FOLLOW_SAME_SPECIES: Instinct(
                type=InstinctType.FOLLOW_SAME_SPECIES,
                base_strength=drive_params.get('social', 0.5),
                threshold=0.4
            ),
            InstinctType.AVOID_LARGER: Instinct(
                type=InstinctType.AVOID_LARGER,
                base_strength=1.0 + drive_params.get('fear', 0.5) * 0.5,
                threshold=0.3
            ),
            InstinctType.EXPLORE_NOVEL: Instinct(
                type=InstinctType.EXPLORE_NOVEL,
                base_strength=0.8 + drive_params.get('curiosity', 0.5),
                threshold=0.15,  # Low threshold - explore often
                cooldown_rate=0.3  # Can explore frequently
            ),
            InstinctType.SEEK_REST: Instinct(
                type=InstinctType.SEEK_REST,
                base_strength=1.0,
                threshold=0.5
            ),
            InstinctType.SEEK_SLEEP: Instinct(
                type=InstinctType.SEEK_SLEEP,
                base_strength=1.2,
                threshold=0.4
            ),
            InstinctType.SURFACE_BREATHE: Instinct(
                type=InstinctType.SURFACE_BREATHE,
                base_strength=2.0,  # Very strong - survival
                threshold=0.1  # Low threshold - react fast
            ),
            InstinctType.FREEZE: Instinct(
                type=InstinctType.FREEZE,
                base_strength=0.5,
                threshold=0.6
            ),
        }
        
        # Last computed outputs
        self.last_outputs: Dict[str, float] = {}
        
        # Reward tracking for learning
        self.last_active_instincts: List[InstinctType] = []
    
    def update(self, dt: float):
        """Update all instinct cooldowns."""
        for instinct in self.instincts.values():
            instinct.update(dt)
    
    def compute(self, sensory_data: Dict, drives: Dict[str, float]) -> Dict[str, float]:
        """
        Compute motor command suggestions from instincts.
        
        Args:
            sensory_data: Current sensory input
            drives: Current drive levels
            
        Returns:
            Dict of action_name -> strength (0-1)
        """
        outputs = {
            'move_left': 0.0,
            'move_right': 0.0,
            'jump': 0.0,
            'eat': 0.0,
            'drink': 0.0,
            'rest': 0.0,
            'sleep': 0.0,
            'flee': 0.0,
            'approach': 0.0,
            'surface': 0.0,  # Swim up to breathe
        }
        
        self.last_active_instincts = []
        
        # === FLEE THREAT ===
        flee_inst = self.instincts[InstinctType.FLEE_THREAT]
        threats = [c for c in sensory_data.get('visible_creatures', []) 
                   if c.get('is_threat', False)]
        if threats and flee_inst.can_activate():
            nearest_threat = min(threats, key=lambda t: t['dist'])
            threat_proximity = 1.0 / (1 + nearest_threat['dist'] / 50)
            
            if threat_proximity * drives.get('safety', 0) > flee_inst.threshold:
                strength = flee_inst.get_effective_strength() * threat_proximity
                # Move away from threat
                if nearest_threat['dx'] > 0:
                    outputs['move_left'] += strength
                else:
                    outputs['move_right'] += strength
                outputs['flee'] += strength
                flee_inst.activate()
                self.last_active_instincts.append(InstinctType.FLEE_THREAT)
        
        # === AVOID HAZARD ===
        hazard_inst = self.instincts[InstinctType.AVOID_HAZARD]
        hazards = sensory_data.get('visible_hazards', [])
        if hazards and hazard_inst.can_activate():
            nearest = min(hazards, key=lambda h: h['dist'])
            proximity = 1.0 / (1 + nearest['dist'] / 30)
            
            if proximity > hazard_inst.threshold:
                strength = hazard_inst.get_effective_strength() * proximity
                if nearest['dx'] > 0:
                    outputs['move_left'] += strength
                else:
                    outputs['move_right'] += strength
                # Jump if hazard is below
                if nearest['dy'] > 0:
                    outputs['jump'] += strength * 0.5
                hazard_inst.activate()
                self.last_active_instincts.append(InstinctType.AVOID_HAZARD)
        
        # === SEEK FOOD ===
        food_inst = self.instincts[InstinctType.SEEK_FOOD]
        hunger = drives.get('hunger', 0)
        foods = sensory_data.get('visible_food', [])
        if foods and hunger > food_inst.threshold and food_inst.can_activate():
            nearest = min(foods, key=lambda f: f['dist'])
            
            strength = food_inst.get_effective_strength() * hunger
            if nearest['dx'] > 5:
                outputs['move_right'] += strength * 0.7
            elif nearest['dx'] < -5:
                outputs['move_left'] += strength * 0.7
            
            # Eat if close
            if nearest['dist'] < 20:
                outputs['eat'] += strength
                
            outputs['approach'] += strength * 0.5
            food_inst.activate()
            self.last_active_instincts.append(InstinctType.SEEK_FOOD)
        
        # === SEEK WATER ===
        water_inst = self.instincts[InstinctType.SEEK_WATER]
        thirst = drives.get('thirst', 0)
        in_water = sensory_data.get('in_water', False)
        if thirst > water_inst.threshold and water_inst.can_activate():
            if in_water:
                outputs['drink'] += water_inst.get_effective_strength() * thirst
                water_inst.activate()
                self.last_active_instincts.append(InstinctType.SEEK_WATER)
            # TODO: Navigate toward water
        
        # === SEEK SHELTER ===
        shelter_inst = self.instincts[InstinctType.SEEK_SHELTER]
        safety_need = drives.get('safety', 0)
        in_shelter = sensory_data.get('in_shelter', False)
        if safety_need > shelter_inst.threshold and not in_shelter:
            # Random exploration toward shelter
            if np.random.random() < 0.1:
                if np.random.random() < 0.5:
                    outputs['move_left'] += shelter_inst.get_effective_strength() * 0.3
                else:
                    outputs['move_right'] += shelter_inst.get_effective_strength() * 0.3
            self.last_active_instincts.append(InstinctType.SEEK_SHELTER)
        
        # === SEEK WARMTH/COOLING ===
        warmth = drives.get('warmth', 0)
        cooling = drives.get('cooling', 0)
        if warmth > 0.3:
            # Move toward warmer areas (right side of world)
            outputs['move_right'] += warmth * 0.3
            self.last_active_instincts.append(InstinctType.SEEK_WARMTH)
        if cooling > 0.3:
            # Move toward cooler areas (left side)
            outputs['move_left'] += cooling * 0.3
            self.last_active_instincts.append(InstinctType.SEEK_COOLING)
        
        # === APPROACH MATE ===
        mate_inst = self.instincts[InstinctType.APPROACH_MATE]
        repro_drive = drives.get('reproduction', 0)
        same_species = [c for c in sensory_data.get('visible_creatures', [])
                        if c.get('is_same_species', False)]
        if same_species and repro_drive > mate_inst.threshold and mate_inst.can_activate():
            nearest = min(same_species, key=lambda c: c['dist'])
            strength = mate_inst.get_effective_strength() * repro_drive
            
            if nearest['dx'] > 5:
                outputs['move_right'] += strength * 0.5
            elif nearest['dx'] < -5:
                outputs['move_left'] += strength * 0.5
                
            outputs['approach'] += strength
            mate_inst.activate()
            self.last_active_instincts.append(InstinctType.APPROACH_MATE)
        
        # === EXPLORE ===
        # Exploration is CONTINUOUS - always contributes to movement when not overridden
        explore_inst = self.instincts[InstinctType.EXPLORE_NOVEL]
        exploration = drives.get('exploration', 0)
        # Exploration is the "default" behavior - always have some urge to move
        exploration = max(exploration, 0.35)  # Minimum exploration urge
        
        # ALWAYS contribute to movement (no cooldown check for basic exploration)
        if exploration > explore_inst.threshold:
            # Check for platform edges
            edge_left = sensory_data.get('edge_left', False)
            edge_right = sensory_data.get('edge_right', False)
            wall_left = sensory_data.get('wall_left', False)
            wall_right = sensory_data.get('wall_right', False)
            fall_distance = sensory_data.get('fall_distance', 0)
            
            # Edge-aware movement - increased multiplier for reliable movement
            strength = explore_inst.get_effective_strength() * exploration * 1.2
            
            # Determine safe directions
            # A drop is only "dangerous" if it's very high (> 100 pixels)
            dangerous_drop = 100
            left_dangerous = edge_left and fall_distance > dangerous_drop
            right_dangerous = edge_right and fall_distance > dangerous_drop
            
            # Choose direction
            if wall_left and wall_right:
                # Trapped between walls - jump
                outputs['jump'] += strength * 1.0
            elif left_dangerous and right_dangerous:
                # Both sides dangerous - must take a risk!
                # Or stay put if there's food nearby
                if sensory_data.get('nearest_food_distance', 1000) < 50:
                    pass  # Stay near food
                elif np.random.random() < 0.5:  # 50% chance to risk moving
                    outputs['jump'] += strength * 0.7  # Jump to get more distance
                    if np.random.random() < 0.5:
                        outputs['move_left'] += strength * 0.8
                    else:
                        outputs['move_right'] += strength * 0.8
            elif wall_left or left_dangerous:
                # Can't go left safely - go right
                outputs['move_right'] += strength
            elif wall_right or right_dangerous:
                # Can't go right safely - go left  
                outputs['move_left'] += strength
            else:
                # Free to move either way - use persistent direction
                # Initialize exploration direction if not set
                if not hasattr(self, '_explore_direction'):
                    self._explore_direction = 1 if np.random.random() < 0.5 else -1
                    self._explore_timer = 0
                
                # Change direction periodically (every 60-150 ticks = 2-5 seconds)
                self._explore_timer += 1
                if self._explore_timer > 60 + np.random.randint(0, 90):
                    self._explore_direction *= -1
                    self._explore_timer = 0
                
                if self._explore_direction > 0:
                    outputs['move_right'] += strength
                else:
                    outputs['move_left'] += strength
            
            # Jump over gaps or when stuck
            can_jump_gap = sensory_data.get('gap_ahead_left', False) or sensory_data.get('gap_ahead_right', False)
            if can_jump_gap and fall_distance < 100:
                # Jumpable gap ahead - try to jump it
                outputs['jump'] += strength * 1.0
            elif wall_left or wall_right:
                # Hit a wall - jump to escape
                outputs['jump'] += strength * 0.7
            elif np.random.random() < 0.15:
                # Occasional random jump
                outputs['jump'] += strength * 0.8
            
            explore_inst.activate()
            self.last_active_instincts.append(InstinctType.EXPLORE_NOVEL)
        
        # === SEEK REST ===
        rest_inst = self.instincts[InstinctType.SEEK_REST]
        rest_need = drives.get('rest', 0)
        if rest_need > rest_inst.threshold:
            outputs['rest'] += rest_inst.get_effective_strength() * rest_need
            # Reduce movement
            outputs['move_left'] *= 0.3
            outputs['move_right'] *= 0.3
            self.last_active_instincts.append(InstinctType.SEEK_REST)
        
        # === SEEK SLEEP ===
        sleep_inst = self.instincts[InstinctType.SEEK_SLEEP]
        sleep_need = drives.get('sleep', 0)
        if sleep_need > sleep_inst.threshold:
            outputs['sleep'] = sleep_inst.get_effective_strength() * sleep_need
            # Stop moving when trying to sleep
            outputs['move_left'] *= 0.1
            outputs['move_right'] *= 0.1
            outputs['jump'] *= 0.0
            self.last_active_instincts.append(InstinctType.SEEK_SLEEP)
        
        # === SURFACE TO BREATHE ===
        breathe_inst = self.instincts[InstinctType.SURFACE_BREATHE]
        breathe_need = drives.get('breathe', 0)
        in_water = sensory_data.get('in_water', False)
        if breathe_need > breathe_inst.threshold and in_water:
            # PANIC - jump/swim upward
            strength = breathe_inst.get_effective_strength() * breathe_need
            outputs['jump'] += strength  # Swim up!
            outputs['surface'] += strength  # Signal urgent surfacing
            # Override other movement - survival first
            outputs['move_left'] *= 0.2
            outputs['move_right'] *= 0.2
            outputs['eat'] = 0
            outputs['rest'] = 0
            outputs['sleep'] = 0
            self.last_active_instincts.append(InstinctType.SURFACE_BREATHE)
        
        # === BASELINE WANDERING ===
        # Even when content, creatures should move around occasionally
        # This provides baseline activity that the brain can modulate
        max_drive = max(outputs['move_left'], outputs['move_right'], 
                       outputs['flee'], outputs['approach'])
        if max_drive < 0.2:  # Not strongly motivated by anything
            # Continuous wandering - much higher chance for active movement
            if np.random.random() < 0.6:  # 60% chance per tick for base movement
                wander_strength = 0.5 + np.random.random() * 0.4
                # Use persistent direction (stored in self)
                if not hasattr(self, '_wander_direction'):
                    self._wander_direction = 1 if np.random.random() < 0.5 else -1
                    self._wander_timer = 0
                
                # Change direction occasionally
                self._wander_timer += 1
                if self._wander_timer > 30 + np.random.randint(0, 60):  # Every 1-3 seconds
                    self._wander_direction *= -1
                    self._wander_timer = 0
                
                if self._wander_direction > 0:
                    outputs['move_right'] += wander_strength
                else:
                    outputs['move_left'] += wander_strength
                    
                # Regular jump chance for exploring platforms
                if np.random.random() < 0.08:
                    outputs['jump'] += 0.6
        
        # === APPLY INSTINCT ARBITRATION ===
        # Mutual inhibition between competing instincts
        if self.last_active_instincts:
            # Build activation map from active instincts
            instinct_activations = {}
            for inst_type in self.last_active_instincts:
                # Use instinct's effective strength as activation
                instinct = self.instincts.get(inst_type)
                if instinct:
                    instinct_activations[inst_type] = instinct.get_effective_strength()
            
            # Run arbitration
            modulated = self.arbitrator.arbitrate(instinct_activations, drives)
            
            # Apply modulation factors to outputs
            # Map instinct types to their primary outputs
            output_map = {
                InstinctType.FLEE_THREAT: ['flee', 'move_left', 'move_right'],
                InstinctType.SEEK_FOOD: ['eat', 'approach'],
                InstinctType.SEEK_WATER: ['drink'],
                InstinctType.APPROACH_MATE: ['approach'],
                InstinctType.EXPLORE_NOVEL: ['move_left', 'move_right', 'jump'],
                InstinctType.SEEK_REST: ['rest'],
                InstinctType.SEEK_SLEEP: ['sleep'],
                InstinctType.SURFACE_BREATHE: ['jump', 'surface'],
            }
            
            for inst_type in self.last_active_instincts:
                if inst_type in modulated and inst_type in output_map:
                    original = instinct_activations.get(inst_type, 1.0)
                    new = modulated[inst_type]
                    if original > 0:
                        ratio = new / original
                        for output_name in output_map[inst_type]:
                            if output_name in outputs:
                                outputs[output_name] *= ratio
        
        # Normalize outputs
        for key in outputs:
            outputs[key] = np.clip(outputs[key], 0, 1)
        
        # CRITICAL: Resolve direction conflict - only one direction at a time
        left = outputs.get('move_left', 0)
        right = outputs.get('move_right', 0)
        if left > 0 and right > 0:
            # Pick stronger direction, zero out the other
            if left > right:
                outputs['move_right'] = 0
            else:
                outputs['move_left'] = 0
        
        self.last_outputs = outputs
        return outputs
    
    def apply_reward(self, reward: float):
        """
        Apply reward to recently active instincts.
        
        Positive reward strengthens, negative weakens.
        """
        for inst_type in self.last_active_instincts:
            self.instincts[inst_type].modulate(reward)
    
    def get_instinct_vector(self) -> np.ndarray:
        """
        Get instinct state as vector for brain input.
        
        Returns 16-element vector of instinct activations.
        """
        vec = np.zeros(16)
        for i, (inst_type, instinct) in enumerate(self.instincts.items()):
            if i >= 16:
                break
            # Encode whether instinct was recently active
            vec[i] = 1.0 if inst_type in self.last_active_instincts else 0.0
        return vec
    
    def to_dict(self) -> Dict:
        """Serialize instinct system."""
        return {
            inst_type.value: {
                'base_strength': inst.base_strength,
                'learned_weight': inst.learned_weight,
                'threshold': inst.threshold,
            }
            for inst_type, inst in self.instincts.items()
        }
    
    @classmethod
    def from_dict(cls, data: Dict, drive_params: Dict = None) -> 'InstinctSystem':
        """Deserialize instinct system."""
        system = cls(drive_params)
        for inst_type_str, inst_data in data.items():
            try:
                inst_type = InstinctType(inst_type_str)
                if inst_type in system.instincts:
                    system.instincts[inst_type].base_strength = inst_data['base_strength']
                    system.instincts[inst_type].learned_weight = inst_data['learned_weight']
                    system.instincts[inst_type].threshold = inst_data.get('threshold', 0.3)
            except (ValueError, KeyError):
                pass
        return system


# =============================================================================
# REWARD SYSTEM
# =============================================================================

class RewardSystem:
    """
    Computes reward signals from drive satisfaction.
    
    This closes the loop: drive -> action -> consequence -> reward -> learning
    """
    
    def __init__(self):
        self.prev_drives: Optional[Dict[str, float]] = None
        self.prev_health: float = 1.0
        self.prev_energy: float = 1.0
        
    def compute_reward(self, 
                       current_drives: Dict[str, float],
                       homeostasis: 'Homeostasis') -> float:
        """
        Compute reward based on drive satisfaction.
        
        Returns:
            Reward signal (-1 to 1)
        """
        reward = 0.0
        
        if self.prev_drives is None:
            self.prev_drives = current_drives.copy()
            self.prev_health = homeostasis.health
            self.prev_energy = homeostasis.energy
            return 0.0
        
        # Reward for reducing drives (satisfying needs)
        for drive_name, current_level in current_drives.items():
            prev_level = self.prev_drives.get(drive_name, current_level)
            delta = prev_level - current_level  # Positive if drive decreased
            reward += delta * 0.5
        
        # Strong reward for eating when hungry
        if current_drives.get('hunger', 0) < self.prev_drives.get('hunger', 0):
            reward += 0.3
        
        # Penalty for taking damage
        health_delta = homeostasis.health - self.prev_health
        if health_delta < 0:
            reward += health_delta * 2  # Negative reward for damage
        
        # Penalty for energy loss
        energy_delta = homeostasis.energy - self.prev_energy
        if energy_delta < -0.1:
            reward -= 0.1
        
        # Bonus for being in shelter when scared
        if current_drives.get('safety', 0) > 0.3:
            # This will be set by the sensory system
            pass
        
        # Pain penalty
        reward -= homeostasis.pain * 0.3
        
        # Update previous state
        self.prev_drives = current_drives.copy()
        self.prev_health = homeostasis.health
        self.prev_energy = homeostasis.energy
        
        return np.clip(reward, -1, 1)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'InstinctType',
    'Instinct',
    'InstinctSystem',
    'RewardSystem',
]
