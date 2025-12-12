"""
Creature Body - Physical body with physiology, drives, and phenotype

This module provides the physical embodiment for a brain:
- Body schema (size, color, limbs, speed from DNA)
- Homeostasis (energy, hunger, fatigue, temperature, pain)
- Physiological drives (hunger, thirst, rest, safety, reproduction)
- Sensorimotor interface (sensors -> brain -> motors)
- Metabolic costs and resource management

A Creature combines a Body with a Brain to create a living entity.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto


# =============================================================================
# PHENOTYPE - Visual appearance from DNA
# =============================================================================

@dataclass
class Phenotype:
    """
    Visual/physical characteristics derived from DNA/RNA.
    
    These determine how the creature looks and its physical capabilities.
    """
    # Size and shape
    size: float = 1.0               # Overall scale (0.5 = small, 2.0 = large)
    width: float = 16.0             # Body width in pixels
    height: float = 24.0            # Body height in pixels
    
    # Color (HSV)
    hue: float = 0.5                # 0-1, color wheel position
    saturation: float = 0.7         # 0-1, color intensity
    brightness: float = 0.8         # 0-1, light/dark
    
    # Pattern
    pattern_type: str = "solid"     # solid, stripes, spots, gradient
    pattern_color: float = 0.3      # Secondary color hue
    pattern_density: float = 0.5    # How much pattern coverage
    
    # Appendages
    limb_count: int = 4             # Number of limbs
    has_tail: bool = True
    tail_length: float = 0.5
    has_fins: bool = False
    has_wings: bool = False
    
    # Movement capabilities
    max_speed: float = 3.0          # Max horizontal speed
    jump_power: float = 8.0         # Jump velocity
    swim_speed: float = 2.0         # Speed in water
    
    # Sensory ranges
    vision_range: float = 100.0     # Pixels
    hearing_range: float = 150.0
    smell_range: float = 80.0
    
    # Metabolism
    metabolic_rate: float = 1.0     # Energy consumption multiplier
    max_energy: float = 100.0       # Maximum energy storage
    digestive_efficiency: float = 1.0 # Nutrient extraction rate
    heat_generation: float = 1.0    # Thermoregulation efficiency
    
    # Behavior (Genes)
    bravery: float = 0.5            # 0.0 = Coward, 1.0 = Brave (affects flee duration)
    
    # Motor Signature (UPGRADE 8: Per-Creature Movement Personality)
    stride_length: float = 1.0      # 0.5-1.5: Short steps vs long strides
    movement_jitter: float = 0.1    # 0.0-0.5: Smooth vs twitchy movement
    reaction_delay: float = 0.15    # 0.05-0.3s: Fast vs slow reactions
    
    def get_rgb(self) -> Tuple[int, int, int]:
        """Convert HSV to RGB for rendering."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(self.hue, self.saturation, self.brightness)
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def get_pattern_rgb(self) -> Tuple[int, int, int]:
        """Get secondary pattern color."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(self.pattern_color, self.saturation * 0.8, self.brightness * 0.7)
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def get_motor_capabilities(self) -> Dict[str, float]:
        """
        Get motor capability levels for brain input.
        
        Returns dict of capability_name -> strength (0-1).
        The brain can learn what actions are available.
        """
        return {
            'can_walk': min(1.0, self.limb_count / 2),     # 0, 0.5, 1+
            'can_run': 1.0 if self.limb_count >= 4 else 0.5 if self.limb_count >= 2 else 0.0,
            'can_jump': min(1.0, self.jump_power / 10) if self.limb_count >= 2 else 0.0,
            'can_swim': min(1.0, self.swim_speed / 3) if self.has_fins else self.swim_speed / 6,
            'can_fly': 1.0 if self.has_wings else 0.0,
            'strength': min(1.0, self.size / 1.5),         # Bigger = stronger
            'agility': min(1.0, 1.5 / max(0.5, self.size)), # Smaller = more agile
            'endurance': min(1.0, self.max_energy / 100),
        }
    
    @classmethod
    def from_body_params(cls, params: Dict[str, Any]) -> 'Phenotype':
        """Create phenotype from body params dict (from DNA development)."""
        return cls(
            size=params.get('size', 1.0),
            width=12 + 8 * params.get('size', 1.0),
            height=18 + 12 * params.get('size', 1.0),
            hue=params.get('color_hue', 0.5),
            saturation=params.get('color_saturation', 0.7),
            brightness=0.8,
            pattern_type=params.get('pattern_type', 'solid'),
            pattern_density=params.get('pattern', 0.5),
            limb_count=params.get('limb_count', 4),
            has_tail=True,
            tail_length=0.5,
            max_speed=2.0 + 3.0 * params.get('speed', 0.5),
            jump_power=6.0 + 6.0 * params.get('speed', 0.5),
            swim_speed=1.0 + 2.0 * params.get('speed', 0.5),
            vision_range=params.get('vision_range', 100),
            hearing_range=params.get('hearing_range', 80),
            smell_range=params.get('smell_range', 60),
            metabolic_rate=params.get('metabolic_rate', 1.0),
            max_energy=params.get('max_energy', 100),
            digestive_efficiency=params.get('digestive_efficiency', 1.0),
            heat_generation=params.get('heat_generation', 1.0),
        )


# =============================================================================
# HOMEOSTASIS - Internal state regulation
# =============================================================================

@dataclass
class Homeostasis:
    """
    Physiological state that must be regulated.
    
    All values are 0-1 representing percentage of optimal.
    Below thresholds trigger drives, extreme values cause damage.
    """
    # Energy/resources
    energy: float = 0.9             # Main energy reserve (start high)
    hydration: float = 0.8          # Water level
    nutrition: float = 0.8          # Food satiation (start well-fed)
    oxygen: float = 1.0             # Breath - depletes underwater
    
    # Physical state
    temperature: float = 0.5        # 0=freezing, 0.5=optimal, 1=overheating
    fatigue: float = 0.1            # 0=rested, 1=exhausted
    health: float = 1.0             # Overall health
    pain: float = 0.0               # Current pain level
    cortisol: float = 0.1           # Stress hormone (0=calm, 1=max stress)
    
    # Sleep state
    sleepiness: float = 0.0         # 0=alert, 1=must sleep
    is_sleeping: bool = False       # Currently asleep
    sleep_depth: float = 0.0        # 0=light, 1=deep (REM)
    dream_intensity: float = 0.0    # How vivid dreams are
    
    # Biological cycles
    age: float = 0.0                # 0=newborn, 1=max lifespan
    fertility: float = 0.0          # 0=infertile, 1=peak fertility
    pregnancy: float = 0.0          # 0=not pregnant, 1=about to give birth
    
    # Immune/healing
    immune_strength: float = 1.0    # Resistance to disease
    healing_rate: float = 0.001     # HP recovery per tick (very slow natural healing)
    
    # Thresholds for drive activation
    HUNGER_THRESHOLD = 0.7   # Start seeking food when nutrition < 70%
    THIRST_THRESHOLD = 0.7   # Start seeking water when hydration < 70%
    FATIGUE_THRESHOLD = 0.7
    SLEEPINESS_THRESHOLD = 0.6      # When creature wants to sleep
    OXYGEN_DANGER = 0.3             # Panic threshold
    COLD_THRESHOLD = 0.3
    HOT_THRESHOLD = 0.7
    PAIN_THRESHOLD = 0.3
    FERTILITY_THRESHOLD = 0.6
    
    def update(self, dt: float, metabolic_rate: float = 1.0, 
               ambient_temp: float = 20.0, activity_level: float = 0.5,
               digestive_efficiency: float = 1.0, heat_generation: float = 1.0):
        """
        Update homeostatic state over time.
        
        Args:
            dt: Time delta
            metabolic_rate: Creature's metabolism
            ambient_temp: Environmental temperature
            activity_level: How active the creature is (0-1)
            digestive_efficiency: Nutrient extraction rate
            heat_generation: Thermoregulation efficiency
        """
        # Metabolism slows during sleep (body conserves resources)
        sleep_multiplier = 0.3 if self.is_sleeping else 1.0
        
        # Energy drain (higher with activity and metabolism)
        # Base drain ~0.01/s at rest, ~0.02/s when active - need to eat every ~50-100s
        energy_drain = 0.01 * metabolic_rate * (0.5 + activity_level) * dt * sleep_multiplier
        self.energy = max(0, self.energy - energy_drain)
        
        # Hydration drain (~0.008/s) - slower during sleep, need water every ~60-120s
        hydration_drain = 0.008 * (1 + 0.5 * activity_level) * dt * sleep_multiplier
        self.hydration = max(0, self.hydration - hydration_drain)
        
        # Nutrition drains over time (stomach empties) - slower during sleep
        # Higher digestive efficiency = slower drain (more thorough digestion)
        nutrition_drain = 0.004 * dt * sleep_multiplier / max(0.1, digestive_efficiency)
        self.nutrition = max(0, self.nutrition - nutrition_drain)
        
        # Nutrition converts to energy (SLOWER than nutrition drain, so must keep eating)
        # Only converts when nutrition > 0.2 (need substantial food to digest)
        if self.nutrition > 0.2 and self.energy < 0.9:
            base_conversion = 0.001 * dt * digestive_efficiency  # Slower conversion
            conversion = min(base_conversion, self.nutrition - 0.2)
            self.nutrition -= conversion
            # More efficient digestion = more energy per unit of food
            energy_gain = conversion * 0.3 * digestive_efficiency
            self.energy = min(1, self.energy + energy_gain)
        
        # Temperature regulation
        optimal_temp = 37.0  # Body temp in Celsius
        
        # Endothermy: Generate internal heat
        internal_heat = 5.0 * heat_generation * metabolic_rate * (0.2 + 0.8 * activity_level)
        
        # Effective ambient temp (factoring in internal heat)
        effective_ambient = ambient_temp + internal_heat
        
        temp_diff = effective_ambient - optimal_temp
        
        # Body tries to regulate but can't fully
        # Larger bodies change temp slower? (ignoring size for now)
        temp_change = temp_diff * 0.001 * dt
        self.temperature = np.clip(0.5 + temp_change * 0.1, 0, 1)
        
        # Extreme temps drain energy (thermoregulation cost)
        if abs(self.temperature - 0.5) > 0.2:
            self.energy -= 0.001 * abs(self.temperature - 0.5) * dt
        
        # Fatigue increases with activity
        fatigue_gain = 0.001 * activity_level * dt
        fatigue_recovery = 0.0005 * (1 - activity_level) * dt
        self.fatigue = np.clip(self.fatigue + fatigue_gain - fatigue_recovery, 0, 1)
        
        # Sleepiness increases with fatigue and time awake
        if not self.is_sleeping:
            sleepiness_gain = 0.0003 * dt * (1 + self.fatigue)
            self.sleepiness = min(1.0, self.sleepiness + sleepiness_gain)
            
            # Auto-sleep when exhausted
            if self.sleepiness > 0.95 and self.fatigue > 0.8:
                self.start_sleep()
        else:
            # Sleeping - recover and dream
            self._sleep_update(dt)
        
        # Pain decays faster now - creatures recover from discomfort
        self.pain = max(0, self.pain - 0.05 * dt)
        
        # Cortisol decays slowly (stress hormone takes time to clear)
        self.cortisol = max(0.1, self.cortisol - 0.02 * dt)  # Never goes below baseline 0.1
        
        # Age (very slow)
        self.age += 0.00001 * dt
        
        # Fertility cycles (peaks, then declines)
        if self.age > 0.1 and self.age < 0.8:
            # Bell curve fertility
            peak_age = 0.4
            self.fertility = max(0, 1 - 4 * (self.age - peak_age)**2)
        else:
            self.fertility = 0
        
        # Starvation damage - die faster when starving
        # Internal discomfort (not flee-worthy, just motivation to eat)
        if self.energy < 0.05:
            self.health -= 0.15 * dt  # Critical starvation - die fast
            self.pain = min(0.3, self.pain + 0.05 * dt)  # Capped mild pain
            self.cortisol = min(1.0, self.cortisol + 0.2 * dt)  # Major stress
        elif self.energy < 0.15:
            self.health -= 0.08 * dt  # Moderate starvation damage
            self.pain = min(0.2, self.pain + 0.02 * dt)  # Very mild
            self.cortisol = min(0.8, self.cortisol + 0.1 * dt)  # Stress buildup
        elif self.energy < 0.3:
            # Warning zone - stress but no damage yet
            self.cortisol = min(0.5, self.cortisol + 0.03 * dt)
        
        # Dehydration damage - more severe than starvation
        if self.hydration < 0.05:
            self.health -= 0.2 * dt  # Critical dehydration
            self.pain = min(0.3, self.pain + 0.05 * dt)
            self.cortisol = min(1.0, self.cortisol + 0.25 * dt)  # Severe stress
        elif self.hydration < 0.15:
            self.health -= 0.1 * dt
            self.pain = min(0.2, self.pain + 0.02 * dt)
            self.cortisol = min(0.8, self.cortisol + 0.12 * dt)
        elif self.hydration < 0.3:
            self.cortisol = min(0.5, self.cortisol + 0.04 * dt)
        
        # Suffocation damage (drowning)
        if self.oxygen < 0.1:
            self.health -= 0.1 * dt
            self.pain = min(1.0, self.pain + 0.15 * dt)
        
        # Additional nutrition-based starvation (when stomach is empty)
        if self.nutrition < 0.05 and self.energy < 0.3:
            # No food in stomach AND low energy = starving
            self.health -= 0.1 * dt
            self.cortisol = min(1.0, self.cortisol + 0.15 * dt)
        
        # Healing ONLY happens when well-fed and hydrated (after damage checks)
        # Cannot heal while starving/dehydrated
        heal_mult = 2.0 if self.is_sleeping else 1.0
        can_heal = (self.energy > 0.4 and self.nutrition > 0.2 and 
                    self.hydration > 0.3 and self.fatigue < 0.8 and
                    self.health < 1.0)
        if can_heal:
            heal_amount = self.healing_rate * self.immune_strength * dt * heal_mult
            self.health = min(1.0, self.health + heal_amount)
            self.energy -= heal_amount * 0.5  # Healing costs energy
    
    def _sleep_update(self, dt: float):
        """Update sleep state."""
        # Recover fatigue and sleepiness
        self.fatigue = max(0, self.fatigue - 0.003 * dt)
        self.sleepiness = max(0, self.sleepiness - 0.002 * dt)
        
        # Sleep depth cycles (REM cycles)
        cycle_period = 50.0  # Time units per sleep cycle
        self.sleep_depth = 0.5 + 0.5 * np.sin(self.age * 1000 / cycle_period * np.pi)
        
        # Dream intensity peaks during deep sleep
        self.dream_intensity = self.sleep_depth * (0.5 + 0.5 * np.random.random())
        
        # Wake up when recovered
        if self.sleepiness < 0.1 and self.fatigue < 0.2:
            self.wake_up()
    
    def start_sleep(self):
        """Enter sleep state."""
        self.is_sleeping = True
        self.sleep_depth = 0.0
        self.dream_intensity = 0.0
    
    def wake_up(self):
        """Exit sleep state."""
        self.is_sleeping = False
        self.sleep_depth = 0.0
        self.dream_intensity = 0.0
    
    def update_oxygen(self, dt: float, in_water: bool, head_above_water: bool = False):
        """Update oxygen levels based on environment."""
        if in_water and not head_above_water:
            # Drowning - lose oxygen
            self.oxygen = max(0, self.oxygen - 0.02 * dt)
        else:
            # Recover oxygen quickly when above water
            self.oxygen = min(1.0, self.oxygen + 0.1 * dt)
    
    def apply_damage(self, amount: float, pain_factor: float = 1.0):
        """Apply damage to creature."""
        self.health = max(0, self.health - amount)
        self.pain = min(1, self.pain + amount * pain_factor)
        # Pain wakes you up
        if self.is_sleeping and self.pain > 0.3:
            self.wake_up()
    
    def eat(self, nutrition_value: float):
        """Consume food."""
        self.nutrition = min(1, self.nutrition + nutrition_value / 100)
    
    def drink(self, amount: float = 0.3):
        """Drink water."""
        self.hydration = min(1, self.hydration + amount)
    
    def rest(self, dt: float):
        """Rest to recover fatigue."""
        self.fatigue = max(0, self.fatigue - 0.01 * dt)
    
    def is_alive(self) -> bool:
        """Check if creature is still alive."""
        return self.health > 0
    
    @property
    def hunger(self) -> float:
        """Hunger level (0 = not hungry, 1 = starving). Inverse of nutrition."""
        if self.nutrition < self.HUNGER_THRESHOLD:
            return 1 - (self.nutrition / self.HUNGER_THRESHOLD)
        return 0
    
    @property
    def thirst(self) -> float:
        """Thirst level (0 = not thirsty, 1 = dehydrated). Inverse of hydration."""
        if self.hydration < self.THIRST_THRESHOLD:
            return 1 - (self.hydration / self.THIRST_THRESHOLD)
        return 0
    
    def get_drive_levels(self) -> Dict[str, float]:
        """
        Calculate drive intensities based on homeostatic state.
        
        Returns dict of drive_name -> intensity (0-1)
        """
        drives = {}
        
        # Hunger: increases as nutrition decreases
        drives['hunger'] = self.hunger
        
        # Thirst
        drives['thirst'] = self.thirst
        
        # Rest (fatigue drive)
        if self.fatigue > self.FATIGUE_THRESHOLD:
            drives['rest'] = (self.fatigue - self.FATIGUE_THRESHOLD) / (1 - self.FATIGUE_THRESHOLD)
        else:
            drives['rest'] = 0
        
        # Thermoregulation
        if self.temperature < self.COLD_THRESHOLD:
            drives['warmth'] = 1 - (self.temperature / self.COLD_THRESHOLD)
            drives['cooling'] = 0
        elif self.temperature > self.HOT_THRESHOLD:
            drives['warmth'] = 0
            drives['cooling'] = (self.temperature - self.HOT_THRESHOLD) / (1 - self.HOT_THRESHOLD)
        else:
            drives['warmth'] = 0
            drives['cooling'] = 0
        
        # Safety (pain avoidance)
        drives['safety'] = self.pain
        
        # Oxygen/breathing (panic when drowning)
        if self.oxygen < self.OXYGEN_DANGER:
            drives['breathe'] = 1 - (self.oxygen / self.OXYGEN_DANGER)
        else:
            drives['breathe'] = 0
        
        # Sleep drive
        if self.sleepiness > self.SLEEPINESS_THRESHOLD:
            drives['sleep'] = (self.sleepiness - self.SLEEPINESS_THRESHOLD) / (1 - self.SLEEPINESS_THRESHOLD)
        else:
            drives['sleep'] = 0
        
        # Reproduction
        if self.fertility > self.FERTILITY_THRESHOLD and self.energy > 0.5 and not self.is_sleeping:
            drives['reproduction'] = self.fertility
        else:
            drives['reproduction'] = 0
        
        # Exploration (when needs are met and not sleeping)
        if self.is_sleeping:
            drives['exploration'] = 0
        else:
            needs_met = 1 - max(drives.get('hunger', 0), drives.get('thirst', 0), 
                               drives.get('rest', 0), drives.get('safety', 0),
                               drives.get('breathe', 0), drives.get('sleep', 0))
            drives['exploration'] = needs_met * 0.5
        
        return drives


# =============================================================================
# MOTOR SYSTEM - Movement and actions
# =============================================================================

class Action(Enum):
    """Possible creature actions."""
    IDLE = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    JUMP = auto()
    EAT = auto()
    DRINK = auto()
    REST = auto()
    ATTACK = auto()
    FLEE = auto()
    MATE = auto()
    CALL = auto()          # Vocalization
    INTERACT = auto()      # Generic interaction
    SPEAK = auto()         # Language


@dataclass
class MotorState:
    """Current motor/movement state."""
    # Position and velocity
    x: float = 400.0
    y: float = 200.0
    vx: float = 0.0
    vy: float = 0.0
    
    # Movement state
    facing_right: bool = True
    on_ground: bool = True
    in_water: bool = False
    near_water: bool = False    # Can drink without being submerged
    in_shelter: bool = False
    
    # Action state
    current_action: Action = Action.IDLE
    action_timer: float = 0.0
    
    # Animation
    animation_frame: int = 0
    animation_timer: float = 0.0


# =============================================================================
# CREATURE BODY - Complete physical entity
# =============================================================================

class CreatureBody:
    """
    Complete physical body for a creature.
    
    Combines:
    - Phenotype (appearance)
    - Homeostasis (internal state)
    - Motor system (movement)
    - Sensors (perception)
    """
    
    def __init__(self, 
                 phenotype: Optional[Phenotype] = None,
                 x: float = 400.0,
                 y: float = 200.0):
        """
        Initialize creature body.
        
        Args:
            phenotype: Visual/physical characteristics
            x, y: Starting position
        """
        self.phenotype = phenotype or Phenotype()
        self.homeostasis = Homeostasis()
        self.motor = MotorState(x=x, y=y)
        
        # Unique ID
        self.id = np.random.randint(0, 2**31)
        
        # Lifetime stats
        self.lifetime = 0.0
        self.food_eaten = 0
        self.distance_traveled = 0.0
        self.offspring_count = 0
        
    def update(self, dt: float, world, brain_output: Optional[Dict] = None):
        """
        Update body state.
        
        Args:
            dt: Time delta
            world: World object for physics/sensing
            brain_output: Optional motor commands from brain
        """
        self.lifetime += dt
        
        # Get world conditions
        ambient_temp = world.get_temperature(self.motor.x, self.motor.y)
        
        # Calculate activity level from motor state
        activity = abs(self.motor.vx) / self.phenotype.max_speed
        
        # Update homeostasis
        # Update homeostasis
        self.homeostasis.update(dt, self.phenotype.metabolic_rate, 
                               ambient_temp, activity,
                               digestive_efficiency=self.phenotype.digestive_efficiency,
                               heat_generation=self.phenotype.heat_generation)
        
        # Check hazards - check tile creature is standing on (y+5 to catch the ground tile)
        # Check hazards/anomalies
        is_hazard, damage, haz_type = world.is_hazard(self.motor.x, self.motor.y + 5)
        if is_hazard:
            if haz_type == "healing":
                # Healing Spring: Heals HP, restores Hydration, but makes you Sleepy
                self.homeostasis.health = min(1.0, self.homeostasis.health + 0.05 * dt)
                self.homeostasis.hydration = min(1.0, self.homeostasis.hydration + 0.05 * dt)
                self.homeostasis.sleepiness = min(1.0, self.homeostasis.sleepiness + 0.1 * dt)
                self.homeostasis.energy = max(0.0, self.homeostasis.energy - 0.01 * dt) # Relaxation uses energy?
                
            elif haz_type == "radiation":
                # Radiation: Low damage, random mutation chance (simulated by color shift for now)
                self.homeostasis.apply_damage(damage * dt * 0.1)
                if np.random.random() < 0.1 * dt:
                     # Mutate color slightly
                     self.phenotype.hue = (self.phenotype.hue + np.random.normal(0, 0.05)) % 1.0
                     
            else:
                # Standard fire/lava
                self.homeostasis.apply_damage(damage * dt * 0.1)
        
        # Process brain output if available
        if brain_output:
            self._process_brain_output(brain_output, world)
        
        # Apply physics
        new_x, new_y, new_vx, new_vy, on_ground = world.apply_physics(
            self.motor.x, self.motor.y,
            self.motor.vx, self.motor.vy,
            dt
        )
        
        # Track distance
        self.distance_traveled += abs(new_x - self.motor.x)
        
        # Update motor state
        self.motor.x = new_x
        self.motor.y = new_y
        self.motor.vx = new_vx
        self.motor.vy = new_vy
        self.motor.on_ground = on_ground
        self.motor.in_water = world.is_water(new_x, new_y)
        
        # Check if near water (for drinking without drowning)
        self.motor.near_water = world.is_water(new_x, new_y + 10) or world.is_water(new_x + 10, new_y) or world.is_water(new_x - 10, new_y)
        
        # Update oxygen based on water state
        # Head above water if in shallow water or at surface
        head_above = not self.motor.in_water or new_vy < 0  # Swimming up
        self.homeostasis.update_oxygen(dt, self.motor.in_water, head_above)
        
        # Swimming in water - can move but slower
        if self.motor.in_water:
            # Water buoyancy - counteract some gravity  
            if self.motor.vy > 0:
                self.motor.vy *= 0.8
            # Can "jump" (swim up) while in water
            self.motor.on_ground = True  # Allow swimming movement
        
        # Friction (reduced to allow momentum)
        if on_ground and not self.motor.in_water:
            self.motor.vx *= 0.95  # Was 0.85, now creatures keep moving
        elif self.motor.in_water:
            self.motor.vx *= 0.96  # Less friction in water
        
        # Sleeping creatures don't animate
        if self.homeostasis.is_sleeping:
            return
        
        # Animation
        self.motor.animation_timer += dt
        if self.motor.animation_timer > 0.1:
            self.motor.animation_timer = 0
            self.motor.animation_frame = (self.motor.animation_frame + 1) % 4
    
    def _process_brain_output(self, output: Dict, world):
        """Process motor commands from brain, constrained by phenotype."""
        # Sleeping creatures can't move (except wake up from danger)
        if self.homeostasis.is_sleeping:
            if output.get('wake', 0) > 0.5 or self.homeostasis.pain > 0.3:
                self.homeostasis.wake_up()
            return
        
        # === MOTOR CONSTRAINTS FROM PHENOTYPE ===
        # Gate motor outputs by physical capability
        can_walk = self.phenotype.limb_count >= 2  # Need at least 2 limbs to walk
        can_jump = self.phenotype.limb_count >= 2 and self.phenotype.jump_power > 0
        can_swim = self.phenotype.has_fins or self.phenotype.swim_speed > 0
        can_fly = self.phenotype.has_wings
        
        # Size affects movement capability
        size_factor = 1.0 / max(0.5, self.phenotype.size)  # Bigger = slower acceleration
        
        # Fatigue reduces effectiveness
        fatigue_factor = 1.0 - self.homeostasis.fatigue * 0.5
        
        # Movement (requires limbs)
        # Use 0.05 threshold - state machine outputs clean signals
        # Direct velocity setting - state machine already handles smoothing
        if can_walk:
            target_speed = self.phenotype.max_speed * size_factor * fatigue_factor
            
            if output.get('move_left', 0) > 0.05:
                # Direct velocity - state machine already smoothed this
                self.motor.vx = -target_speed * output['move_left']
                self.motor.facing_right = False
                
            elif output.get('move_right', 0) > 0.05:
                # Direct velocity - state machine already smoothed this
                self.motor.vx = target_speed * output['move_right']
                self.motor.facing_right = True
        else:
            # Limbless creatures can only wiggle/slither slowly
            if output.get('move_left', 0) > 0.2:
                self.motor.vx = -0.5 * output['move_left']
                self.motor.facing_right = False
            if output.get('move_right', 0) > 0.2:
                self.motor.vx = 0.5 * output['move_right']
                self.motor.facing_right = True
        
        # Jump (requires legs AND being on ground/in water)
        # Use 0.3 threshold - jump should require slightly stronger intent
        if can_jump and output.get('jump', 0) > 0.3 and (self.motor.on_ground or self.motor.in_water):
            jump_power = self.phenotype.jump_power * fatigue_factor
            if self.motor.in_water:
                if can_swim:
                    jump_power *= 0.8  # Fins help in water
                else:
                    jump_power *= 0.4  # Struggle without fins
            self.motor.vy = -jump_power
        elif not can_jump and output.get('jump', 0) > 0.3:
            # No legs - can't jump, but can push off ground weakly
            if self.motor.on_ground:
                self.motor.vy = -2.0  # Tiny hop
        
        # Swimming (in water, better with fins)
        if self.motor.in_water and output.get('surface', 0) > 0.5:
            swim_power = self.phenotype.swim_speed
            if not can_swim:
                swim_power *= 0.3  # Struggle to surface
            self.motor.vy = -swim_power * 2  # Swim up
        
        # Eating
        if output.get('eat', 0) > 0.5:
            nearby_food = world.find_food_nearby(
                self.motor.x, self.motor.y, 
                self.phenotype.size * 20
            )
            if nearby_food:
                food = nearby_food[0]
                nutrition = world.eat_food(food, 0.3)
                self.homeostasis.eat(nutrition)
                self.food_eaten += 1
        
        # Drinking (near water OR in water)
        # Added Instincual Drinking override if thirsty and near water
        want_to_drink = output.get('drink', 0) > 0.5
        instinct_drink = self.homeostasis.hydration < 0.4 and (self.motor.near_water or self.motor.in_water)
        
        if (want_to_drink or instinct_drink) and (self.motor.in_water or self.motor.near_water):
            self.homeostasis.drink(0.3)
        
        # Sleep action
        if output.get('sleep', 0) > 0.5 and self.homeostasis.sleepiness > 0.4:
            self.homeostasis.start_sleep()
    
    def get_sensory_input(self, world, other_creatures: List['CreatureBody'] = None) -> Dict:
        """
        Gather sensory information from environment.
        
        Returns dict suitable for brain input.
        """
        # Get world sensory data
        data = world.get_sensory_data(
            self.motor.x, self.motor.y,
            self.phenotype.vision_range,
            self.phenotype.hearing_range,
            self.phenotype.smell_range
        )
        
        # Add internal state
        data['internal'] = {
            'energy': self.homeostasis.energy,
            'nutrition': self.homeostasis.nutrition,
            'hydration': self.homeostasis.hydration,
            'fatigue': self.homeostasis.fatigue,
            'pain': self.homeostasis.pain,
            'cortisol': self.homeostasis.cortisol,
            'temperature': self.homeostasis.temperature,
            'health': self.homeostasis.health,
            'fertility': self.homeostasis.fertility,
        }
        
        # Add drives
        data['drives'] = self.homeostasis.get_drive_levels()
        
        # Add motor state
        data['motor'] = {
            'on_ground': self.motor.on_ground,
            'in_water': self.motor.in_water,
            'in_shelter': self.motor.in_shelter,
            'vx': self.motor.vx,
            'vy': self.motor.vy,
        }
        
        # Add world bounds
        data['world_bounds'] = {
            'min_x': getattr(world, 'min_x', None),
            'max_x': getattr(world, 'max_x', None)
        }
        
        # Add visible creatures
        if other_creatures:
            for other in other_creatures:
                if other.id == self.id:
                    continue
                dx = other.motor.x - self.motor.x
                dy = other.motor.y - self.motor.y
                dist = np.sqrt(dx**2 + dy**2)
                if dist <= self.phenotype.vision_range:
                    is_threat = other.phenotype.size > self.phenotype.size * 1.2
                    data['visible_creatures'].append({
                        'dx': dx,
                        'dy': dy,
                        'dist': dist,
                        'size': other.phenotype.size,
                        'is_threat': is_threat,
                        'is_same_species': abs(other.phenotype.hue - self.phenotype.hue) < 0.1
                    })
        
        return data
    
    def to_brain_input(self, sensory_data: Dict) -> np.ndarray:
        """
        Convert sensory data to brain input vector.
        
        Creates a fixed-size array from variable sensory information.
        """
        # Fixed size: 64 elements
        vec = np.zeros(64)
        
        # Internal state (0-7)
        internal = sensory_data.get('internal', {})
        vec[0] = internal.get('energy', 0.5)
        vec[1] = internal.get('nutrition', 0.5)
        vec[2] = internal.get('hydration', 0.5)
        vec[3] = internal.get('fatigue', 0.0)
        vec[4] = internal.get('pain', 0.0)
        vec[5] = internal.get('temperature', 0.5)
        vec[6] = internal.get('health', 1.0)
        vec[7] = internal.get('fertility', 0.0)
        
        # Drives (8-17)
        drives = sensory_data.get('drives', {})
        vec[8] = drives.get('hunger', 0.0)
        vec[9] = drives.get('thirst', 0.0)
        vec[10] = drives.get('rest', 0.0)
        vec[11] = drives.get('warmth', 0.0)
        vec[12] = drives.get('cooling', 0.0)
        vec[13] = drives.get('safety', 0.0)
        vec[14] = drives.get('reproduction', 0.0)
        vec[15] = drives.get('exploration', 0.0)
        
        # Environment (18-25)
        vec[18] = sensory_data.get('light', 0.5)
        vec[19] = sensory_data.get('temperature', 20) / 40  # Normalize
        vec[20] = 1.0 if sensory_data.get('on_ground', False) else 0.0
        vec[21] = 1.0 if sensory_data.get('in_water', False) else 0.0
        vec[22] = 1.0 if sensory_data.get('in_shelter', False) else 0.0
        vec[23] = sensory_data.get('time_of_day', 0.5)
        
        # Wall Sensors (24-25)
        # Proximity: 0=far (>=200px), 1=touching
        wall_dist = sensory_data.get('nearest_wall_dist', 200)
        proximity = max(0, 1.0 - (wall_dist / 200.0))
        vec[24] = proximity
        vec[25] = sensory_data.get('nearest_wall_dir', 0.0)
        
        # Motor state (26-29)
        motor = sensory_data.get('motor', {})
        vec[26] = motor.get('vx', 0) / 5  # Normalize
        vec[27] = motor.get('vy', 0) / 10
        
        # Nearest food (30-35)
        foods = sensory_data.get('visible_food', [])
        if foods:
            nearest = min(foods, key=lambda f: f['dist'])
            vec[30] = nearest['dx'] / 100  # Normalized direction
            vec[31] = nearest['dy'] / 100
            vec[32] = 1.0 / (1 + nearest['dist'] / 50)  # Proximity
            vec[33] = nearest['nutrition'] / 100
            vec[34] = 1.0 if nearest['type'] == 'plant' else 0.0
        
        # Nearest hazard (36-40)
        hazards = sensory_data.get('visible_hazards', [])
        if hazards:
            nearest = min(hazards, key=lambda h: h['dist'])
            vec[36] = nearest['dx'] / 100
            vec[37] = nearest['dy'] / 100
            vec[38] = 1.0 / (1 + nearest['dist'] / 50)
            vec[39] = nearest['damage'] / 20
        
        # Nearest creature (41-47)
        creatures = sensory_data.get('visible_creatures', [])
        if creatures:
            nearest = min(creatures, key=lambda c: c['dist'])
            vec[41] = nearest['dx'] / 100
            vec[42] = nearest['dy'] / 100
            vec[43] = 1.0 / (1 + nearest['dist'] / 50)
            vec[44] = nearest['size']
            vec[45] = 1.0 if nearest['is_threat'] else 0.0
            vec[46] = 1.0 if nearest['is_same_species'] else 0.0
        
        return vec
    
    def is_alive(self) -> bool:
        """Check if creature is alive."""
        return self.homeostasis.is_alive()
    
    def get_collision_rect(self) -> Tuple[float, float, float, float]:
        """Get bounding box for collision detection."""
        hw = self.phenotype.width / 2
        hh = self.phenotype.height / 2
        return (self.motor.x - hw, self.motor.y - hh, 
                self.phenotype.width, self.phenotype.height)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'Phenotype',
    'Homeostasis', 
    'Action',
    'MotorState',
    'CreatureBody',
]
