"""
World Simulation - 2D Environment for Creatures

A tile-based world with:
- Terrain types (ground, water, hazard, shelter)
- Food spawners
- Temperature zones
- Day/night cycle
- Physics (gravity, collision)

The world provides sensory data to creatures and receives motor commands.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum, auto
import json

try:
    from brain.quadtree import QuadTree, Rect, Point
    _HAS_QUADTREE = True
except ImportError:
    _HAS_QUADTREE = False


# =============================================================================
# WORLD CONSTANTS
# =============================================================================

class TileType(Enum):
    """Types of terrain tiles."""
    EMPTY = 0           # Air/void
    GROUND = 1          # Solid ground
    WATER = 2           # Water (swimming/drowning)
    HAZARD = 3          # Damage zone (lava, spikes)
    SHELTER = 4         # Safe zone (reduced predation)
    FOOD_PLANT = 5      # Edible vegetation
    FOOD_MEAT = 6       # Edible meat/prey
    NEST = 7            # Breeding spot
    STONE = 8           # Constructed wall



class Weather(Enum):
    """Weather conditions."""
    CLEAR = 0
    RAIN = 1
    STORM = 2
    SNOW = 3
    HEAT_WAVE = 4


class FoodType(Enum):
    """Types of food with different effects."""
    PLANT = "plant"
    MEAT = "meat"
    SWEET_BERRY = "sweet_berry"    # Reward (Dopamine)
    BITTER_BERRY = "bitter_berry"  # Aversion (No nutritional value)
    POISON_BERRY = "poison_berry"  # Punishment (Pain/Cortisol)


class ToolType(Enum):
    """Types of tools creatures can pick up and use."""
    STICK = "stick"     # Long reach, poking, digging
    STONE = "stone"     # Heavy, throwing, breaking
    LEAF = "leaf"       # Carrying water, cover
    SHELL = "shell"     # Scooping, protection
    BONE = "bone"       # Digging, weapon
    # System 6: Constructed Tools
    NEST = "nest"       # Resting spot (2 Sticks)
    HAMMER = "hammer"   # Stick + Stone
    SHARP_ROCK = "sharp_rock" # Stone + Stone
    SPEAR = "spear"     # Stick + Sharp Rock


@dataclass
class ToolObject:
    """A tool item in the world that creatures can pick up and use."""
    x: float
    y: float
    tool_type: ToolType = ToolType.STICK
    weight: float = 0.5         # Affects carry speed (0-1)
    durability: float = 1.0     # How much use left (0-1)
    held_by: Optional[str] = None  # Creature ID if held
    
    # Tool properties (what the tool is good at)
    reach: float = 0.0          # Extra reach distance (sticks)
    damage: float = 0.0         # Damage when thrown (stones)
    carry_capacity: float = 0.0 # Can carry stuff (leaves, shells)
    
    # Throwing physics
    throwing: bool = False      # Currently in flight
    throw_vx: float = 0.0       # Velocity while thrown
    throw_vy: float = 0.0
    throw_power: float = 1.0    # Power of throw
    thrower_id: Optional[str] = None  # Who threw it (avoid self-damage)
    
    # Throwing physics
    throwing: bool = False      # Currently in flight
    throw_vx: float = 0.0       # Velocity while thrown
    throw_vy: float = 0.0
    throw_power: float = 1.0    # Power of throw
    
    def __post_init__(self):
        """Set tool properties based on type."""
        if self.tool_type == ToolType.STICK:
            self.reach = 50.0
            self.weight = 0.3
            self.damage = 5.0
        elif self.tool_type == ToolType.STONE:
            self.reach = 0.0
            self.weight = 0.7
            self.damage = 15.0
        elif self.tool_type == ToolType.LEAF:
            self.reach = 0.0
            self.weight = 0.1
            self.carry_capacity = 0.3
        elif self.tool_type == ToolType.SHELL:
            self.reach = 10.0
            self.weight = 0.4
            self.carry_capacity = 0.5
            self.damage = 8.0
        elif self.tool_type == ToolType.BONE:
            self.reach = 30.0
            self.weight = 0.5
            self.damage = 12.0
        # System 6 Properties
        elif self.tool_type == ToolType.NEST:
            self.reach = 0.0
            self.weight = 0.8
            self.carry_capacity = 0.8
            self.damage = 0.0
        elif self.tool_type == ToolType.HAMMER:
            self.reach = 20.0
            self.weight = 0.6
            self.damage = 25.0
        elif self.tool_type == ToolType.SHARP_ROCK:
            self.reach = 0.0
            self.weight = 0.5
            self.damage = 30.0
        elif self.tool_type == ToolType.SPEAR:
            self.reach = 90.0
            self.weight = 0.4
            self.damage = 20.0
    
    def use(self, dt: float = 0.1) -> bool:
        """Use the tool, reducing durability. Returns True if still usable."""
        self.durability -= 0.05 * dt
        return self.durability > 0
    
    def is_available(self) -> bool:
        """Check if tool is available to pick up."""
        return self.held_by is None and self.durability > 0 and not getattr(self, 'throwing', False)
    
    def get_hold_position(self, creature_x: float, creature_y: float, 
                         creature_width: float, facing_right: bool) -> tuple:
        """Get position where tool should be displayed when held."""
        # Attach to front of creature (ahead of center)
        offset_x = creature_width * 0.6 if facing_right else -creature_width * 0.6
        offset_y = -5  # Slightly above center
        return (creature_x + offset_x, creature_y + offset_y)


@dataclass
class FoodSource:
    """A food item in the world."""
    x: float
    y: float
    nutrition: float = 0.5          # Energy provided (0-1 scale)
    type: FoodType = FoodType.PLANT # "plant", "meat", etc.
    spoilage_rate: float = 0.001    # How fast it decays
    remaining: float = 1.0          # 0-1, how much left
    
    # Cultivation properties
    is_planted: bool = False
    growth_stage: float = 1.0       # 0.0=seed, 1.0=grown
    max_nutrition: float = 0.5      # Target nutrition when grown
    
    def decay(self, dt: float = 1.0):
        """Food decays over time."""
        # Meat spoils faster
        rate = self.spoilage_rate * 2.0 if self.type == FoodType.MEAT else self.spoilage_rate
        self.remaining -= rate * dt
        return self.remaining > 0

    def update(self, dt: float) -> bool:
        """Update food state (growth and decay)."""
        # Growth logic
        if self.is_planted and self.growth_stage < 1.0:
            # Growth rate: 100 seconds to mature
            growth_rate = 0.5 * dt  # Tuning: 0.01 = 100s. 0.1 = 10s.
            self.growth_stage += growth_rate * 0.02 # Slower
            
            if self.growth_stage >= 1.0:
                self.growth_stage = 1.0
                
            # Nutrition scales with growth
            self.nutrition = self.max_nutrition * self.growth_stage
            
            # Plants growing don't decay
            return True
            
        # Normal decay
        return self.decay(dt)


@dataclass 
class Hazard:
    """A hazardous zone."""
    x: float
    y: float
    width: float
    height: float
    damage: float = 10.0            # Damage per second
    type: str = "fire"              # fire, poison, cold, etc.


@dataclass
class Shelter:
    """A safe zone."""
    x: float
    y: float
    width: float
    height: float
    temperature_mod: float = 0.0    # Temperature modification
    safety: float = 0.8             # Predation reduction


@dataclass
class WaterSource:
    """A water pool object at a specific world position (not tile-based)."""
    x: float            # Center X in world coordinates
    y: float            # Center Y in world coordinates
    width: float = 150.0   # Width in pixels
    height: float = 50.0   # Height in pixels (visual)
    
    def contains_point(self, px: float, py: float, radius: float = 0.0) -> bool:
        """Check if a point (with optional radius) overlaps this water source."""
        return (abs(px - self.x) < (self.width / 2 + radius) and
                abs(py - self.y) < (self.height / 2 + radius))


# =============================================================================
# WORLD CLASS
# =============================================================================

class World:
    """
    2D tile-based world simulation.
    
    Coordinate system:
    - (0,0) is top-left
    - x increases right
    - y increases down
    - Gravity pulls down (+y)
    """
    
    def __init__(self, 
                 width: int = 800,
                 height: int = 400,
                 tile_size: int = 16,
                 gravity: float = 400.0):
        """
        Initialize world.
        
        Args:
            width: World width in pixels
            height: World height in pixels
            tile_size: Size of each tile in pixels
            gravity: Gravity strength (pixels/second²)
        """
        self.width = width
        self.height = height
        
        # Horizontal Boundaries (None = Infinite)
        self.min_x = None
        self.max_x = None
        self.zone_width = None  # Width of each zone (set by GUI from background image width)
        
        self.tile_size = tile_size
        self.gravity = gravity
        
        # Ensure minimum world size
        min_tiles = 15
        self.width = max(width, min_tiles * tile_size)
        self.height = max(height, min_tiles * tile_size)
        
        # Tile grid
        self.tiles_x = self.width // tile_size
        self.tiles_y = self.height // tile_size
        self.tiles = np.zeros((self.tiles_y, self.tiles_x), dtype=np.int8)
        
        # Temperature map (baseline, modified by weather/time)
        self.temperature = np.ones((self.tiles_y, self.tiles_x)) * 20.0  # 20°C baseline
        
        # Light level (0-1, affected by day/night)
        self.light_level = 1.0
        
        # Time tracking
        self.time = 0.0                 # World time in seconds
        self.day_length = 300.0         # 5 minutes = 1 day
        
        # Weather
        self.weather = Weather.CLEAR
        self.weather_timer = 0.0
        
        # Dynamic objects
        self.food_sources: List[FoodSource] = []
        self.water_sources: List['WaterSource'] = []  # Water objects across zones
        self.hazards: List[Hazard] = []
        self.shelters: List[Shelter] = []
        self.tools: List[ToolObject] = []  # TIER 4: Tool objects
        
        # Food spawning
        self.food_spawn_rate = 0.01    # Probability per tick (reduced for less clutter)
        self.max_food = 30             # Reduced max food
        
        # Tool spawning
        self.tool_spawn_rate = 0.002   # Less common than food
        self.max_tools = 15            # Max tools in world
        
        # Spatial Partitioning
        if _HAS_QUADTREE:
            self.food_quadtree = QuadTree(Rect(0, 0, self.width, self.height))
        
        # Generate initial world
        self._generate_terrain()
    
    def _generate_terrain(self):
        """Generate procedural terrain - FLAT ground only."""
        # Ground at 70% from top
        ground_level = int(self.tiles_y * 0.7)
        
        # Fill ground - simple flat terrain
        self.tiles[ground_level:, :] = TileType.GROUND.value
        
        # Add random Surface Walls (Stone Pillars)
        for _ in range(6):
            rx = np.random.randint(5, self.tiles_x - 5)
            h = np.random.randint(2, 5)
            # Build pillar up from ground
            for y in range(ground_level - h, ground_level):
                if y >= 0:
                    self.tiles[y, rx] = TileType.STONE.value
                    
        # Add random Caves (Holes in ground)
        for _ in range(10):
            rx = np.random.randint(0, self.tiles_x - 1)
            ry = np.random.randint(ground_level + 1, self.tiles_y - 1)
            self.tiles[ry, rx] = TileType.EMPTY.value
            # Make it 2x2 roughly
            self.tiles[ry, rx+1] = TileType.EMPTY.value
            self.tiles[ry+1, rx] = TileType.EMPTY.value
        
        # NO PLATFORMS - removed to prevent invisible platform issues
        
        # Water and food are spawned later in respawn_across_zones() 
        # after world bounds are set from indexed backgrounds
        
        # Spawn initial tools (sticks and stones) in the starting zone
        # 1-2 sticks and 1-2 stones scattered around (reduced to half)
        n_sticks = np.random.randint(1, 3)
        n_stones = np.random.randint(1, 3)
        min_spacing = 80  # Minimum distance between tools
        
        for _ in range(n_sticks):
            for attempt in range(50):  # Try up to 50 times to find non-overlapping spot
                spawn_x = np.random.randint(50, self.width - 50)
                spawn_y = (ground_level - 1) * self.tile_size
                
                # Check if this position overlaps with existing tools
                overlap = False
                for existing_tool in self.tools:
                    dist = np.sqrt((existing_tool.x - spawn_x)**2 + (existing_tool.y - spawn_y)**2)
                    if dist < min_spacing:
                        overlap = True
                        break
                
                if not overlap:
                    self.tools.append(ToolObject(
                        x=float(spawn_x),
                        y=float(spawn_y),
                        tool_type=ToolType.STICK
                    ))
                    break
        
        for _ in range(n_stones):
            for attempt in range(50):
                spawn_x = np.random.randint(50, self.width - 50)
                spawn_y = (ground_level - 1) * self.tile_size
                
                # Check if this position overlaps with existing tools
                overlap = False
                for existing_tool in self.tools:
                    dist = np.sqrt((existing_tool.x - spawn_x)**2 + (existing_tool.y - spawn_y)**2)
                    if dist < min_spacing:
                        overlap = True
                        break
                
                if not overlap:
                    self.tools.append(ToolObject(
                        x=float(spawn_x),
                        y=float(spawn_y),
                        tool_type=ToolType.STONE
                    ))
                    break
        
        print(f"[World] Spawned {len([t for t in self.tools if t.tool_type == ToolType.STICK])} sticks and {len([t for t in self.tools if t.tool_type == ToolType.STONE])} stones at ground level {ground_level}")
        
        # Temperature zones
        # Left side cooler
        self.temperature[:, :self.tiles_x//3] -= 10
        # Right side warmer  
        self.temperature[:, 2*self.tiles_x//3:] += 10
    
    def plant_seed(self, x: float, y: float):
        """Plant a seed at the location."""
        # Align to ground nicely
        ground_y = (self.tiles_y - 19) * self.tile_size # Approx ground level
        
        # Check if too close to other food
        for food in self.food_sources:
             if abs(food.x - x) < 20 and abs(food.y - ground_y) < 20:
                 return False # Too close
                 
        seed = FoodSource(
            x=x,
            y=ground_y,
            type=FoodType.PLANT,
            is_planted=True,
            growth_stage=0.0,
            nutrition=0.05,
            max_nutrition=0.8, # Cultivated plants are better?
            remaining=1.0
        )
        self.food_sources.append(seed)
        return True
    
    def _spawn_food(self):
        """Spawn a food item at a valid location across all world zones."""
        if len(self.food_sources) >= self.max_food:
            return
        
        # Determine spawn range based on world bounds (set by indexed backgrounds)
        # No padding - spawn across full zone bounds
        if self.min_x is not None and self.max_x is not None:
            spawn_min_x = self.min_x
            spawn_max_x = self.max_x
            # DEBUG: Print bounds being used
            if len(self.food_sources) == 0:
                print(f"[Food Spawn] Using bounds: {spawn_min_x} to {spawn_max_x}")
        else:
            spawn_min_x = 0
            spawn_max_x = self.width
            print(f"[Food Spawn] WARNING: No bounds set, using width={self.width}")
        
        # Find a valid spawn location
        for _ in range(20):  # Max attempts
            # Random x across all zones
            spawn_x = spawn_min_x + np.random.random() * (spawn_max_x - spawn_min_x)
            
            # Convert to local tile coordinates for ground check
            # We check tiles in the "home" zone (0 to width) - terrain is the same across zones
            local_x = spawn_x % self.width
            tx = int(local_x / self.tile_size)
            tx = min(tx, self.tiles_x - 1)  # Clamp to valid range
            
            # Search from bottom up for a valid spawn position (just above ground)
            for ty in range(self.tiles_y - 2, 0, -1):
                # Check if this tile is empty with ground below
                if (self.tiles[ty, tx] == TileType.EMPTY.value and 
                    self.tiles[ty + 1, tx] == TileType.GROUND.value):
                    
                    # Determine type - only berry types (which have images)
                    r = np.random.random()
                    if r < 0.5:
                        food_type = FoodType.SWEET_BERRY
                        nutrition = 0.6 + np.random.random() * 0.3 # High value
                    elif r < 0.8:
                        food_type = FoodType.BITTER_BERRY
                        nutrition = 0.1 # Low value
                    else:
                        food_type = FoodType.POISON_BERRY
                        nutrition = 0.0 # No value
                    
                    # Spawn at the actual world x, with y based on tile
                    self.food_sources.append(FoodSource(
                        x=spawn_x,
                        y=ty * self.tile_size + self.tile_size // 2,
                        nutrition=nutrition,
                        type=food_type,
                        spoilage_rate=0.0005
                    ))
                    return  # Successfully spawned, exit function

    def _spawn_tool(self):
        """Spawn a tool item at a valid location. TIER 4: Tool Use."""
        if len(self.tools) >= self.max_tools:
            return
        
        # Spawn across full zone bounds - no padding
        if self.min_x is not None and self.max_x is not None:
            spawn_min_x = self.min_x
            spawn_max_x = self.max_x
        else:
            spawn_min_x = 0
            spawn_max_x = self.width
        
        # Find valid spawn location (same logic as food)
        for _ in range(20):
            spawn_x = spawn_min_x + np.random.random() * (spawn_max_x - spawn_min_x)
            local_x = spawn_x % self.width
            tx = int(local_x / self.tile_size)
            
            for ty in range(self.tiles_y - 1, 0, -1):
                if (self.tiles[ty, tx] == TileType.GROUND.value and 
                    self.tiles[ty-1, tx] == TileType.EMPTY.value):
                    
                    # Random tool type (sticks and stones most common)
                    r = np.random.random()
                    if r < 0.4:
                        tool_type = ToolType.STICK
                    elif r < 0.7:
                        tool_type = ToolType.STONE
                    elif r < 0.85:
                        tool_type = ToolType.LEAF
                    elif r < 0.95:
                        tool_type = ToolType.SHELL
                    else:
                        tool_type = ToolType.BONE
                    
                    self.tools.append(ToolObject(
                        x=spawn_x,
                        y=ty * self.tile_size - self.tile_size // 2,
                        tool_type=tool_type
                    ))
                    return

    def get_nearby_tools(self, x: float, y: float, radius: float = 100.0) -> List[ToolObject]:
        """Get tools within radius of position. TIER 4."""
        nearby = []
        for tool in self.tools:
            if tool.is_available():
                dist = np.sqrt((tool.x - x)**2 + (tool.y - y)**2)
                if dist < radius:
                    nearby.append(tool)
        return nearby
    
    def pickup_tool(self, tool: ToolObject, creature_id: str) -> bool:
        """Creature picks up a tool. Returns True if successful."""
        if tool in self.tools and tool.is_available():
            tool.held_by = creature_id
            return True
        return False
    
    def drop_tool(self, tool: ToolObject, x: float, y: float):
        """Creature drops a tool at position."""
        if tool in self.tools:
            tool.held_by = None
            tool.x = x
            tool.y = y
            # Enable physics so it falls
            tool.throwing = True
            tool.throw_vx = 0
            tool.throw_vy = 0
    
    def throw_tool(self, tool: ToolObject, start_x: float, start_y: float, 
                   direction: float, power: float = 1.0, thrower_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Throw a tool in a direction with physics simulation.
        
        Args:
            tool: Tool to throw
            start_x, start_y: Starting position
            direction: Angle in radians (0 = right, pi = left)
            power: Throw strength (0-1)
            thrower_id: ID of creature throwing (to avoid self-damage)
        
        Returns info about where it landed and what it hit.
        """
        if tool not in self.tools:
            return {'success': False}
        
        # Release from creature
        tool.held_by = None
        tool.throwing = True
        tool.throw_power = power
        
        # Calculate initial velocity based on weight and power
        # Lighter objects travel farther, heavier do more damage
        base_speed = 300 * power  # pixels per second
        speed = base_speed * (1.5 - tool.weight)  # Lighter = faster
        
        tool.throw_vx = np.cos(direction) * speed
        tool.throw_vy = np.sin(direction) * speed
        tool.x = start_x
        tool.y = start_y
        
        return {
            'success': True,
            'throwing': True,
            'thrower_id': thrower_id
        }
    
    def update_thrown_tools(self, dt: float, creatures: List = None) -> List[Dict[str, Any]]:
        """
        Update physics for thrown tools and check for hits.
        
        Args:
            dt: Time delta
            creatures: List of creatures to check for hits
        
        Returns list of hit events with damage info.
        """
        hit_events = []
        
        for tool in self.tools:
            if not tool.throwing:
                continue
            
            # Store previous position for collision check
            prev_x, prev_y = tool.x, tool.y
            
            # Apply physics
            tool.throw_vy += 400 * dt  # Gravity (pixels/s²)
            tool.x += tool.throw_vx * dt
            tool.y += tool.throw_vy * dt
            
            # Air resistance
            tool.throw_vx *= (1.0 - 0.5 * dt)
            tool.throw_vy *= (1.0 - 0.2 * dt)
            
            # Check ground collision
            if tool.y >= self.height - 20:
                tool.y = self.height - 20
                tool.throwing = False
                tool.throw_vx = 0
                tool.throw_vy = 0
                tool.durability -= 0.05  # Minor damage from landing
                continue
            
            # Check creature hits
            if creatures:
                for creature in creatures:
                    if not hasattr(creature, 'motor') or not hasattr(creature, 'homeostasis'):
                        continue
                    
                    # Skip if this creature threw it (avoid self-damage)
                    creature_id = str(getattr(creature, 'id', None))
                    if creature_id == getattr(tool, 'thrower_id', None):
                        continue
                    
                    # Get creature bounds
                    cx, cy = creature.motor.x, creature.motor.y
                    cw = getattr(creature.phenotype, 'width', 20)
                    ch = getattr(creature.phenotype, 'height', 30)
                    
                    # Simple AABB collision
                    if (tool.x > cx - cw/2 and tool.x < cx + cw/2 and
                        tool.y > cy - ch/2 and tool.y < cy + ch/2):
                        
                        # Calculate damage based on velocity and weight
                        velocity_mag = np.sqrt(tool.throw_vx**2 + tool.throw_vy**2)
                        speed_factor = min(velocity_mag / 300, 1.0)  # Normalize to throw speed
                        
                        # Base damage from tool properties scaled by speed and weight
                        base_damage = tool.damage * speed_factor * (0.5 + tool.weight)
                        
                        # Check for head hit (top 30% of creature)
                        head_zone = cy - ch * 0.35
                        is_head_hit = tool.y < head_zone
                        
                        if is_head_hit:
                            # Head hits: 3x damage, heavy objects (stone/bone) can instant kill
                            damage = base_damage * 3.0
                            
                            # Instant death if heavy object hits head hard
                            if tool.weight > 0.6 and speed_factor > 0.7:
                                damage = creature.homeostasis.health * 2  # Overkill
                        else:
                            # Body hit: normal damage
                            damage = base_damage
                        
                        # Apply damage
                        creature.homeostasis.apply_damage(damage)
                        
                        # Stop tool
                        tool.throwing = False
                        tool.throw_vx = 0
                        tool.throw_vy = 0
                        tool.durability -= 0.15  # Hitting creature damages tool
                        
                        # Record event
                        hit_events.append({
                            'creature_id': getattr(creature, 'id', None),
                            'tool_type': tool.tool_type.value,
                            'damage': damage,
                            'is_head_hit': is_head_hit,
                            'is_lethal': damage >= creature.homeostasis.health,
                            'position': (tool.x, tool.y)
                        })
                        
                        break  # Tool stops at first hit
        
        return hit_events

    def respawn_across_zones(self):
        """Clear and respawn food/water across all indexed background zones.
        Call this after min_x/max_x bounds are set.
        
        Uses per-zone boundaries for spawning - no padding, zones connect seamlessly.
        """
        if self.min_x is None or self.max_x is None:
            return  # Bounds not set yet
        
        # Get zone boundaries
        zone_boundaries = getattr(self, 'zone_boundaries', None)
        
        if not zone_boundaries:
            # Fallback - create single zone
            zone_boundaries = {0: {'start': self.min_x, 'end': self.max_x, 'width': self.max_x - self.min_x}}
        
        num_zones = len(zone_boundaries)
        print(f"[Respawn] {num_zones} zones, world: {self.min_x} to {self.max_x}")
            
        # Reinitialize Quadtree to cover all zones
        if _HAS_QUADTREE:
            total_width = self.max_x - self.min_x
            self.food_quadtree = QuadTree(Rect(self.min_x, 0, total_width, self.height))
            
        # Clear existing food
        self.food_sources.clear()
        
        # Spawn food across zones
        food_per_zone = max(1, self.max_food // (num_zones * 3))
        for _ in range(food_per_zone * num_zones):
            self._spawn_food()
        
        # TIER 4: Spawn tools across zones
        min_spacing = 80
        
        # Spawn 1-2 sticks and 1-2 stones per zone - NO PADDING
        for zone_idx, zone_info in zone_boundaries.items():
            zone_start = zone_info['start']
            zone_end = zone_info['end']
            spawn_width = zone_end - zone_start
            
            if spawn_width <= 0:
                continue
            
            # 1-2 sticks per zone
            for _ in range(np.random.randint(1, 3)):
                if len(self.tools) >= self.max_tools:
                    break
                    
                # Try to find non-overlapping position
                for attempt in range(50):
                    spawn_x = zone_start + np.random.random() * spawn_width
                    spawn_y = (self.tiles_y - 19) * self.tile_size  # One row above ground
                    
                    # Check overlap with existing tools
                    overlap = False
                    for existing_tool in self.tools:
                        dist = np.sqrt((existing_tool.x - spawn_x)**2 + (existing_tool.y - spawn_y)**2)
                        if dist < min_spacing:
                            overlap = True
                            break
                    
                    if not overlap:
                        self.tools.append(ToolObject(
                            x=float(spawn_x),
                            y=float(spawn_y),
                            tool_type=ToolType.STICK
                        ))
                        break
            
            # 1-2 stones per zone
            for _ in range(np.random.randint(1, 3)):
                if len(self.tools) >= self.max_tools:
                    break
                    
                # Try to find non-overlapping position
                for attempt in range(50):
                    spawn_x = zone_start + np.random.random() * spawn_width
                    spawn_y = (self.tiles_y - 19) * self.tile_size  # One row above ground
                    
                    # Check overlap with existing tools
                    overlap = False
                    for existing_tool in self.tools:
                        dist = np.sqrt((existing_tool.x - spawn_x)**2 + (existing_tool.y - spawn_y)**2)
                        if dist < min_spacing:
                            overlap = True
                            break
                    
                    if not overlap:
                        self.tools.append(ToolObject(
                            x=float(spawn_x),
                            y=float(spawn_y),
                            tool_type=ToolType.STONE
                        ))
                        break
        
        # Clear any existing water tiles from tile array (we use water_sources objects now)
        self.tiles[self.tiles == TileType.WATER.value] = TileType.EMPTY.value
            
        # Clear existing water sources
        self.water_sources.clear()
        
        # Spawn exactly ONE water source in zone 0 - creatures must travel back to drink
        ground_y = (self.tiles_y - 18) * self.tile_size + self.tile_size // 2  # At ground level
        
        # Get zone 0's width from zone_boundaries
        zone_0_info = zone_boundaries.get(0, zone_boundaries.get(min(zone_boundaries.keys())))
        zone_0_start = zone_0_info['start']
        zone_0_width = zone_0_info['width']
        
        # Single water pool in zone 0 only (flattened appearance - 50% larger)
        water_x = zone_0_start + np.random.randint(int(zone_0_width * 0.25), int(zone_0_width * 0.75))
        water_width = 225.0  # 150 * 1.5 = 225
        water_height = 45.0  # 30 * 1.5 = 45
        
        self.water_sources.append(WaterSource(
            x=float(water_x),
            y=float(ground_y),
            width=float(water_width),
            height=float(water_height)
        ))
        
        print(f"[Respawn] Created {len(self.water_sources)} water source(s) at x={water_x}")
        
        # NOTE: We no longer add water tiles to the tile array - only water_sources are used
        # This prevents water from appearing in every chunk when tiles are repeated

    def update(self, dt: float = 1.0):
        """
        Update world state.
        
        Args:
            dt: Time delta in simulation units
        """
        self.time += dt
        
        # Day/night cycle
        day_progress = (self.time % self.day_length) / self.day_length
        # Light peaks at noon (0.5), darkest at midnight (0.0/1.0)
        self.light_level = 0.3 + 0.7 * np.sin(day_progress * np.pi)
        
        # Weather updates
        self.weather_timer -= dt
        if self.weather_timer <= 0:
            self._change_weather()
        
        # Food spawning
        if np.random.random() < self.food_spawn_rate * dt:
            self._spawn_food()
        
        # Tool spawning (TIER 4)
        if np.random.random() < self.tool_spawn_rate * dt:
            self._spawn_tool()
        
        # Food decay/growth
        self.food_sources = [f for f in self.food_sources if f.update(dt)]
        
        # Tool durability decay for held tools
        self.tools = [t for t in self.tools if t.durability > 0]
        
        # System 6: Check interactions (crafting)
        self._check_crafting_interactions()
        
        # Update Quadtree
        
        # Update Quadtree
        if _HAS_QUADTREE:
            self.food_quadtree.clear()
            for food in self.food_sources:
                self.food_quadtree.insert(Point(food.x, food.y, food))
        
        # Temperature updates based on weather/time
        self._update_temperature()
    
    def _change_weather(self):
        """Randomly change weather."""
        self.weather = np.random.choice(list(Weather))
        self.weather_timer = 30 + np.random.random() * 60  # 30-90 seconds
    
    def _update_temperature(self):
        """Update temperature based on conditions."""
        # Reset to baseline
        base_temp = 20.0
        
        # Day/night effect
        base_temp += 10 * (self.light_level - 0.5)
        
        # Weather effect
        # (Rest of update_temperature implementation...)
    
    def _check_crafting_interactions(self):
        """
        Check for tool-tool interactions for crafting.
        If compatible tools are stacked (close proximity basic physics), merge them.
        """
        to_remove = []  # Changed from set to list since ToolObject is not hashable
        to_add = []
        
        # Naive N^2 check (n is small, ~15)
        # Only check tools on the ground (available)
        available_tools = [t for t in self.tools if t.is_available() and t not in to_remove]
        
        for i, t1 in enumerate(available_tools):
            if t1 in to_remove: continue
            
            for j, t2 in enumerate(available_tools):
                if i >= j: continue
                if t2 in to_remove: continue
                
                # Check distance
                dist = np.sqrt((t1.x - t2.x)**2 + (t1.y - t2.y)**2)
                if dist < 25.0:  # Stacked proximity
                    # Check recipe
                    result_type = self._get_crafting_result(t1.tool_type, t2.tool_type)
                    
                    if result_type:
                        # CRAFT!
                        to_remove.append(t1)
                        to_remove.append(t2)
                        
                        # Create new item at center
                        new_tool = ToolObject(
                            x=(t1.x + t2.x) / 2,
                            y=(t1.y + t2.y) / 2,
                            tool_type=result_type
                        )
                        to_add.append(new_tool)
                        print(f"[Crafting] Created {result_type.name} from {t1.tool_type.name} + {t2.tool_type.name}")
                        break # One reaction per tool per tick
        
        # Apply changes
        if to_remove:
            self.tools = [t for t in self.tools if t not in to_remove]
            self.tools.extend(to_add)
            
    def _get_crafting_result(self, t1: ToolType, t2: ToolType) -> Optional[ToolType]:
        """Define crafting recipes."""
        types = sorted([t1, t2], key=lambda t: t.value)
        
        # Stick + Stick -> Nest
        if types == [ToolType.STICK, ToolType.STICK]:
            return ToolType.NEST
            
        # Stick + Stone -> Hammer
        if types == sorted([ToolType.STICK, ToolType.STONE], key=lambda t: t.value):
            return ToolType.HAMMER
            
        # Stone + Stone -> Sharp Rock
        if types == [ToolType.STONE, ToolType.STONE]:
            return ToolType.SHARP_ROCK
            
        # Stick + Sharp Rock -> Spear
        if types == sorted([ToolType.STICK, ToolType.SHARP_ROCK], key=lambda t: t.value):
            return ToolType.SPEAR
            
        return None
        if self.weather == Weather.RAIN:
            base_temp -= 5
        elif self.weather == Weather.STORM:
            base_temp -= 10
        elif self.weather == Weather.SNOW:
            base_temp -= 20
        elif self.weather == Weather.HEAT_WAVE:
            base_temp += 15
        
        # Apply to temperature map (with spatial variation)
        self.temperature = np.ones_like(self.temperature) * base_temp
        self.temperature[:, :self.tiles_x//3] -= 10
        self.temperature[:, 2*self.tiles_x//3:] += 10
    
    def get_tile(self, x: float, y: float) -> TileType:
        """Get tile type at world coordinates (wraps across zones)."""
        ty = int(y // self.tile_size)
        if ty < 0 or ty >= self.tiles_y:
            return TileType.EMPTY
        
        # Wrap x coordinate to local tile space (tiles repeat across all zones)
        local_x = x % self.width
        tx = int(local_x // self.tile_size)
        tx = max(0, min(tx, self.tiles_x - 1))  # Clamp to valid range
        
        return TileType(self.tiles[ty, tx])
    
    def get_temperature(self, x: float, y: float) -> float:
        """Get temperature at world coordinates (wraps across zones)."""
        ty = int(y // self.tile_size)
        if ty < 0 or ty >= self.tiles_y:
            return 20.0
            
        # Wrap x coordinate to local tile space
        local_x = x % self.width
        tx = int(local_x // self.tile_size)
        tx = max(0, min(tx, self.tiles_x - 1))
        
        return float(self.temperature[ty, tx])
    
    def is_solid(self, x: float, y: float) -> bool:
        """Check if position is solid (collision)."""
        # Out of bounds at bottom = solid (prevents falling through world)
        ty = int(y // self.tile_size)
        if ty >= self.tiles_y:
            return True  # Bottom of world is solid
        if ty < 0:
            return False  # Above world = not solid
        
        # Wrap x coordinate to local tile space (tiles repeat across all zones)
        local_x = x % self.width
        tx = int(local_x // self.tile_size)
        tx = max(0, min(tx, self.tiles_x - 1))  # Clamp to valid range
        
        tile = self.tiles[ty, tx]
        return tile == TileType.GROUND.value or tile == TileType.STONE.value

    
    def is_hazard(self, x: float, y: float) -> Tuple[bool, float, str]:
        """Check if position is hazardous, return (is_hazard, damage, type)."""
        tile = self.get_tile(x, y)
        if tile == TileType.HAZARD:
            # Find specific hazard for damage value
            for h in self.hazards:
                if h.x <= x < h.x + h.width and h.y <= y < h.y + h.height:
                    return True, h.damage, h.type
            return True, 10.0, "fire" # Default
        return False, 0.0, "none"
    
    def is_water(self, x: float, y: float) -> bool:
        """Check if position is water (checks water_sources first, then tiles)."""
        # Check water_sources objects first (zone-specific water)
        if hasattr(self, 'water_sources'):
            for water in self.water_sources:
                if water.contains_point(x, y):
                    return True
        
        # Fallback: check tile array (zone 0 only for backward compatibility)
        if x < 0 or x >= self.width:
            return False
        return self.get_tile_nowrap(x, y) == TileType.WATER
    
    def get_tile_nowrap(self, x: float, y: float) -> TileType:
        """Get tile type at world coordinates (NO wrapping - returns EMPTY outside bounds)."""
        if x < 0 or x >= self.width:
            return TileType.EMPTY
        ty = int(y // self.tile_size)
        if ty < 0 or ty >= self.tiles_y:
            return TileType.EMPTY
        tx = int(x // self.tile_size)
        tx = max(0, min(tx, self.tiles_x - 1))
        return TileType(self.tiles[ty, tx])
    
    def find_food_nearby(self, x: float, y: float, radius: float) -> List[FoodSource]:
        """Find food sources within radius."""
        if _HAS_QUADTREE:
            # First get candidates from quadtree (square query)
            dist_sq = radius * radius
            candidates = self.food_quadtree.query(Rect(x - radius, y - radius, radius * 2, radius * 2))
            
            # Filter by exact distance
            nearby = []
            for food in candidates:
                d_sq = (food.x - x)**2 + (food.y - y)**2
                if d_sq <= dist_sq:
                    nearby.append(food)
            return nearby
        else:
            nearby = []
            for food in self.food_sources:
                dist = np.sqrt((food.x - x)**2 + (food.y - y)**2)
                if dist <= radius:
                    nearby.append(food)
            return nearby
    
    def eat_food(self, food: FoodSource, amount: float = 0.3) -> float:
        """
        Consume food, return nutrition gained.
        
        Args:
            food: The food source to eat
            amount: Fraction to consume (0-1)
            
        Returns:
            Nutrition value gained
        """
        actual_amount = min(amount, food.remaining)
        food.remaining -= actual_amount
        return food.nutrition * actual_amount
    
    def get_sensory_data(self, x: float, y: float, 
                         vision_range: float = 100,
                         hearing_range: float = 150,
                         smell_range: float = 80) -> Dict[str, Any]:
        """
        Get sensory data for a creature at position.
        
        Returns dict with:
        - visible_food: list of (dx, dy, type, nutrition)
        - visible_hazards: list of (dx, dy, type)
        - visible_creatures: list of (dx, dy, size, is_threat)
        - temperature: local temperature
        - light: current light level
        - on_ground: bool
        - in_water: bool
        - near_water: bool (within drinking distance)
        - in_shelter: bool
        """
        # Check for water in/around creature
        in_water = self.is_water(x, y)
        near_water = (
            in_water or
            self.is_water(x, y + 10) or  # Below feet
            self.is_water(x - 10, y) or  # Left
            self.is_water(x + 10, y) or  # Right  
            self.is_water(x, y + 20)     # Further below
        )
        
        data = {
            'visible_food': [],
            'visible_hazards': [],
            'visible_creatures': [],  # Filled by game loop
            'temperature': self.get_temperature(x, y),
            'light': self.light_level,
            'weather': self.weather.name,
            'on_ground': self.is_solid(x, y + 1),
            'in_water': in_water,
            'near_water': near_water,
            'in_shelter': False,
            'time_of_day': (self.time % self.day_length) / self.day_length,
        }
        
        # Edge detection - check if there's ground in each direction
        # This helps creatures on platforms know when to turn around or jump
        check_distance = 20  # Pixels to look ahead
        data['edge_left'] = not self.is_solid(x - check_distance, y + 10)  # No ground to left
        data['edge_right'] = not self.is_solid(x + check_distance, y + 10)  # No ground to right
        data['wall_left'] = self.is_solid(x - 5, y)  # Wall directly left
        data['wall_right'] = self.is_solid(x + 5, y)  # Wall directly right
        data['gap_ahead_left'] = data['edge_left'] and not self.is_solid(x - 40, y)  # Jumpable gap
        data['gap_ahead_right'] = data['edge_right'] and not self.is_solid(x + 40, y)
        
        # Wall proximity (normalized 0-1, where 1 is touching, 0 is far)
        dist_left = x
        dist_right = self.width - x
        data['wall_dist_left'] = dist_left
        data['wall_dist_right'] = dist_right
        data['nearest_wall_dist'] = min(dist_left, dist_right)
        # Direction to nearest wall (-1 left, 1 right)
        data['nearest_wall_dir'] = -1.0 if dist_left < dist_right else 1.0
        
        # Distance to ground below (for fall detection)
        fall_distance = 0
        test_y = y + 1
        while test_y < self.height and not self.is_solid(x, test_y) and fall_distance < 200:
            fall_distance += 1
            test_y += 1
        data['fall_distance'] = fall_distance
        
        # Check shelter
        for shelter in self.shelters:
            if (shelter.x <= x < shelter.x + shelter.width and 
                shelter.y <= y < shelter.y + shelter.height):
                data['in_shelter'] = True
                break
        
        # Find visible food (affected by light)
        effective_vision = vision_range * (0.3 + 0.7 * self.light_level)
        nearest_food_dist = 1000
        nearest_food_dx = 0
        food_touching = False
        
        for food in self.food_sources:
            dx = food.x - x
            dy = food.y - y
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= effective_vision:
                data['visible_food'].append({
                    'dx': dx,
                    'dy': dy,
                    'dist': dist,
                    'type': food.type,
                    'nutrition': food.nutrition * food.remaining
                })
                # Track nearest food for brainstem
                if dist < nearest_food_dist:
                    nearest_food_dist = dist
                    nearest_food_dx = dx
                # Check if touching (within eating range)
                if dist < 25:
                    food_touching = True
        
        # Add brainstem-relevant food info
        data['nearest_food_distance'] = nearest_food_dist
        data['food_direction'] = np.sign(nearest_food_dx) if nearest_food_dist < 500 else 0
        data['food_touching'] = food_touching
        
        # Find nearby hazards
        for hazard in self.hazards:
            hx = hazard.x + hazard.width / 2
            hy = hazard.y + hazard.height / 2
            dx = hx - x
            dy = hy - y
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= effective_vision:
                data['visible_hazards'].append({
                    'dx': dx,
                    'dy': dy,
                    'dist': dist,
                    'type': hazard.type,
                    'damage': hazard.damage
                })
        
        return data
    
    def apply_physics(self, x: float, y: float, 
                      vx: float, vy: float,
                      dt: float = 1.0) -> Tuple[float, float, float, float, bool]:
        """
        Apply physics to an entity.
        
        Args:
            x, y: Current position
            vx, vy: Current velocity
            dt: Time delta
            
        Returns:
            new_x, new_y, new_vx, new_vy, on_ground
        """
        # Apply gravity (pixels/second² * seconds = pixels/second)
        vy += self.gravity * dt
        
        # Terminal velocity (pixels/second)
        vy = min(vy, 500.0)
        
        # In water, reduce gravity effect
        if self.is_water(x, y):
            vy *= 0.9
            vx *= 0.95
        
        # New position
        new_x = x + vx * dt
        new_y = y + vy * dt
        
        # Collision detection with sweep to prevent tunneling
        on_ground = False
        
        # Vertical collision (check along path)
        if vy > 0:  # Falling
            # Check multiple points along fall path to prevent tunneling
            steps = max(1, int(abs(vy * dt) / self.tile_size) + 1)
            step_y = (new_y - y) / steps
            check_y = y
            for _ in range(steps):
                check_y += step_y
                if self.is_solid(x, check_y):
                    on_ground = True
                    # Snap to top of the solid tile
                    tile_y = int(check_y // self.tile_size)
                    new_y = tile_y * self.tile_size - 1
                    vy = 0
                    break
            else:
                # No collision found along path
                pass
        elif self.is_solid(x, new_y):
            # Moving up into ceiling
            new_y = y
            vy = 0
        
        # Horizontal collision
        if self.is_solid(new_x, min(new_y, y)):
            new_x = x
            vx = 0
        
        # World bounds - Horizontal Clamping (if limits set)
        if self.min_x is not None and new_x < self.min_x:
            new_x = self.min_x
            vx = 0
        if self.max_x is not None and new_x >= self.max_x:
            new_x = self.max_x - 1  # Keep inside the boundary
            vx = 0
            
        if new_y <= 0:
            new_y = 0
            vy = max(0, vy)
        elif new_y >= self.height - 1:
            new_y = self.height - 1
            vy = min(0, vy)
        
        # Final ground check (safety net)
        if not on_ground and self.is_solid(new_x, new_y + 1):
            on_ground = True
        
        return new_x, new_y, vx, vy, on_ground
    
    def to_dict(self) -> Dict:
        """Serialize world state."""
        data = {
            'width': self.width,
            'height': self.height,
            'tile_size': self.tile_size,
            'tiles': self.tiles.tolist(),
            'time': self.time,
            'weather': self.weather.name,
            'food_sources': [
                {'x': f.x, 'y': f.y, 'nutrition': f.nutrition, 
                 'type': f.type.value if hasattr(f.type, 'value') else str(f.type), 
                 'remaining': f.remaining}
                for f in self.food_sources
            ]
        }
        
        # Add water_sources if they exist
        if hasattr(self, 'water_sources') and self.water_sources:
            data['water_sources'] = [
                {'x': w.x, 'y': w.y, 'width': w.width, 'height': w.height}
                for w in self.water_sources
            ]
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'World':
        """Deserialize world state."""
        world = cls(data['width'], data['height'], data['tile_size'])
        world.tiles = np.array(data['tiles'], dtype=np.int8)
        world.time = data.get('time', 0)
        world.weather = Weather[data.get('weather', 'CLEAR')]
        
        world.food_sources = [
            FoodSource(
                x=f['x'], y=f['y'],
                nutrition=f['nutrition'],
                type=f['type'],
                remaining=f['remaining']
            )
            for f in data.get('food_sources', [])
        ]
        
        # Load water_sources if they exist in save data
        world.water_sources = []
        if 'water_sources' in data:
            world.water_sources = [
                WaterSource(
                    x=w['x'], y=w['y'],
                    width=w['width'], height=w['height']
                )
                for w in data['water_sources']
            ]
        
        return world
    
    def render_to_array(self) -> np.ndarray:
        """
        Render world to RGB array for display.
        
        Returns:
            (height, width, 3) uint8 array
        """
        # Color mapping for tiles
        colors = {
            TileType.EMPTY.value: (40, 44, 52),        # Dark grey (sky)
            TileType.GROUND.value: (101, 67, 33),     # Brown
            TileType.WATER.value: (64, 164, 223),     # Blue
            TileType.HAZARD.value: (255, 87, 34),     # Orange/red
            TileType.SHELTER.value: (76, 175, 80),    # Green
            TileType.FOOD_PLANT.value: (139, 195, 74), # Light green
            TileType.FOOD_MEAT.value: (244, 67, 54),  # Red
            TileType.NEST.value: (255, 193, 7),       # Yellow
        }
        
        # Create image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Fill tiles
        for ty in range(self.tiles_y):
            for tx in range(self.tiles_x):
                tile_val = self.tiles[ty, tx]
                color = colors.get(tile_val, (128, 128, 128))
                
                y1 = ty * self.tile_size
                y2 = (ty + 1) * self.tile_size
                x1 = tx * self.tile_size
                x2 = (tx + 1) * self.tile_size
                
                
                # Override specific types
                if tile_val == TileType.HAZARD.value:
                     # Check actual hazard object covering this tile
                     # Just approximate color for tiles since we don't have per-tile metadata easily mapped here
                     # We can iterate hazards
                     cx = x1 + self.tile_size//2
                     cy = y1 + self.tile_size//2
                     for h in self.hazards:
                         if h.x <= cx < h.x + h.width and h.y <= cy < h.y + h.height:
                             color = self.get_hazard_color(h.type)
                             break
                             
                img[y1:y2, x1:x2] = color
        
        # Apply lighting
        light_factor = 0.5 + 0.5 * self.light_level
        img = (img * light_factor).astype(np.uint8)
        
        # Draw food sources
        for food in self.food_sources:
            fx, fy = int(food.x), int(food.y)
            if 0 <= fx < self.width and 0 <= fy < self.height:
                size = int(4 + 4 * food.remaining)
                color = (139, 195, 74) if food.type == "plant" else (244, 67, 54)
                for dy in range(-size//2, size//2):
                    for dx in range(-size//2, size//2):
                        px, py = fx + dx, fy + dy
                        if 0 <= px < self.width and 0 <= py < self.height:
                            img[py, px] = color
        
        return img

    def get_hazard_color(self, type_name: str) -> Tuple[int, int, int]:
        if type_name == "radiation": return (100, 255, 0)   # Neon Green
        if type_name == "healing": return (255, 105, 180)   # Pink
        return (255, 87, 34) # Orange default

    def dig(self, x: float, y: float) -> bool:
        """
        Dig at position (turn GROUND to EMPTY).
        Returns True if successful.
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
            
        ty = int(y // self.tile_size)
        tx = int(x // self.tile_size)
        
        # Clamp
        if ty < 0 or ty >= self.tiles_y or tx < 0 or tx >= self.tiles_x:
            return False
            
        if self.tiles[ty, tx] == TileType.GROUND.value:
            self.tiles[ty, tx] = TileType.EMPTY.value
            return True
        return False
        
    def build(self, x: float, y: float, type: TileType = TileType.STONE) -> bool:
        """
        Build at position (turn EMPTY to type).
        Returns True if successful.
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
            
        ty = int(y // self.tile_size)
        tx = int(x // self.tile_size)
        
        # Clamp
        if ty < 0 or ty >= self.tiles_y or tx < 0 or tx >= self.tiles_x:
            return False
            
        # Only build in empty space
        # Also ensure we aren't building inside a creature (collision check?)
        # For now, just tile check
        if self.tiles[ty, tx] == TileType.EMPTY.value:
            self.tiles[ty, tx] = type.value
            return True
        return False
        
    def trigger_disaster(self, type: str = "earthquake"):
        """Trigger a disaster event (System 3)."""
        if type == "earthquake":
            # Shake the world - break some walls, creatue rubble?
            # For now, just break 10% of walls
            for y in range(self.tiles_y):
                for x in range(self.tiles_x):
                    if self.tiles[y, x] == TileType.STONE.value:
                        if np.random.random() < 0.1: # 10% chance to crumble
                            self.tiles[y, x] = TileType.EMPTY.value
            print("Earthquake triggered! Walls crumbled.")
        elif type == "heat_wave":
            self.weather = Weather.HEAT_WAVE



# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'TileType',
    'Weather',
    'FoodSource',
    'Hazard',
    'Shelter',
    'World',
]

