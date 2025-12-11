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


class Weather(Enum):
    """Weather conditions."""
    CLEAR = 0
    RAIN = 1
    STORM = 2
    SNOW = 3
    HEAT_WAVE = 4


@dataclass
class FoodSource:
    """A food item in the world."""
    x: float
    y: float
    nutrition: float = 0.5          # Energy provided (0-1 scale)
    type: str = "plant"             # "plant" or "meat"
    spoilage_rate: float = 0.001    # How fast it decays
    remaining: float = 1.0          # 0-1, how much left
    
    def decay(self, dt: float = 1.0):
        """Food decays over time."""
        self.remaining -= self.spoilage_rate * dt
        return self.remaining > 0


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
        self.hazards: List[Hazard] = []
        self.shelters: List[Shelter] = []
        
        # Food spawning
        self.food_spawn_rate = 0.04    # Probability per tick (Increased from 0.005)
        self.max_food = 80             # Increased max food
        
        # Spatial Partitioning
        if _HAS_QUADTREE:
            self.food_quadtree = QuadTree(Rect(0, 0, self.width, self.height))
        
        # Generate initial world
        self._generate_terrain()
    
    def _generate_terrain(self):
        """Generate procedural terrain - FLAT ground only."""
        # Ground at bottom third - FLAT, no platforms
        ground_level = self.tiles_y * 2 // 3
        
        # Fill ground - simple flat terrain
        self.tiles[ground_level:, :] = TileType.GROUND.value
        
        # NO PLATFORMS - removed to prevent invisible platform issues
        
        # Add water pool (only if world is large enough)
        # Water is placed ABOVE ground level so creatures can walk to edge
        if self.tiles_x > 15:
            water_x = np.random.randint(5, max(6, self.tiles_x // 2))
            water_w = np.random.randint(4, min(10, self.tiles_x - water_x))
            # Water sits on TOP of ground (ground_level - 1), not inside it
            self.tiles[ground_level - 1, water_x:water_x+water_w] = TileType.WATER.value
        
        # Hazards disabled for now - can add back later
        # (Removed radiation and healing zones)
        
        # Add some initial food
        for _ in range(5):  # More food
            self._spawn_food()
        
        # Temperature zones
        # Left side cooler
        self.temperature[:, :self.tiles_x//3] -= 10
        # Right side warmer  
        self.temperature[:, 2*self.tiles_x//3:] += 10
    
    def _spawn_food(self):
        """Spawn a food item at a valid location."""
        if len(self.food_sources) >= self.max_food:
            return
        
        # Find ground tiles
        for _ in range(20):  # Max attempts
            tx = np.random.randint(0, self.tiles_x)
            ty = np.random.randint(0, self.tiles_y - 1)
            
            # Check if above ground
            if (self.tiles[ty, tx] == TileType.EMPTY.value and 
                self.tiles[ty + 1, tx] == TileType.GROUND.value):
                
                food_type = "plant" if np.random.random() < 0.8 else "meat"
                nutrition = 0.3 + np.random.random() * 0.5  # 0.3-0.8 scale
                
                self.food_sources.append(FoodSource(
                    x=tx * self.tile_size + self.tile_size // 2,
                    y=ty * self.tile_size + self.tile_size // 2,
                    nutrition=nutrition,
                    type=food_type,
                    spoilage_rate=0.0005 if food_type == "plant" else 0.002
                ))
                return
    
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
        
        # Food decay
        self.food_sources = [f for f in self.food_sources if f.decay(dt)]
        
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
        """Get tile type at world coordinates."""
        tx = int(x // self.tile_size)
        ty = int(y // self.tile_size)
        if 0 <= tx < self.tiles_x and 0 <= ty < self.tiles_y:
            return TileType(self.tiles[ty, tx])
        return TileType.EMPTY
    
    def get_temperature(self, x: float, y: float) -> float:
        """Get temperature at world coordinates."""
        tx = int(x // self.tile_size)
        ty = int(y // self.tile_size)
        if 0 <= tx < self.tiles_x and 0 <= ty < self.tiles_y:
            return float(self.temperature[ty, tx])
        return 20.0
    
    def is_solid(self, x: float, y: float) -> bool:
        """Check if position is solid (collision)."""
        # Out of bounds at bottom or sides = solid (prevents falling through world)
        tx = int(x // self.tile_size)
        ty = int(y // self.tile_size)
        if ty >= self.tiles_y or tx < 0 or tx >= self.tiles_x:
            return True  # Bottom of world or out of horizontal bounds = solid
        if ty < 0:
            return False  # Above world = not solid (can jump up)
        
        tile = self.get_tile(x, y)
        return tile == TileType.GROUND
    
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
        """Check if position is water."""
        return self.get_tile(x, y) == TileType.WATER
    
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
        
        # World bounds
        new_x = max(0, min(self.width - 1, new_x))
        new_y = max(0, min(self.height - 1, new_y))
        
        # Final ground check (safety net)
        if not on_ground and self.is_solid(new_x, new_y + 1):
            on_ground = True
        
        return new_x, new_y, vx, vy, on_ground
    
    def to_dict(self) -> Dict:
        """Serialize world state."""
        return {
            'width': self.width,
            'height': self.height,
            'tile_size': self.tile_size,
            'tiles': self.tiles.tolist(),
            'time': self.time,
            'weather': self.weather.name,
            'food_sources': [
                {'x': f.x, 'y': f.y, 'nutrition': f.nutrition, 
                 'type': f.type, 'remaining': f.remaining}
                for f in self.food_sources
            ]
        }
    
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
