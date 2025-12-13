"""
Game Tab - 2D World Visualization and Creature Simulation

This tab provides:
- Visual rendering of the 2D world
- Creature spawning and management
- Real-time simulation controls
- Creature inspection and breeding
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import threading
import queue
import time
from collections import deque

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QGridLayout, QScrollArea, QSlider, QSpinBox,
    QListWidget, QListWidgetItem, QSplitter, QFrame, QComboBox,
    QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, QRect, QPointF, QPoint
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap, QMouseEvent, QFont, QFontMetrics, QPainterPath, QTransform

import sys
import json
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain.world import World, TileType, FoodSource, FoodType
from brain.creature import CreatureBody, Phenotype, Homeostasis
from brain.instincts import InstinctSystem, RewardSystem
from brain.embodiment import EmbodiedBrain
from brain.brainstem import Brainstem
from brain.behavior_state import BehaviorStateMachine, BehaviorState
from brain.movement_controller import MovementController, MovementGoal
from brain.language_decoder import NeuralLanguageDecoder
from brain.three_system_brain import ThreeSystemBrain, BrainConfig
from brain.social_learning import SocialLearningSystem, CulturalKnowledge
from brain.mental_map import MentalMap
from brain.procedural_language import ProceduralLanguageSystem, CulturalLanguage
from brain.procedural_language import ProceduralLanguageSystem, CulturalLanguage
from brain.lineage import LineageSystem
from brain.culture import CulturalMemory
from brain.migration import save_creature_to_file

# Try to import settings for visual config
try:
    from gui.settings_tab import (
        CreatureVisualConfig, BodyPart, AgeStage, apply_hue_to_pixmap,
        WorldObjectType, WorldObjectConfig
    )
    _HAS_SETTINGS = True
except ImportError:
    _HAS_SETTINGS = False
    CreatureVisualConfig = None
    WorldObjectType = None
    WorldObjectType = None
    WorldObjectType = None
    WorldObjectConfig = None

from gui.graph_widget import StatsGraphWidget
from gui.brain_inspector import BrainInspectorWindow
from gui.sound_manager import SoundManager



class ToolType:
    """God-mode tool types."""
    SELECT = "Cursor (Select)"
    PAINT_WALL = "Paint Wall"
    PAINT_GROUND = "Paint Ground"
    PAINT_WATER = "Paint Water"
    PAINT_RADIATION = "Paint Radiation"
    PAINT_HEALING = "Paint Healing"
    SPAWN_FOOD = "Spawn Food"
    SPAWN_TEMPLATE = "Spawn Template"
    DRAG = "Drag/Move"
    
    @classmethod
    def all_tools(cls):
        return [cls.SELECT, cls.PAINT_WALL, cls.PAINT_GROUND, 
                cls.PAINT_WATER, cls.PAINT_RADIATION, cls.PAINT_HEALING, 
                cls.SPAWN_FOOD, cls.SPAWN_TEMPLATE, cls.DRAG]


# =============================================================================
# LIVING CREATURE - Brain + Body + Instincts + Brainstem + State Machine
# =============================================================================

@dataclass
class LivingCreature:
    """A complete creature with brainstem, instincts, state machine, and optional cortex."""
    body: CreatureBody
    brain: Any  # ThreeSystemBrain or None for instinct-only
    instincts: InstinctSystem
    reward_system: RewardSystem
    brainstem: Brainstem = None  # Survival foundation
    behavior_state: BehaviorStateMachine = None  # KEY: State machine for stable behavior
    movement: MovementController = None  # NEW: High-level movement controller
    embodied_brain: Any = None  # EmbodiedBrain for full neural control
    last_step_data: Dict = None # Data from last step for inspection
    name: str = "Creature"
    generation: int = 0
    use_neural_control: bool = False  # Whether to use brain for behavior
    language: NeuralLanguageDecoder = None # Feature: Language
    procedural_language: ProceduralLanguageSystem = None  # TIER 4: Emergent Communication
    current_speech: str = None
    current_gesture: str = None
    speech_timer: float = 0.0
    
    # State tracking for failure handling (prevent getting stuck)
    consecutive_drink_fails: int = 0
    consecutive_eat_fails: int = 0
    forced_explore_timer: float = 0.0  # If >0, force WANDER/EXPLORE
    social_learning: SocialLearningSystem = None  # TIER 3: Cultural Evolution
    mental_map: MentalMap = None  # TIER 3: Environmental Intelligence
    cultural_memory: CulturalMemory = None # TIER 3: Social Learning
    decay_timer: float = 0.0  # TIER 4: Decomposition timer for dead bodies
    last_damage_source: str = 'none'  # Track damage source for behavior state machine
    
    def __post_init__(self):
        if not self.name or self.name == "Creature":
            self.name = f"Creature_{self.body.id % 10000:04d}"
        # Every creature needs a brainstem - it's not optional for survival
        if self.brainstem is None:
            self.brainstem = Brainstem()
            
        # Initialize traits from DNA
        if hasattr(self.body, 'phenotype') and hasattr(self.body.phenotype, 'bravery'):
            self.brainstem.set_phenotype_traits(self.body.phenotype.bravery)
            
        # Every creature needs a state machine for stable behavior
        if self.behavior_state is None:
            self.behavior_state = BehaviorStateMachine()
        # Every creature needs a movement controller
        if self.movement is None:
            self.movement = MovementController()
        # Init language
        if self.language is None:
            self.language = NeuralLanguageDecoder()
        # TIER 3: Every creature can learn from others
        if self.social_learning is None:
            self.social_learning = SocialLearningSystem(self.body.id)
        # TIER 3: Every creature builds mental maps
        if self.mental_map is None:
            self.mental_map = MentalMap(self.body.id)
        # TIER 4: Every creature can develop language
        if self.procedural_language is None:
            self.procedural_language = ProceduralLanguageSystem(str(self.body.id), innovation_rate=0.15)
        # TIER 3: Cultural Memory
        if self.cultural_memory is None:
            self.cultural_memory = CulturalMemory()



# =============================================================================
# WORLD RENDERER
# =============================================================================

class WorldRenderer(QWidget):
    """Widget that renders the 2D world and creatures."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.world: Optional[World] = None
        self.creatures: List[LivingCreature] = []
        self.selected_creature: Optional[LivingCreature] = None
        self.selected_secondary: Optional[LivingCreature] = None
        self.paused: bool = True  # Don't animate until playing
        
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        
        # Camera
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 1.0
        
        # Visual configuration for creatures (from settings tab)
        self.visual_config: Optional[CreatureVisualConfig] = None if not _HAS_SETTINGS else CreatureVisualConfig()
        self.sprite_cache: Dict[str, QPixmap] = {}  # Cache for loaded/tinted sprites
        self.world_object_images: Dict[str, List[QPixmap]] = {}  # Cache for world object images
        self.world_object_frame: int = 0  # Animation frame counter
        
        # Background image
        self.background_image = None
        self._load_background()
        
        # God Mode State
        self.current_tool = ToolType.SELECT
        self.current_template = None # Dict containing creature data
        self.is_dragging = False
        self.last_mouse_pos = None
        
    def _get_world_object_images(self, obj_type: str) -> List[QPixmap]:
        """Get cached images for a world object type."""
        if obj_type not in self.world_object_images:
            images = []
            if self.visual_config and _HAS_SETTINGS:
                config = self.visual_config.get_world_object(obj_type)
                if config and config.image_paths:
                    for path in config.image_paths:
                        try:
                            pixmap = QPixmap(path)
                            if not pixmap.isNull():
                                images.append(pixmap)
                        except Exception:
                            pass
            # Only cache if we actually found images - otherwise check again next frame
            if images:
                self.world_object_images[obj_type] = images
            return images
        return self.world_object_images[obj_type]
    
    def _get_world_object_config(self, obj_type: str):
        """Get config for a world object type."""
        if self.visual_config and _HAS_SETTINGS:
            return self.visual_config.get_world_object(obj_type)
        return None
    
    def _clear_world_object_cache(self):
        """Clear the world object image cache (call when config changes)."""
        self.world_object_images.clear()
        
    def _find_2d_regions(self, tile_type: int) -> List[tuple]:
        """Find rectangular 2D regions of a tile type.
        Returns list of (start_tx, start_ty, width, height) tuples."""
        regions = []
        visited = set()
        
        for ty in range(self.world.tiles_y):
            for tx in range(self.world.tiles_x):
                if (tx, ty) in visited:
                    continue
                
                if self.world.tiles[ty, tx] == tile_type:
                    # Found top-left of a potential region
                    start_tx = tx
                    start_ty = ty
                    
                    # Measure width
                    width = 0
                    while (tx + width < self.world.tiles_x and 
                           self.world.tiles[ty, tx + width] == tile_type and
                           (tx + width, ty) not in visited):
                        width += 1
                    
                    # Measure height (checking if subsequent rows match this width)
                    height = 1
                    while (ty + height < self.world.tiles_y):
                        # Check if row below matches the width
                        row_match = True
                        for w in range(width):
                            if (self.world.tiles[ty + height, tx + w] != tile_type or
                                (tx + w, ty + height) in visited):
                                row_match = False
                                break
                        if row_match:
                            height += 1
                        else:
                            break
                    
                    # Mark as visited
                    for h in range(height):
                        for w in range(width):
                            visited.add((tx + w, ty + h))
                    
                    regions.append((start_tx, start_ty, width, height))
        
        return regions
    
    def _load_background(self):
        """Load background images from visual config."""
        self.background_images: Dict[int, QPixmap] = {}
        
        # Default fallback
        try:
            from pathlib import Path
            base_path = Path(__file__).parent.parent
            default_bg_path = base_path / "images" / "BackgroundForest.png"
            if default_bg_path.exists():
                self.background_image = QPixmap(str(default_bg_path)) # Keep legacy for safety
            else:
                self.background_image = QPixmap()
        except Exception as e:
            print(f"Failed to load default background: {e}")
            self.background_image = QPixmap()

        # Load indexed backgrounds from config
        if self.visual_config and hasattr(self.visual_config, 'indexed_backgrounds'):
            for idx, path in self.visual_config.indexed_backgrounds.items():
                try:
                    pix = QPixmap(path)
                    if not pix.isNull():
                        self.background_images[idx] = pix
                except Exception as e:
                    print(f"Failed to load background {idx}: {e}")
                    
        # If no indexed backgrounds but we have a default/legacy, assign to 0
        if 0 not in self.background_images and not self.background_image.isNull():
             self.background_images[0] = self.background_image
             
        # Calculate world boundaries based on indices
        if self.background_images:
            indices = list(self.background_images.keys())
            self.min_world_idx = min(indices)
            self.max_world_idx = max(indices)
        else:
            self.min_world_idx = 0
            self.max_world_idx = 0
            
        # Push to world if exists
        self._apply_bounds_to_world()

    def _apply_bounds_to_world(self):
        """Apply calculated horizontal bounds to the physics world.
        
        All backgrounds are scaled to match the tallest one's height.
        Widths are adjusted proportionally to maintain aspect ratio.
        Zones connect seamlessly with no padding.
        """
        if self.world:
            self.zone_boundaries = {}
            
            if self.background_images:
                sorted_indices = sorted(self.background_images.keys())
                
                # Find the tallest background
                max_height = 0
                for idx in sorted_indices:
                    bg = self.background_images[idx]
                    if not bg.isNull() and bg.height() > max_height:
                        max_height = bg.height()
                
                if max_height == 0:
                    max_height = 1080  # Default
                
                # Build zones with scaled widths (all normalized to max_height)
                current_x = 0
                for idx in sorted_indices:
                    bg = self.background_images[idx]
                    if not bg.isNull():
                        native_w = bg.width()
                        native_h = bg.height()
                        # Scale factor to make this bg's height match max_height
                        height_scale = max_height / native_h
                        scaled_width = int(native_w * height_scale)
                        
                        self.zone_boundaries[idx] = {
                            'start': current_x,
                            'end': current_x + scaled_width,
                            'width': scaled_width,
                            'height': max_height,  # All zones same height
                            'native_width': native_w,
                            'native_height': native_h,
                            'height_scale': height_scale
                        }
                        current_x += scaled_width
                
                if self.zone_boundaries:
                    self.world.min_x = 0
                    self.world.max_x = current_x
                    self.world.zone_boundaries = self.zone_boundaries
                    print(f"[World] Zones (height-normalized): {[(idx, z['start'], z['end'], z['width']) for idx, z in self.zone_boundaries.items()]}")
                else:
                    self._set_default_bounds()
            else:
                self._set_default_bounds()
            
            # Respawn food/water across all zones
            self.world.respawn_across_zones()
    
    def _set_default_bounds(self):
        """Set default world bounds when no backgrounds are loaded."""
        self.world.min_x = 0
        self.world.max_x = 1920
        self.world.zone_boundaries = {0: {'start': 0, 'end': 1920, 'width': 1920, 'height': 1080}}
        self.zone_boundaries = self.world.zone_boundaries
            # print(f"World Bounds set to: {self.world.min_x} to {self.world.max_x}")
        
    def set_world(self, world: World):
        """Set the world to render."""
        self.world = world
        self._apply_bounds_to_world()
        self.update()
    
    def set_creatures(self, creatures: List[LivingCreature]):
        """Set creatures to render."""
        self.creatures = creatures
        self.update()
    
    def paintEvent(self, event):
        """Render the world and creatures."""
        # Increment animation frame for world objects (only when playing)
        if not self.paused:
            self.world_object_frame += 1
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.world is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                           "No world loaded\nClick 'New World' to create one")
            painter.end()
            return
        
        # Get zone boundaries (all normalized to same height)
        zone_boundaries = getattr(self, 'zone_boundaries', None) or getattr(self.world, 'zone_boundaries', None)
        
        # All zones have the same normalized height
        world_height = 1080  # Default
        if zone_boundaries:
            first_zone = zone_boundaries.get(min(zone_boundaries.keys()))
            if first_zone:
                world_height = first_zone.get('height', 1080)
        
        # Scale: screen height = world height (at zoom=1)
        scale = (self.height() / world_height) * self.zoom
        
        # Camera offset - camera_x is in world coordinates
        offset_x = -self.camera_x * scale
        
        # === BACKGROUND RENDERING (All Same Height) ===
        # Draw void background first
        painter.fillRect(self.rect(), QColor(30, 30, 40))
        
        if zone_boundaries:
            for idx in sorted(zone_boundaries.keys()):
                zone_info = zone_boundaries[idx]
                bg_pixmap = self.background_images.get(idx)
                
                if bg_pixmap and not bg_pixmap.isNull():
                    # Zone's normalized dimensions
                    zone_width = zone_info['width']  # Already scaled to match height
                    zone_height = zone_info['height']  # All same height
                    
                    # Screen dimensions
                    screen_w = int(zone_width * scale)
                    screen_h = int(zone_height * scale)
                    
                    # Screen position from zone start
                    screen_x = int(offset_x + zone_info['start'] * scale)
                    
                    # Align to bottom of screen (ground stays at bottom)
                    screen_y = self.height() - screen_h
                    
                    # Only render if visible
                    if screen_x + screen_w > 0 and screen_x < self.width():
                        # Scale image to fill zone (already aspect-correct from zone calc)
                        scaled_bg = bg_pixmap.scaled(
                            screen_w, screen_h,
                            Qt.AspectRatioMode.IgnoreAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                        painter.drawPixmap(screen_x, screen_y, scaled_bg)

        # === WORLD RENDERING ===
        
        # Align world objects with backgrounds (both anchored at bottom)
        screen_world_h = int(world_height * scale)
        offset_y = self.height() - screen_world_h

        # Draw tiles
        tile_size = self.world.tile_size * scale
        
        # Tile colors based on type and lighting
        light = self.world.light_level
        
        # If we have a background image, only draw water/hazards/shelter on top
        has_bg = bool(self.background_images) or (self.background_image and not self.background_image.isNull())
        
        # Track which tiles we've drawn with stretched images (non-tile mode)
        drawn_tiles = set()
        
        # First pass: draw stretched images for non-tile-mode objects
        if _HAS_SETTINGS:
            tile_type_map = {
                TileType.WATER.value: WorldObjectType.WATER,
                TileType.HAZARD.value: WorldObjectType.HAZARD,
                TileType.GROUND.value: WorldObjectType.GROUND,
                TileType.SHELTER.value: WorldObjectType.SHELTER,
            }
            
            for tile_val, obj_type in tile_type_map.items():
                config = self._get_world_object_config(obj_type)
                images = self._get_world_object_images(obj_type)
                
                # Only do region-based drawing if NOT tile mode and we have images
                if config and images and not config.tile_mode:
                    # Use 2D regions for better object merging (e.g. ponds)
                    regions = self._find_2d_regions(tile_val)
                    frame_idx = (self.world_object_frame // max(1, int(config.animation_speed * 30))) % len(images)
                    img = images[frame_idx]
                    
                    for start_tx, start_ty, width, height in regions:
                        # Calculate region bounds
                        rx = int(offset_x + start_tx * tile_size)
                        ry = int(offset_y + start_ty * tile_size)
                        rw = int(width * tile_size)
                        rh = int(height * tile_size)
                        
                        # For non-tiled objects, use aspect-ratio preserving scaling
                        # centered on the region, prioritizing the image's look over exact region filling.
                        if width > 0 and height > 0:
                            img_aspect = img.width() / img.height() if img.height() > 0 else 1.0
                            
                            # Reference size: Use a fixed relative height (e.g. 4 tiles) modified by config scale
                            # This ensures size depends on settings, not random generation.
                            ref_height_tiles = 4.0
                            target_h = int(tile_size * ref_height_tiles * config.scale)
                            target_w = int(target_h * img_aspect)
                            
                            scaled_img = img.scaled(target_w, target_h, 
                                                   Qt.AspectRatioMode.KeepAspectRatio,
                                                   Qt.TransformationMode.SmoothTransformation)
                            
                            # Center the image on the region
                            center_x = rx + rw // 2
                            center_y = ry + rh // 2
                            img_x = center_x - target_w // 2
                            img_y = center_y - target_h // 2
                            
                            painter.drawPixmap(img_x, img_y, scaled_img)
                        
                        # Mark these tiles as drawn
                        for h in range(height):
                            for w in range(width):
                                drawn_tiles.add((start_tx + w, start_ty + h))
        
        # Second pass: draw individual tiles (tiled mode or fallback colors)
        # Tiles only exist in zone 0 - they don't wrap to other zones
        for ty in range(self.world.tiles_y):
            for tx in range(self.world.tiles_x):
                # Skip if already drawn with stretched image
                if (tx, ty) in drawn_tiles:
                    continue
                
                tile = self.world.tiles[ty, tx]
                
                # Skip empty tiles if using background image (Background shows through)
                if has_bg and tile == TileType.EMPTY.value:
                    continue
                
                x = int(offset_x + tx * tile_size)
                y = int(offset_y + ty * tile_size)
                ts = int(tile_size) + 1
                
                # Skip if off-screen (optimization)
                if x + ts < 0 or x > self.width():
                    continue
                
                # Try to use images for world objects (tile mode)
                obj_type = None
                if tile == TileType.WATER.value and _HAS_SETTINGS:
                    obj_type = WorldObjectType.WATER
                elif tile == TileType.HAZARD.value and _HAS_SETTINGS:
                    obj_type = WorldObjectType.HAZARD
                elif tile == TileType.GROUND.value and _HAS_SETTINGS:
                    obj_type = WorldObjectType.GROUND
                elif tile == TileType.SHELTER.value and _HAS_SETTINGS:
                    obj_type = WorldObjectType.SHELTER
                
                # Draw with image if available and in tile mode
                if obj_type:
                    config = self._get_world_object_config(obj_type)
                    images = self._get_world_object_images(obj_type)
                    
                    # If we have images but NOT in tile mode, skip this tile
                    # (it was already drawn in the stretched region pass)
                    if images and config and not config.tile_mode:
                        continue  # Already drawn as stretched region
                    
                    if images and config and config.tile_mode:
                        # Pick frame for animation
                        frame_idx = (self.world_object_frame // max(1, int(config.animation_speed * 30))) % len(images)
                        img = images[frame_idx]
                        
                        # Apply scale
                        scaled_ts = int(ts * config.scale)
                        scaled = img.scaled(scaled_ts, scaled_ts, 
                                          Qt.AspectRatioMode.IgnoreAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
                        
                        # Draw centered
                        img_x = x + (ts - scaled_ts) // 2
                        img_y = y + (ts - scaled_ts) // 2
                        painter.drawPixmap(img_x, img_y, scaled)
                        continue  # Skip color fallback
                
                # Base colors for tiles that need rendering (fallback)
                if tile == TileType.EMPTY.value:
                    # Sky color based on time of day
                    r = int(40 + 60 * light)
                    g = int(44 + 100 * light)
                    b = int(80 + 120 * light)
                    color = QColor(r, g, b)
                elif tile == TileType.GROUND.value:
                    color = QColor(int(80 * light + 20), int(55 * light + 12), int(27 * light + 6))
                elif tile == TileType.WATER.value:
                    # Semi-transparent water so background shows through
                    color = QColor(int(50 * light + 14), int(130 * light + 34), int(180 * light + 43), 180)
                elif tile == TileType.STONE.value:
                    # Stone block
                    color = QColor(int(100 * light), int(100 * light), int(110 * light))
                elif tile == TileType.HAZARD.value:
                    # Hazards glow
                    color = QColor(255, int(80 + 40 * np.sin(self.world.time * 3)), 34, 200)
                elif tile == TileType.SHELTER.value:
                    color = QColor(int(60 * light + 16), int(140 * light + 35), int(60 * light + 20), 150)
                else:
                    color = QColor(128, 128, 128)
                
                painter.fillRect(x, y, ts, ts, color)
        
        # Draw food sources
        # Pre-fetch generic config
        generic_food_images = self._get_world_object_images(WorldObjectType.FOOD) if _HAS_SETTINGS else []
        generic_food_config = self._get_world_object_config(WorldObjectType.FOOD) if _HAS_SETTINGS else None
        
        for food in self.world.food_sources:
            x = int(offset_x + food.x * scale)
            y = int(offset_y + food.y * scale)
            # Base size: 60-120 pixels depending on remaining (50% larger), scaled by view zoom
            base_size = int((60 + 60 * food.remaining) * scale)
            
            # Apply growth scaling (Cultivation)
            growth = getattr(food, 'growth_stage', 1.0)
            if growth < 1.0:
                base_size = int(base_size * (0.3 + 0.7 * growth)) # Start small (30%)
            
            # Determine image key based on specific type
            # e.g. "food_sweet_berry", "food_plant"
            food_type_val = food.type.value if hasattr(food.type, 'value') else str(food.type)
            type_key = f"food_{food_type_val}"
            
            specific_images = self._get_world_object_images(type_key)
            specific_config = self._get_world_object_config(type_key)
            
            # Decide which images to use
            active_images = specific_images if specific_images else generic_food_images
            active_config = specific_config if specific_images else generic_food_config
            
            scale_factor = active_config.scale if active_config else 1.0
            size = int(base_size * scale_factor)
            
            # Try to use food image
            if active_images:
                anim_speed = active_config.animation_speed if active_config else 0.2
                frame_idx = (self.world_object_frame // max(1, int(anim_speed * 30))) % len(active_images)
                img = active_images[frame_idx]
                scaled = img.scaled(int(size), int(size), Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
                painter.drawPixmap(int(x - scaled.width()/2), int(y - scaled.height()/2), scaled)
            else:
                # Fallback to colored ellipse
                # Fallback to colored ellipse
                if food.type == FoodType.PLANT:
                    color = QColor(139, 195, 74)
                elif food.type == FoodType.SWEET_BERRY:
                    color = QColor(255, 60, 60) # Red
                elif food.type == FoodType.BITTER_BERRY:
                    color = QColor(60, 60, 255) # Blue
                elif food.type == FoodType.POISON_BERRY:
                    color = QColor(0, 255, 0) # Toxic Green
                else: # MEAT
                    color = QColor(244, 67, 54)
                
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(0, 0, 0, 100)))
                painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))
        
        # Draw tools (sticks, stones, etc.) - TIER 4
        for tool in self.world.tools:
            # Only draw if not being held by a creature (thrown tools ARE drawn)
            if tool.held_by is not None:
                continue
            
            x = int(offset_x + tool.x * scale)
            y = int(offset_y + tool.y * scale)
            
            # Get tool type key for image lookup (e.g., "tool_stick", "tool_stone")
            tool_type_val = tool.tool_type.value if hasattr(tool.tool_type, 'value') else str(tool.tool_type)
            type_key = f"tool_{tool_type_val}"
            
            # Try to load images for this specific tool type
            tool_images = self._get_world_object_images(type_key) if _HAS_SETTINGS else []
            tool_config = self._get_world_object_config(type_key) if _HAS_SETTINGS else None
            
            # Base size for tools (50% larger)
            base_size = int(60 * scale)
            scale_factor = tool_config.scale if tool_config else 1.0
            tool_size = int(base_size * scale_factor)
            
            # Try to use tool image
            if tool_images:
                anim_speed = tool_config.animation_speed if tool_config else 0.2
                frame_idx = (self.world_object_frame // max(1, int(anim_speed * 30))) % len(tool_images)
                img = tool_images[frame_idx]
                
                # Special sizing for different tool types
                if tool_type_val == "stick":
                    # Sticks should be wider
                    img_w = int(tool_size * 1.5)
                    img_h = int(tool_size * 0.4)
                else:
                    img_w = img_h = tool_size
                
                scaled = img.scaled(int(img_w), int(img_h), 
                                   Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
                painter.drawPixmap(int(x - scaled.width()/2), int(y - scaled.height()/2), scaled)
            else:
                # Fallback to procedural drawing (50% larger)
                if tool_type_val == "stick":
                    # Draw stick as brown rectangle
                    tool_w = int(90 * scale)  # 60 * 1.5
                    tool_h = int(12 * scale)  # 8 * 1.5
                    painter.setBrush(QBrush(QColor(139, 90, 43)))  # Brown
                    painter.setPen(QPen(QColor(80, 50, 20), 2))
                    painter.drawRect(int(x - tool_w/2), int(y - tool_h/2), tool_w, tool_h)
                    
                elif tool_type_val == "stone":
                    # Draw stone as grey circle
                    painter.setBrush(QBrush(QColor(120, 120, 130)))  # Grey
                    painter.setPen(QPen(QColor(60, 60, 70), 2))
                    painter.drawEllipse(int(x - tool_size/2), int(y - tool_size/2), tool_size, tool_size)
                    
                elif tool_type_val == "leaf":
                    # Draw leaf as green diamond
                    painter.setBrush(QBrush(QColor(100, 200, 80)))  # Green
                    painter.setPen(QPen(QColor(50, 100, 40), 2))
                    points = [
                        QPoint(x, int(y - tool_size/2)),
                        QPoint(int(x + tool_size/2), y),
                        QPoint(x, int(y + tool_size/2)),
                        QPoint(int(x - tool_size/2), y)
                    ]
                    painter.drawPolygon(points)
                    
                elif tool_type_val == "shell":
                    # Draw shell as beige oval
                    tool_w = int(52 * scale)  # 35 * 1.5
                    tool_h = int(38 * scale)  # 25 * 1.5
                    painter.setBrush(QBrush(QColor(245, 222, 179)))  # Beige
                    painter.setPen(QPen(QColor(180, 150, 120), 2))
                    painter.drawEllipse(int(x - tool_w/2), int(y - tool_h/2), tool_w, tool_h)
                    
                elif tool_type_val == "bone":
                    # Draw bone as white dumbbell shape
                    tool_w = int(75 * scale)  # 50 * 1.5
                    tool_h = int(15 * scale)  # 10 * 1.5
                    painter.setBrush(QBrush(QColor(240, 230, 220)))  # Off-white
                    painter.setPen(QPen(QColor(200, 190, 180), 2))
                    painter.drawRect(int(x - tool_w/2), int(y - tool_h/2), tool_w, tool_h)
                    # End caps
                    cap_size = int(22 * scale)  # 15 * 1.5
                    painter.drawEllipse(int(x - tool_w/2 - cap_size/4), int(y - cap_size/2), cap_size, cap_size)
                    painter.drawEllipse(int(x + tool_w/2 - cap_size*0.75), int(y - cap_size/2), cap_size, cap_size)
                
                # System 6: New Tools Fallbacks
                elif tool_type_val == "nest":
                    # Nest: Concentric brown circles
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(QColor(101, 67, 33), 3))
                    painter.drawEllipse(int(x - tool_size/2), int(y - tool_size/2), tool_size, tool_size)
                    painter.drawEllipse(int(x - tool_size/3), int(y - tool_size/3), int(tool_size*0.66), int(tool_size*0.66))
                    
                elif tool_type_val == "hammer":
                    # Hammer: Stick handle + Stone head
                    # Handle
                    painter.setBrush(QBrush(QColor(139, 90, 43)))
                    painter.setPen(QPen(QColor(80, 50, 20), 2))
                    painter.drawRect(int(x - 4), int(y - tool_size/2), 8, int(tool_size))
                    # Head
                    painter.setBrush(QBrush(QColor(100, 100, 100)))
                    painter.drawRect(int(x - 15), int(y - tool_size/2), 30, 20)
                    
                elif tool_type_val == "sharp_rock":
                    # Sharp Rock: Triangle
                    painter.setBrush(QBrush(QColor(140, 140, 140)))
                    painter.setPen(QPen(QColor(60, 60, 60), 2))
                    points = [
                        QPoint(x, int(y - tool_size/2)),
                        QPoint(int(x + tool_size/2), int(y + tool_size/2)),
                        QPoint(int(x - tool_size/2), int(y + tool_size/2))
                    ]
                    painter.drawPolygon(points)
                    
                elif tool_type_val == "spear":
                    # Spear: Long stick + triangle tip
                    # Stick
                    painter.setBrush(QBrush(QColor(139, 90, 43))) 
                    painter.setPen(QPen(QColor(80, 50, 20), 2))
                    painter.drawRect(int(x - 3), int(y - tool_size), 6, int(tool_size * 2))
                    # Tip
                    painter.setBrush(QBrush(QColor(180, 180, 180)))
                    points = [
                        QPoint(x, int(y - tool_size - 10)),
                        QPoint(int(x + 6), int(y - tool_size)),
                        QPoint(int(x - 6), int(y - tool_size))
                    ]
                    painter.drawPolygon(points)

            # Draw Label (System 6 Requirement)
            # Reusing images requires labels
            if scale > 0.5:
                # Draw label just above tool center with fixed small gap
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(QFont("Arial", 8))
                label = tool_type_val.replace('_', ' ').title()
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(label)
                th = fm.height()
                
                # Fixed small gap above tool center (5px gap)
                label_y = int(y - th - 5)
                
                # Draw bg directly above tool center
                painter.fillRect(int(x - tw/2 - 2), label_y, tw + 4, th, QColor(0, 0, 0, 150))
                
                # Draw text
                painter.drawText(int(x - tw/2), label_y + th - 2, label)
        
        # Draw water sources (zone-specific water pools)
        if hasattr(self.world, 'water_sources'):
            water_images = self._get_world_object_images(WorldObjectType.WATER) if _HAS_SETTINGS else []
            water_config = self._get_world_object_config(WorldObjectType.WATER) if _HAS_SETTINGS else None
            
            # Get scale factor from settings (default 1.0)
            water_scale_factor = water_config.scale if water_config else 1.0
            
            for water in self.world.water_sources:
                x = int(offset_x + water.x * scale)
                y = int(offset_y + water.y * scale)
                w = int(water.width * scale * water_scale_factor)
                h = int(water.height * scale * water_scale_factor)
                
                # Try to use water image
                if water_images and water_config:
                    anim_speed = water_config.animation_speed
                    frame_idx = (self.world_object_frame // max(1, int(anim_speed * 30))) % len(water_images)
                    img = water_images[frame_idx]
                    
                    # Scale water image to fit the water source size
                    scaled = img.scaled(w, h, Qt.AspectRatioMode.IgnoreAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
                    painter.drawPixmap(int(x - w/2), int(y - h/2), scaled)
                else:
                    # Fallback: draw blue rectangle
                    painter.setBrush(QBrush(QColor(66, 165, 245, 180)))  # Semi-transparent blue
                    painter.setPen(QPen(QColor(33, 150, 243, 200), 2))
                    painter.drawRect(int(x - w/2), int(y - h/2), w, h)
        
        # Draw creatures
        for creature in self.creatures:
            self._draw_creature(painter, creature, scale, offset_x, offset_y)
        
        # Draw Mating Hearts
        mating_pairs = getattr(self, 'mating_pairs', [])
        if mating_pairs:
            painter.setFont(QFont("Segoe UI Emoji", 24))
            for pair in mating_pairs:
                if not pair['p1'].body.is_alive() or not pair['p2'].body.is_alive():
                    continue
                
                # Calculate center position
                c1x = int(offset_x + pair['p1'].body.motor.x * scale)
                c1y = int(offset_y + pair['p1'].body.motor.y * scale)
                c2x = int(offset_x + pair['p2'].body.motor.x * scale)
                c2y = int(offset_y + pair['p2'].body.motor.y * scale)
                
                mx = (c1x + c2x) // 2
                my = min(c1y, c2y) - 30 # Floating above
                
                # Bobbing animation
                offset = int(np.sin(self.world.time * 10) * 5)
                
                painter.drawText(mx - 15, my + offset, "💕")

        # Draw ZZZ interaction (Sleep)
        painter.setFont(QFont("Segoe UI", 12)) 
        for creature in self.creatures:
            if creature.body.homeostasis.is_sleeping and creature.body.is_alive():
                cx = int(offset_x + creature.body.motor.x * scale)
                cy = int(offset_y + creature.body.motor.y * scale)
                
                # Animate ZZZs
                t = self.world.time
                for i in range(3):
                    # Staggered Zs
                    z_offset = (t * 2 + i * 2) % 6  # 0 to 6 seconds cycle
                    if z_offset < 2.0: # Visible for 2 seconds
                        alpha = int(255 * (1 - z_offset/2.0))
                        painter.setPen(QColor(150, 150, 220, alpha))
                        
                        dx = int(10 + i * 8 + np.sin(t * 3 + i) * 5)
                        dy = int(-60 - z_offset * 15)  # Start higher (-60 instead of -20)
                        
                        painter.drawText(cx + dx, cy + dy, "z")
                
                # Dream Sparkles (Visualizing Replay)
                if np.random.random() < 0.1:
                    painter.setBrush(QBrush(QColor(255, 255, 100, 200))) # Yellow spark
                    painter.setPen(Qt.PenStyle.NoPen)
                    sx = cx + int(np.random.normal(0, 15))
                    sy = cy - 65 + int(np.random.normal(0, 10))  # Higher up (-65 instead of -25)
                    painter.drawEllipse(sx, sy, 3, 3)
        
        # Draw weather overlay
        if self.world.weather.name == "RAIN":
            painter.setPen(QPen(QColor(100, 150, 255, 50)))
            for _ in range(50):
                rx = np.random.randint(0, self.width())
                ry = np.random.randint(0, self.height())
                painter.drawLine(rx, ry, rx - 2, ry + 10)
        elif self.world.weather.name == "SNOW":
            painter.setPen(QPen(QColor(255, 255, 255, 100)))
            for _ in range(30):
                rx = np.random.randint(0, self.width())
                ry = np.random.randint(0, self.height())
                painter.drawEllipse(rx, ry, 3, 3)
        
        # Draw time indicator
        painter.setPen(QColor(200, 200, 200))
        time_str = f"Day {int(self.world.time / self.world.day_length)} "
        day_progress = (self.world.time % self.world.day_length) / self.world.day_length
        if day_progress < 0.25:
            time_str += "🌅 Dawn"
        elif day_progress < 0.5:
            time_str += "☀️ Day"
        elif day_progress < 0.75:
            time_str += "🌅 Dusk"
        else:
            time_str += "🌙 Night"
        painter.drawText(10, 20, time_str)
        
        # Weather indicator
        weather_icons = {
            "CLEAR": "☀️",
            "RAIN": "🌧️",
            "STORM": "⛈️",
            "SNOW": "❄️",
            "HEAT_WAVE": "🔥"
        }
        painter.drawText(10, 40, f"Weather: {weather_icons.get(self.world.weather.name, '?')} {self.world.weather.name}")
        
        # Creature count
        alive = sum(1 for c in self.creatures if c.body.is_alive())
        painter.drawText(10, 60, f"Creatures: {alive}/{len(self.creatures)}")
        
        # === DAY/NIGHT OVERLAY ===
        # Darken screen at night
        if self.world.light_level < 1.0:
            darkness = int((1.0 - self.world.light_level) * 180) # Max darkness alpha 180/255
            painter.fillRect(self.rect(), QColor(0, 0, 20, darkness))
            
        painter.end()
    
    def _draw_creature(self, painter: QPainter, creature: LivingCreature, 
                       scale: float, offset_x: float, offset_y: float):
        """Draw a single creature (2.25x size for visibility - 50% larger than before)."""
        body = creature.body
        pheno = body.phenotype
        
        # 2.25x SIZE MULTIPLIER for visibility (50% larger than 1.5x)
        SIZE_MULT = 2.25
        
        if not body.is_alive():
            # Dead creatures are grey and faded
            color = QColor(100, 100, 100, 100)
        else:
            r, g, b = pheno.get_rgb()
            color = QColor(r, g, b)
        
        x = int(offset_x + body.motor.x * scale)
        y = int(offset_y + body.motor.y * scale)
        w = int(pheno.width * scale * pheno.size * SIZE_MULT)
        h = int(pheno.height * scale * pheno.size * SIZE_MULT)
        
        # Try to draw with sprites from visual config
        if self._draw_creature_sprites(painter, creature, x, y, w, h, scale, SIZE_MULT):
            # Sprites were drawn, just add selection indicator and health bar
            # Scaling box up to match sprite visual size (approx 2.5x physics body)
            vis_w = int(w * 2.5)
            vis_h = int(h * 2.5)
            
            if creature == self.selected_creature:
                painter.setPen(QPen(QColor(255, 255, 0), 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                # Align bottom of box with feet (y + h//2)
                box_top = (y + h//2) - vis_h
                painter.drawRect(x - vis_w//2, box_top, vis_w, vis_h)
            elif creature == self.selected_secondary:
                painter.setPen(QPen(QColor(0, 255, 255), 3)) # Cyan for secondary
                painter.setBrush(Qt.BrushStyle.NoBrush)
                # Align bottom of box with feet (y + h//2)
                box_top = (y + h//2) - vis_h
                painter.drawRect(x - vis_w//2, box_top, vis_w, vis_h)
        else:
            # Fall back to procedural drawing
            # Body
            painter.setBrush(QBrush(color))
            outline_color = QColor(0, 0, 0)
            if creature == self.selected_creature:
                outline_color = QColor(255, 255, 0)
            elif creature == self.selected_secondary:
                outline_color = QColor(0, 255, 255)
            
            painter.setPen(QPen(outline_color, 3 if (creature == self.selected_creature or creature == self.selected_secondary) else 2))
            
            # Main body ellipse
            painter.drawEllipse(x - w//2, y - h//2, w, h)
            
            # Pattern (scaled for larger size)
            if pheno.pattern_type == "stripes":
                pr, pg, pb = pheno.get_pattern_rgb()
                painter.setPen(QPen(QColor(pr, pg, pb), 2))
                for i in range(int(3 * pheno.pattern_density)):
                    stripe_y = y - h//2 + int((i + 1) * h / (3 * pheno.pattern_density + 1))
                    painter.drawLine(x - w//3, stripe_y, x + w//3, stripe_y)
            elif pheno.pattern_type == "spots":
                pr, pg, pb = pheno.get_pattern_rgb()
                painter.setBrush(QBrush(QColor(pr, pg, pb)))
                painter.setPen(Qt.PenStyle.NoPen)
                spot_size = int(4 * SIZE_MULT)
                # Use creature's hash as seed for consistent spot positions
                rng = np.random.RandomState(hash(id(creature)) & 0xFFFFFFFF)
                for _ in range(int(5 * pheno.pattern_density)):
                    sx = x + rng.randint(-w//3, max(1, w//3))
                    sy = y + rng.randint(-h//3, max(1, h//3))
                    painter.drawEllipse(sx - spot_size//2, sy - spot_size//2, spot_size, spot_size)
            
            # Eyes (indicate facing direction) - scaled for 1.5x size
            eye_size = int(8 * SIZE_MULT)
            pupil_size = int(4 * SIZE_MULT)
            eye_offset = int(6 * SIZE_MULT) if body.motor.facing_right else int(-6 * SIZE_MULT)
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawEllipse(x + eye_offset - eye_size//2, y - h//4 - eye_size//2, eye_size, eye_size)
            
            # Pupil
            painter.setBrush(QBrush(QColor(0, 0, 0)))
            pupil_offset = int(2 * SIZE_MULT) if body.motor.facing_right else int(-2 * SIZE_MULT)
            painter.drawEllipse(x + eye_offset + pupil_offset - pupil_size//2, y - h//4 - pupil_size//2, pupil_size, pupil_size)
            
            # Limbs (simple lines) - scaled for 1.5x size
            if pheno.limb_count >= 2:
                painter.setPen(QPen(color.darker(120), int(3 * SIZE_MULT)))
                # Legs
                leg_phase = body.motor.animation_frame * 0.5
                leg_length = int(12 * scale * SIZE_MULT)
                for i in range(min(2, pheno.limb_count)):
                    leg_x = x + (w//4 if i == 0 else -w//4)
                    leg_offset = int(6 * np.sin(leg_phase + i * np.pi) * scale * SIZE_MULT)
                    painter.drawLine(leg_x, y + h//2, leg_x + leg_offset, y + h//2 + leg_length)
        
        # Health bar (only if not full)
        if body.homeostasis.health < 1.0:
            bar_w = w
            bar_h = 3
            painter.fillRect(x - bar_w//2, y - h//2 - 8, bar_w, bar_h, QColor(100, 0, 0))
            painter.fillRect(x - bar_w//2, y - h//2 - 8, 
                           int(bar_w * body.homeostasis.health), bar_h, QColor(0, 200, 0))
        
        # Energy indicator (small bar)
        if body.homeostasis.energy < 0.5:
            bar_w = int(w * 0.6)
            painter.fillRect(x - bar_w//2, y - h//2 - 4, 
                           int(bar_w * body.homeostasis.energy * 2), 2, QColor(255, 200, 0))
        
        # Fatigue indicator (Blue bar) - Only if tired
        if body.homeostasis.fatigue > 0.3:
            bar_w = int(w * 0.6)
            # Draw empty background
            painter.fillRect(x - bar_w//2, y - h//2 - 2, bar_w, 2, QColor(0, 0, 50))
            painter.fillRect(x - bar_w//2, y - h//2 - 2, 
                           int(bar_w * body.homeostasis.fatigue), 2, QColor(100, 100, 255))

        # NSM: Consolidation Progress Bar (Purple)
        # Only visible when sleeping
        if body.homeostasis.is_sleeping:
            progress = 0.0
            # Try to get progress from brain's sleep manager
            if hasattr(creature, 'embodied_brain') and creature.embodied_brain and \
               hasattr(creature.embodied_brain.brain, 'sleep_manager'):
                progress = creature.embodied_brain.brain.sleep_manager.consolidation_progress
            
            # Use max(0, min(1, ...)) to clamp
            progress = max(0.0, min(1.0, progress))
            
            bar_w = int(w * 0.8)
            # Purple background
            painter.fillRect(x - bar_w//2, y - h//2 - 14, bar_w, 4, QColor(30, 0, 40))
            # Bright purple fill
            painter.fillRect(x - bar_w//2, y - h//2 - 14, 
                           int(bar_w * progress), 4, QColor(180, 80, 255))
            
            # Text label
            if scale > 0.8:
                painter.setFont(QFont("Arial", 7))
                painter.setPen(QColor(220, 200, 255))
                painter.drawText(x - bar_w//2, y - h//2 - 16, "CONSOLIDATING")
            painter.fillRect(x - bar_w//2, y - h//2 - 12, bar_w, 2, QColor(0, 0, 50, 100))
            # Draw fill
            painter.fillRect(x - bar_w//2, y - h//2 - 12, 
                           int(bar_w * body.homeostasis.fatigue), 2, QColor(100, 100, 255))
        
        # Draw held tool attached to creature
        if hasattr(creature, 'held_tool') and creature.held_tool is not None:
            self._draw_held_tool(painter, creature, x, y, w, h, scale, body.motor.facing_right)
        
        # Feature: Draw speech bubble
        if creature.current_speech:
             self._draw_speech_bubble(painter, x, y, creature.current_speech, scale)

    def _draw_speech_bubble(self, painter, x, y, text, scale):
        """Draw a speech bubble above the creature."""
        # Bubble config
        padding = 5
        font = QFont("Arial", 8)
        painter.setFont(font)
        fm = QFontMetrics(font)
        text_w = fm.horizontalAdvance(text)
        text_h = fm.height()
        
        bubble_w = text_w + padding * 2
        bubble_h = text_h + padding * 2
        
        # Position above creature (adjusted for scale)
        # Assuming creature height 'h' is roughly 20-30 pixels at scale 1
        bx = int(x - bubble_w / 2)
        by = int(y - 50 * scale - bubble_h) 
        
        # Draw bubble
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 210))
        painter.drawRoundedRect(bx, by, bubble_w, bubble_h, 5, 5)
        
        # Draw tail
        path = QPainterPath()
        tail_tip_y = int(y - 30 * scale)
        path.moveTo(tail_tip_y, tail_tip_y) 
        path.moveTo(bx + bubble_w/2, by + bubble_h + 5) # Point
        path.lineTo(bx + bubble_w/2 - 5, by + bubble_h) # Base left
        path.lineTo(bx + bubble_w/2 + 5, by + bubble_h) # Base right
        # That path logic is a bit messy, let's simplify
        
        # Draw text
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(bx + padding, by + padding + fm.ascent() - 2, text)
    
    def _draw_held_tool(self, painter: QPainter, creature: 'LivingCreature',
                        x: int, y: int, w: int, h: int, scale: float, facing_right: bool):
        """Draw a tool held by the creature."""
        tool = creature.held_tool
        if tool is None:
            return
        
        # Get tool type first (needed for positioning)
        tool_type_val = tool.tool_type.value if hasattr(tool.tool_type, 'value') else str(tool.tool_type)
        
        # Position tool at creature center (no offset - attach directly)
        tool_x = x
        tool_y = y  # Centered vertically (at creature center)
        type_key = f"tool_{tool_type_val}"
        
        # Try to load images for this specific tool type
        tool_images = self._get_world_object_images(type_key) if _HAS_SETTINGS else []
        tool_config = self._get_world_object_config(type_key) if _HAS_SETTINGS else None
        
        # Size for held tools (match ground size)
        base_size = int(60 * scale)
        scale_factor = tool_config.scale if tool_config else 1.0
        tool_size = int(base_size * scale_factor)
        
        if tool_images:
            anim_speed = tool_config.animation_speed if tool_config else 0.2
            frame_idx = (self.world_object_frame // max(1, int(anim_speed * 30))) % len(tool_images)
            img = tool_images[frame_idx]
            
            # Sizing
            if tool_type_val == "stick":
                # Match ground tool dimensions
                img_w = int(tool_size * 1.5)
                img_h = int(tool_size * 0.4)
            else:
                img_w = img_h = tool_size
            
            scaled = img.scaled(int(img_w), int(img_h), 
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
            
            # Flip image if facing left
            if not facing_right:
                scaled = scaled.transformed(QTransform().scale(-1, 1))
            
            painter.drawPixmap(int(tool_x - img_w // 2), int(tool_y - img_h // 2), scaled)
        else:
            # Fallback: draw simple shapes
            if tool_type_val == "stick":
                # Brown rectangle (stick)
                stick_w = int(tool_size * 1.2)
                stick_h = int(tool_size * 0.2)
                painter.setBrush(QBrush(QColor(139, 90, 43)))  # Brown
                painter.setPen(QPen(QColor(80, 50, 20), 1))
                # Rotate slightly for held appearance
                painter.save()
                painter.translate(tool_x, tool_y)
                angle = -30 if facing_right else 30
                painter.rotate(angle)
                painter.drawRect(-stick_w // 2, -stick_h // 2, stick_w, stick_h)
                painter.restore()
            elif tool_type_val == "stone":
                # Gray ellipse (stone)
                stone_size = int(tool_size * 0.6)
                painter.setBrush(QBrush(QColor(120, 120, 120)))
                painter.setPen(QPen(QColor(80, 80, 80), 1))
                painter.drawEllipse(tool_x - stone_size // 2, tool_y - stone_size // 2, 
                                   stone_size, stone_size)
            elif tool_type_val == "leaf":
                # Green ellipse (leaf)
                leaf_w = int(tool_size * 0.8)
                leaf_h = int(tool_size * 0.5)
                painter.setBrush(QBrush(QColor(34, 139, 34)))  # Forest green
                painter.setPen(QPen(QColor(0, 100, 0), 1))
                painter.drawEllipse(tool_x - leaf_w // 2, tool_y - leaf_h // 2, 
                                   leaf_w, leaf_h)
            else:
                # Generic tool indicator
                painter.setBrush(QBrush(QColor(200, 200, 200)))
                painter.setPen(QPen(QColor(100, 100, 100), 1))
                painter.drawRect(tool_x - tool_size // 4, tool_y - tool_size // 4,
                                tool_size // 2, tool_size // 2)

    def _draw_creature_sprites(self, painter: QPainter, creature: LivingCreature,
                                x: int, y: int, w: int, h: int, 
                                scale: float, size_mult: float) -> bool:
        """
        Draw creature using sprite images from visual config.
        
        Returns True if sprites were drawn, False to fall back to procedural.
        """
        if not _HAS_SETTINGS or not self.visual_config:
            return False
        
        body = creature.body
        pheno = body.phenotype
        age = body.homeostasis.age if hasattr(body.homeostasis, 'age') else 0.5
        
        # Get DNA value for image variant selection (using hue as default)
        dna_value = pheno.hue if hasattr(pheno, 'hue') else 0.5
        
        # Check if we have a full body sprite for current age
        full_body_sprite = self.visual_config.get_sprite(BodyPart.FULL_BODY, age)
        
        if full_body_sprite and full_body_sprite.image_paths:
            # Use full body sprite - get variant based on DNA
            image_path = full_body_sprite.get_image_for_dna(dna_value)
            if image_path:
                self._draw_single_sprite(painter, creature, full_body_sprite, image_path,
                                         x, y, w, h, scale, size_mult, pheno)
                return True
        
        # Otherwise, draw individual parts
        # Check if we have any sprites configured for this age
        has_sprites = False
        for part in BodyPart.INDIVIDUAL_PARTS:
            sprite_config = self.visual_config.get_sprite(part, age)
            if sprite_config and sprite_config.image_paths:
                has_sprites = True
                break
        
        if not has_sprites:
            return False
        
        # Collect all sprites with z-order
        sprites_to_draw = []
        
        for part in BodyPart.INDIVIDUAL_PARTS:
            # Get sprite for current age
            sprite_config = self.visual_config.get_sprite(part, age)
            
            if not sprite_config or not sprite_config.image_paths:
                continue
            
            # Get the image variant based on DNA gene
            dna_gene = sprite_config.dna_gene
            gene_value = getattr(pheno, dna_gene, dna_value) if hasattr(pheno, dna_gene) else dna_value
            image_path = sprite_config.get_image_for_dna(gene_value)
            
            if not image_path:
                continue
            
            # Get or create cached pixmap with DNA coloring
            cache_key = f"{image_path}_{pheno.hue:.2f}_{pheno.saturation:.2f}_{pheno.brightness:.2f}"
            
            if cache_key not in self.sprite_cache:
                # Load and color the sprite
                try:
                    pixmap = QPixmap(image_path)
                    if not pixmap.isNull():
                        # Apply DNA-based coloring
                        if sprite_config.apply_hue or sprite_config.apply_saturation or sprite_config.apply_brightness:
                            hue_shift = pheno.hue if sprite_config.apply_hue else 0
                            sat_mult = pheno.saturation if sprite_config.apply_saturation else 1.0
                            bright_mult = pheno.brightness if sprite_config.apply_brightness else 1.0
                            pixmap = apply_hue_to_pixmap(pixmap, hue_shift, sat_mult, bright_mult)
                        self.sprite_cache[cache_key] = pixmap
                except Exception as e:
                    print(f"Failed to load sprite {image_path}: {e}")
                    continue
            
            if cache_key in self.sprite_cache:
                pixmap = self.sprite_cache[cache_key]
                sprites_to_draw.append((sprite_config.z_order, part, pixmap, sprite_config))
        
        if not sprites_to_draw:
            return False
        
        # Sort by z-order
        sprites_to_draw.sort(key=lambda x: x[0])
        
        # Draw sprites
        for z_order, part, pixmap, config in sprites_to_draw:
            # Calculate position based on part and offsets
            sprite_w = int(pixmap.width() * config.scale * size_mult * scale / 2)
            sprite_h = int(pixmap.height() * config.scale * size_mult * scale / 2)
            
            sprite_x = x + int(config.offset_x * scale) - sprite_w // 2
            sprite_y = y + int(config.offset_y * scale) - sprite_h // 2
            
            # Flip horizontally if facing left
            if not body.motor.facing_right:
                painter.save()
                painter.translate(sprite_x + sprite_w, sprite_y)
                painter.scale(-1, 1)
                painter.drawPixmap(0, 0, sprite_w, sprite_h, pixmap)
                painter.restore()
            else:
                painter.drawPixmap(sprite_x, sprite_y, sprite_w, sprite_h, pixmap)
        
        # Grey out if dead
        if not body.is_alive():
            painter.fillRect(x - w//2, y - h//2, w, h, QColor(100, 100, 100, 150))
        
        return True
    
    def _draw_single_sprite(self, painter: QPainter, creature: LivingCreature,
                            sprite_config: 'BodyPartSprite', image_path: str,
                            x: int, y: int, w: int, h: int,
                            scale: float, size_mult: float, pheno) -> None:
        """Draw a single full-body sprite for a creature with transparency support."""
        body = creature.body
        
        # Check if sleeping for rotation
        is_sleeping = (hasattr(creature, 'behavior_state') and 
                      creature.behavior_state.state == BehaviorState.SLEEPING)
        
        # Enable smooth scaling and alpha blending for transparency
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        
        # Get or create cached pixmap with DNA coloring
        cache_key = f"{image_path}_{pheno.hue:.2f}_{pheno.saturation:.2f}_{pheno.brightness:.2f}"
        
        if cache_key not in self.sprite_cache:
            try:
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    # Apply DNA-based coloring
                    if sprite_config.apply_hue or sprite_config.apply_saturation or sprite_config.apply_brightness:
                        hue_shift = pheno.hue if sprite_config.apply_hue else 0
                        sat_mult = pheno.saturation if sprite_config.apply_saturation else 1.0
                        bright_mult = pheno.brightness if sprite_config.apply_brightness else 1.0
                        pixmap = apply_hue_to_pixmap(pixmap, hue_shift, sat_mult, bright_mult)
                    self.sprite_cache[cache_key] = pixmap
            except Exception as e:
                print(f"Failed to load sprite {image_path}: {e}")
                return
        
        if cache_key not in self.sprite_cache:
            return
        
        pixmap = self.sprite_cache[cache_key]
        
        # Calculate size
        sprite_w = int(pixmap.width() * sprite_config.scale * size_mult * scale / 2)
        sprite_h = int(pixmap.height() * sprite_config.scale * size_mult * scale / 2)
        
        sprite_x = x + int(sprite_config.offset_x * scale) - sprite_w // 2
        
        # Align vertically: Bottom-align to feet unless sleeping
        if is_sleeping:
             sprite_y = y + int(sprite_config.offset_y * scale) - sprite_h // 2
        else:
             # Bottom align: feet at y + h//2
             sprite_y = (y + h // 2) - sprite_h + int(sprite_config.offset_y * scale)
        
        # Rotate -90 degrees when sleeping (lying on back - cute!)
        if is_sleeping:
            painter.save()
            painter.translate(x, y)
            painter.rotate(-90)  # Rotate -90 degrees (lying on back)
            painter.drawPixmap(-sprite_w // 2, -sprite_h // 2, sprite_w, sprite_h, pixmap)
            painter.restore()
        # Flip horizontally if facing left
        elif not body.motor.facing_right:
            painter.save()
            painter.translate(sprite_x + sprite_w, sprite_y)
            painter.scale(-1, 1)
            painter.drawPixmap(0, 0, sprite_w, sprite_h, pixmap)
            painter.restore()
        else:
            painter.drawPixmap(sprite_x, sprite_y, sprite_w, sprite_h, pixmap)
        
        # Grey out if dead
        if not body.is_alive():
            painter.fillRect(x - w//2, y - h//2, w, h, QColor(100, 100, 100, 150))
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for tools."""
        if self.world is None:
            return
            
        self.last_mouse_pos = event.pos()
        
        # Calculate world coordinates
        world_x, world_y = self._to_world_coords(event.pos())
        
        # Handle Tools
        if self.current_tool == ToolType.SELECT:
            # Check for modifier keys
            modifiers = event.modifiers()
            is_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
            
            if is_ctrl:
                 # Try to select secondary
                 prev_selected = self.selected_creature
                 self._select_creature_at(world_x, world_y)
                 # If we clicked something and it wasn't the primary
                 if self.selected_creature and self.selected_creature != prev_selected:
                     self.selected_secondary = self.selected_creature
                     self.selected_creature = prev_selected # Restore primary
                 elif self.selected_creature == prev_selected:
                     # Clicked same one, maybe toggle? or just keep it
                     pass
            else:
                self._select_creature_at(world_x, world_y)
                self.selected_secondary = None # Clear secondary logic
            
        elif self.current_tool == ToolType.DRAG:
            # Try to grab a creature
            self._select_creature_at(world_x, world_y)
            if self.selected_creature:
                self.is_dragging = True
                
        elif self.current_tool == ToolType.SPAWN_FOOD:
            # Pick a random food type - only use types with images configured
            food_types = [
                FoodType.SWEET_BERRY,
                FoodType.BITTER_BERRY,
                FoodType.POISON_BERRY,
            ]
            food_type = food_types[np.random.randint(0, len(food_types))]
            self.world.food_sources.append(FoodSource(
                x=world_x, y=world_y,
                nutrition=0.5, type=food_type
            ))
            
        elif self.current_tool == ToolType.SPAWN_TEMPLATE:
            # Spawn creature from template data
            if self.current_template and self.parent():
                try:
                    # Call parent (GameTab) to spawn
                    self.parent().spawn_creature_from_data(self.current_template, world_x, world_y)
                except AttributeError:
                    print("Cannot spawn: Parent not GameTab or missing method")
            
        elif self.current_tool == ToolType.PAINT_WALL:
            # Right Click = Eraser (Empty/Air) to remove walls. Left click = Ground (Blocker?)
            # Wait, "Paint Wall" usually means making an obstacle.
            # In this engine, what stops movement?
            # It seems 'GROUND' is the default floor. 'EMPTY' is air (no floor?).
            # Actually, let's assume PAINT_WALL -> EMPTY (VOID/WALL) and PAINT_GROUND -> GROUND.
            # If the world is a cave, EMPTY is wall.
            if event.button() == Qt.MouseButton.RightButton:
                self._paint_tile(world_x, world_y, TileType.EMPTY.value) # Erase -> Empty
            else:
                 self._paint_tile(world_x, world_y, TileType.STONE.value) # Paint -> Stone
                 
        elif self.current_tool == ToolType.PAINT_GROUND:
             if event.button() == Qt.MouseButton.RightButton:
                self._paint_tile(world_x, world_y, TileType.EMPTY.value)
             else:
                self._paint_tile(world_x, world_y, TileType.GROUND.value)
             
        elif self.current_tool == ToolType.PAINT_WATER:
             if event.button() == Qt.MouseButton.RightButton:
                self._paint_tile(world_x, world_y, TileType.GROUND.value)
             else:
                self._paint_tile(world_x, world_y, TileType.WATER.value)
             
        elif self.current_tool == ToolType.PAINT_RADIATION:
             is_right = event.button() == Qt.MouseButton.RightButton
             self._paint_hazard_tool(world_x, world_y, is_right, "radiation", 2.0)
             
        elif self.current_tool == ToolType.PAINT_HEALING:
             is_right = event.button() == Qt.MouseButton.RightButton
             self._paint_hazard_tool(world_x, world_y, is_right, "healing", -5.0)
        
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move (dragging/painting)."""
        if self.world is None:
            return

        world_x, world_y = self._to_world_coords(event.pos())
        
        # Track mouse delta for panning
        if self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
        else:
            delta = QPointF(0, 0)
        self.last_mouse_pos = event.pos()
        
        if (event.buttons() & Qt.MouseButton.MiddleButton) or \
           (event.buttons() & Qt.MouseButton.RightButton and self.current_tool == ToolType.SELECT):
            self.camera_x -= delta.x()
            self.camera_y -= delta.y()
            self._clamp_camera() # Enforce bounds
            self.update()
            return

        if self.current_tool == ToolType.DRAG and self.is_dragging and self.selected_creature:
            # Move creature
            self.selected_creature.body.motor.x = world_x
            self.selected_creature.body.motor.y = world_y
            self.selected_creature.body.motor.vx = 0
            self.selected_creature.body.motor.vy = 0
            
        elif self.current_tool in [ToolType.PAINT_WALL, ToolType.PAINT_GROUND, 
                                 ToolType.PAINT_WATER, ToolType.PAINT_RADIATION, ToolType.PAINT_HEALING]:
            # Paint while dragging
             val = TileType.GROUND.value 
             # Determine value based on tool and button
             is_right = event.buttons() & Qt.MouseButton.RightButton
             
             if self.current_tool == ToolType.PAINT_WATER:
                 val = TileType.GROUND.value if is_right else TileType.WATER.value
                 self._paint_tile(world_x, world_y, val)
             elif self.current_tool == ToolType.PAINT_RADIATION:
                 self._paint_hazard_tool(world_x, world_y, is_right, "radiation", 2.0)
             elif self.current_tool == ToolType.PAINT_HEALING:
                 self._paint_hazard_tool(world_x, world_y, is_right, "healing", -5.0)
             elif self.current_tool == ToolType.PAINT_GROUND:
                 val = TileType.EMPTY.value if is_right else TileType.GROUND.value
                 self._paint_tile(world_x, world_y, val)
             elif self.current_tool == ToolType.PAINT_WALL: 
                 val = TileType.EMPTY.value if is_right else TileType.STONE.value
                 self._paint_tile(world_x, world_y, val)
             
        self.update()
        
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        self.is_dragging = False

    def wheelEvent(self, event):
        """Handle zoom with mouse wheel and pan with touchpad."""
        # Horizontal scroll (Touchpad two-finger swipe) -> Pan
        delta_x = event.angleDelta().x()
        if abs(delta_x) > 0:
            self.camera_x -= delta_x
            self._clamp_camera()
            self.update()
            
        # Vertical scroll (Mouse wheel or Touchpad pinch/scroll) -> Zoom
        # Note: Some touchpads send ctrl+wheel for pinch zoom
        delta_y = event.angleDelta().y()
        if abs(delta_y) > 0:
            zoom_factor = 1.1 if delta_y > 0 else 0.9
            self.zoom *= zoom_factor
            # Strict zoom limit: 1.0 (100%) to 5.0 (500%)
            self.zoom = np.clip(self.zoom, 1.0, 5.0)
            self._clamp_camera() # Zoom changes scale, so re-clamp
            self.update()
            
    def _clamp_camera(self):
        """Restrict camera_x to valid world bounds."""
        if not self.world:
            return
            
        # Get normalized world dimensions from zone_boundaries
        zone_boundaries = getattr(self, 'zone_boundaries', None) or getattr(self.world, 'zone_boundaries', None)
        
        if zone_boundaries:
            # All zones have same normalized height
            first_zone = zone_boundaries.get(min(zone_boundaries.keys()))
            world_height = first_zone.get('height', 1080) if first_zone else 1080
            world_width = self.world.max_x - self.world.min_x
        else:
            world_height = self.world.height
            world_width = self.world.width
        
        # Scale based on screen height to world height
        scale = (self.height() / world_height) * self.zoom
        
        # Visible width in world coordinates
        visible_world_w = self.width() / scale
        
        # Camera can pan from 0 to (world_width - visible_width)
        # So you can see from left edge to right edge
        max_camera_x = max(0, world_width - visible_world_w)
        
        self.camera_x = max(0, min(max_camera_x, self.camera_x))

    def _to_world_coords(self, pos) -> Tuple[float, float]:
        # Get normalized world dimensions from zone_boundaries
        zone_boundaries = getattr(self, 'zone_boundaries', None) or getattr(self.world, 'zone_boundaries', None)
        
        if zone_boundaries:
            first_zone = zone_boundaries.get(min(zone_boundaries.keys()))
            world_height = first_zone.get('height', 1080) if first_zone else 1080
        else:
            world_height = self.world.height
        
        # Same scale as paintEvent
        scale = (self.height() / world_height) * self.zoom
        
        offset_x = -self.camera_x * scale
        # Align to bottom
        screen_world_h = int(world_height * scale)
        offset_y = self.height() - screen_world_h - self.camera_y
        
        wx = (pos.x() - offset_x) / scale
        wy = (pos.y() - offset_y) / scale
        return wx, wy

    def _select_creature_at(self, x, y):
        self.selected_creature = None
        for creature in self.creatures:
            body = creature.body
            dist = np.sqrt((body.motor.x - x)**2 + (body.motor.y - y)**2)
            if dist < body.phenotype.width * body.phenotype.size * 1.5: # 1.5 tolerance
                self.selected_creature = creature
                break

    def _paint_tile(self, x, y, tile_value):
        """Paint a single tile at world coords."""
        tx = int(x // self.world.tile_size)
        ty = int(y // self.world.tile_size)
        
        if 0 <= tx < self.world.tiles_x and 0 <= ty < self.world.tiles_y:
            self.world.tiles[ty, tx] = tile_value
            
    def _paint_hazard_tool(self, x, y, is_erase, haz_type, damage):
        """Helper to paint hazard tiles and objects."""
        # This is a simplification. A real editor would merge hazards.
        # Here we just spawn 1-tile hazards.
        from brain.world import Hazard
        
        if is_erase:
            self._paint_tile(x, y, TileType.GROUND.value)
            # Remove hazards at this location
            self.world.hazards = [h for h in self.world.hazards 
                                if not (h.x <= x < h.x + h.width and h.y <= y < h.y + h.height)]
        else:
            self._paint_tile(x, y, TileType.HAZARD.value)
            # Add hazard object if not one already here
            already_exists = False
            for h in self.world.hazards:
                if h.x <= x < h.x + h.width and h.y <= y < h.y + h.height:
                    already_exists = True
                    break
            
            if not already_exists:
                # Align to tile grid
                tx = int(x // self.world.tile_size) * self.world.tile_size
                ty = int(y // self.world.tile_size) * self.world.tile_size
                
                self.world.hazards.append(Hazard(
                    x=tx, y=ty,
                    width=self.world.tile_size,
                    height=self.world.tile_size,
                    damage=damage,
                    type=haz_type
                ))
    
    def set_visual_config(self, config):
        """Set visual configuration from settings tab."""
        self.visual_config = config
        self.sprite_cache.clear()  # Clear cache to reload with new settings
        self._clear_world_object_cache()  # Clear world object cache too
        
        # Update background if changed
        if config and config.background_path:
            self.background_image = QPixmap(config.background_path)
        
        self.update()


# =============================================================================
# THREADED UPDATE MANAGER
# =============================================================================

@dataclass
class UpdateRates:
    """Configuration for update loop frequencies.
    
    Multi-threaded simulation decouples:
    - Physics (60Hz): Gravity, collisions, movement - needs high frequency for accuracy
    - Brain (10Hz): Neural processing, learning - computationally heavy, can be slower
    - Metabolism (1Hz): Hunger, thirst, energy - slow biological processes
    """
    physics_hz: float = 60.0  # Physics updates per second
    brain_hz: float = 10.0    # Brain updates per second
    metabolism_hz: float = 1.0  # Metabolism updates per second
    
    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz
    
    @property 
    def brain_dt(self) -> float:
        return 1.0 / self.brain_hz
    
    @property
    def metabolism_dt(self) -> float:
        return 1.0 / self.metabolism_hz


class ThreadedUpdateManager:
    """
    Decoupled multi-rate simulation manager.
    
    Separates physics (60Hz), brain (10Hz), and metabolism (1Hz) updates
    for 30-50% performance improvement. Uses producer-consumer pattern
    with thread-safe queues for creature state updates.
    
    Architecture:
    - Main thread: Rendering, UI, orchestration
    - Physics thread: High-frequency movement, collisions
    - Brain thread: Neural processing, learning (heaviest computation)
    - Metabolism is updated inline with physics (simple calculations)
    
    Thread Safety:
    - Creature state uses copy-on-read pattern
    - Physics writes position/velocity atomically via locks
    - Brain results queued for main thread application
    """
    
    def __init__(self, rates: Optional[UpdateRates] = None):
        self.rates = rates or UpdateRates()
        
        # Thread control
        self._running = False
        self._paused = True
        self._speed_multiplier = 1.0
        
        # Threads
        self._physics_thread: Optional[threading.Thread] = None
        self._brain_thread: Optional[threading.Thread] = None
        
        # Thread-safe queues for results
        self._brain_results: queue.Queue = queue.Queue(maxsize=100)
        self._physics_lock = threading.RLock()  # Reentrant for nested calls
        
        # Creature references (set by GameTab)
        self._creatures: List[Any] = []
        self._world: Optional[Any] = None
        
        # Timing accumulators for multi-rate updates
        self._physics_accumulator = 0.0
        self._brain_accumulator = 0.0
        self._metabolism_accumulator = 0.0
        
        # Performance metrics
        self._physics_times: deque = deque(maxlen=60)
        self._brain_times: deque = deque(maxlen=30)
        self._frame_times: deque = deque(maxlen=60)
        self._last_frame_time = time.perf_counter()
        
        # Callbacks for physics substep (set by GameTab)
        self._physics_substep_callback: Optional[callable] = None
        self._brain_update_callback: Optional[callable] = None
    
    def start(self, creatures: List, world: Any,
              physics_callback: callable, brain_callback: callable):
        """Start the threaded update loops."""
        if self._running:
            return
            
        self._creatures = creatures
        self._world = world
        self._physics_substep_callback = physics_callback
        self._brain_update_callback = brain_callback
        self._running = True
        self._paused = False
        
        # Start worker threads
        self._physics_thread = threading.Thread(
            target=self._physics_loop,
            name="PhysicsThread",
            daemon=True
        )
        self._brain_thread = threading.Thread(
            target=self._brain_loop, 
            name="BrainThread",
            daemon=True
        )
        
        self._physics_thread.start()
        self._brain_thread.start()
    
    def stop(self):
        """Stop all update threads."""
        self._running = False
        if self._physics_thread:
            self._physics_thread.join(timeout=0.5)
        if self._brain_thread:
            self._brain_thread.join(timeout=0.5)
        self._physics_thread = None
        self._brain_thread = None
    
    def pause(self):
        """Pause updates."""
        self._paused = True
    
    def resume(self):
        """Resume updates."""
        self._paused = False
    
    def set_speed(self, speed: float):
        """Set simulation speed multiplier."""
        self._speed_multiplier = max(0.1, min(10.0, speed))
    
    def update_creatures(self, creatures: List):
        """Update creature list (thread-safe)."""
        with self._physics_lock:
            self._creatures = creatures
    
    def _physics_loop(self):
        """Physics update thread - runs at physics_hz."""
        target_dt = self.rates.physics_dt
        last_time = time.perf_counter()
        
        while self._running:
            if self._paused:
                time.sleep(0.01)
                last_time = time.perf_counter()
                continue
            
            current_time = time.perf_counter()
            elapsed = current_time - last_time
            last_time = current_time
            
            # Apply speed multiplier
            scaled_dt = target_dt * self._speed_multiplier
            
            # Run physics update
            start = time.perf_counter()
            try:
                with self._physics_lock:
                    if self._physics_substep_callback and self._world:
                        self._physics_substep_callback(scaled_dt)
                        
                        # Inline metabolism update (cheap, runs with physics)
                        self._metabolism_accumulator += elapsed * self._speed_multiplier
                        if self._metabolism_accumulator >= self.rates.metabolism_dt:
                            self._update_metabolism(self._metabolism_accumulator)
                            self._metabolism_accumulator = 0.0
            except Exception as e:
                pass  # Don't crash the thread
            
            self._physics_times.append(time.perf_counter() - start)
            
            # Sleep to maintain target rate
            sleep_time = target_dt - (time.perf_counter() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _brain_loop(self):
        """Brain update thread - runs at brain_hz."""
        target_dt = self.rates.brain_dt
        
        while self._running:
            if self._paused:
                time.sleep(0.01)
                continue
            
            start = time.perf_counter()
            
            try:
                # Process brains for all creatures
                with self._physics_lock:
                    creatures_snapshot = list(self._creatures)
                
                for creature in creatures_snapshot:
                    if not hasattr(creature, 'body') or not creature.body.is_alive():
                        continue
                    if not hasattr(creature, 'embodied_brain') or creature.embodied_brain is None:
                        continue
                    
                    try:
                        if self._brain_update_callback:
                            result = self._brain_update_callback(
                                creature, target_dt * self._speed_multiplier
                            )
                            if result:
                                self._brain_results.put_nowait((creature, result))
                    except queue.Full:
                        pass  # Skip if queue full
                    except Exception:
                        pass  # Don't crash on brain errors
                        
            except Exception:
                pass
            
            self._brain_times.append(time.perf_counter() - start)
            
            # Sleep to maintain target rate
            elapsed = time.perf_counter() - start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _update_metabolism(self, dt: float):
        """Update creature metabolism (runs inline with physics)."""
        for creature in self._creatures:
            if not hasattr(creature, 'body') or not creature.body.is_alive():
                continue
            
            try:
                body = creature.body
                # Metabolism updates are handled in homeostasis.update()
                # which is already called in physics substep
                pass
            except Exception:
                pass
    
    def apply_brain_results(self):
        """Apply queued brain results (call from main thread)."""
        applied = 0
        while not self._brain_results.empty() and applied < 10:
            try:
                creature, result = self._brain_results.get_nowait()
                # Brain results contain last_step_data for inspection
                if hasattr(creature, 'embodied_brain') and creature.embodied_brain:
                    creature.embodied_brain.last_step_data = result
                applied += 1
            except queue.Empty:
                break
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance metrics."""
        physics_avg = np.mean(self._physics_times) * 1000 if self._physics_times else 0
        brain_avg = np.mean(self._brain_times) * 1000 if self._brain_times else 0
        
        return {
            'physics_ms': physics_avg,
            'brain_ms': brain_avg,
            'physics_hz_actual': 1000 / physics_avg if physics_avg > 0 else 0,
            'brain_hz_actual': 1000 / brain_avg if brain_avg > 0 else 0,
        }
    
    @property
    def is_running(self) -> bool:
        return self._running and not self._paused


# =============================================================================
# GAME TAB
# =============================================================================

class GameTab(QWidget):
    """Main game tab with world simulation.
    
    Supports two simulation modes:
    - Single-threaded (default): All updates in main timer callback
    - Multi-threaded: Physics (60Hz), Brain (10Hz), Metabolism (1Hz) on separate threads
    
    Multi-threaded mode provides 30-50% performance improvement for large populations.
    """
    
    def __init__(self, brain=None, use_threading: bool = True):
        super().__init__()
        self.template_brain = brain  # Used as template for new creatures
        
        self.world: Optional[World] = None
        self.creatures: List[LivingCreature] = []
        self.paused = True
        self.simulation_speed = 1.0
        
        # Multi-threaded simulation (UPGRADE 10)
        self.use_threading = use_threading
        self.threaded_manager: Optional[ThreadedUpdateManager] = None
        if use_threading:
            self.threaded_manager = ThreadedUpdateManager(UpdateRates(
                physics_hz=60.0,
                brain_hz=10.0,
                metabolism_hz=1.0
            ))
        
        # TIER 3: Cultural Evolution - shared knowledge across all creatures
        self.cultural_knowledge = CulturalKnowledge()
        
        # TIER 4: Cultural Language - emergent communication conventions
        self.cultural_language = CulturalLanguage()
        
        # Mating animation tracking
        self.mating_pairs = [] # List of {p1, p2, timer, start_time}

        # SYSTEM 9: Lineage Tracking
        self.lineage_system = LineageSystem()

        self.setup_ui()
        
        # Simulation timer (lower rate when multi-threaded since physics is separate)
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.simulation_step)
        # 30 FPS for rendering, physics runs at 60Hz on separate thread
        self.sim_timer.start(33 if not use_threading else 33)
        
        # Sound Manager
        self.sound_manager = SoundManager()
        
        # Init settings tab with config
        if _HAS_SETTINGS:
            # We need to access the settings tab to update its config if needed
            # But normally it's separate. However, for sound settings, we might want to pass the manager.
            pass
        self.sound_manager.load_sounds()
    
    def update_visual_config(self, config):
        """Update visual configuration from settings tab."""
        self.world_renderer.set_visual_config(config)
    
    def setup_ui(self):
        """Set up the game UI."""
        layout = QHBoxLayout(self)
        
        # Left: World view
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # World renderer
        self.world_renderer = WorldRenderer()
        # Link mating pairs for rendering
        self.world_renderer.mating_pairs = self.mating_pairs
        left_layout.addWidget(self.world_renderer, stretch=1)
        
        # Controls bar
        controls = QHBoxLayout()
        
        self.new_world_btn = QPushButton("🌍 New World")
        self.new_world_btn.clicked.connect(self.create_new_world)
        controls.addWidget(self.new_world_btn)
        
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self.toggle_pause)
        controls.addWidget(self.play_btn)
        
        self.spawn_btn = QPushButton("🥚 Spawn Creature")
        self.spawn_btn.clicked.connect(self.spawn_creature_with_brain)  # Always spawn with full brain
        controls.addWidget(self.spawn_btn)
        
        # Gender selector for spawn
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Random", "Male", "Female"])
        controls.addWidget(self.gender_combo)
        
        # Brain size selector for spawn
        self.brain_size_combo = QComboBox()
        self.brain_size_combo.addItems(["Micro", "Small", "Medium", "Large"])
        self.brain_size_combo.setToolTip("Brain size: Micro=800, Small=2K, Medium=5K, Large=10K neurons")
        controls.addWidget(self.brain_size_combo)

        controls.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 200)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.update_speed)
        self.speed_slider.setMaximumWidth(100)
        controls.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("1.0x")
        controls.addWidget(self.speed_label)
        
        controls.addStretch()
        left_layout.addLayout(controls)
        
        # Stats Graph
        self.stats_graph = StatsGraphWidget()
        left_layout.addWidget(self.stats_graph)
        
        # Right: Info panel
        right_widget = QWidget()
        right_widget.setMinimumWidth(280)
        right_widget.setMaximumWidth(350)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)
        
        
        # God Mode Tools
        tools_group = QGroupBox("God Mode Tools")
        tools_layout = QVBoxLayout(tools_group)
        
        tools_layout.addWidget(QLabel("Select Tool:"))
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(ToolType.all_tools())
        self.tool_combo.currentTextChanged.connect(self._on_tool_changed)
        tools_layout.addWidget(self.tool_combo)
        
        tools_layout.addWidget(QLabel("Right-click to Erase/Delete with paint tools."))
        
        right_layout.addWidget(tools_group)

        # Population stats
        pop_group = QGroupBox("Population")
        pop_layout = QGridLayout(pop_group)
        
        self.alive_label = QLabel("Alive: 0")
        self.dead_label = QLabel("Dead: 0")
        self.food_label = QLabel("Food: 0")
        self.gen_label = QLabel("Max Gen: 0")
        
        pop_layout.addWidget(self.alive_label, 0, 0)
        pop_layout.addWidget(self.dead_label, 0, 1)
        pop_layout.addWidget(self.food_label, 1, 0)
        pop_layout.addWidget(self.gen_label, 1, 1)
        
        right_layout.addWidget(pop_group)
        
        # Selected creature info
        self.creature_group = QGroupBox("Selected Creature")
        creature_layout = QVBoxLayout(self.creature_group)
        creature_layout.setSpacing(2)
        
        self.creature_name = QLabel("None selected")
        self.creature_name.setWordWrap(True)
        self.creature_gender = QLabel("Gender: -")
        self.creature_health = QLabel("Health: -")
        self.creature_energy = QLabel("Energy: -")
        self.creature_hunger = QLabel("Hunger: -")
        self.creature_thirst = QLabel("Thirst: -")
        self.creature_state = QLabel("State: -")
        self.creature_age = QLabel("Age: -")
        self.creature_mating = QLabel("Mating: -")  # Mating stats
        
        creature_layout.addWidget(self.creature_name)
        creature_layout.addWidget(self.creature_gender)
        creature_layout.addWidget(self.creature_health)
        creature_layout.addWidget(self.creature_energy)
        creature_layout.addWidget(self.creature_hunger)
        creature_layout.addWidget(self.creature_thirst)
        creature_layout.addWidget(self.creature_state)
        creature_layout.addWidget(self.creature_age)
        creature_layout.addWidget(self.creature_mating)
        
        # Brain info
        self.creature_brain = QLabel("Brain: -")
        self.creature_brain.setWordWrap(True)
        creature_layout.addWidget(self.creature_brain)
        
        # Held tool info
        self.creature_tool = QLabel("Holding: -")
        creature_layout.addWidget(self.creature_tool)
        
        # Creature actions
        action_layout = QHBoxLayout()
        self.feed_btn = QPushButton("🍎 Feed")
        self.feed_btn.clicked.connect(self.feed_selected)
        self.breed_btn = QPushButton("💕 Breed")
        self.breed_btn.clicked.connect(self.breed_selected)
        action_layout.addWidget(self.feed_btn)
        action_layout.addWidget(self.breed_btn)
        
        self.inspect_btn = QPushButton("🧠 Inspect Brain")
        self.inspect_btn.clicked.connect(self.inspect_selected)
        creature_layout.addWidget(self.inspect_btn)
        
        self.save_tmpl_btn = QPushButton("💾 Save Template")
        self.save_tmpl_btn.clicked.connect(self.save_selected_template)
        creature_layout.addWidget(self.save_tmpl_btn) # Add to vertical layout below buttons?
        # Or add to action layout? action layout is horizontal.
        # "Feed", "Breed", "Save".
        # Let's add it to action_layout.
        # Wait, I originally replaced adding to action_layout.
        
        creature_layout.addLayout(action_layout)
        
        right_layout.addWidget(self.creature_group)
        
        # Creature list
        list_group = QGroupBox("Creatures (double-click to rename)")
        list_layout = QVBoxLayout(list_group)
        
        self.creature_list = QListWidget()
        self.creature_list.itemClicked.connect(self.select_creature_from_list)
        self.creature_list.itemDoubleClicked.connect(self.rename_creature_from_list)
        list_layout.addWidget(self.creature_list)
        
        right_layout.addWidget(list_group)
        
        # Save/Load buttons
        save_group = QGroupBox("Save/Load")
        save_layout = QHBoxLayout(save_group)
        
        self.save_btn = QPushButton("💾 Save Game")
        self.save_btn.clicked.connect(self.save_game)
        self.load_btn = QPushButton("📂 Load Game")
        self.load_btn.clicked.connect(self.load_game)
        save_layout.addWidget(self.save_btn)
        save_layout.addWidget(self.load_btn)
        
        right_layout.addWidget(save_group)
        
        right_layout.addStretch()
        
        # Add to main layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 300])
        
        layout.addWidget(splitter)
        
        # Info update timer
        self.info_timer = QTimer()
        self.info_timer.timeout.connect(self.update_info)
        self.info_timer.start(200)
    
    
    def _on_tool_changed(self, tool_name):
        """Handle tool selection change."""
        self.world_renderer.current_tool = tool_name
        
        if tool_name == ToolType.SPAWN_TEMPLATE:
            # Open file dialog to load template
            save_dir = "creature_templates"
            os.makedirs(save_dir, exist_ok=True)
            
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Load Creature Template", save_dir, "JSON Files (*.json)"
            )
            
            if filepath:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    self.world_renderer.current_template = data
                    print(f"Loaded template from {filepath}")
                except Exception as e:
                    print(f"Failed to load template: {e}")
                    self.tool_combo.setCurrentText(ToolType.SELECT)
            else:
                self.tool_combo.setCurrentText(ToolType.SELECT)
        else:
            self.world_renderer.current_template = None
            
    def inspect_selected(self):
        """Open brain inspector for selected creature."""
        selected = self.world_renderer.selected_creature
        if not selected or not selected.body.is_alive():
            return
            
        if not selected.embodied_brain:
            print("Selected creature has no embodied brain (instinct only).")
            return
            
        # Create inspector window (keep reference so it doesn't GC)
        self.inspector = BrainInspectorWindow(selected, self)
        self.inspector.show()
    
    def closeEvent(self, event):
        """Clean up when tab is closed."""
        self._stop_threaded_simulation()
        super().closeEvent(event)
    
    def __del__(self):
        """Destructor - ensure threads are stopped."""
        self._stop_threaded_simulation()
        
    def create_new_world(self):
        """Create a new world."""
        # Stop any existing threaded simulation
        self._stop_threaded_simulation()
        
        # Ensure paused state when creating new world
        self.paused = True
        self.play_btn.setText("▶️ Play")
        
        # World height must match background height (1080) for proper alignment
        self.world = World(width=1920, height=1080)
        self.world_renderer.set_world(self.world)
        self.world_renderer.paused = True  # Sync paused state to renderer
        self.creatures = []
        self.world_renderer.set_creatures(self.creatures)
        self.update_creature_list()
        
        # Start threaded simulation if enabled (it will be paused)
        if self.use_threading and self.threaded_manager:
            self._start_threaded_simulation()
            self.threaded_manager.pause()  # Ensure it starts paused
    
    def toggle_pause(self):
        """Toggle simulation pause."""
        self.paused = not self.paused
        self.play_btn.setText("⏸️ Pause" if not self.paused else "▶️ Play")
        
        # Sync paused state to renderer (stops animations)
        self.world_renderer.paused = self.paused
        
        # Sync threaded manager pause state
        if self.threaded_manager:
            if self.paused:
                self.threaded_manager.pause()
            else:
                self.threaded_manager.resume()
    
    def update_speed(self, value):
        """Update simulation speed."""
        self.simulation_speed = value / 10.0
        self.speed_label.setText(f"{self.simulation_speed:.1f}x")
        
        # Sync speed to threaded manager
        if self.threaded_manager:
            self.threaded_manager.set_speed(self.simulation_speed)
    
    def spawn_creature_dialog(self):
        """Show dialog to configure creature spawn options."""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QComboBox
        from brain.creature import Gender
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Spawn Creature")
        dialog.setMinimumWidth(300)
        
        layout = QFormLayout(dialog)
        
        # Gender selection
        gender_combo = QComboBox()
        gender_combo.addItems(["♂ Male", "♀ Female", "🎲 Random"])
        layout.addRow("Gender:", gender_combo)
        
        # Brain size selection
        brain_combo = QComboBox()
        brain_combo.addItems(["None (Instinct Only)", "Micro (~800 neurons)", "Small (~3,200)", "Medium (~6,400)", "Large (~16,000)"])
        brain_combo.setCurrentIndex(1)  # Default to Micro
        layout.addRow("Brain Size:", brain_combo)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get selections
            gender_idx = gender_combo.currentIndex()
            if gender_idx == 0:
                gender = Gender.MALE
            elif gender_idx == 1:
                gender = Gender.FEMALE
            else:
                gender = Gender.random()
            
            brain_idx = brain_combo.currentIndex()
            brain_scales = [None, 'micro', 'small', 'medium', 'large']
            brain_scale = brain_scales[brain_idx]
            
            # Spawn with these settings
            self.spawn_creature_with_options(gender=gender, brain_scale=brain_scale)
    
    def spawn_creature_with_options(self, gender=None, brain_scale='micro'):
        """Spawn a creature with specific gender and brain size."""
        if self.world is None:
            self.create_new_world()
        
        from brain.creature import Gender
        
        # Find a safe spawn location
        x, y = self._find_safe_spawn_location()
        
        # Create random phenotype with specified gender
        if gender is None:
            gender = Gender.random()
        
        phenotype = Phenotype(
            size=0.7 + np.random.random() * 0.6,
            hue=np.random.random(),
            saturation=0.5 + np.random.random() * 0.5,
            pattern_type=np.random.choice(["solid", "stripes", "spots", "gradient"]),
            pattern_density=np.random.random(),
            max_speed=2.0 + np.random.random() * 3,
            jump_power=6 + np.random.random() * 6,
            metabolic_rate=0.8 + np.random.random() * 0.4,
            gender=gender,
        )
        
        body = CreatureBody(phenotype=phenotype, x=x, y=y)
        
        # === HORMONAL DIMORPHISM (1) - Gender-specific hormone baselines ===
        if gender == Gender.MALE:
            body.homeostasis.testosterone_base = 0.7 + np.random.random() * 0.2
            body.homeostasis.estrogen_base = 0.2 + np.random.random() * 0.2
        else:
            body.homeostasis.testosterone_base = 0.2 + np.random.random() * 0.2
            body.homeostasis.estrogen_base = 0.7 + np.random.random() * 0.2
        
        # Random drive params with gender dimorphism (10)
        drive_params = {
            'hunger': 0.3 + np.random.random() * 0.4,
            'fear': 0.3 + np.random.random() * 0.4,
            'curiosity': 0.3 + np.random.random() * 0.4,
            'social': 0.3 + np.random.random() * 0.4,
        }
        
        # Gender affects baseline drives
        if gender == Gender.MALE:
            drive_params['fear'] = max(0.1, drive_params['fear'] - 0.1)
            drive_params['curiosity'] = min(0.9, drive_params['curiosity'] + 0.1)
        else:
            drive_params['social'] = min(0.9, drive_params['social'] + 0.1)
        
        instincts = InstinctSystem(drive_params)
        reward_system = RewardSystem()
        
        # Create brain if requested
        brain = None
        embodied = None
        if brain_scale:
            try:
                embodied = EmbodiedBrain(brain=None, body=body, brain_scale=brain_scale)
                brain = embodied.brain
            except Exception as e:
                print(f"Error creating brain: {e}")
        
        creature = LivingCreature(
            body=body,
            brain=brain,
            instincts=instincts,
            reward_system=reward_system,
            generation=0,
            use_neural_control=brain is not None,
            name=f"{gender.symbol} Creature_{len(self.creatures)+1}"
        )
        
        if embodied:
            creature.embodied_brain = embodied
        
        self.creatures.append(creature)
        self.lineage_system.register_birth(creature)
        self.world_renderer.set_creatures(self.creatures)
        self.update_creature_list()
    
    def spawn_creature(self):
        """Spawn a single creature at random location."""
        if self.world is None:
            self.create_new_world()
        
        # Find a safe spawn location (on ground, not water/hazard)
        x, y = self._find_safe_spawn_location()
        
        # Import Gender for assignment
        from brain.creature import Gender
        
        # Create random phenotype with random gender
        phenotype = Phenotype(
            size=0.7 + np.random.random() * 0.6,
            hue=np.random.random(),
            saturation=0.5 + np.random.random() * 0.5,
            pattern_type=np.random.choice(["solid", "stripes", "spots", "gradient"]),
            pattern_density=np.random.random(),
            max_speed=2.0 + np.random.random() * 3,
            jump_power=6 + np.random.random() * 6,
            metabolic_rate=0.8 + np.random.random() * 0.4,
            gender=Gender.random(),  # Random gender assignment
        )
        
        body = CreatureBody(phenotype=phenotype, x=x, y=y)
        
        # Random drive params for instincts
        drive_params = {
            'hunger': 0.3 + np.random.random() * 0.4,
            'fear': 0.3 + np.random.random() * 0.4,
            'curiosity': 0.3 + np.random.random() * 0.4,
            'social': 0.3 + np.random.random() * 0.4,
        }
        
        instincts = InstinctSystem(drive_params)
        reward_system = RewardSystem()
        
        creature = LivingCreature(
            body=body,
            brain=None,  # Instinct-only for now
            instincts=instincts,
            reward_system=reward_system,
            generation=0
        )
        
        self.creatures.append(creature)
        self.lineage_system.register_birth(creature)
        self.world_renderer.set_creatures(self.creatures)
        self.update_creature_list()
    
    def _find_safe_spawn_location(self) -> tuple:
        """Find a safe spawn location on solid ground, not water/hazard."""
        for _ in range(50):  # Max attempts
            x = np.random.randint(50, self.world.width - 50)
            
            # Scan down from top to find ground
            for y in range(20, self.world.height - 20):
                tx = int(x / self.world.tile_size)
                ty = int(y / self.world.tile_size)
                ty_below = min(ty + 1, self.world.tiles_y - 1)
                
                if tx < 0 or tx >= self.world.tiles_x:
                    continue
                if ty < 0 or ty >= self.world.tiles_y:
                    continue
                
                tile_here = self.world.tiles[ty, tx]
                tile_below = self.world.tiles[ty_below, tx]
                
                # Valid spawn: empty here, solid ground below (not water/hazard)
                if (tile_here == TileType.EMPTY.value and 
                    tile_below == TileType.GROUND.value):
                    return (x, y - 5)  # Slightly above ground
        
        # Fallback: center of world, high up
        return (self.world.width // 2, 50)
    
    def spawn_creatures(self, count: int):
        """Spawn multiple creatures."""
        for _ in range(count):
            self.spawn_creature()
            
    def _trigger_speech(self, creature: LivingCreature, context_text: str, mood: str = "neutral"):
        """
        Trigger speech generation for a creature using procedural language.
        
        ENHANCED: First checks if the creature's brain has learned word associations
        from observing others. If the creature heard "apa" while watching someone drink,
        and now THEY are drinking, the "apa" pattern may activate and they'll say it.
        
        This is emergent language - words spread through observation + neural association.
        """
        if not creature.procedural_language:
            return
        
        # Parse context to extract need/target
        context = {'need': 'idle', 'target': None}
        
        # Map keywords to needs
        if 'food' in context_text or 'hungry' in context_text or 'eat' in context_text:
            context['need'] = 'food'
            context['target'] = 'food'
        elif 'water' in context_text or 'thirst' in context_text or 'drink' in context_text:
            context['need'] = 'water'
            context['target'] = 'water'
        elif 'sleep' in context_text or 'tired' in context_text or 'rest' in context_text:
            context['need'] = 'sleep'
        elif 'pain' in context_text or 'hurt' in context_text or 'danger' in context_text:
            context['need'] = 'danger'
            context['target'] = 'danger'
        elif 'love' in context_text or 'family' in context_text or 'baby' in context_text:
            context['need'] = 'social'
            context['target'] = 'mate'
        
        # === NEURAL RETRIEVAL: Check if brain has learned word for this action ===
        # Present the action pattern to brain and see if it activates a word pattern
        retrieved_word = None
        if creature.embodied_brain is not None:
            try:
                brain = creature.embodied_brain.brain
                behavior = creature.behavior_state.state.name if creature.behavior_state else 'unknown'
                
                # Encode current action as pattern
                action_pattern = self._encode_action_pattern(context['need'], behavior)
                
                # Present to cortex and get response
                if hasattr(brain, 'cortex'):
                    result = brain.cortex.process(
                        input_data=action_pattern,
                        current_time=brain.current_time,
                        learning_enabled=False  # Just retrieving, not learning
                    )
                    
                    # Check if any learned words have similar activation pattern
                    # by comparing response to known word patterns in lexicon
                    cortex_response = result.get('activation', np.zeros(100))
                    
                    best_match_word = None
                    best_similarity = 0.3  # Threshold for recall
                    
                    for word in creature.procedural_language.lexicon:
                        word_pattern = self._encode_word_pattern(word)
                        # Simple similarity: dot product of patterns
                        similarity = np.dot(cortex_response[:len(word_pattern)], word_pattern)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_word = word
                    
                    if best_match_word:
                        retrieved_word = best_match_word
                        
            except Exception:
                pass  # Fall back to procedural generation
        
        # Use retrieved word if found, otherwise generate new utterance
        if retrieved_word:
            word = retrieved_word
            # Get associated gesture from lexicon
            symbol = creature.procedural_language.lexicon.get(word)
            gesture = symbol.gesture if symbol else None
        else:
            # Generate utterance using procedural system (may innovate new word)
            word, gesture = creature.procedural_language.generate_utterance(context)
        
        # Set speech bubble and gesture
        creature.current_speech = word
        creature.current_gesture = gesture
        creature.speech_timer = 3.0  # Show for 3 seconds
    
    
    def spawn_creature_with_brain(self):
        """Spawn a creature with a neural brain (EmbodiedBrain)."""
        if self.world is None:
            self.create_new_world()
        
        # Find safe spawn location
        x, y = self._find_safe_spawn_location()
        
        # Determine gender from dropdown
        from brain.creature import Gender
        gender_text = self.gender_combo.currentText()
        if gender_text == "Male":
            gender = Gender.MALE
        elif gender_text == "Female":
            gender = Gender.FEMALE
        else:
            gender = Gender.random()
        
        # Determine brain size from dropdown
        brain_size = self.brain_size_combo.currentText().lower()  # 'micro', 'small', 'medium', 'large'
        
        # Create random phenotype
        phenotype = Phenotype(
            size=0.7 + np.random.random() * 0.6,
            hue=np.random.random(),
            saturation=0.5 + np.random.random() * 0.5,
            pattern_type=np.random.choice(["solid", "stripes", "spots", "gradient"]),
            pattern_density=np.random.random(),
            gender=gender,
            max_speed=2.0 + np.random.random() * 3,
            jump_power=6 + np.random.random() * 6,
            metabolic_rate=0.8 + np.random.random() * 0.4,
        )
        
        body = CreatureBody(phenotype=phenotype, x=x, y=y)
        
        # === HORMONAL DIMORPHISM (1) - Gender-specific hormone baselines ===
        if gender == Gender.MALE:
            body.homeostasis.testosterone_base = 0.7 + np.random.random() * 0.2
            body.homeostasis.estrogen_base = 0.2 + np.random.random() * 0.2
        else:
            body.homeostasis.testosterone_base = 0.2 + np.random.random() * 0.2
            body.homeostasis.estrogen_base = 0.7 + np.random.random() * 0.2

        # Create EmbodiedBrain (brain + body wiring)
        try:
            embodied = EmbodiedBrain(brain=None, body=body, brain_scale=brain_size)
        except Exception as e:
            print(f"Failed to create brain: {e}")
            # Fall back to instinct-only
            self.spawn_creature()
            return

        # Still create instincts as backup/blending
        drive_params = {
            'hunger': 0.3 + np.random.random() * 0.4,
            'fear': 0.3 + np.random.random() * 0.4,
            'curiosity': 0.3 + np.random.random() * 0.4,
            'social': 0.3 + np.random.random() * 0.4,
        }
        
        # Gender affects baseline drives (10)
        if gender == Gender.MALE:
            drive_params['fear'] = max(0.1, drive_params['fear'] - 0.1)
            drive_params['curiosity'] = min(0.9, drive_params['curiosity'] + 0.1)
        else:
            drive_params['social'] = min(0.9, drive_params['social'] + 0.1)
            
        instincts = InstinctSystem(drive_params)
        reward_system = RewardSystem()
        
        creature = LivingCreature(
            body=body,
            brain=embodied.brain,
            instincts=instincts,
            reward_system=reward_system,
            embodied_brain=embodied,
            generation=0,
            use_neural_control=False  # STATE MACHINE controls movement, not raw brain output
        )
        
        self.creatures.append(creature)
        self.lineage_system.register_birth(creature)
        self.world_renderer.set_creatures(self.creatures)
        self.update_creature_list()
        print(f"Spawned {creature.name} with state machine control")

    def simulation_step(self):
        """Run one simulation step.
        
        In single-threaded mode: Runs physics, brain, metabolism at render rate (~30 FPS)
        In multi-threaded mode: Physics/brain run on separate threads, this just renders + applies results
        """
        if self.paused or self.world is None:
            return
        
        # Multi-threaded mode: Physics runs on its own thread, we just apply results
        if self.use_threading and self.threaded_manager and self.threaded_manager.is_running:
            # Apply any queued brain results
            self.threaded_manager.apply_brain_results()
            
            # Update world time
            self.world.update(0.033 * self.simulation_speed)
            
            # Process mating pairs (timer updates, baby spawning)
            self._process_mating_pairs(0.033 * self.simulation_speed)
            
            # Redraw
            self.world_renderer.update()
            return
        
        # Single-threaded mode: Everything runs here
        # Fixed physics timestep to prevent tunneling
        # At high speeds, run multiple sub-steps
        base_dt = 0.033  # ~30 FPS base
        
        # Calculate how many sub-steps we need
        # Physics works best with dt <= 0.05 to prevent tunneling
        max_physics_dt = 0.05
        total_dt = base_dt * self.simulation_speed
        
        # At speed 1, use a reasonable minimum dt so things fall properly
        if self.simulation_speed < 2:
            # Low speed: use single step with minimum dt
            physics_dt = max(0.02, total_dt)  # Minimum dt for visible movement
            sub_steps = 1
        else:
            # High speed: subdivide into safe physics steps
            sub_steps = max(1, int(np.ceil(total_dt / max_physics_dt)))
            physics_dt = total_dt / sub_steps
        
        # Run sub-steps
        for _ in range(sub_steps):
            self._run_simulation_substep(physics_dt)
        
        # Process mating pairs (timer updates, baby spawning)
        self._process_mating_pairs(total_dt)
        
        # Update world time (only once per frame)
        self.world.update(total_dt)
        
        # Redraw
        self.world_renderer.update()
    
    def _start_threaded_simulation(self):
        """Start the multi-threaded simulation loops."""
        if not self.threaded_manager or not self.world:
            return
        
        # Define physics callback
        def physics_callback(dt):
            self._run_simulation_substep(dt)
        
        # Define brain callback (returns result for queuing)
        def brain_callback(creature, dt):
            return self._update_creature_brain(creature, dt)
        
        self.threaded_manager.start(
            creatures=self.creatures,
            world=self.world,
            physics_callback=physics_callback,
            brain_callback=brain_callback
        )
    
    def _stop_threaded_simulation(self):
        """Stop the multi-threaded simulation loops."""
        if self.threaded_manager:
            self.threaded_manager.stop()
    
    def _update_creature_brain(self, creature, dt: float) -> Optional[Dict]:
        """
        Update a single creature's brain (used by threaded brain loop).
        
        Returns brain result dict for queueing, or None on error.
        """
        if creature.embodied_brain is None:
            return None
        
        try:
            # Get sensory data for brain
            all_bodies = [c.body for c in self.creatures if c.body.is_alive()]
            sensory_data = creature.body.get_sensory_input(self.world, all_bodies)
            
            # Encode for brain
            brain_input = creature.embodied_brain.sensory_encoder.encode(sensory_data, creature.body)
            
            # Get drives for neuromodulation
            drives = creature.body.homeostasis.get_drive_levels()
            
            # Compute neuromodulator changes from body state
            neuro_changes = creature.embodied_brain.neuro_bridge.compute_neuromodulator_changes(
                drives, sensory_data.get('internal', {})
            )
            
            # Apply neuromodulator changes to brain
            creature.embodied_brain._apply_neuromodulator_changes(neuro_changes)
            
            # Process through brain (learning happens here)
            brain_result = creature.embodied_brain.brain.process_raw(
                brain_input,
                dt=dt,
                learning_enabled=True
            )
            
            # === NSM: Set age and metabolic plasticity ===
            brain = creature.embodied_brain.brain
            if hasattr(brain, 'set_creature_age'):
                # Normalize age (assuming max lifespan ~300 seconds for full maturity)
                normalized_age = min(1.0, creature.body.lifetime / 300.0)
                brain.set_creature_age(normalized_age)
            
            if hasattr(brain, 'set_metabolic_plasticity'):
                brain.set_metabolic_plasticity(creature.body.phenotype.metabolic_rate)
            
            # === NSM: Pain-Based Reward Tagging ===
            # Tag synapses based on current experience for sleep consolidation
            if hasattr(brain, 'tag_reward'):
                pain = sensory_data.get('pain', 0)
                damage_source = sensory_data.get('damage_source', 'general')
                
                # Pain → negative tag (weaken these pathways during sleep)
                if pain > 0.1:
                    brain.tag_reward(-pain, context=f"pain:{damage_source}")
                
                # Food reward → positive tag (strengthen these pathways)
                if getattr(creature.body, 'food_eaten', 0) > 0:
                    brain.tag_reward(0.5, context="eating")
                
                # Water reward → positive tag
                if getattr(creature.body, 'water_consumed', 0) > 0:
                    brain.tag_reward(0.3, context="drinking")
                
                # Accumulate fatigue for sleep system
                if hasattr(brain, 'accumulate_fatigue'):
                    cortisol = neuro_changes.get('cortisol', 0)
                    brain.accumulate_fatigue(dt, cortisol)
            
            # === NSM: Sleep Sync ===
            if hasattr(brain, 'sync_sleep_state'):
                 brain.sync_sleep_state(creature.body.homeostasis.is_sleeping)
            
            # Legacy reward signal
            if getattr(creature.body, 'food_eaten', 0) > 0:
                if hasattr(brain, 'receive_reward'):
                    brain.receive_reward(0.5)
            
            # Return result for inspection
            return {
                'sensory_input': brain_input,
                'brain_output': brain_result.get('reservoir_output', np.zeros(64)) if brain_result else np.zeros(64)
            }
            
        except Exception as e:
            return None

    def get_threading_stats(self) -> Optional[Dict[str, float]]:
        """Get multi-threading performance statistics."""
        if self.threaded_manager:
            return self.threaded_manager.get_performance_stats()
        return None
    
    def _run_simulation_substep(self, dt: float):
        """Run a single physics sub-step using HIGH-LEVEL movement controller."""
        # Get all creature bodies for visibility
        all_bodies = [c.body for c in self.creatures if c.body.is_alive()]
        
        # Gather all hazard positions for movement controllers
        all_hazards = [(h.x + h.width/2, h.y + h.height/2, max(h.width, h.height)/2) 
                       for h in self.world.hazards]
        
        # Update each creature
        for creature in self.creatures: 
            if not creature.body.is_alive():
                # TIER 4: Decomposition / Cleanup
                # We only increment logic here. Removal happens in update_info (Main Thread)
                # to avoid "list modified during iteration" errors in the renderer.
                if not getattr(creature, 'death_registered', False):
                    self.lineage_system.register_death(creature)
                    creature.death_registered = True
                
                creature.decay_timer += dt
                continue
            
            # Feature: Update speech timer
            if creature.current_speech:
                creature.speech_timer -= dt
                if creature.speech_timer <= 0:
                    creature.current_speech = None
            
            # Feature: Forced Explore (anti-stuck mechanism)
            if creature.forced_explore_timer > 0:
                creature.forced_explore_timer -= dt
                creature.movement.set_goal_wander()
                creature.behavior_state.state = BehaviorState.WANDER
                
                # Still update physics and metabolism
                creature.movement.update(dt, [], [])
                creature.body.update(dt, self.world, creature.movement.get_motor_output())
                self._update_metabolism(dt)
                continue  # SKIP other AI logic while forced exploring
            
            # Get creature state
            cx = creature.body.motor.x
            cy = creature.body.motor.y
            on_ground = creature.body.motor.on_ground
            in_water = creature.body.motor.in_water
            
            # Update movement controller's knowledge of hazards
            creature.movement.set_hazards(all_hazards)
            
            # Get drives for decision making
            drives = creature.body.homeostasis.get_drive_levels()
            hunger = drives.get('hunger', 0)
            thirst = drives.get('thirst', 0)
            pain = creature.body.homeostasis.pain
            
            # === TIER 3: Social Learning Update ===
            # Update what creatures this one is watching
            if creature.social_learning is not None:
                # Build list of visible creatures for observation
                visible_creatures = []
                for other in self.creatures:
                    if other == creature or not other.body.is_alive():
                        continue
                    other_x = other.body.motor.x
                    other_y = other.body.motor.y
                    dist = np.sqrt((other_x - cx)**2 + (other_y - cy)**2)
                    if dist < creature.social_learning.observation_range:
                        # Mutation: Check collision/mating
                        if dist < 30:
                            my_rep = drives.get('reproduction', 0)
                            other_rep = other.body.homeostasis.get_drive_levels().get('reproduction', 0)
                            if my_rep > 0 and other_rep > 0:
                                self._start_mating(creature, other)

                        visible_creatures.append({
                            'creature': other,
                            'dist': dist,
                            'rel_x': other_x - cx,
                            'rel_y': other_y - cy,
                            'behavior': other.behavior_state.state.name if other.behavior_state else 'unknown',
                            'is_eating': hasattr(other.body, 'food_eaten') and other.body.food_eaten > 0,
                            'is_drinking': hasattr(other.body, 'water_consumed') and other.body.water_consumed > 0,
                            'health_delta': getattr(other.body.homeostasis, '_last_health_delta', 0),
                            'hunger_before': getattr(other.body.homeostasis, '_last_hunger', hunger),
                            'hunger_after': other.body.homeostasis.hunger,
                        })
                
                # Update observer's watched list
                creature.social_learning.update_watched_creatures(
                    observer_x=cx, observer_y=cy,
                    visible_creatures=visible_creatures
                )
                
                # Detect success events from watched creatures
                creature.social_learning.detect_success_events(
                    visible_creatures=visible_creatures,
                    cultural_knowledge=self.cultural_knowledge
                )
            
            # === TIER 3: Mental Map Update ===
            # Update creature's internal spatial memory
            if creature.mental_map is not None:
                creature.mental_map.update_position(cx, cy)
                # Decay old memories slightly each frame
                creature.mental_map.decay_memories(dt * 0.1)  # Slow decay
            
            # === HIGH-LEVEL AI DECISION ===
            # Find nearest resources (combine direct sensing with mental map)
            nearest_food = self._find_nearest_food(cx, cy)
            nearest_water = self._find_nearest_water(cx, cy)
            nearest_hazard = self._find_nearest_hazard(cx, cy)
            
            # If no direct food/water visible, check mental map
            if nearest_food is None and creature.mental_map is not None:
                remembered_food = creature.mental_map.get_nearest_food()
                if remembered_food:
                    nearest_food = (remembered_food[0], remembered_food[1])
            
            if nearest_water is None and creature.mental_map is not None:
                remembered_water = creature.mental_map.get_nearest_water()
                if remembered_water:
                    nearest_water = (remembered_water[0], remembered_water[1])
            
            # Check mental map for known dangers (even if not currently visible)
            if creature.mental_map is not None and nearest_hazard is None:
                remembered_danger = creature.mental_map.get_nearest_danger(max_dist=150)
                if remembered_danger:
                    nearest_hazard = (remembered_danger[0], remembered_danger[1])
            
            # === MATING CHECK: Skip all behavior if actively mating ===
            # Mating pairs are frozen in place - don't override their state
            is_mating = False
            for pair in self.mating_pairs:
                if pair['p1'] == creature or pair['p2'] == creature:
                    is_mating = True
                    creature.movement.set_goal_stay()
                    creature.body.motor.vx = 0
                    break
            
            if is_mating:
                # Still update physics but skip all behavior decisions
                movement_result = creature.movement.update(
                    dt, cx, cy, on_ground, in_water, self.world
                )
                creature.body.motor.vx = 0  # Force stop
                
                # Apply physics (gravity, collisions) but no movement
                new_x, new_y, new_vx, new_vy, new_on_ground = self.world.apply_physics(
                    creature.body.motor.x, creature.body.motor.y,
                    0, creature.body.motor.vy,  # Zero horizontal velocity
                    dt
                )
                creature.body.motor.x = new_x
                creature.body.motor.y = new_y
                creature.body.motor.vx = 0
                creature.body.motor.vy = new_vy
                creature.body.motor.on_ground = new_on_ground
                continue  # Skip all other behavior logic
            
            # === TIER 2: Predictive Pain Avoidance ===
            # Check if brain predicts pain BEFORE it happens
            predicted_pain = 0.0
            if creature.embodied_brain is not None:
                brain = creature.embodied_brain.brain
                if hasattr(brain, 'predict_pain'):
                    predicted_pain = brain.predict_pain()
                    
                    # Update predictor with actual pain (for learning)
                    if hasattr(brain, 'update_pain_predictor'):
                        brain.update_pain_predictor(pain)
            
            # Priority-based goal selection (simple and clear)
            # 1. FLEE if in pain OR if brain predicts pain
            # Bravery: Higher bravery = Higher pain threshold
            # Range 0.0-1.0. 1.0 = 2x threshold (0.2). 0.0 = 0.5x threshold (0.05).
            flee_threshold = 0.1 * (0.5 + 1.5 * creature.body.phenotype.bravery)
            
            # Desperation: If starving, ignore pain (Rabbit eats Bear)
            # If hunger > 0.8, threshold increases massively (need serious pain to flee)
            if hunger > 0.8:
                flee_threshold = 0.4
            
            # Combine actual pain with predicted pain (predicted is weighted less)
            effective_pain = max(pain, predicted_pain * 0.7)
            
            # Sleep override: Don't sleep if critically thirsty/hungry (survival priority)
            critical_thirst = creature.body.homeostasis.thirst > 0.7
            critical_hunger = creature.body.homeostasis.hunger > 0.7
            can_sleep = not (critical_thirst or critical_hunger)
                
            if effective_pain > flee_threshold:
                if nearest_hazard:
                    creature.movement.set_goal_flee_from(nearest_hazard[0], nearest_hazard[1])
                else:
                    # Just run in current direction (Flee from point BEHIND us)
                    # If vx is 0, pick a random point behind
                    if abs(creature.body.motor.vx) > 0.1:
                        flee_from_x = cx - np.sign(creature.body.motor.vx) * 100
                    else:
                        flee_from_x = cx - 100 if np.random.random() > 0.5 else cx + 100
                        
                    creature.movement.set_goal_flee_from(flee_from_x, cy)
            
            # 2. SLEEP - Check if already sleeping OR need to sleep (survival allows override)
            elif can_sleep and (creature.behavior_state.state == BehaviorState.SLEEPING or creature.body.homeostasis.energy < 0.2):
                # Start or continue sleeping
                creature.movement.set_goal_stay()  # Stop moving
                creature.behavior_state.state = BehaviorState.SLEEPING
                
                # Use the homeostasis sleep system for realistic metabolism
                if not creature.body.homeostasis.is_sleeping:
                    creature.body.homeostasis.start_sleep()
                    # Play sleep sound
                    self.sound_manager.play('sleep', 0.2)
                    # Trigger speech
                    self._trigger_speech(creature, "sleep tired rest", mood="tired")
                    
                    # NSM: Start sleep consolidation for neural brain
                    if creature.embodied_brain is not None and hasattr(creature.embodied_brain.brain, 'enter_sleep'):
                        creature.embodied_brain.brain.enter_sleep()
                
                # Energy recovery is now handled in homeostasis._sleep_update()
                # No need to duplicate it here
                
                # EMERGENCY: Wake up if critically thirsty/hungry (survival override)
                critical_thirst = creature.body.homeostasis.thirst > 0.85
                critical_hunger = creature.body.homeostasis.hunger > 0.85
                emergency_wake = critical_thirst or critical_hunger
                
                # Wake up when: (normal rest) OR (critical survival need)
                normal_wake = creature.body.homeostasis.energy > 0.6 and creature.body.homeostasis.fatigue < 0.2
                
                # Sleep deprivation penalty if forced awake (emergency)
                if emergency_wake and creature.body.homeostasis.fatigue > 0.5:
                     creature.body.homeostasis.stress += 0.2 # Waking up tired is stressful
                
                if normal_wake or emergency_wake:
                    creature.body.homeostasis.wake_up()
                    creature.behavior_state.state = BehaviorState.IDLE
                    
                    # NSM: Perform sleep consolidation when waking
                    if creature.embodied_brain is not None and hasattr(creature.embodied_brain.brain, 'sleep_consolidation'):
                        try:
                            consol_result = creature.embodied_brain.brain.sleep_consolidation(sleep_duration=1.0)
                            if consol_result.get('strengthened', 0) > 0 or consol_result.get('pruned', 0) > 0:
                                print(f"[NSM] {creature.name} consolidated: "
                                      f"+{consol_result.get('strengthened', 0)} -{consol_result.get('pruned', 0)} synapses")
                        except Exception as e:
                            pass  # Don't crash on consolidation errors
            
            # 3. SEEK MATE (Condition: High reproduction drive & Energy & Maturity)
            # Use 'reproduction' drive from homeostasis which already checks fertility & energy
            # TUNING: Lowered threshold from 0.4 to 0.3 for easier breeding
            elif drives.get('reproduction', 0) > 0.3 and creature.body.lifetime > 20: 
                # Check for opposite gender mates
                from brain.creature import Gender
                my_gender = creature.body.phenotype.gender
                
                # Scan for mate
                best_mate = None
                min_dist = 200 # Vision range for mating
                
                for other in self.creatures:
                    if other == creature or not other.body.is_alive(): 
                        continue
                    
                    # Gender check - must be opposite
                    other_gender = other.body.phenotype.gender
                    if my_gender and other_gender and my_gender == other_gender:
                        continue  # Same gender, skip
                        
                    # 1. Similarity check (species)
                    if abs(other.body.phenotype.hue - creature.body.phenotype.hue) > 0.2:
                        continue
                        
                    # 2. Distance check
                    dist = np.sqrt((other.body.motor.x - cx)**2 + (other.body.motor.y - cy)**2)
                    if dist > min_dist:
                        continue
                        
                    # 3. Preference (can add more complex choice here)
                    min_dist = dist
                    best_mate = other
                
                if best_mate:
                    if min_dist < 50:
                        # Close enough to mate!
                        creature.behavior_state.state = BehaviorState.MATING
                        self._start_mating(creature, best_mate)
                        creature.movement.set_goal_stay()
                        creature.movement.speed_multiplier = 1.0
                    elif min_dist < 100:
                        # In courting range - do MATE_CALLING display
                        creature.behavior_state.state = BehaviorState.MATE_CALLING
                        creature.movement.set_goal_stay()  # Stop to display
                        
                        # Courtship behaviors (brain-driven, but we trigger the context)
                        # 1. Vocalizations - trigger speech about attraction
                        if np.random.random() < 0.1:  # 10% chance per tick
                            self._trigger_speech(creature, "love mate attraction", mood="excited")
                        
                        # 2. Visual display - small "dance" movements
                        if np.random.random() < 0.05:
                            creature.movement.speed_multiplier = 0.3
                            # Quick left-right shimmy
                            creature.body.motor.vx = 30 * (1 if np.random.random() > 0.5 else -1)
                        
                        # 3. Tool display - if holding a tool, show it off
                        if hasattr(creature, 'held_tool') and creature.held_tool:
                            # Having a tool makes you more attractive (brain can learn this)
                            if creature.embodied_brain is not None:
                                try:
                                    creature.embodied_brain.brain.learning.receive_reward(0.02)
                                except Exception:
                                    pass
                    else:
                        # Too far - approach mate
                        creature.movement.set_goal_go_to(best_mate.body.motor.x, best_mate.body.motor.y)
                        creature.movement.speed_multiplier = 1.0
                else:
                    # No mate found - do MATE_CALLING to attract one
                    if np.random.random() < 0.3:  # Sometimes call out
                        creature.behavior_state.state = BehaviorState.MATE_CALLING
                        if np.random.random() < 0.1:
                            self._trigger_speech(creature, "seeking mate lonely", mood="longing")
            
            # 4. SEEK WATER if thirsty (Priority over food)
            elif thirst > 0.3:
                # Desperation scaling: the thirstier, the faster and farther we search
                desperation = (thirst - 0.3) / 0.7  # 0.0 at threshold, 1.0 when dying
                speed_multiplier = 1.0 + desperation * 2.0  # Up to 3x speed when desperate
                
                if nearest_water:
                    # Clear search target when water is found
                    if hasattr(creature, '_water_search_target'):
                        creature._water_search_target = None
                    
                    # Check if close enough to drink (stop and drink)
                    # Must be within actual drinking range (50px) not just detection range
                    water_dist = np.sqrt((nearest_water[0] - cx)**2 + (nearest_water[1] - cy)**2)
                    if water_dist < 50:
                        # Close enough - STOP and drink
                        creature.movement.set_goal_stay()
                        creature.behavior_state.state = BehaviorState.SEEKING_WATER
                        
                        # Try to drink
                        success = self._try_drink_water(creature, nearest_water[0], nearest_water[1])
                        
                        if success:
                            creature.consecutive_drink_fails = 0
                        else:
                            creature.consecutive_drink_fails += 1
                            if creature.consecutive_drink_fails > 3:
                                # Stuck! Force explore
                                creature.forced_explore_timer = 2.0 + np.random.random() * 2.0
                                creature.consecutive_drink_fails = 0
                                # Push away slightly
                                creature.body.motor.vx = (np.random.random() - 0.5) * 50
                                creature.body.motor.vy = (np.random.random() - 0.5) * 50
                    else:
                        # Still approaching - move toward water (faster when desperate)
                        # Add slight randomness to target to prevent stacking
                        target_x = nearest_water[0] + (np.random.random() - 0.5) * 20
                        target_y = nearest_water[1] + (np.random.random() - 0.5) * 20
                        creature.movement.set_goal_go_to(target_x, target_y)
                        creature.movement.speed_multiplier = speed_multiplier
                        creature.behavior_state.state = BehaviorState.SEEKING_WATER
                else:
                    # Thirsty but no water visible - DESPERATELY explore
                    # Desperation makes us search farther and pick new targets more often
                    needs_new_target = False
                    
                    if not hasattr(creature, '_water_search_target') or creature._water_search_target is None:
                        needs_new_target = True
                    else:
                        # Check if we reached target
                        target_x, target_y = creature._water_search_target
                        dist_to_target = np.sqrt((target_x - cx)**2 + (target_y - cy)**2)
                        # When desperate, pick new targets sooner (don't go all the way)
                        retarget_threshold = 100 - desperation * 70  # 100px when calm, 30px when dying
                        if dist_to_target < retarget_threshold:
                            needs_new_target = True
                    
                    if needs_new_target:
                        # Search distance increases with desperation
                        base_dist = 400 + desperation * 600  # 400-1000 pixels based on desperation
                        search_dist = base_dist + np.random.random() * 300
                        search_angle = np.random.random() * 2 * np.pi
                        target_x = cx + search_dist * np.cos(search_angle)
                        target_y = cy + search_dist * np.sin(search_angle)
                        creature._water_search_target = (target_x, target_y)
                    
                    # Always move toward search target with desperation speed
                    target_x, target_y = creature._water_search_target
                    creature.movement.set_goal_go_to(target_x, target_y)
                    creature.movement.speed_multiplier = speed_multiplier
                    creature.behavior_state.state = BehaviorState.SEEKING_WATER

            # 5. SEEK FOOD if hungry (NOT sleeping)
            elif hunger > 0.3:
                # Desperation scaling: the hungrier, the faster and farther we search
                desperation = (hunger - 0.3) / 0.7  # 0.0 at threshold, 1.0 when starving
                speed_multiplier = 1.0 + desperation * 2.0  # Up to 3x speed when desperate
                
                if nearest_food:
                    # Clear search target when food is found
                    if hasattr(creature, '_food_search_target'):
                        creature._food_search_target = None
                    
                    # Check if close enough to eat (stop and eat)
                    # Must be within actual eating range (70px) not just detection range
                    food_dist = np.sqrt((nearest_food[0] - cx)**2 + (nearest_food[1] - cy)**2)
                    if food_dist < 70:
                        # Close enough - STOP and eat
                        creature.movement.set_goal_stay()
                        creature.behavior_state.state = BehaviorState.SEEKING_FOOD
                        
                        # Try to eat
                        success = self._try_eat_food(creature)
                        
                        if success:
                            creature.consecutive_eat_fails = 0
                        else:
                            creature.consecutive_eat_fails += 1
                            if creature.consecutive_eat_fails > 3:
                                # Stuck! Force explore
                                creature.forced_explore_timer = 2.0 + np.random.random() * 2.0
                                creature.consecutive_eat_fails = 0
                                # Push away slightly
                                creature.body.motor.vx = (np.random.random() - 0.5) * 50
                                creature.body.motor.vy = (np.random.random() - 0.5) * 50
                    else:
                        # Still approaching - move toward food (faster when desperate)
                        # Add slight randomness to target to prevent stacking
                        target_x = nearest_food[0] + (np.random.random() - 0.5) * 20
                        target_y = nearest_food[1] + (np.random.random() - 0.5) * 20
                        creature.movement.set_goal_go_to(target_x, target_y)
                        creature.movement.speed_multiplier = speed_multiplier
                        creature.behavior_state.state = BehaviorState.SEEKING_FOOD
                else:
                    # Hungry but no food visible - DESPERATELY explore
                    # Desperation makes us search farther and pick new targets more often
                    needs_new_target = False
                    
                    if not hasattr(creature, '_food_search_target') or creature._food_search_target is None:
                        needs_new_target = True
                    else:
                        # Check if we reached target
                        target_x, target_y = creature._food_search_target
                        dist_to_target = np.sqrt((target_x - cx)**2 + (target_y - cy)**2)
                        # When desperate, pick new targets sooner (explore more areas)
                        retarget_threshold = 100 - desperation * 70  # 100px when calm, 30px when starving
                        if dist_to_target < retarget_threshold:
                            needs_new_target = True
                    
                    if needs_new_target:
                        # Search distance increases with desperation
                        base_dist = 400 + desperation * 600  # 400-1000 pixels based on desperation
                        search_dist = base_dist + np.random.random() * 300
                        search_angle = np.random.random() * 2 * np.pi
                        target_x = cx + search_dist * np.cos(search_angle)
                        target_y = cy + search_dist * np.sin(search_angle)
                        creature._food_search_target = (target_x, target_y)
                    
                    # Always move toward search target with desperation speed
                    target_x, target_y = creature._food_search_target
                    creature.movement.set_goal_go_to(target_x, target_y)
                    creature.movement.speed_multiplier = speed_multiplier
                    creature.behavior_state.state = BehaviorState.SEEKING_FOOD
            
            # 6. SOCIAL LEARNING - imitate successful others or use cultural knowledge
            elif creature.social_learning is not None and creature.social_learning.should_imitate():
                # Check if we should imitate someone
                imitation_target = creature.social_learning.get_imitation_target()
                if imitation_target:
                    # Follow the successful creature
                    creature.movement.set_goal_go_to(imitation_target['x'], imitation_target['y'])
                    creature.movement.speed_multiplier = 1.0  # Normal speed
                    creature.behavior_state.state = BehaviorState.SOCIALIZING
                else:
                    # Use cultural knowledge - go to known food/water spots
                    known_food = self.cultural_knowledge.get_nearest_food(cx, cy)
                    known_water = self.cultural_knowledge.get_nearest_water(cx, cy)
                    
                    if known_food and hunger > 0.15:
                        creature.movement.set_goal_go_to(known_food[0], known_food[1])
                        creature.movement.speed_multiplier = 1.0  # Normal speed
                    elif known_water and thirst > 0.15:
                        creature.movement.set_goal_go_to(known_water[0], known_water[1])
                        creature.movement.speed_multiplier = 1.0  # Normal speed
                    else:
                        creature.movement.set_goal_wander()
                        creature.movement.speed_multiplier = 1.0  # Normal speed
            
            # 7. WANDER otherwise
            else:
                creature.movement.set_goal_wander()
                creature.movement.speed_multiplier = 1.0  # Normal speed
                
                # === TOOL PICKUP - Opportunistic collection while wandering ===
                # Check for nearby tools and pick them up
                if not hasattr(creature, 'held_tool') or creature.held_tool is None:
                    nearby_tools = self.world.get_nearby_tools(cx, cy, radius=40)
                    if nearby_tools:
                        tool = nearby_tools[0]
                        if self.world.pickup_tool(tool, str(creature.body.id)):
                            creature.held_tool = tool
                            # Small reward for collecting tool (curiosity satisfied)
                            if creature.embodied_brain is not None:
                                try:
                                    if hasattr(creature.embodied_brain.brain, 'learning'):
                                        creature.embodied_brain.brain.learning.receive_reward(0.1)
                                except Exception:
                                    pass

            # === AUTONOMOUS TOOL DROP/THROW DECISIONS (Brain-Based) ===
            # Each creature's brain influences their tool decisions differently
            if hasattr(creature, 'held_tool') and creature.held_tool is not None:
                tool = creature.held_tool
                should_drop = False
                should_throw = False
                throw_direction = 0.0
                
                # Get brain chemistry for decision-making (each creature is different)
                drop_threshold = 0.5  # Default
                throw_threshold = 0.7  # Default
                impulsivity = 0.5  # Default
                
                if creature.embodied_brain and hasattr(creature.embodied_brain, 'brain'):
                    try:
                        brain = creature.embodied_brain.brain
                        if hasattr(brain, 'learning') and hasattr(brain.learning, 'neuromod'):
                            neuromod = brain.learning.neuromod
                            levels = neuromod.levels if hasattr(neuromod, 'levels') else {}
                            
                            # Brain chemistry affects thresholds
                            dopamine = levels.get('dopamine', 0.5)
                            serotonin = levels.get('serotonin', 0.5)
                            cortisol = levels.get('cortisol', 0.3)
                            norepinephrine = levels.get('norepinephrine', 0.3)
                            
                            # High dopamine = less likely to drop (enjoying the tool)
                            # Low serotonin = more impulsive (throw more readily)
                            # High cortisol = stressed, might drop or throw
                            # High norepinephrine = alert, defensive throwing
                            
                            drop_threshold = 0.3 + dopamine * 0.4 - cortisol * 0.2
                            throw_threshold = 0.9 - norepinephrine * 0.3 - cortisol * 0.2
                            impulsivity = 0.3 + (1.0 - serotonin) * 0.5 + cortisol * 0.2
                    except Exception:
                        pass
                
                # Brain-influenced decisions based on creature state
                current_state = creature.behavior_state.state if hasattr(creature, 'behavior_state') else None
                
                # Eating/drinking - drop chance based on how much they value the tool vs food
                if current_state == BehaviorState.SEEKING_FOOD and hunger > drop_threshold:
                    should_drop = True
                elif current_state == BehaviorState.SEEKING_WATER and thirst > drop_threshold:
                    should_drop = True
                elif current_state == BehaviorState.MATING:
                    # Some creatures might keep their tool, impulsive ones drop it
                    if np.random.random() < impulsivity:
                        should_drop = True
                    
                # Exhaustion - impulsive creatures drop sooner
                energy_drop_point = 0.05 + (1.0 - impulsivity) * 0.15  # 0.05 to 0.2
                if creature.body.homeostasis.energy < energy_drop_point:
                    should_drop = True
                    
                # Threat response - brain decides fight (throw) or flight (drop)
                if hasattr(creature, 'threat_level') and creature.threat_level > throw_threshold:
                    # Fight response (throw at threat)
                    should_throw = True
                    if hasattr(creature, 'threat_direction'):
                        throw_direction = creature.threat_direction
                    else:
                        throw_direction = 0 if creature.body.motor.facing_right else np.pi
                        
                # Impulsive random actions based on brain state
                # High impulsivity + stress = might randomly throw
                if impulsivity > 0.6 and np.random.random() < 0.005 * impulsivity:
                    # Impulsive throw!
                    should_throw = True
                    throw_direction = np.random.random() * 2 * np.pi
                
                # Execute drop or throw
                if should_throw:
                    self.world.throw_tool(tool, cx, cy, throw_direction, power=0.5 + impulsivity * 0.4, 
                                         thrower_id=str(creature.body.id))
                    creature.held_tool = None
                elif should_drop:
                    self.world.drop_tool(tool, cx, cy)
                    creature.held_tool = None

            
            # === CONTROL: NEURAL VS GOAL-BASED ===
            if creature.use_neural_control and hasattr(creature, 'last_step_data') and creature.last_step_data and 'brain_output' in creature.last_step_data:
                # NEURAL CONTROL: Brain drives motor directly
                brain_output = creature.last_step_data['brain_output']
                creature.body._process_brain_output(brain_output)
                # Facing direction updated by brain (usually)
            else:
                # GOAL-BASED CONTROL: Movement controller executes high-level goal
                movement_result = creature.movement.update(
                    dt, cx, cy, on_ground, in_water, self.world
                )
                
                # Apply movement directly to creature
                creature.body.motor.vx = movement_result['vx']
                if movement_result['vy'] is not None:
                    creature.body.motor.vy = movement_result['vy']
                if movement_result['jump'] and on_ground:
                    creature.body.motor.vy = -creature.movement.jump_power
                
                # Update facing direction
                creature.body.motor.facing_right = creature.movement.facing_right
            
            # Water no longer blocks or causes drowning - it's just for drinking
            # (Removed swim-up logic)
            
            # Facing update handled above
            
            # Apply physics (gravity, collisions)
            new_x, new_y, new_vx, new_vy, new_on_ground = self.world.apply_physics(
                creature.body.motor.x, creature.body.motor.y,
                creature.body.motor.vx, creature.body.motor.vy,
                dt
            )
            
            # Update creature position
            creature.body.motor.x = new_x
            creature.body.motor.y = new_y
            creature.body.motor.vx = new_vx
            creature.body.motor.vy = new_vy
            creature.body.motor.on_ground = new_on_ground
            creature.body.motor.in_water = self.world.is_water(new_x, new_y)
            
            # Update homeostasis (hunger, thirst, etc)
            ambient_temp = self.world.get_temperature(new_x, new_y)
            activity = abs(new_vx) / 100.0
            
            # Get aging setting
            aging_speed = 1.0
            if self.visual_config:
                aging_speed = getattr(self.visual_config, 'aging_speed', 1.0)
                
            creature.body.homeostasis.update(dt, creature.body.phenotype.metabolic_rate, 
                                             ambient_temp, activity,
                                             aging_speed_setting=aging_speed)
                                             
            # === MIGRATION (Walk off edge to new world) ===
            if new_x > self.world.width + 50 or new_x < -50:
                 # Auto-export and remove
                 if self.lineage_system:
                     self.lineage_system.register_death(creature) # Track as "left"
                 
                 # Save to migrations folder
                 mig_path = Path("migrations") / f"{creature.name}_{int(time.time())}.creature"
                 save_creature_to_file(creature, mig_path)
                 
                 # Remove from world (mark for death/removal)
                 # We can't safely remove from list during iteration, so we cheat:
                 creature.body.homeostasis.health = -100 # Kill it so cleanup removes it
                 print(f"[Migration] {creature.name} migrated to {mig_path}")
                 continue
            
            # Check hazards
            is_hazard, damage, hazard_type = self.world.is_hazard(new_x, new_y + 5)
            if is_hazard:
                creature.body.homeostasis.apply_damage(damage * dt * 0.1)
                creature.last_damage_source = 'hazard'  # Track for behavior state machine
                # TIER 3: Update cultural knowledge with danger location
                self.cultural_knowledge.add_danger_location(new_x, new_y, intensity=damage)
                # TIER 3: Update personal mental map with danger
                if creature.mental_map is not None:
                    creature.mental_map.remember_danger(new_x, new_y, severity=damage,
                                                         timestamp=self.world.time if self.world else 0)
            else:
                # Decay damage source if not taking damage
                creature.last_damage_source = 'none'
            
            # Update lifetime
            creature.body.lifetime += dt
            
            # === BRAIN LEARNING (runs in background, doesn't control movement) ===
            # This is where the neural network learns from experience
            # Movement is controlled by the high-level controller above
            # But the brain learns associations: fire → pain → cortisol → weaken synapses
            
            # OPTIMIZATION: If threading is enabled, this logic runs in background thread (10Hz).
            # We skip it here to avoid duplication (60Hz + 10Hz) and race conditions.
            should_run_brain = not (self.use_threading and self.threaded_manager)
            
            if creature.embodied_brain is not None and should_run_brain:
                try:
                    # Get sensory data for brain
                    all_bodies = [c.body for c in self.creatures if c.body.is_alive()]
                    sensory_data = creature.body.get_sensory_input(self.world, all_bodies)
                    
                    # Encode for brain
                    brain_input = creature.embodied_brain.sensory_encoder.encode(sensory_data, creature.body)
                    
                    # Compute neuromodulator changes from body state
                    # This is where PAIN → CORTISOL happens!
                    neuro_changes = creature.embodied_brain.neuro_bridge.compute_neuromodulator_changes(
                        drives, sensory_data.get('internal', {})
                    )
                    
                    # Apply neuromodulator changes to brain
                    creature.embodied_brain._apply_neuromodulator_changes(neuro_changes)
                    
                    # Process through brain (learning happens here via three-factor rule)
                    # High cortisol during pain = LTD boost = weaken "fire approach" synapses
                    brain_result = creature.embodied_brain.brain.process_raw(
                        brain_input,
                        dt=dt,
                        learning_enabled=True
                    )
                    
                    # === NSM: Set age and metabolic plasticity ===
                    brain = creature.embodied_brain.brain
                    if hasattr(brain, 'set_creature_age'):
                        # Normalize age (assuming max lifespan ~300 seconds for full maturity)
                        normalized_age = min(1.0, creature.body.lifetime / 300.0)
                        brain.set_creature_age(normalized_age)
                    
                    if hasattr(brain, 'set_metabolic_plasticity'):
                        brain.set_metabolic_plasticity(creature.body.phenotype.metabolic_rate)
                    
                    # === NSM: Pain-Based Reward Tagging ===
                    # Tag synapses based on current experience for sleep consolidation
                    if hasattr(brain, 'tag_reward'):
                        # Pain → negative tag (weaken these pathways during sleep)
                        if pain > 0.1:
                            brain.tag_reward(-pain, context=f"pain:{hazard_type if is_hazard else 'general'}")
                        
                        # Food reward → positive tag (strengthen these pathways)
                        if creature.body.food_eaten > 0:
                            brain.tag_reward(0.5, context="eating")
                        
                        # Water reward → positive tag
                        if hasattr(creature.body, 'water_consumed') and creature.body.water_consumed > 0:
                            brain.tag_reward(0.3, context="drinking")
                        
                        # Accumulate fatigue for sleep system
                        if hasattr(brain, 'accumulate_fatigue'):
                            brain.accumulate_fatigue(dt, creature.body.homeostasis.cortisol)
                    
                    # Legacy reward signal
                    if creature.body.food_eaten > 0:
                        if hasattr(brain, 'receive_reward'):
                            brain.receive_reward(0.5)
                    
                    # Store data for brain inspector visualization
                    if brain_result:
                        creature.embodied_brain.last_step_data = {
                            'sensory_input': brain_input,
                            'brain_output': brain_result.get('reservoir_output', np.zeros(64)),
                            'drives': drives,
                            'neuro_changes': neuro_changes,
                            'reward': brain_result.get('body_reward', 0),
                        }
                        
                except Exception as e:
                    import traceback
                    traceback.print_exc()  # Print errors for debugging
        
        # Cleanup of dead creatures is now handled in update_info (Main Thread)
        # using creature.decay_timer to avoid race conditions.
        pass
    
    def _combine_motor_outputs(self, brainstem: Dict, instincts: Dict) -> Dict:
        """
        Combine brainstem and instinct outputs.
        
        Priority rules:
        - If brainstem is in survival mode, it dominates
        - Otherwise, max of each channel
        - eat/flee from brainstem always wins
        - IMPORTANT: Resolve move_left vs move_right to single direction
        """
        result = {}
        survival_mode = brainstem.get('survival_mode', False)
        
        # First gather all values
        for key in ['move_left', 'move_right', 'jump', 'eat']:
            bs_val = brainstem.get(key, 0)
            inst_val = instincts.get(key, 0)
            
            if survival_mode:
                # Brainstem dominates in survival mode
                result[key] = bs_val if bs_val > 0.1 else inst_val
            else:
                # Take max of both
                result[key] = max(bs_val, inst_val)
        
        # CRITICAL: Resolve move_left vs move_right conflict
        # Only allow one direction at a time to prevent oscillation
        left = result.get('move_left', 0)
        right = result.get('move_right', 0)
        
        if left > 0 and right > 0:
            # Both directions active - pick the stronger one
            if left > right:
                result['move_right'] = 0
            elif right > left:
                result['move_left'] = 0
            else:
                # Equal - pick based on survival mode or random
                if survival_mode:
                    # In survival, prefer brainstem's choice
                    if brainstem.get('move_left', 0) > brainstem.get('move_right', 0):
                        result['move_right'] = 0
                    else:
                        result['move_left'] = 0
                else:
                    # No preference - just pick right
                    result['move_left'] = 0
        
        return result
    
    def _enrich_sensory_data(self, creature: LivingCreature, sensory_data: Dict) -> Dict:
        """Add position information for state machine targeting."""
        cx, cy = creature.body.motor.x, creature.body.motor.y
        
        # Find nearest food position
        # Use Quadtree optimization if available (radius 500 is generous view)
        nearest_food_list = self.world.find_food_nearby(cx, cy, 500)
        
        nearest_food = None
        nearest_food_dist = float('inf')
        
        for food in nearest_food_list:
            if food.remaining <= 0:
                continue
            dx = food.x - cx
            dy = food.y - cy
            dist = np.sqrt(dx**2 + dy**2)
            if dist < nearest_food_dist:
                nearest_food_dist = dist
                nearest_food = food
        
        if nearest_food:
            sensory_data['nearest_food_x'] = nearest_food.x
            sensory_data['nearest_food_y'] = nearest_food.y
            sensory_data['nearest_food_distance'] = nearest_food_dist
        
        # Find nearest water position
        # Scan nearby tiles for water
        tile_size = self.world.tile_size
        best_water_dist = float('inf')
        best_water_x = None
        best_water_y = None
        
        for dx in range(-5, 6):
            for dy in range(-3, 4):
                check_x = int(cx / tile_size) + dx
                check_y = int(cy / tile_size) + dy
                if 0 <= check_x < self.world.tiles_x and 0 <= check_y < self.world.tiles_y:
                    if self.world.tiles[check_y, check_x] == 2:  # WATER
                        water_x = check_x * tile_size + tile_size // 2
                        water_y = check_y * tile_size + tile_size // 2
                        dist = np.sqrt((water_x - cx)**2 + (water_y - cy)**2)
                        if dist < best_water_dist:
                            best_water_dist = dist
                            best_water_x = water_x
                            best_water_y = water_y
        
        if best_water_x is not None:
            sensory_data['nearest_water_x'] = best_water_x
            sensory_data['nearest_water_y'] = best_water_y
            sensory_data['nearest_water_distance'] = best_water_dist
        
        # Check if in water
        sensory_data['in_water'] = creature.body.motor.in_water
        
        # Add pain level for danger detection (CRITICAL for fleeing fire)
        sensory_data['pain'] = creature.body.homeostasis.pain
        sensory_data['damage_source'] = creature.last_damage_source  # For behavior state machine
        
        # Find nearest hazard for flee targeting
        nearest_hazard = None
        nearest_hazard_dist = float('inf')
        for hazard in self.world.hazards:
            hx = hazard.x + hazard.width / 2
            hy = hazard.y + hazard.height / 2
            dist = np.sqrt((hx - cx)**2 + (hy - cy)**2)
            if dist < nearest_hazard_dist:
                nearest_hazard_dist = dist
                nearest_hazard = hazard
        
        if nearest_hazard:
            sensory_data['nearest_hazard_x'] = nearest_hazard.x + nearest_hazard.width / 2
            sensory_data['nearest_hazard_y'] = nearest_hazard.y + nearest_hazard.height / 2
            sensory_data['nearest_hazard_distance'] = nearest_hazard_dist
        else:
            sensory_data['nearest_hazard_distance'] = float('inf')
        
        # Add creature position for flee calculations
        sensory_data['creature_x'] = cx
        sensory_data['creature_y'] = cy
        
        return sensory_data
    
    def _find_nearest_food(self, cx: float, cy: float):
        """Find nearest food across all zones. Returns (x, y, distance) or None."""
        world_width = self.world.width
        nearest = None
        nearest_dist = float('inf')
        
        # Determine which zones to search based on creature position
        creature_zone = int(cx // world_width)
        min_x = self.world.min_x if self.world.min_x is not None else 0
        max_x = self.world.max_x if self.world.max_x is not None else world_width
        
        # Search in current zone and adjacent zones
        zones_to_check = []
        for z in range(creature_zone - 1, creature_zone + 2):
            zone_start = z * world_width
            if zone_start >= min_x - world_width and zone_start < max_x:
                zones_to_check.append(zone_start)
        
        # Search food in each zone
        for zone_offset in zones_to_check:
            # Calculate search center in this zone's coordinate space
            local_cx = (cx - zone_offset) % world_width + zone_offset
            
            # Use Quadtree with large radius
            nearby_food = self.world.find_food_nearby(local_cx, cy, 600)
            
            for food in nearby_food:
                if food.remaining <= 0:
                    continue
                dx = food.x - cx
                dy = food.y - cy
                dist = np.sqrt(dx**2 + dy**2)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = (food.x, food.y, dist)
        
        return nearest
    
    def _find_nearest_water(self, cx: float, cy: float):
        """Find nearest water source object across all zones. Returns (x, y, distance) or None."""
        if not hasattr(self.world, 'water_sources') or not self.world.water_sources:
            # Fallback: search tiles if no water_sources exist (old worlds)
            return self._find_nearest_water_tile(cx, cy)
        
        best = None
        best_dist = float('inf')
        
        # Search all water source objects
        for water in self.world.water_sources:
            dist = np.sqrt((water.x - cx)**2 + (water.y - cy)**2)
            if dist < best_dist:
                best_dist = dist
                best = (water.x, water.y, dist)
        
        return best
    
    def _find_nearest_water_tile(self, cx: float, cy: float):
        """Fallback: Find nearest water tile (for old worlds without water_sources)."""
        tile_size = self.world.tile_size
        world_width = self.world.width
        best = None
        best_dist = float('inf')
        
        min_x = self.world.min_x if self.world.min_x is not None else 0
        max_x = self.world.max_x if self.world.max_x is not None else world_width
        
        zones_to_check = []
        creature_zone = int(cx // world_width)
        for z in range(creature_zone - 2, creature_zone + 3):
            zone_start = z * world_width
            if zone_start >= min_x - world_width and zone_start < max_x + world_width:
                zones_to_check.append(zone_start)
        
        search_range_x = 50
        search_range_y = 25
        
        for zone_offset in zones_to_check:
            for dx in range(-search_range_x, search_range_x + 1):
                for dy in range(-search_range_y, search_range_y + 1):
                    local_cx = cx - zone_offset
                    check_x = int(local_cx / tile_size) + dx
                    check_y = int(cy / tile_size) + dy
                    if 0 <= check_x < self.world.tiles_x and 0 <= check_y < self.world.tiles_y:
                        if self.world.tiles[check_y, check_x] == 2:  # WATER
                            water_x = zone_offset + check_x * tile_size + tile_size // 2
                            water_y = check_y * tile_size + tile_size // 2
                            dist = np.sqrt((water_x - cx)**2 + (water_y - cy)**2)
                            if dist < best_dist:
                                best_dist = dist
                                best = (water_x, water_y, dist)
        return best
    
    def _find_nearest_hazard(self, cx: float, cy: float):
        """Find nearest hazard. Returns (x, y, distance) or None."""
        nearest = None
        nearest_dist = float('inf')
        for hazard in self.world.hazards:
            hx = hazard.x + hazard.width / 2
            hy = hazard.y + hazard.height / 2
            dist = np.sqrt((hx - cx)**2 + (hy - cy)**2)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = (hx, hy, dist)
        return nearest

    def _try_eat_food(self, creature: LivingCreature):
        """Attempt to eat nearby food. Creature should already be stopped."""
        cx, cy = creature.body.motor.x, creature.body.motor.y
        
        for food in self.world.food_sources[:]:  # Copy list for safe removal
            dx = food.x - cx
            dy = food.y - cy
            dist = np.sqrt(dx**2 + dy**2)
            
            # Large eating range - within 100 pixels total distance
            if dist < 100:
                # STOP the creature while eating
                creature.body.motor.vx = 0
                creature.body.motor.target_vx = 0
                
                # Eat the food
                nutrition = min(food.remaining, 0.3)  # Bite size
                food.remaining -= nutrition
                
                # Apply to creature (clamp to 0-1 range)
                food_value = nutrition * food.nutrition
                
                # Effect multipliers
                reward_mult = 1.0
                poison_damage = 0.0
                cortisol_spike = 0.0
                
                if food.type == FoodType.SWEET_BERRY:
                    reward_mult = 3.0 # Big reward
                elif food.type == FoodType.BITTER_BERRY:
                    reward_mult = -0.5 # Aversion
                elif food.type == FoodType.POISON_BERRY:
                    reward_mult = -1.0 # Punishment (mild)
                    poison_damage = 0.15 # Significant damage per bite
                    cortisol_spike = 0.4 # Major stress
                
                # Apply physiological effects
                creature.body.homeostasis.nutrition = min(
                    1.0, creature.body.homeostasis.nutrition + food_value
                )
                creature.body.homeostasis.energy = min(
                    1.0, creature.body.homeostasis.energy + food_value * 0.3
                )
                creature.body.food_eaten += 1
                
                if poison_damage > 0:
                     creature.body.homeostasis.apply_damage(poison_damage)
                     creature.body.homeostasis.pain = min(1.0, creature.body.homeostasis.pain + 0.5)
                     creature.body.homeostasis.cortisol = min(1.0, creature.body.homeostasis.cortisol + cortisol_spike)
                     # Poison prevents nutrition gain (vomiting)
                     creature.body.homeostasis.nutrition = max(0, creature.body.homeostasis.nutrition - food_value * 2)
                     self.sound_manager.play('hurt', 0.5)
                     
                # Reward brainstem (dopamine learning)
                final_reward = nutrition * food.nutrition * reward_mult
                
                # Play eat sound
                if np.random.random() < 0.3:
                    self.sound_manager.play('eat', 0.3)
                    # Trigger speech
                    self._trigger_speech(creature, "food tasty good", mood="happy")
                    
                    # TIER 4: Reinforce successful communication if nearby creatures respond
                    if creature.current_speech:
                        self._check_communication_success(creature, 'food', food.x, food.y)
                    
                if cortisol_spike > 0:
                    # Direct cortisol injection if supported, else rely on pain
                    pass 

                creature.brainstem.receive_food_reward(final_reward)
                
                # Also reward instincts
                creature.instincts.apply_reward(final_reward)
                
                # TIER 3: Update cultural knowledge with food location
                if poison_damage == 0:  # Only share good food locations
                    self.cultural_knowledge.add_food_location(food.x, food.y, food.nutrition)
                else:  # Share danger locations for poison
                    self.cultural_knowledge.add_danger_location(food.x, food.y, intensity=0.5)
                
                # TIER 3: Update personal mental map
                if creature.mental_map is not None:
                    if poison_damage == 0:
                        creature.mental_map.remember_food(food.x, food.y, food.nutrition, 
                                                          timestamp=self.world.time if self.world else 0)
                    else:
                        creature.mental_map.remember_danger(food.x, food.y, severity=poison_damage,
                                                             timestamp=self.world.time if self.world else 0)
                
                # Remove depleted food
                if food.remaining <= 0.01:
                    self.world.food_sources.remove(food)
                
                return True  # Only eat one food per frame
                
        return False
    
    def _try_drink_water(self, creature: LivingCreature, water_x: float = None, water_y: float = None):
        """Attempt to drink if near water."""
        cx, cy = creature.body.motor.x, creature.body.motor.y
        near_water = False
        
        # Check water_sources objects first (preferred)
        if hasattr(self.world, 'water_sources') and self.world.water_sources:
            for water in self.world.water_sources:
                if water.contains_point(cx, cy, radius=70):
                    near_water = True
                    break
        
        # If water position was provided, also check distance to that specific location
        if not near_water and water_x is not None and water_y is not None:
            water_dist = np.sqrt((water_x - cx)**2 + (water_y - cy)**2)
            near_water = water_dist < 70  # Within 70px of known water source
        
        # Fallback: Check if creature is near any water tile
        if not near_water:
            for dx in range(-70, 71, 10):
                for dy in range(-70, 71, 10):
                    if self.world.is_water(cx + dx, cy + dy):
                        near_water = True
                        break
                if near_water:
                    break
        
        if near_water:
            # STOP the creature while drinking
            creature.body.motor.vx = 0
            creature.body.motor.target_vx = 0
            
            # Drink water - restore hydration
            drink_amount = 0.2  # Per drink action (faster drinking)
            creature.body.homeostasis.hydration = min(
                1.0, creature.body.homeostasis.hydration + drink_amount
            )
            
            # TIER 3: Update cultural knowledge with water location
            self.cultural_knowledge.add_water_location(cx, cy)
            
            # TIER 3: Update personal mental map
            if creature.mental_map is not None:
                creature.mental_map.remember_water(cx, cy, quality=1.0,
                                                    timestamp=self.world.time if self.world else 0)
            
            # Play drink sound
            if np.random.random() < 0.3:
                self.sound_manager.play('drink', 0.3)
            
            # Reward brainstem (dopamine learning)
            if creature.brainstem:
                creature.brainstem.receive_water_reward(drink_amount)
                
                # Trigger speech
                self._trigger_speech(creature, "water refresh drink", mood="calm")
                
                # TIER 4: Reinforce successful communication
                if creature.current_speech:
                    self._check_communication_success(creature, 'water', cx, cy)
                
            return True
            
        return False
    
    def update_info(self):
        """Update info panel."""
        if self.world is None:
            return
            
        # TIER 4: Cleanup dead bodies (Main Thread)
        for c in list(self.creatures):
            if hasattr(c, 'decay_timer') and c.decay_timer > 5.0:
                self.creatures.remove(c)
                # Cleanup selection
                if self.world_renderer.selected_creature == c:
                    self.world_renderer.selected_creature = None
                if self.world_renderer.selected_secondary == c:
                    self.world_renderer.selected_secondary = None
        
        # Update population labels
        total = len(self.creatures)
        alive = sum(1 for c in self.creatures if c.body.is_alive())
        dead = total - alive
        food_count = len(self.world.food_sources) if self.world else 0
        max_gen = max([c.generation for c in self.creatures], default=0)
        
        self.alive_label.setText(f"Alive: {alive}")
        self.dead_label.setText(f"Dead: {dead}")
        self.food_label.setText(f"Food: {food_count}")
        self.gen_label.setText(f"Max Gen: {max_gen}")
        
        # Update Graph
        avg_energy = 0
        if alive > 0:
            avg_energy = sum(c.body.homeostasis.energy for c in self.creatures if c.body.is_alive()) / alive
            
        if hasattr(self, 'stats_graph'):
            self.stats_graph.add_data_point(alive, food_count, avg_energy)
        
        # Update selected creature info
        selected = self.world_renderer.selected_creature
        if selected and selected.body.is_alive():
            body = selected.body
            h = body.homeostasis
            
            # Get current behavioral state
            state_name = selected.behavior_state.state.name if selected.behavior_state else "UNKNOWN"
            
            # Calculate hunger/thirst as "how empty" (inverse of nutrition/hydration)
            hunger_pct = (1.0 - h.nutrition) * 100
            thirst_pct = (1.0 - h.hydration) * 100
            
            self.creature_name.setText(f"Name: {selected.name}")
            
            # Gender display
            gender_text = "Gender: Unknown"
            if hasattr(selected.body.phenotype, 'gender') and selected.body.phenotype.gender:
                g = selected.body.phenotype.gender
                gender_text = f"Gender: {g.symbol} {g.value.capitalize()}"
            self.creature_gender.setText(gender_text)
            
            self.creature_health.setText(f"Health: {h.health:.0%}")
            self.creature_energy.setText(f"Energy: {h.energy:.0%}")
            self.creature_hunger.setText(f"Hunger: {hunger_pct:.0f}% (Food: {h.nutrition:.0%})")
            self.creature_thirst.setText(f"Thirst: {thirst_pct:.0f}% (Water: {h.hydration:.0%})")
            self.creature_state.setText(f"State: {state_name}")
            self.creature_age.setText(f"Age: {body.lifetime:.0f}s (Gen {selected.generation})")
            
            # Mating stats
            cooldown = h.mating_cooldown if hasattr(h, 'mating_cooldown') else 0
            times_mated = h.times_mated if hasattr(h, 'times_mated') else 0
            offspring = selected.body.offspring_count if hasattr(selected.body, 'offspring_count') else 0
            if cooldown > 0:
                self.creature_mating.setText(f"Mating: {times_mated}x | Cooldown: {cooldown:.0f}s | Offspring: {offspring}")
            else:
                fertility = h.fertility if hasattr(h, 'fertility') else 0
                self.creature_mating.setText(f"Mating: {times_mated}x | Fertility: {fertility:.0%} | Offspring: {offspring}")
            
            # Brain info
            if selected.embodied_brain:
                brain = selected.embodied_brain.brain
                chems = brain.get_chemicals() if hasattr(brain, 'get_chemicals') else {}
                da = chems.get('dopamine', 0)
                cort = chems.get('cortisol', 0)
                # Get dynamic neuron count from dashboard data
                if hasattr(brain, 'get_dashboard_data'):
                    dash = brain.get_dashboard_data()
                    neurons = dash.get('neurons', {}).get('total', 0)
                    born = dash.get('neurons', {}).get('born', 0)
                    died = dash.get('neurons', {}).get('died', 0)
                    self.creature_brain.setText(f"🧠 Neurons: {neurons} (+{born}/-{died}) | DA: {da:.0%} | Cortisol: {cort:.0%}")
                else:
                    self.creature_brain.setText(f"🧠 DA: {da:.0%} | Cortisol: {cort:.0%}")
            else:
                # Show homeostasis cortisol for brainstem-only creatures
                cort = body.homeostasis.cortisol
                self.creature_brain.setText(f"🧠 Brainstem only | Cortisol: {cort:.0%}")
            
            # Held tool info
            if hasattr(selected, 'held_tool') and selected.held_tool is not None:
                tool = selected.held_tool
                tool_name = tool.tool_type.value if hasattr(tool.tool_type, 'value') else str(tool.tool_type)
                self.creature_tool.setText(f"Holding: 🔧 {tool_name.capitalize()}")
            else:
                self.creature_tool.setText("Holding: Nothing")
        else:
            self.creature_name.setText("None selected")
            self.creature_gender.setText("Gender: -")
            self.creature_health.setText("Health: -")
            self.creature_energy.setText("Energy: -")
            self.creature_hunger.setText("Hunger: -")
            self.creature_thirst.setText("Thirst: -")
            self.creature_state.setText("State: -")
            self.creature_age.setText("Age: -")
            self.creature_mating.setText("Mating: -")
            self.creature_brain.setText("Brain: -")
            self.creature_tool.setText("Holding: -")
            
        # Smart update creature list (icons only) to prevent flickering
        if self.creature_list.count() != len(self.creatures):
            self.update_creature_list()
        else:
            for i in range(self.creature_list.count()):
                item = self.creature_list.item(i)
                # Ensure data is valid (PyQt sometimes loses UserRole if not careful, but here it should be fine)
                creature = item.data(Qt.ItemDataRole.UserRole)
                if not creature: 
                    continue
                    
                # Update status icon
                status = "💀" if not creature.body.is_alive() else "🟢"
                current_text = item.text()
                if not current_text.startswith(status):
                     # Rebuild text
                     gender_sym = ""
                     if hasattr(creature.body.phenotype, 'gender') and creature.body.phenotype.gender:
                         gender_sym = creature.body.phenotype.gender.symbol + " "
                     new_text = f"{status} {gender_sym}{creature.name}"
                     item.setText(new_text)
    
    def update_creature_list(self):
        """Update the creature list widget."""
        self.creature_list.clear()
        for creature in self.creatures:
            status = "💀" if not creature.body.is_alive() else "🟢"
            # Show gender symbol
            gender_sym = ""
            if hasattr(creature.body.phenotype, 'gender') and creature.body.phenotype.gender:
                gender_sym = creature.body.phenotype.gender.symbol + " "
            item = QListWidgetItem(f"{status} {gender_sym}{creature.name}")
            item.setData(Qt.ItemDataRole.UserRole, creature)
            self.creature_list.addItem(item)
        
        # Sync creature list with threaded manager
        if self.threaded_manager:
            self.threaded_manager.update_creatures(self.creatures)
    
    def select_creature_from_list(self, item):
        """Select creature from list click."""
        creature = item.data(Qt.ItemDataRole.UserRole)
        self.world_renderer.selected_creature = creature
        self.world_renderer.update()
    
    def rename_creature_from_list(self, item):
        """Rename creature on double-click."""
        from PyQt6.QtWidgets import QInputDialog
        
        creature = item.data(Qt.ItemDataRole.UserRole)
        if not creature:
            return
        
        # Get current name (strip status icons)
        current_name = creature.name
        
        # Show input dialog
        new_name, ok = QInputDialog.getText(
            self, 
            "Rename Creature",
            f"Enter new name for {current_name}:",
            text=current_name
        )
        
        if ok and new_name.strip():
            # Update creature name
            creature.name = new_name.strip()
            
            # Refresh list
            self.update_creature_list()
    
    def feed_selected(self):
        """Feed the selected creature."""
        selected = self.world_renderer.selected_creature
        if selected and selected.body.is_alive():
            selected.body.homeostasis.eat(50)
    
    def breed_selected(self):
        """Attempt to breed selected creature."""
        selected = self.world_renderer.selected_creature
        secondary = self.world_renderer.selected_secondary
        
        if not selected or not selected.body.is_alive():
            return
            
        # 1. Manual Pair (Primary + Secondary)
        if secondary and secondary.body.is_alive() and secondary != selected:
             self._breed_pair(selected, secondary)
             return
        
        # 2. Find nearby same-species creature
        for other in self.creatures:
            if other == selected or not other.body.is_alive():
                continue
            
            dist = np.sqrt(
                (other.body.motor.x - selected.body.motor.x)**2 +
                (other.body.motor.y - selected.body.motor.y)**2
            )
            
            # Check if same species (similar hue)
            hue_diff = abs(other.body.phenotype.hue - selected.body.phenotype.hue)
            if dist < 50 and hue_diff < 0.15:
                # Breed!
                self._breed_pair(selected, other)
                return
    
    def _start_mating(self, parent1: LivingCreature, parent2: LivingCreature):
        """Initiate mating sequence (visuals -> delay -> baby)."""
        # 1. Age Check (Prevent babies making babies)
        # Using puberty_onset_age (usually ~0.1-0.3)
        if parent1.body.homeostasis.age < parent1.body.homeostasis.puberty_onset_age:
            return
        if parent2.body.homeostasis.age < parent2.body.homeostasis.puberty_onset_age:
            return

        # 2. Existing pre-checks (Gender, Cooldown, Awake, Count)
        # We duplicate key checks here to fail fast
        if parent1.body.homeostasis.is_sleeping or parent2.body.homeostasis.is_sleeping:
            return
            
        from brain.creature import Gender
        g1 = parent1.body.phenotype.gender
        g2 = parent2.body.phenotype.gender
        if g1 and g2 and g1 == g2:
            return
            
        if parent1.body.homeostasis.mating_cooldown > 0 or parent2.body.homeostasis.mating_cooldown > 0:
            return

        # 3. Start Sequence
        # Check if already mating
        for pair in self.mating_pairs:
            if pair['p1'] == parent1 or pair['p2'] == parent1 or pair['p1'] == parent2 or pair['p2'] == parent2:
                return

        # Set state to MATING (stops movement)
        if hasattr(parent1, 'behavior_state'):
            from brain.behavior_state import BehaviorState
            parent1.behavior_state.state = BehaviorState.MATING
        if hasattr(parent2, 'behavior_state'):
            from brain.behavior_state import BehaviorState
            parent2.behavior_state.state = BehaviorState.MATING
            
        # Add to tracking
        self.mating_pairs.append({
            'p1': parent1, 
            'p2': parent2, 
            'timer': 2.5, # 2.5 seconds of "love"
            'start_time': self.world.time if self.world else 0
        })
        
        # Play sound (One cute pop or loop?)
        if np.random.random() < 0.7:
             self.sound_manager.play('breed', 0.5)

    def _process_mating_pairs(self, dt: float):
        """Update ongoing mating sequences."""
        if not self.mating_pairs:
            return
            
        for pair in self.mating_pairs[:]: # Copy for removal
            p1 = pair['p1']
            p2 = pair['p2']
            
            # Validation: Are they still close and alive?
            if not p1.body.is_alive() or not p2.body.is_alive():
                self.mating_pairs.remove(pair)
                continue
                
            dist = np.sqrt((p1.body.motor.x - p2.body.motor.x)**2 + (p1.body.motor.y - p2.body.motor.y)**2)
            if dist > 60: # Broke up (moved apart)
                self.mating_pairs.remove(pair)
                continue
                
            # Freeze them (redundant if state is MATING, but safe)
            p1.body.motor.vx = 0
            p2.body.motor.vx = 0
            
            # Update Timer
            pair['timer'] -= dt
            
            # Finish?
            if pair['timer'] <= 0:
                # Spawn Baby!
                self._breed_pair(p1, p2)
                
                # Reset states
                if hasattr(p1, 'behavior_state'):
                    from brain.behavior_state import BehaviorState
                    p1.behavior_state.state = BehaviorState.IDLE
                if hasattr(p2, 'behavior_state'):
                    from brain.behavior_state import BehaviorState
                    p2.behavior_state.state = BehaviorState.IDLE
                
                self.mating_pairs.remove(pair)

    def _breed_pair(self, parent1: LivingCreature, parent2: LivingCreature):
        """
        Attempt to create offspring from two parents with genetic inheritance.
        
        GATING:
        - Both must be awake (not sleeping)
        - Both must have no mating cooldown
        - Max babies per pair: 3
        - Mating success chance: 60%
        - Sets 30s cooldown after mating
        - Visual: Hearts float up from both creatures
        """
        from brain.genetics_helper import create_child_phenotype, create_child_brain_config, blend_values
        
        # === PRE-CHECKS: Can they mate? ===
        
        # 1. Both must be awake
        if parent1.body.homeostasis.is_sleeping or parent2.body.homeostasis.is_sleeping:
            return  # Can't mate while sleeping!
        
        # 2. Must be opposite genders
        from brain.creature import Gender
        g1 = parent1.body.phenotype.gender
        g2 = parent2.body.phenotype.gender
        if g1 is None or g2 is None:
            # If either has no gender assigned, randomly assign one
            if g1 is None:
                parent1.body.phenotype.gender = Gender.random()
                g1 = parent1.body.phenotype.gender
            if g2 is None:
                parent2.body.phenotype.gender = Gender.random()
                g2 = parent2.body.phenotype.gender
        
        if g1 == g2:
            return  # Same gender - can't mate
        
        # 3. Both must have no cooldown
        if parent1.body.homeostasis.mating_cooldown > 0 or parent2.body.homeostasis.mating_cooldown > 0:
            return  # Still in refractory period
        
        # 3. Check if this pair has exceeded max babies
        p1_id = str(parent1.body.id)
        p2_id = str(parent2.body.id)
        pair_key = tuple(sorted([p1_id, p2_id]))
        
        if not hasattr(self, '_pair_offspring_count'):
            self._pair_offspring_count = {}
        
        MAX_BABIES_PER_PAIR = 3
        if self._pair_offspring_count.get(pair_key, 0) >= MAX_BABIES_PER_PAIR:
            return  # This pair has had enough babies
        
        # 4. Success chance (not every mating attempt works)
        MATING_SUCCESS_CHANCE = 0.6
        if np.random.random() > MATING_SUCCESS_CHANCE:
            # Mating attempt failed - still apply short cooldown
            parent1.body.homeostasis.mating_cooldown = 10.0  # 10s cooldown on failure
            parent2.body.homeostasis.mating_cooldown = 10.0
            return
        
        # === MATING BEHAVIOR: Visual feedback ===
        # Set both to MATING state
        parent1.behavior_state.state = BehaviorState.SOCIALIZING  # TODO: Add MATING state
        parent2.behavior_state.state = BehaviorState.SOCIALIZING
        
        # Trigger heart particles (visual feedback)
        self._spawn_heart_particles(parent1.body.motor.x, parent1.body.motor.y - 30)
        self._spawn_heart_particles(parent2.body.motor.x, parent2.body.motor.y - 30)
        
        # === OFFSPRING CREATION ===
        # Calculate litter size based on parents' tendency
        h1 = parent1.body.homeostasis
        h2 = parent2.body.homeostasis
        avg_tendency = (getattr(h1, 'litter_size_tendency', 1.5) + getattr(h2, 'litter_size_tendency', 1.5)) / 2
        
        # Check remaining limit (MAX_BABIES_PER_PAIR = 3)
        import random
        base_litter = int(round(avg_tendency + np.random.normal(0, 0.5)))
        num_babies = max(1, min(3, base_litter))
        
        print(f'[Breeding] Spawning {num_babies} babies (Target: {avg_tendency:.1f})')
        
        for i in range(num_babies):
            p1 = parent1.body.phenotype
            p2 = parent2.body.phenotype
            
            # 1. Phenotype Inheritance
            child_phenotype = create_child_phenotype(p1, p2)
            
            # Assign random gender to child (50/50 chance)
            child_phenotype.gender = Gender.random()
            
            # Spawn near parents blending position
            x = (parent1.body.motor.x + parent2.body.motor.x) / 2
            # Litter spread
            if num_babies > 1:
                x += np.random.uniform(-10, 10)
            y = min(parent1.body.motor.y, parent2.body.motor.y) - 20
            if num_babies > 1:
                y += np.random.uniform(-5, 5)
            body = CreatureBody(phenotype=child_phenotype, x=x, y=y)
            
            # === HORMONAL DIMORPHISM (1) - Gender-specific hormone baselines ===
            if child_phenotype.gender == Gender.MALE:
                body.homeostasis.testosterone_base = 0.7 + np.random.random() * 0.2  # 0.7-0.9
                body.homeostasis.estrogen_base = 0.2 + np.random.random() * 0.2      # 0.2-0.4
            else:  # FEMALE
                body.homeostasis.testosterone_base = 0.2 + np.random.random() * 0.2  # 0.2-0.4
                body.homeostasis.estrogen_base = 0.7 + np.random.random() * 0.2      # 0.7-0.9
            
            # === EPIGENETIC TRAUMA INHERITANCE (2) ===
            # Offspring inherit stress markers from both parents
            h1 = parent1.body.homeostasis
            h2 = parent2.body.homeostasis
            
            # Inherit average of parents' trauma, with some random variation
            inherited_trauma = (h1.trauma_markers + h2.trauma_markers) / 2
            body.homeostasis.inherited_caution = min(1.0, inherited_trauma * 0.1 + np.random.random() * 0.1)
            body.homeostasis.inherited_stress = min(1.0, 
                (h1.starvation_memory + h2.starvation_memory) / 4 +
                (h1.pain_memory + h2.pain_memory) / 4
            )
            
            # If either parent nearly died from starvation, child is more cautious about food
            if h1.starvation_memory > 0.5 or h2.starvation_memory > 0.5:
                body.homeostasis.starvation_memory = max(h1.starvation_memory, h2.starvation_memory) * 0.6
            
            # If either parent suffered extreme pain, child has heightened stress reactivity
            if h1.pain_memory > 0.5 or h2.pain_memory > 0.5:
                body.homeostasis.pain_memory = max(h1.pain_memory, h2.pain_memory) * 0.5
                body.homeostasis.inherited_stress += 0.1
            
            # === SEXUAL STRATEGY INHERITANCE (4) ===
            # Blend parents' reproductive strategies with mutation
            body.homeostasis.reproductive_strategy = blend_values(
                h1.reproductive_strategy, h2.reproductive_strategy, 0.1, 0.0, 1.0
            )
            body.homeostasis.litter_size_tendency = blend_values(
                h1.litter_size_tendency, h2.litter_size_tendency, 0.2, 1.0, 5.0
            )
            body.homeostasis.parental_investment = blend_values(
                h1.parental_investment, h2.parental_investment, 0.1, 0.0, 1.0
            )
            body.homeostasis.parent_bond_strength = blend_values(
                h1.parent_bond_strength, h2.parent_bond_strength, 0.1, 0.0, 1.0
            )
            
            # === ESTRUS CYCLE INHERITANCE (5) ===
            # Females inherit cycle length from mother (with variation)
            if child_phenotype.gender == Gender.FEMALE:
                mother_h = h2 if parent2.body.phenotype.gender == Gender.FEMALE else h1
                body.homeostasis.estrus_cycle_length = mother_h.estrus_cycle_length * (0.8 + np.random.random() * 0.4)
            
            # === NEUROTRANSMITTER RECEPTOR INHERITANCE (3) ===
            body.homeostasis.dopamine_receptor_density = blend_values(
                h1.dopamine_receptor_density, h2.dopamine_receptor_density, 0.1, 0.5, 1.5
            )
            body.homeostasis.serotonin_receptor_density = blend_values(
                h1.serotonin_receptor_density, h2.serotonin_receptor_density, 0.1, 0.5, 1.5
            )
            body.homeostasis.cortisol_receptor_density = blend_values(
                h1.cortisol_receptor_density, h2.cortisol_receptor_density, 0.1, 0.5, 1.5
            )
            body.homeostasis.oxytocin_receptor_density = blend_values(
                h1.oxytocin_receptor_density, h2.oxytocin_receptor_density, 0.1, 0.5, 1.5
            )
            
            # === MATE PREFERENCE INHERITANCE (8) ===
            # Inherit preferences with mutation - runaway sexual selection
            body.homeostasis.preferred_hue = blend_values(
                h1.preferred_hue, h2.preferred_hue, 0.15, 0.0, 1.0  # Higher mutation for preferences
            )
            body.homeostasis.preferred_size = blend_values(
                h1.preferred_size, h2.preferred_size, 0.1, 0.5, 2.0
            )
            body.homeostasis.preference_strength = blend_values(
                h1.preference_strength, h2.preference_strength, 0.1, 0.0, 1.0
            )
            # Display trait slightly mutates - Fisherian runaway
            body.homeostasis.display_trait_value = blend_values(
                h1.display_trait_value, h2.display_trait_value, 0.15, 0.0, 2.0
            )
            
            # === TOOL PREFERENCE INHERITANCE (9) ===
            # Inherit tool preferences from parents (75% chance from same-gender parent)
            same_gender_parent = parent1 if parent1.body.phenotype.gender == child_phenotype.gender else parent2
            body.homeostasis.preferred_tool_type = same_gender_parent.body.homeostasis.preferred_tool_type
            body.homeostasis.tool_specialization = blend_values(
                h1.tool_specialization, h2.tool_specialization, 0.1, 0.0, 1.0
            )
            
            # === INBREEDING TRACKING (12) ===
            import uuid
            body.homeostasis.genetic_lineage_id = str(uuid.uuid4())[:8]
            body.homeostasis.parent_lineage_ids = [
                h1.genetic_lineage_id or str(uuid.uuid4())[:8],
                h2.genetic_lineage_id or str(uuid.uuid4())[:8]
            ]
            # Calculate inbreeding coefficient
            if h1.genetic_lineage_id and h2.genetic_lineage_id:
                # Same lineage = related
                if h1.genetic_lineage_id == h2.genetic_lineage_id:
                    body.homeostasis.inbreeding_coefficient = 0.5  # Siblings
                elif (h1.genetic_lineage_id in h2.parent_lineage_ids or 
                      h2.genetic_lineage_id in h1.parent_lineage_ids):
                    body.homeostasis.inbreeding_coefficient = 0.25  # Parent-child
                elif set(h1.parent_lineage_ids) & set(h2.parent_lineage_ids):
                    body.homeostasis.inbreeding_coefficient = 0.125  # Cousins
                else:
                    body.homeostasis.inbreeding_coefficient = 0.0  # Unrelated
            
            # === BODY SIZE INHERITANCE (14) ===
            body.homeostasis.body_size_mult = blend_values(
                h1.body_size_mult, h2.body_size_mult, 0.1, 0.5, 2.0
            )
            # Size affects phenotype
            child_phenotype.size *= body.homeostasis.body_size_mult
            
            # === LIFETIME HORMONAL ARCS INHERITANCE (7) ===
            body.homeostasis.puberty_onset_age = blend_values(
                h1.puberty_onset_age, h2.puberty_onset_age, 0.05, 0.1, 0.3
            )
            body.homeostasis.elder_onset_age = blend_values(
                h1.elder_onset_age, h2.elder_onset_age, 0.05, 0.5, 0.9
            )
            
            # === GUT-BRAIN AXIS INHERITANCE (5) ===
            # Offspring inherit maternal microbiome (like real birth/nursing)
            mother_h = h2 if parent2.body.phenotype.gender == Gender.FEMALE else h1
            
            # Start with mother's microbiome composition (with some variation)
            body.homeostasis.microbiome_lactobacillus = mother_h.microbiome_lactobacillus * (0.7 + np.random.random() * 0.3)
            body.homeostasis.microbiome_bifidobacteria = mother_h.microbiome_bifidobacteria * (0.7 + np.random.random() * 0.3)
            body.homeostasis.microbiome_enterococcus = mother_h.microbiome_enterococcus * (0.7 + np.random.random() * 0.3)
            body.homeostasis.microbiome_pathogenic = mother_h.microbiome_pathogenic * 0.5  # Less pathogenic (clean start)
            body.homeostasis.microbiome_diversity = mother_h.microbiome_diversity * 0.8  # Diversity takes time to build
            body.homeostasis.gut_health = 0.9  # Babies start with healthy guts
            
            # 2. Instinct Inheritance - kept local for now as it depends on instincts dict
            drive_params = {}
            for key in ['hunger', 'fear', 'curiosity', 'social', 'reproduction', 'fatigue']:
                p1_val = None
                p2_val = None
                
                for k, v in parent1.instincts.instincts.items():
                    if key in str(k).lower():
                        p1_val = v
                        break
                for k, v in parent2.instincts.instincts.items():
                    if key in str(k).lower():
                        p2_val = v
                        break
                
                if p1_val and p2_val:
                    drive_params[key] = blend_values(p1_val.learned_weight, p2_val.learned_weight, 0.1, 0.1, 2.0)
                else:
                    drive_params[key] = 0.5
            
            # === SEXUAL DIMORPHISM BRAIN WIRING (10) ===
            # Gender affects baseline instinct weights
            if child_phenotype.gender == Gender.MALE:
                drive_params['fear'] = max(0.1, drive_params.get('fear', 0.5) - 0.1)  # Males less fearful
                drive_params['curiosity'] = min(0.9, drive_params.get('curiosity', 0.5) + 0.1)  # More exploratory
            else:
                drive_params['social'] = min(0.9, drive_params.get('social', 0.5) + 0.1)  # Females more social
                drive_params['fear'] = min(0.9, drive_params.get('fear', 0.5) + 0.05)  # More cautious
    
            instincts = InstinctSystem(drive_params)
            reward_system = RewardSystem()
            
            # 3. Brain Inheritance (Neuro-Evolution with Structural Memory)
            # ALL babies get a brain - even if parents are brainstem-only
            child_brain = None
            use_neural = True  # Always give babies a brain
            
            # Get actual brain references (could be in .brain or .embodied_brain.brain)
            p1_brain = parent1.brain
            if p1_brain is None and parent1.embodied_brain is not None:
                p1_brain = parent1.embodied_brain.brain
                
            p2_brain = parent2.brain
            if p2_brain is None and parent2.embodied_brain is not None:
                p2_brain = parent2.embodied_brain.brain
            
            # Determine brain scale - inherit from parent if available, otherwise small
            parent_has_brain = (p1_brain and hasattr(p1_brain, 'config')) or \
                              (p2_brain and hasattr(p2_brain, 'config'))
            
            if parent_has_brain:
                # Get parent configs or defaults
                c1 = p1_brain.config if p1_brain and hasattr(p1_brain, 'config') else BrainConfig()
                c2 = p2_brain.config if p2_brain and hasattr(p2_brain, 'config') else BrainConfig()
                
                # Evolve brain hyperparameters
                child_config = create_child_brain_config(c1, c2)
                
                # Instantiate brain with evolved config
                try:
                    child_brain = ThreeSystemBrain(config=child_config)
                    print(f"[Birth] Created brain for child with {child_config.num_columns} columns")
                    
                    # NSM: Inherit neural topology from parents
                    # Get structural snapshot from fitter parent (or random)
                    donor_brain = p1_brain if parent1.body.homeostasis.health > parent2.body.homeostasis.health else p2_brain
                    if donor_brain and hasattr(donor_brain, 'get_structural_snapshot'):
                        try:
                            parent_snapshot = donor_brain.get_structural_snapshot()
                            # Mutation rate scales with generation (older lineages more stable)
                            child_generation = max(parent1.generation, parent2.generation) + 1
                            mutation_rate = max(0.05, 0.2 - 0.01 * child_generation)
                            child_brain.load_structural_snapshot(parent_snapshot, mutation_rate=mutation_rate)
                            print(f"[NSM] Child inherited neural topology from parent")
                        except Exception as e:
                            print(f"[NSM] Structural inheritance failed: {e}")
                            
                except Exception as e:
                    print(f"Error creating brain for child: {e}")
                    import traceback
                    traceback.print_exc()
                    child_brain = None
            
            # Fallback: if no brain created yet, create a fresh one
            if child_brain is None:
                try:
                    from brain import create_brain
                    child_brain = create_brain(scale="small")
                    print(f"[Birth] Created fallback small brain for offspring")
                except Exception as e:
                    print(f"Error creating fallback brain: {e}")
                    child_brain = None
            
            # Wrap in EmbodiedBrain if we have a brain
            child_embodied = None
            if child_brain:
                try:
                    child_embodied = EmbodiedBrain(brain=child_brain, body=body)
                except Exception as e:
                    print(f"Failed to embody child brain: {e}")
    
            # 4. Create Offspring
            child = LivingCreature(
                body=body,
                brain=child_brain,
                instincts=instincts,
                reward_system=reward_system,
                embodied_brain=child_embodied,
                generation=max(parent1.generation, parent2.generation) + 1,
                use_neural_control=use_neural or (child_brain is not None)
            )
            
            # Initialize embodied brain connection if neural
            # Note: child_embodied is already created above with correct arguments
            if child.brain and child.embodied_brain is None:
                 child.embodied_brain = EmbodiedBrain(brain=child.brain, body=child.body)
                 
            # TIER 3: Mental Map Inheritance - pass spatial knowledge to offspring
            if child.mental_map is not None:
                # Inherit from fitter parent's mental map
                donor_parent = parent1 if parent1.body.homeostasis.health > parent2.body.homeostasis.health else parent2
                if donor_parent.mental_map is not None:
                    inherited_map = donor_parent.mental_map.get_inheritable_map(strength=0.4)
                    child.mental_map.merge_from(inherited_map, trust_level=0.6)
                    # Also merge some knowledge from the other parent
                    other_parent = parent2 if donor_parent == parent1 else parent1
                    if other_parent.mental_map is not None:
                        other_map = other_parent.mental_map.get_inheritable_map(strength=0.2)
                        child.mental_map.merge_from(other_map, trust_level=0.3)
            
            # Inherit Lineage
            if parent1.body.homeostasis.genetic_lineage_id:
                child.body.homeostasis.genetic_lineage_id = parent1.body.homeostasis.genetic_lineage_id
                
            self.creatures.append(child)
            self.lineage_system.register_birth(child, parent_ids=[str(parent1.body.id), str(parent2.body.id)])
            parent1.body.offspring_count += 1
            parent2.body.offspring_count += 1
            
            # Track pair offspring count
            self._pair_offspring_count[pair_key] = self._pair_offspring_count.get(pair_key, 0) + 1
            
            # Play breed sound
            self.sound_manager.play('breed', 0.6)
            # Trigger speech
            self._trigger_speech(parent1, "love family baby", mood="excited")
            
        # 5. Apply Reproduction Cost + Cooldown
        MATING_COOLDOWN = 30.0  # 30 seconds before can mate again
        
        # Significant energy investment
        cost = 0.35 * (1.0 + (num_babies - 1) * 0.5)  # Scale for litter
        parent1.body.homeostasis.energy = max(0.1, parent1.body.homeostasis.energy - cost)
        parent2.body.homeostasis.energy = max(0.1, parent2.body.homeostasis.energy - cost)
        
        # Set mating cooldown (refractory period)
        parent1.body.homeostasis.mating_cooldown = MATING_COOLDOWN
        parent2.body.homeostasis.mating_cooldown = MATING_COOLDOWN
        parent1.body.homeostasis.times_mated += 1
        parent2.body.homeostasis.times_mated += 1
        parent1.body.homeostasis.last_mate_id = p2_id
        parent2.body.homeostasis.last_mate_id = p1_id
        
        self.world_renderer.set_creatures(self.creatures)
        self.update_creature_list()
    
    def _spawn_heart_particles(self, x: float, y: float):
        """Spawn floating heart particles at position for mating visual feedback."""
        # Add to particle system if we have one, otherwise just log
        if hasattr(self, 'particles'):
            for i in range(3):
                self.particles.append({
                    'type': 'heart',
                    'x': x + np.random.uniform(-20, 20),
                    'y': y,
                    'vy': -30 - np.random.uniform(0, 20),  # Float upward
                    'life': 2.0,  # 2 seconds
                    'alpha': 1.0
                })
        # If no particle system, the hearts simply won't show (graceful degradation)

    def save_game(self):
        import json
        import os
        from datetime import datetime
        
        if self.world is None:
            return
        
        # Create saves directory
        save_dir = "game_saves"
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"game_{timestamp}.json")
        
        # Serialize world
        world_data = self.world.to_dict()
        
        # Serialize creatures (simplified - no brain state for now)
        creatures_data = []
        for c in self.creatures:
            if c.body.is_alive():
                creatures_data.append({
                    'name': c.name,
                    'generation': c.generation,
                    'x': c.body.motor.x,
                    'y': c.body.motor.y,
                    'phenotype': {
                        'size': c.body.phenotype.size,
                        'hue': c.body.phenotype.hue,
                        'saturation': c.body.phenotype.saturation,
                        'pattern_type': c.body.phenotype.pattern_type,
                        'pattern_density': c.body.phenotype.pattern_density,
                        'max_speed': c.body.phenotype.max_speed,
                        'jump_power': c.body.phenotype.jump_power,
                        'metabolic_rate': c.body.phenotype.metabolic_rate,
                    },
                    'homeostasis': {
                        'energy': c.body.homeostasis.energy,
                        'nutrition': c.body.homeostasis.nutrition,
                        'hydration': c.body.homeostasis.hydration,
                        'health': c.body.homeostasis.health,
                        'fatigue': c.body.homeostasis.fatigue,
                        'sleepiness': c.body.homeostasis.sleepiness,
                    },
                    'stats': {
                        'lifetime': c.body.lifetime,
                        'food_eaten': c.body.food_eaten,
                        'distance': c.body.distance_traveled,
                        'offspring': c.body.offspring_count,
                    },
                    'instincts': c.instincts.to_dict(),
                    'use_neural': c.use_neural_control,
                })
        
        save_data = {
            'version': 1,
            'timestamp': timestamp,
            'world': world_data,
            'creatures': creatures_data,
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Game saved to {filepath}")
    
    def load_game(self):
        """Load a saved game."""
        save_dir = "game_saves"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Game", save_dir, "JSON Files (*.json)"
        )
        
        if not filepath:
            return
        
        # Stop any existing simulation and ensure paused state
        self._stop_threaded_simulation()
        self.paused = True
        self.play_btn.setText("▶️ Play")
        self.world_renderer.paused = True
        
        try:
            with open(filepath, 'r') as f:
                save_data = json.load(f)
            
            # Restore world
            self.world = World.from_dict(save_data['world'])
            self.world_renderer.set_world(self.world)
            
            # Restore creatures
            self.creatures = []
            for c_data in save_data['creatures']:
                phenotype = Phenotype(
                    size=c_data['phenotype']['size'],
                    hue=c_data['phenotype']['hue'],
                    saturation=c_data['phenotype']['saturation'],
                    pattern_type=c_data['phenotype']['pattern_type'],
                    pattern_density=c_data['phenotype']['pattern_density'],
                    max_speed=c_data['phenotype']['max_speed'],
                    jump_power=c_data['phenotype']['jump_power'],
                    metabolic_rate=c_data['phenotype']['metabolic_rate'],
                )
                
                body = CreatureBody(
                    phenotype=phenotype, 
                    x=c_data['x'], 
                    y=c_data['y']
                )
                
                # Restore homeostasis
                h_data = c_data.get('homeostasis', {})
                if h_data:
                    body.homeostasis.energy = h_data.get('energy', 1.0)
                    body.homeostasis.nutrition = h_data.get('nutrition', 1.0)
                    body.homeostasis.hydration = h_data.get('hydration', 1.0)
                    body.homeostasis.health = h_data.get('health', 1.0)
                    body.homeostasis.fatigue = h_data.get('fatigue', 0.0)
                    body.homeostasis.sleepiness = h_data.get('sleepiness', 0.0)
                
                # Restore stats
                s_data = c_data.get('stats', {})
                if s_data:
                    body.lifetime = s_data.get('lifetime', 0)
                    body.food_eaten = s_data.get('food_eaten', 0)
                    body.distance_traveled = s_data.get('distance', 0)
                    body.offspring_count = s_data.get('offspring', 0)

                # Restore Instincts
                inst_data = c_data.get('instincts', {})
                instincts = InstinctSystem.from_dict(inst_data)
                
                reward_system = RewardSystem()
                
                 # Recreate creature
                try:
                    embodied = EmbodiedBrain(brain=None, body=body, brain_scale='micro')
                    creature = LivingCreature(
                        body=body,
                        brain=embodied.brain,
                        instincts=instincts,
                        reward_system=reward_system,
                        embodied_brain=embodied,
                        generation=c_data.get('generation', 0),
                        use_neural_control=c_data.get('use_neural', False),
                        name=c_data.get('name', 'Creature')
                    )
                except Exception:
                     creature = LivingCreature(
                        body=body,
                        brain=None,
                        instincts=instincts,
                        reward_system=reward_system,
                        generation=c_data.get('generation', 0),
                        name=c_data.get('name', 'Creature')
                    )
                
                self.creatures.append(creature)
                
            self.world_renderer.set_creatures(self.creatures)
            self.update_creature_list()
            print(f"Loaded game from {filepath}")
            
        except Exception as e:
            print(f"Failed to load game: {e}")
            import traceback
            traceback.print_exc()

    def save_selected_template(self):
        """Save selected creature as a template."""
        selected = self.world_renderer.selected_creature
        if not selected or not selected.body.is_alive():
            return
            
        save_dir = "creature_templates"
        os.makedirs(save_dir, exist_ok=True)
        
        # Serialize logic (shared with save_game basically)
        c = selected
        data = {
            'name': c.name,
            'phenotype': {
                'size': c.body.phenotype.size,
                'hue': c.body.phenotype.hue,
                'saturation': c.body.phenotype.saturation,
                'pattern_type': c.body.phenotype.pattern_type,
                'pattern_density': c.body.phenotype.pattern_density,
                'max_speed': c.body.phenotype.max_speed,
                'jump_power': c.body.phenotype.jump_power,
                'metabolic_rate': c.body.phenotype.metabolic_rate,
            },
            'instincts': c.instincts.to_dict(),
        }
        
        filename = f"{c.name.replace(' ', '_')}_{int(c.body.phenotype.hue*100)}.json"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Template saved to {filepath}")
        
    def spawn_creature_from_data(self, data, x, y):
        """Spawn a creature from template data at x, y."""
        p_data = data['phenotype']
        phenotype = Phenotype(
            size=p_data['size'],
            hue=p_data['hue'],
            saturation=p_data['saturation'],
            pattern_type=p_data['pattern_type'],
            pattern_density=p_data['pattern_density'],
            max_speed=p_data['max_speed'],
            jump_power=p_data['jump_power'],
            metabolic_rate=p_data['metabolic_rate'],
        )
        
        body = CreatureBody(phenotype=phenotype, x=x, y=y)
        
        # Load instincts
        instinct_data = data.get('instincts', {})
        instincts = InstinctSystem.from_dict(instinct_data) if instinct_data else InstinctSystem()
        
        reward_system = RewardSystem()
        
        # Brain logic
        try:
            embodied = EmbodiedBrain(brain=None, body=body, brain_scale='micro')
            creature = LivingCreature(
                body=body,
                brain=embodied.brain,
                instincts=instincts,
                reward_system=reward_system,
                embodied_brain=embodied,
                generation=0,
                use_neural_control=True,  # ENABLE NEURAL CONTROL
                name=data.get('name', 'TemplateCreature')
            )
        except Exception:
             creature = LivingCreature(
                body=body,
                brain=None,
                instincts=instincts,
                reward_system=reward_system,
                generation=0,
                name=data.get('name', 'TemplateCreature')
            )
            
        self.creatures.append(creature)
        self.world_renderer.set_creatures(self.creatures)
        self.update_creature_list()

    def _check_communication_success(self, speaker: LivingCreature, meaning: str, x: float, y: float):
        """
        Check if nearby creatures respond to speaker's utterance.
        
        ENHANCED: Listeners now form NEURAL associations between:
        - The heard word pattern
        - The observed action (what speaker is doing)
        
        This creates true emergent language - words get meaning from observed context.
        Later, when the listener performs the same action, the word-pattern may activate.
        """
        if not speaker.current_speech or not speaker.procedural_language:
            return
        
        word = speaker.current_speech
        gesture = speaker.current_gesture
        speaker_x = speaker.body.motor.x
        speaker_y = speaker.body.motor.y
        
        # Get speaker's current behavior state for context
        speaker_behavior = speaker.behavior_state.state.name if speaker.behavior_state else 'unknown'
        
        # Check nearby creatures
        communication_range = 300  # Distance within which creatures can hear/see
        listeners_responded = 0
        
        for listener in self.creatures:
            if listener == speaker or not listener.body.is_alive():
                continue
            
            # Check if in range
            dist = np.sqrt((listener.body.motor.x - speaker_x)**2 + 
                          (listener.body.motor.y - speaker_y)**2)
            if dist > communication_range:
                continue
            
            # === REAL OBSERVATION: What can the listener actually SEE? ===
            # The listener doesn't "know" the meaning - they have to INFER it
            # from observable behavior:
            # - Speaker's behavior state (SEEKING_FOOD, SEEKING_WATER, etc.)
            # - What objects are near the speaker (food, water, etc.)
            # - Speaker's body language (gestures)
            
            # Infer meaning from observable behavior state
            observed_action = None
            if 'FOOD' in speaker_behavior or 'EATING' in speaker_behavior:
                observed_action = 'food'
            elif 'WATER' in speaker_behavior or 'DRINKING' in speaker_behavior:
                observed_action = 'water'
            elif 'SLEEP' in speaker_behavior:
                observed_action = 'sleep'
            elif 'FLEE' in speaker_behavior or 'PAIN' in speaker_behavior.upper():
                observed_action = 'danger'
            elif 'SOCIAL' in speaker_behavior or 'MATE' in speaker_behavior:
                observed_action = 'social'
            else:
                # Check what's near the speaker to infer context
                for food in self.world.foods:
                    food_dist = np.sqrt((food.x - speaker_x)**2 + (food.y - speaker_y)**2)
                    if food_dist < 60:  # Speaker is near food
                        observed_action = 'food'
                        break
                
                if observed_action is None:
                    for water in self.world.water_sources:
                        water_dist = np.sqrt((water.x - speaker_x)**2 + (water.y - speaker_y)**2)
                        if water_dist < 60:  # Speaker is near water
                            observed_action = 'water'
                            break
            
            # Listener hears the word - provide OBSERVABLE context
            if listener.procedural_language:
                context = {
                    'speaker_id': str(speaker.body.id),
                    'observable_action': observed_action,  # What listener INFERS from observation
                    'speaker_behavior': speaker_behavior,  # Full behavior state (visible)
                    'speaker_x': speaker_x,
                    'speaker_y': speaker_y,
                }
                understood_meaning = listener.procedural_language.receive_utterance(word, gesture, context)
                
                # === NEURAL ASSOCIATION LEARNING ===
                # The listener's brain forms connections between:
                # 1. The auditory pattern of the word
                # 2. The visual pattern of what the speaker is doing
                # ONLY learn if we could actually observe what they're doing!
                if observed_action and listener.embodied_brain is not None:
                    try:
                        brain = listener.embodied_brain.brain
                        
                        # Encode the word as an auditory pattern
                        word_pattern = self._encode_word_pattern(word)
                        
                        # Encode the OBSERVED action as a pattern
                        action_pattern = self._encode_action_pattern(observed_action, speaker_behavior)
                        
                        # Present both patterns to brain simultaneously
                        # This creates Hebbian association: "neurons that fire together wire together"
                        if hasattr(brain, 'cortex'):
                            # Create combined input that associates word with action
                            combined_input = word_pattern * 0.5 + action_pattern * 0.5
                            
                            # Process through cortex - this strengthens connections
                            # between word-responsive and action-responsive neurons
                            brain.cortex.process(
                                input_data=combined_input,
                                current_time=brain.current_time,
                                learning_enabled=True,
                                modulation={'plasticity_mod': 1.5}  # Enhanced plasticity for learning
                            )
                            
                            # Small reward for successful learning
                            if hasattr(brain, 'learning'):
                                brain.learning.receive_reward(0.05)
                    except Exception:
                        pass  # Don't crash on brain errors
                
                # Check if listener responded appropriately (moving towards same resource)
                if understood_meaning == meaning:
                    # Success! Listener understood
                    listeners_responded += 1
                    
                    # Register with cultural language

                    self.cultural_language.register_communication(
                        str(speaker.body.id), word, meaning, success=True
                    )
        
        # Reinforce speaker's symbol based on how many listeners responded
        if listeners_responded > 0:
            reward = min(1.0, listeners_responded * 0.3)  # Up to 1.0 for 3+ listeners
            speaker.procedural_language.reinforce_communication(word, success=True, reward=reward)
        else:
            # No one responded - weak negative reinforcement
            speaker.procedural_language.reinforce_communication(word, success=False, reward=-0.1)
    
    def _encode_word_pattern(self, word: str) -> np.ndarray:
        """
        Encode a word as a neural activation pattern.
        
        Each word gets a unique pattern based on its characters.
        This simulates auditory cortex activation from hearing the word.
        """
        # Default input dimension for brain
        pattern_dim = 100
        pattern = np.zeros(pattern_dim)
        
        # Hash each character to activate specific neurons
        for i, char in enumerate(word):
            # Multiple neurons per character for robustness
            base_idx = hash(char) % (pattern_dim - 3)
            pattern[base_idx] = 0.8
            pattern[(base_idx + 1) % pattern_dim] = 0.5
            pattern[(base_idx + 7 * (i + 1)) % pattern_dim] = 0.6
        
        # Normalize
        if np.max(pattern) > 0:
            pattern = pattern / np.max(pattern)
        
        return pattern
    
    def _encode_action_pattern(self, meaning: str, behavior: str) -> np.ndarray:
        """
        Encode an observed action as a neural activation pattern.
        
        Different actions (drinking, eating, fleeing) activate different
        regions of the pattern, simulating visual cortex response to observed behavior.
        """
        pattern_dim = 100
        pattern = np.zeros(pattern_dim)
        
        # Map meanings to different regions of the neural space
        action_regions = {
            'food': (0, 20),
            'water': (20, 40),
            'danger': (40, 60),
            'sleep': (60, 75),
            'social': (75, 90),
        }
        
        # Activate region based on meaning
        region = action_regions.get(meaning, (90, 100))
        start, end = region
        pattern[start:end] = 0.7 + 0.3 * np.random.random(end - start)
        
        # Behavior state adds additional activation
        behavior_hash = hash(behavior) % 30
        pattern[behavior_hash:behavior_hash + 5] += 0.3
        
        # Normalize
        pattern = np.clip(pattern, 0, 1)
        if np.max(pattern) > 0:
            pattern = pattern / np.max(pattern)
        
        return pattern
    
    def update_visual_config(self, config):
        """Update visual configuration from settings."""
        if hasattr(self, 'world_renderer'):
            self.world_renderer.visual_config = config
            self.world_renderer.sprite_cache.clear()
            self.world_renderer.world_object_images.clear() # Clear object cache too
            self.world_renderer.background_images.clear()   # Clear bg cache
            self.world_renderer._load_background()
            self.world_renderer.update()


