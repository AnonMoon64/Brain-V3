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
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QGridLayout, QScrollArea, QSlider, QSpinBox,
    QListWidget, QListWidgetItem, QSplitter, QFrame, QComboBox,
    QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, QRect, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap, QMouseEvent

import sys
import json
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain.world import World, TileType, FoodSource
from brain.creature import CreatureBody, Phenotype, Homeostasis
from brain.instincts import InstinctSystem, RewardSystem
from brain.embodiment import EmbodiedBrain
from brain.brainstem import Brainstem
from brain.behavior_state import BehaviorStateMachine, BehaviorState
from brain.movement_controller import MovementController, MovementGoal

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
    behavior_state: BehaviorStateMachine = None  # KEY: State machine for stable behavior
    movement: MovementController = None  # NEW: High-level movement controller
    embodied_brain: Any = None  # EmbodiedBrain for full neural control
    last_step_data: Dict = None # Data from last step for inspection
    name: str = "Creature"
    generation: int = 0
    use_neural_control: bool = False  # Whether to use brain for behavior
    
    def __post_init__(self):
        if not self.name or self.name == "Creature":
            self.name = f"Creature_{self.body.id % 10000:04d}"
        # Every creature needs a brainstem - it's not optional for survival
        if self.brainstem is None:
            self.brainstem = Brainstem()
        # Every creature needs a state machine for stable behavior
        if self.behavior_state is None:
            self.behavior_state = BehaviorStateMachine()
        # Every creature needs a movement controller
        if self.movement is None:
            self.movement = MovementController()



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
        
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        
        # Camera
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 1.0
        
        # Background image
        self.background_image = None
        self._load_background()
        
        # Visual configuration for creatures (from settings tab)
        self.visual_config: Optional[CreatureVisualConfig] = None if not _HAS_SETTINGS else CreatureVisualConfig()
        self.sprite_cache: Dict[str, QPixmap] = {}  # Cache for loaded/tinted sprites
        self.world_object_images: Dict[str, List[QPixmap]] = {}  # Cache for world object images
        self.world_object_frame: int = 0  # Animation frame counter
        
        # God Mode State
        self.current_tool = ToolType.SELECT
        self.current_template = None # Dict containing creature data
        self.is_dragging = False
        self.last_mouse_pos = None
        
    def _get_world_object_images(self, obj_type: str) -> List[QPixmap]:
        """Get cached images for a world object type."""
        if obj_type not in self.world_object_images:
            self.world_object_images[obj_type] = []
            if self.visual_config and _HAS_SETTINGS:
                config = self.visual_config.get_world_object(obj_type)
                if config and config.image_paths:
                    for path in config.image_paths:
                        try:
                            pixmap = QPixmap(path)
                            if not pixmap.isNull():
                                self.world_object_images[obj_type].append(pixmap)
                        except Exception:
                            pass
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
        """Load background image from images folder."""
        try:
            import os
            # Try to find the images folder
            base_path = Path(__file__).parent.parent
            bg_path = base_path / "images" / "BackgroundForest.png"
            if bg_path.exists():
                self.background_image = QPixmap(str(bg_path))
            else:
                print(f"Background image not found at {bg_path}")
        except Exception as e:
            print(f"Failed to load background: {e}")
        
    def set_world(self, world: World):
        """Set the world to render."""
        self.world = world
        self.update()
    
    def set_creatures(self, creatures: List[LivingCreature]):
        """Set creatures to render."""
        self.creatures = creatures
        self.update()
    
    def paintEvent(self, event):
        """Render the world and creatures."""
        # Increment animation frame for world objects
        self.world_object_frame += 1
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background - use image if available
        if self.background_image and not self.background_image.isNull():
            # Scale and draw background to fill the widget
            scaled_bg = self.background_image.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding
            )
            # Center the background
            x_offset = (self.width() - scaled_bg.width()) // 2
            y_offset = (self.height() - scaled_bg.height()) // 2
            painter.drawPixmap(x_offset, y_offset, scaled_bg)
        else:
            painter.fillRect(self.rect(), QColor(30, 30, 40))
        
        if self.world is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                           "No world loaded\nClick 'New World' to create one")
            painter.end()
            return
        
        # Scale to fit
        scale_x = self.width() / self.world.width
        scale_y = self.height() / self.world.height
        scale = min(scale_x, scale_y) * self.zoom
        
        offset_x = (self.width() - self.world.width * scale) / 2 - self.camera_x
        offset_y = (self.height() - self.world.height * scale) / 2 - self.camera_y
        
        # Draw tiles
        tile_size = self.world.tile_size * scale
        
        # Tile colors based on type and lighting
        light = self.world.light_level
        
        # If we have a background image, only draw water/hazards/shelter on top
        has_bg = self.background_image and not self.background_image.isNull()
        
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
        for ty in range(self.world.tiles_y):
            for tx in range(self.world.tiles_x):
                # Skip if already drawn with stretched image
                if (tx, ty) in drawn_tiles:
                    continue
                    
                tile = self.world.tiles[ty, tx]
                
                # Skip empty and ground tiles if using background image
                if has_bg and tile in (TileType.EMPTY.value, TileType.GROUND.value):
                    continue
                
                x = int(offset_x + tx * tile_size)
                y = int(offset_y + ty * tile_size)
                ts = int(tile_size) + 1
                
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
                elif tile == TileType.HAZARD.value:
                    # Hazards glow
                    color = QColor(255, int(80 + 40 * np.sin(self.world.time * 3)), 34, 200)
                elif tile == TileType.SHELTER.value:
                    color = QColor(int(60 * light + 16), int(140 * light + 35), int(60 * light + 20), 150)
                else:
                    color = QColor(128, 128, 128)
                
                painter.fillRect(x, y, ts, ts, color)
        
        # Draw food sources
        food_images = self._get_world_object_images(WorldObjectType.FOOD) if _HAS_SETTINGS else []
        food_config = self._get_world_object_config(WorldObjectType.FOOD) if _HAS_SETTINGS else None
        food_scale = food_config.scale if food_config else 1.0
        
        for food in self.world.food_sources:
            x = int(offset_x + food.x * scale)
            y = int(offset_y + food.y * scale)
            base_size = int(4 + 6 * food.remaining) * scale
            size = int(base_size * food_scale)
            
            # Try to use food image
            if food_images:
                anim_speed = food_config.animation_speed if food_config else 0.2
                frame_idx = (self.world_object_frame // max(1, int(anim_speed * 30))) % len(food_images)
                img = food_images[frame_idx]
                scaled = img.scaled(int(size), int(size), Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
                painter.drawPixmap(int(x - scaled.width()/2), int(y - scaled.height()/2), scaled)
            else:
                # Fallback to colored ellipse
                if food.type == "plant":
                    color = QColor(139, 195, 74)
                else:
                    color = QColor(244, 67, 54)
                
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(0, 0, 0, 100)))
                painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))
        
        # Draw creatures
        for creature in self.creatures:
            self._draw_creature(painter, creature, scale, offset_x, offset_y)
        
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
        
        painter.end()
    
    def _draw_creature(self, painter: QPainter, creature: LivingCreature, 
                       scale: float, offset_x: float, offset_y: float):
        """Draw a single creature (1.5x size for visibility)."""
        body = creature.body
        pheno = body.phenotype
        
        # 1.5x SIZE MULTIPLIER for visibility (reduced from 3x)
        SIZE_MULT = 1.5
        
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
            if creature == self.selected_creature:
                painter.setPen(QPen(QColor(255, 255, 0), 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(x - w//2 - 2, y - h//2 - 2, w + 4, h + 4)
        else:
            # Fall back to procedural drawing
            # Body
            painter.setBrush(QBrush(color))
            outline_color = QColor(255, 255, 0) if creature == self.selected_creature else QColor(0, 0, 0)
            painter.setPen(QPen(outline_color, 3 if creature == self.selected_creature else 2))
            
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
                for _ in range(int(5 * pheno.pattern_density)):
                    sx = x + np.random.randint(-w//3, w//3)
                    sy = y + np.random.randint(-h//3, h//3)
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
        
        # Check if we have a full body sprite first
        # Check ALL age stages since user might only configure one stage
        full_body_sprite = None
        for check_stage in AgeStage.ALL_STAGES:
            stage_idx = AgeStage.ALL_STAGES.index(check_stage)
            test_sprite = self.visual_config.get_sprite(BodyPart.FULL_BODY, stage_idx * 0.25)
            if test_sprite and test_sprite.image_paths:
                full_body_sprite = test_sprite
                break
        
        if full_body_sprite and full_body_sprite.image_paths:
            # Use full body sprite - get variant based on DNA
            image_path = full_body_sprite.get_image_for_dna(dna_value)
            if image_path:
                self._draw_single_sprite(painter, creature, full_body_sprite, image_path,
                                         x, y, w, h, scale, size_mult, pheno)
                return True
        
        # Otherwise, draw individual parts
        # Check if we have any sprites configured (check all age stages)
        has_sprites = False
        for part in BodyPart.INDIVIDUAL_PARTS:
            for check_stage in AgeStage.ALL_STAGES:
                stage_idx = AgeStage.ALL_STAGES.index(check_stage)
                sprite_config = self.visual_config.get_sprite(part, stage_idx * 0.25)
                if sprite_config and sprite_config.image_paths:
                    has_sprites = True
                    break
            if has_sprites:
                break
        
        if not has_sprites:
            return False
        
        # Collect all sprites with z-order
        sprites_to_draw = []
        
        for part in BodyPart.INDIVIDUAL_PARTS:
            # Check all stages for this part
            sprite_config = None
            for check_stage in AgeStage.ALL_STAGES:
                stage_idx = AgeStage.ALL_STAGES.index(check_stage)
                test_sprite = self.visual_config.get_sprite(part, stage_idx * 0.25)
                if test_sprite and test_sprite.image_paths:
                    sprite_config = test_sprite
                    break
            
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
        sprite_y = y + int(sprite_config.offset_y * scale) - sprite_h // 2
        
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
            self._select_creature_at(world_x, world_y)
            
        elif self.current_tool == ToolType.DRAG:
            # Try to grab a creature
            self._select_creature_at(world_x, world_y)
            if self.selected_creature:
                self.is_dragging = True
                
        elif self.current_tool == ToolType.SPAWN_FOOD:
            self.world.food_sources.append(FoodSource(
                x=world_x, y=world_y,
                nutrition=0.5, type="plant" if np.random.random() > 0.3 else "meat"
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
                self._paint_tile(world_x, world_y, TileType.GROUND.value) # Erase wall -> Ground
            else:
                 self._paint_tile(world_x, world_y, TileType.EMPTY.value) # Paint wall -> Empty
                 
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
                 val = TileType.GROUND.value if is_right else TileType.EMPTY.value
                 self._paint_tile(world_x, world_y, val)
             
        self.update()
        
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        self.is_dragging = False

    def _to_world_coords(self, pos) -> Tuple[float, float]:
        scale_x = self.width() / self.world.width
        scale_y = self.height() / self.world.height
        scale = min(scale_x, scale_y) * self.zoom
        
        offset_x = (self.width() - self.world.width * scale) / 2 - self.camera_x
        offset_y = (self.height() - self.world.height * scale) / 2 - self.camera_y
        
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
# GAME TAB
# =============================================================================

class GameTab(QWidget):
    """Main game tab with world simulation."""
    
    def __init__(self, brain=None):
        super().__init__()
        self.template_brain = brain  # Used as template for new creatures
        
        self.world: Optional[World] = None
        self.creatures: List[LivingCreature] = []
        self.paused = True
        self.simulation_speed = 1.0
        
        self.setup_ui()
        
        # Simulation timer
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.simulation_step)
        self.sim_timer.start(33)  # ~30 FPS
    
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
        self.creature_health = QLabel("Health: -")
        self.creature_energy = QLabel("Energy: -")
        self.creature_hunger = QLabel("Hunger: -")
        self.creature_thirst = QLabel("Thirst: -")
        self.creature_state = QLabel("State: -")
        self.creature_age = QLabel("Age: -")
        
        creature_layout.addWidget(self.creature_name)
        creature_layout.addWidget(self.creature_health)
        creature_layout.addWidget(self.creature_energy)
        creature_layout.addWidget(self.creature_hunger)
        creature_layout.addWidget(self.creature_thirst)
        creature_layout.addWidget(self.creature_state)
        creature_layout.addWidget(self.creature_age)
        
        # Brain info
        self.creature_brain = QLabel("Brain: -")
        self.creature_brain.setWordWrap(True)
        creature_layout.addWidget(self.creature_brain)
        
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
        list_group = QGroupBox("Creatures")
        list_layout = QVBoxLayout(list_group)
        
        self.creature_list = QListWidget()
        self.creature_list.itemClicked.connect(self.select_creature_from_list)
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
        
    def create_new_world(self):
        """Create a new world."""
        self.world = World(width=800, height=400)
        self.world_renderer.set_world(self.world)
        self.creatures = []
        self.world_renderer.set_creatures(self.creatures)
        self.update_creature_list()
    
    def toggle_pause(self):
        """Toggle simulation pause."""
        self.paused = not self.paused
        self.play_btn.setText("⏸️ Pause" if not self.paused else "▶️ Play")
    
    def update_speed(self, value):
        """Update simulation speed."""
        self.simulation_speed = value / 10.0
        self.speed_label.setText(f"{self.simulation_speed:.1f}x")
    
    def spawn_creature(self):
        """Spawn a single creature at random location."""
        if self.world is None:
            self.create_new_world()
        
        # Find a safe spawn location (on ground, not water/hazard)
        x, y = self._find_safe_spawn_location()
        
        # Create random phenotype
        phenotype = Phenotype(
            size=0.7 + np.random.random() * 0.6,
            hue=np.random.random(),
            saturation=0.5 + np.random.random() * 0.5,
            pattern_type=np.random.choice(["solid", "stripes", "spots", "gradient"]),
            pattern_density=np.random.random(),
            max_speed=2.0 + np.random.random() * 3,
            jump_power=6 + np.random.random() * 6,
            metabolic_rate=0.8 + np.random.random() * 0.4,
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
    
    def spawn_creature_with_brain(self):
        """Spawn a creature with a neural brain (EmbodiedBrain)."""
        if self.world is None:
            self.create_new_world()
        
        # Find safe spawn location
        x, y = self._find_safe_spawn_location()
        
        # Create random phenotype
        phenotype = Phenotype(
            size=0.7 + np.random.random() * 0.6,
            hue=np.random.random(),
            saturation=0.5 + np.random.random() * 0.5,
            pattern_type=np.random.choice(["solid", "stripes", "spots", "gradient"]),
            pattern_density=np.random.random(),
            max_speed=2.0 + np.random.random() * 3,
            jump_power=6 + np.random.random() * 6,
            metabolic_rate=0.8 + np.random.random() * 0.4,
        )
        
        body = CreatureBody(phenotype=phenotype, x=x, y=y)
        
        # Create EmbodiedBrain (brain + body wiring)
        try:
            embodied = EmbodiedBrain(brain=None, body=body, brain_scale='micro')
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
        self.world_renderer.set_creatures(self.creatures)
        self.update_creature_list()
        print(f"Spawned {creature.name} with state machine control")

    def simulation_step(self):
        """Run one simulation step."""
        if self.paused or self.world is None:
            return
        
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
        
        # Update world time (only once per frame)
        self.world.update(total_dt)
        
        # Redraw
        self.world_renderer.update()
    
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
                continue
            
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
            
            # === HIGH-LEVEL AI DECISION ===
            # Find nearest resources
            nearest_food = self._find_nearest_food(cx, cy)
            nearest_water = self._find_nearest_water(cx, cy)
            nearest_hazard = self._find_nearest_hazard(cx, cy)
            
            # Priority-based goal selection (simple and clear)
            # 1. FLEE if in pain (from fire)
            if pain > 0.1:
                if nearest_hazard:
                    creature.movement.set_goal_flee_from(nearest_hazard[0], nearest_hazard[1])
                else:
                    # Just run in current direction
                    creature.movement.set_goal_flee_from(cx, cy)
            
            # 2. SLEEP - Check if already sleeping OR need to sleep
            elif creature.behavior_state.state == BehaviorState.SLEEPING or creature.body.homeostasis.energy < 0.2:
                # Start or continue sleeping
                creature.movement.set_goal_stay()  # Stop moving
                creature.behavior_state.state = BehaviorState.SLEEPING
                
                # Use the homeostasis sleep system for realistic metabolism
                if not creature.body.homeostasis.is_sleeping:
                    creature.body.homeostasis.start_sleep()
                
                # While sleeping: slower metabolism = less hunger/thirst increase
                # Energy recovers, but hunger/thirst frozen (body conserves)
                creature.body.homeostasis.energy = min(
                    1.0, creature.body.homeostasis.energy + 0.015 * dt
                )
                # Reduce fatigue faster while sleeping
                creature.body.homeostasis.fatigue = max(
                    0, creature.body.homeostasis.fatigue - 0.01 * dt
                )
                
                # Wake up when energy is restored AND rested
                if creature.body.homeostasis.energy > 0.6 and creature.body.homeostasis.fatigue < 0.3:
                    creature.body.homeostasis.wake_up()
                    creature.behavior_state.state = BehaviorState.IDLE
            
            # 3. SEEK FOOD if hungry (NOT sleeping)
            elif hunger > 0.3 and nearest_food:
                creature.movement.set_goal_go_to(nearest_food[0], nearest_food[1])
                # ALWAYS try to eat if near food (don't wait for exact arrival)
                self._try_eat_food(creature)
            
            # 4. SEEK WATER if thirsty (NOT sleeping)
            elif thirst > 0.3 and nearest_water:
                creature.movement.set_goal_go_to(nearest_water[0], nearest_water[1])
                # ALWAYS try to drink if near water
                self._try_drink_water(creature)
            
            # 5. WANDER otherwise
            else:
                creature.movement.set_goal_wander()
            
            # === MOVEMENT CONTROLLER EXECUTES GOAL ===
            movement_result = creature.movement.update(
                dt, cx, cy, on_ground, in_water, self.world
            )
            
            # Apply movement directly to creature
            creature.body.motor.vx = movement_result['vx']
            if movement_result['vy'] is not None:
                creature.body.motor.vy = movement_result['vy']
            if movement_result['jump'] and on_ground:
                creature.body.motor.vy = -creature.movement.jump_power
            
            # Water no longer blocks or causes drowning - it's just for drinking
            # (Removed swim-up logic)
            
            # Update facing direction
            creature.body.motor.facing_right = creature.movement.facing_right
            
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
            creature.body.homeostasis.update(dt, creature.body.phenotype.metabolic_rate, 
                                             ambient_temp, activity)
            
            # Check hazards
            is_hazard, damage, hazard_type = self.world.is_hazard(new_x, new_y + 5)
            if is_hazard:
                creature.body.homeostasis.apply_damage(damage * dt * 0.1)
            
            # Update lifetime
            creature.body.lifetime += dt
            
            # === BRAIN LEARNING (runs in background, doesn't control movement) ===
            # This is where the neural network learns from experience
            # Movement is controlled by the high-level controller above
            # But the brain learns associations: fire → pain → cortisol → weaken synapses
            if creature.embodied_brain is not None:
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
                    
                    # Give reward signal for eating
                    if creature.body.food_eaten > 0:
                        creature.embodied_brain.brain.receive_reward(0.5)
                        
                    if brain_result:
                         # Store for inspector
                         creature.embodied_brain.last_step_data = {
                             'sensory_input': brain_input,
                             'brain_output': brain_result.get('reservoir_output', np.zeros(64)), # output from process_raw is dict
                         }
                        
                except Exception as e:
                    pass  # Brain errors shouldn't crash the game
        
        # Mark death time when creature dies (using world time)
        current_time = self.world.time if self.world else 0
        for c in self.creatures:
            if not c.body.is_alive() and not hasattr(c, '_death_time'):
                c._death_time = current_time
        
        # Remove dead creatures after 3 seconds (for death animation)
        # Keep creatures that are alive OR have been dead less than 3 seconds
        self.creatures = [c for c in self.creatures 
                         if c.body.is_alive() or (current_time - getattr(c, '_death_time', current_time)) < 3]
    
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
        """Find nearest food. Returns (x, y, distance) or None."""
        # Use Quadtree with large radius
        nearest_food_list = self.world.find_food_nearby(cx, cy, 600)
        
        nearest = None
        nearest_dist = float('inf')
        
        for food in nearest_food_list:
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
        """Find nearest water tile. Returns (x, y, distance) or None."""
        tile_size = self.world.tile_size
        best = None
        best_dist = float('inf')
        
        for dx in range(-10, 11):
            for dy in range(-6, 7):
                check_x = int(cx / tile_size) + dx
                check_y = int(cy / tile_size) + dy
                if 0 <= check_x < self.world.tiles_x and 0 <= check_y < self.world.tiles_y:
                    if self.world.tiles[check_y, check_x] == 2:  # WATER
                        # Go to CENTER of water tile
                        water_x = check_x * tile_size + tile_size // 2
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
        """Attempt to eat nearby food."""
        cx, cy = creature.body.motor.x, creature.body.motor.y
        
        for food in self.world.food_sources[:]:  # Copy list for safe removal
            dx = food.x - cx
            dy = food.y - cy
            dist = np.sqrt(dx**2 + dy**2)
            
            # Large eating range - within 80 pixels total distance
            if dist < 80:
                # Eat the food
                nutrition = min(food.remaining, 0.3)  # Bite size
                food.remaining -= nutrition
                
                # Apply to creature (clamp to 0-1 range)
                food_value = nutrition * food.nutrition
                creature.body.homeostasis.nutrition = min(
                    1.0, creature.body.homeostasis.nutrition + food_value
                )
                creature.body.homeostasis.energy = min(
                    1.0, creature.body.homeostasis.energy + food_value * 0.3
                )
                creature.body.food_eaten += 1
                
                # Reward brainstem (dopamine learning)
                reward = nutrition * food.nutrition
                creature.brainstem.receive_food_reward(reward)
                
                # Also reward instincts
                creature.instincts.apply_reward(reward)
                
                # Remove depleted food
                if food.remaining <= 0.01:
                    self.world.food_sources.remove(food)
                
                break  # Only eat one food per frame
    
    def _try_drink_water(self, creature: LivingCreature):
        """Attempt to drink if near water."""
        cx, cy = creature.body.motor.x, creature.body.motor.y
        
        # Check if creature is near any water tile (check many points around creature)
        near_water = False
        for dx in range(-60, 61, 15):  # Check in 15-pixel increments
            for dy in range(-60, 61, 15):
                if self.world.is_water(cx + dx, cy + dy):
                    near_water = True
                    break
            if near_water:
                break
        
        if near_water:
            # Drink water - restore hydration
            drink_amount = 0.2  # Per drink action (faster drinking)
            creature.body.homeostasis.hydration = min(
                1.0, creature.body.homeostasis.hydration + drink_amount
            )
            
            # Reward brainstem (dopamine learning)
            if creature.brainstem:
                creature.brainstem.receive_water_reward(drink_amount)
    
    def update_info(self):
        """Update info panel."""
        if self.world is None:
            return
        
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
            self.creature_health.setText(f"Health: {h.health:.0%}")
            self.creature_energy.setText(f"Energy: {h.energy:.0%}")
            self.creature_hunger.setText(f"Hunger: {hunger_pct:.0f}% (Food: {h.nutrition:.0%})")
            self.creature_thirst.setText(f"Thirst: {thirst_pct:.0f}% (Water: {h.hydration:.0%})")
            self.creature_state.setText(f"State: {state_name}")
            self.creature_age.setText(f"Age: {body.lifetime:.0f}s (Gen {selected.generation})")
            
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
                self.creature_brain.setText("🧠 No brain")
        else:
            self.creature_name.setText("None selected")
            self.creature_health.setText("Health: -")
            self.creature_energy.setText("Energy: -")
            self.creature_hunger.setText("Hunger: -")
            self.creature_thirst.setText("Thirst: -")
            self.creature_state.setText("State: -")
            self.creature_age.setText("Age: -")
            self.creature_brain.setText("Brain: -")
    
    def update_creature_list(self):
        """Update the creature list widget."""
        self.creature_list.clear()
        for creature in self.creatures:
            status = "💀" if not creature.body.is_alive() else "🟢"
            item = QListWidgetItem(f"{status} {creature.name}")
            item.setData(Qt.ItemDataRole.UserRole, creature)
            self.creature_list.addItem(item)
    
    def select_creature_from_list(self, item):
        """Select creature from list click."""
        creature = item.data(Qt.ItemDataRole.UserRole)
        self.world_renderer.selected_creature = creature
        self.world_renderer.update()
    
    def feed_selected(self):
        """Feed the selected creature."""
        selected = self.world_renderer.selected_creature
        if selected and selected.body.is_alive():
            selected.body.homeostasis.eat(50)
    
    def breed_selected(self):
        """Attempt to breed selected creature."""
        selected = self.world_renderer.selected_creature
        if not selected or not selected.body.is_alive():
            return
        
        # Find nearby same-species creature
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
    
    def _breed_pair(self, parent1: LivingCreature, parent2: LivingCreature):
        """Create offspring from two parents."""
        p1 = parent1.body.phenotype
        p2 = parent2.body.phenotype
        
        # Blend phenotypes with mutation
        def blend(a, b, mutation=0.1):
            base = (a + b) / 2
            return np.clip(base + np.random.normal(0, mutation), 0, 1)
        
        child_phenotype = Phenotype(
            size=blend(p1.size, p2.size, 0.1),
            hue=blend(p1.hue, p2.hue, 0.05),
            saturation=blend(p1.saturation, p2.saturation, 0.1),
            pattern_type=np.random.choice([p1.pattern_type, p2.pattern_type]),
            pattern_density=blend(p1.pattern_density, p2.pattern_density),
            max_speed=blend(p1.max_speed / 5, p2.max_speed / 5) * 5,
            jump_power=blend(p1.jump_power / 12, p2.jump_power / 12) * 12,
            metabolic_rate=blend(p1.metabolic_rate, p2.metabolic_rate, 0.05),
        )
        
        # Spawn near parents
        x = (parent1.body.motor.x + parent2.body.motor.x) / 2
        y = min(parent1.body.motor.y, parent2.body.motor.y) - 20
        
        body = CreatureBody(phenotype=child_phenotype, x=x, y=y)
        
        # Blend instinct strengths
        drive_params = {}
        for key in ['hunger', 'fear', 'curiosity', 'social']:
            p1_val = parent1.instincts.instincts.get(
                next((k for k in parent1.instincts.instincts.keys() if key in k.value.lower()), None),
                None
            )
            p2_val = parent2.instincts.instincts.get(
                next((k for k in parent2.instincts.instincts.keys() if key in k.value.lower()), None),
                None
            )
            if p1_val and p2_val:
                drive_params[key] = blend(p1_val.learned_weight, p2_val.learned_weight, 0.1)
            else:
                drive_params[key] = 0.5
        
        instincts = InstinctSystem(drive_params)
        reward_system = RewardSystem()
        
        child = LivingCreature(
            body=body,
            brain=None,
            instincts=instincts,
            reward_system=reward_system,
            generation=max(parent1.generation, parent2.generation) + 1
        )
        
        self.creatures.append(child)
        parent1.body.offspring_count += 1
        parent2.body.offspring_count += 1
        
        self.world_renderer.set_creatures(self.creatures)
        self.update_creature_list()

    def save_game(self):
        """Save the current game state."""
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
                use_neural_control=False,
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


