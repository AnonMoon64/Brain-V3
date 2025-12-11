"""
Settings Tab - Customizable Body Parts and Visual Configuration

This tab provides:
- Background image selection
- Body part sprite configuration linked to DNA
- DNA-driven hue/saturation/brightness on sprites
- Age stage variants (baby, adult, elder)
- Body part mapping to DNA genes
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QGridLayout, QScrollArea, QSlider, QSpinBox,
    QListWidget, QListWidgetItem, QSplitter, QFrame, QComboBox,
    QFileDialog, QLineEdit, QCheckBox, QTabWidget, QColorDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter, QIcon


# =============================================================================
# BODY PART DEFINITIONS
# =============================================================================

class BodyPart:
    """Definition of a body part type."""
    FULL_BODY = "full_body"  # Single sprite for entire creature
    HEAD = "head"
    TORSO = "torso"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    LEFT_LEG = "left_leg"
    RIGHT_LEG = "right_leg"
    TAIL = "tail"
    EYES = "eyes"
    MOUTH = "mouth"
    
    # Full body first - if set, individual parts are ignored
    ALL_PARTS = [FULL_BODY, HEAD, TORSO, LEFT_ARM, RIGHT_ARM, LEFT_LEG, RIGHT_LEG, TAIL, EYES, MOUTH]
    
    # Parts only (for individual rendering)
    INDIVIDUAL_PARTS = [HEAD, TORSO, LEFT_ARM, RIGHT_ARM, LEFT_LEG, RIGHT_LEG, TAIL, EYES, MOUTH]
    
    # Which DNA genes affect which body parts
    DNA_MAPPINGS = {
        FULL_BODY: ["color_hue", "color_saturation", "body_size", "pattern_type"],
        HEAD: ["color_hue", "color_saturation", "pattern_type"],
        TORSO: ["body_size", "color_hue", "color_saturation", "pattern_type"],
        LEFT_ARM: ["limb_count", "muscle_strength", "color_hue"],
        RIGHT_ARM: ["limb_count", "muscle_strength", "color_hue"],
        LEFT_LEG: ["limb_count", "movement_speed", "color_hue"],
        RIGHT_LEG: ["limb_count", "movement_speed", "color_hue"],
        TAIL: ["color_hue", "pattern_type"],
        EYES: ["vision_range", "color_hue"],
        MOUTH: ["color_hue"],
    }


class AgeStage:
    """Age stages for body part variants."""
    BABY = "baby"       # 0-20% of lifespan
    JUVENILE = "juvenile"  # 20-40%
    ADULT = "adult"     # 40-70%
    ELDER = "elder"     # 70-100%
    
    ALL_STAGES = [BABY, JUVENILE, ADULT, ELDER]
    
    @staticmethod
    def from_age(age: float) -> str:
        """Get age stage from normalized age (0-1)."""
        if age < 0.2:
            return AgeStage.BABY
        elif age < 0.4:
            return AgeStage.JUVENILE
        elif age < 0.7:
            return AgeStage.ADULT
        else:
            return AgeStage.ELDER


@dataclass
class BodyPartSprite:
    """A sprite configuration for a body part with multiple image variants."""
    part: str                       # Which body part
    stage: str                      # Age stage
    image_paths: List[str] = field(default_factory=list)  # List of image file paths (variants)
    dna_gene: str = "color_hue"     # Which DNA gene selects variant and affects coloring
    offset_x: int = 0               # Offset from body center
    offset_y: int = 0
    scale: float = 1.0              # Scale multiplier
    z_order: int = 0                # Draw order (higher = on top)
    apply_hue: bool = True          # Apply DNA hue to image
    apply_saturation: bool = True   # Apply DNA saturation
    apply_brightness: bool = True   # Apply DNA brightness
    
    @property
    def image_path(self) -> Optional[str]:
        """Get first image path for backwards compatibility."""
        return self.image_paths[0] if self.image_paths else None
    
    def get_image_for_dna(self, dna_value: float) -> Optional[str]:
        """Get image variant based on DNA value (0-1)."""
        if not self.image_paths:
            return None
        # Map DNA value to image index
        idx = int(dna_value * len(self.image_paths)) % len(self.image_paths)
        return self.image_paths[idx]
    
    def add_image(self, path: str):
        """Add an image variant."""
        if path and path not in self.image_paths:
            self.image_paths.append(path)
    
    def remove_image(self, path: str):
        """Remove an image variant."""
        if path in self.image_paths:
            self.image_paths.remove(path)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'part': self.part,
            'stage': self.stage,
            'image_paths': self.image_paths,
            'dna_gene': self.dna_gene,
            'offset_x': self.offset_x,
            'offset_y': self.offset_y,
            'scale': self.scale,
            'z_order': self.z_order,
            'apply_hue': self.apply_hue,
            'apply_saturation': self.apply_saturation,
            'apply_brightness': self.apply_brightness,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BodyPartSprite':
        """Deserialize from dictionary."""
        # Handle old format with single image_path
        if 'image_path' in data and 'image_paths' not in data:
            data['image_paths'] = [data['image_path']] if data['image_path'] else []
            del data['image_path']
        return cls(**data)


@dataclass
class WorldObjectConfig:
    """Configuration for a world object type (water, fire, food)."""
    object_type: str                    # water, hazard, food, ground
    image_paths: List[str] = field(default_factory=list)  # List of images (animated or variants)
    tile_mode: bool = True              # Tile the image or stretch
    animation_speed: float = 0.2        # Seconds per frame (if multiple images)
    scale: float = 1.0
    
    def add_image(self, path: str):
        if path and path not in self.image_paths:
            self.image_paths.append(path)
    
    def remove_image(self, path: str):
        if path in self.image_paths:
            self.image_paths.remove(path)
    
    def to_dict(self) -> Dict:
        return {
            'object_type': self.object_type,
            'image_paths': self.image_paths,
            'tile_mode': self.tile_mode,
            'animation_speed': self.animation_speed,
            'scale': self.scale,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WorldObjectConfig':
        return cls(**data)


# World object types
class WorldObjectType:
    WATER = "water"
    HAZARD = "hazard"  # Fire, lava, etc
    FOOD = "food"
    GROUND = "ground"
    SHELTER = "shelter"
    
    ALL_TYPES = [WATER, HAZARD, FOOD, GROUND, SHELTER]


@dataclass
class CreatureVisualConfig:
    """Complete visual configuration for creatures and world objects."""
    background_path: Optional[str] = None
    body_parts: Dict[str, Dict[str, BodyPartSprite]] = field(default_factory=dict)
    # structure: body_parts[part_name][age_stage] = BodyPartSprite
    world_objects: Dict[str, WorldObjectConfig] = field(default_factory=dict)
    # structure: world_objects[object_type] = WorldObjectConfig
    
    def __post_init__(self):
        # Initialize default empty structure for body parts
        if not self.body_parts:
            for part in BodyPart.ALL_PARTS:
                self.body_parts[part] = {}
                for stage in AgeStage.ALL_STAGES:
                    self.body_parts[part][stage] = BodyPartSprite(
                        part=part, 
                        stage=stage,
                        z_order=self._default_z_order(part)
                    )
        # Initialize default empty structure for world objects
        if not self.world_objects:
            for obj_type in WorldObjectType.ALL_TYPES:
                self.world_objects[obj_type] = WorldObjectConfig(object_type=obj_type)
    
    def _default_z_order(self, part: str) -> int:
        """Get default z-order for a body part."""
        orders = {
            BodyPart.TORSO: 0,
            BodyPart.LEFT_LEG: 1,
            BodyPart.RIGHT_LEG: 1,
            BodyPart.LEFT_ARM: 2,
            BodyPart.RIGHT_ARM: 2,
            BodyPart.TAIL: -1,
            BodyPart.HEAD: 3,
            BodyPart.EYES: 4,
            BodyPart.MOUTH: 4,
        }
        return orders.get(part, 0)
    
    def get_sprite(self, part: str, age: float) -> Optional[BodyPartSprite]:
        """Get the sprite for a body part at given age."""
        stage = AgeStage.from_age(age)
        if part in self.body_parts and stage in self.body_parts[part]:
            return self.body_parts[part][stage]
        return None
    
    def set_sprite(self, part: str, stage: str, sprite: BodyPartSprite):
        """Set a sprite configuration."""
        if part not in self.body_parts:
            self.body_parts[part] = {}
        self.body_parts[part][stage] = sprite
    
    def get_world_object(self, obj_type: str) -> Optional[WorldObjectConfig]:
        """Get the config for a world object type."""
        return self.world_objects.get(obj_type)
    
    def set_world_object(self, obj_type: str, config: WorldObjectConfig):
        """Set a world object configuration."""
        self.world_objects[obj_type] = config
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        parts_dict = {}
        for part, stages in self.body_parts.items():
            parts_dict[part] = {}
            for stage, sprite in stages.items():
                parts_dict[part][stage] = sprite.to_dict()
        
        objects_dict = {}
        for obj_type, config in self.world_objects.items():
            objects_dict[obj_type] = config.to_dict()
        
        return {
            'background_path': self.background_path,
            'body_parts': parts_dict,
            'world_objects': objects_dict,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CreatureVisualConfig':
        """Deserialize from dictionary."""
        config = cls(background_path=data.get('background_path'))
        if 'body_parts' in data:
            for part, stages in data['body_parts'].items():
                config.body_parts[part] = {}
                for stage, sprite_data in stages.items():
                    config.body_parts[part][stage] = BodyPartSprite.from_dict(sprite_data)
        if 'world_objects' in data:
            for obj_type, obj_data in data['world_objects'].items():
                config.world_objects[obj_type] = WorldObjectConfig.from_dict(obj_data)
        return config
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'CreatureVisualConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


def apply_hue_to_pixmap(pixmap: QPixmap, hue: float, saturation: float = 1.0, 
                        brightness: float = 1.0) -> QPixmap:
    """
    Apply DNA hue/saturation/brightness to a pixmap.
    
    hue: 0-1 (maps to 0-360 degrees on color wheel)
    saturation: 0-1 (0 = grayscale, 1 = full color)
    brightness: 0-1 (0 = black, 1 = full brightness)
    """
    if pixmap.isNull():
        return pixmap
    
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    
    import colorsys
    
    for y in range(image.height()):
        for x in range(image.width()):
            pixel = image.pixelColor(x, y)
            if pixel.alpha() == 0:
                continue  # Skip transparent pixels
            
            # Get original HSV
            r, g, b = pixel.red() / 255, pixel.green() / 255, pixel.blue() / 255
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            
            # Apply DNA modifications
            # Shift hue (additive)
            new_h = (h + hue) % 1.0
            # Modulate saturation (multiplicative)
            new_s = min(1.0, s * saturation)
            # Modulate brightness (multiplicative)
            new_v = min(1.0, v * brightness)
            
            # Convert back to RGB
            new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, new_s, new_v)
            
            new_color = QColor(
                int(new_r * 255),
                int(new_g * 255),
                int(new_b * 255),
                pixel.alpha()
            )
            image.setPixelColor(x, y, new_color)
    
    return QPixmap.fromImage(image)


# =============================================================================
# SETTINGS TAB WIDGET
# =============================================================================

class SettingsTab(QWidget):
    """Settings tab for customizing creature visuals and game options."""
    
    # Signal emitted when settings change
    settings_changed = pyqtSignal()
    
    # Default autosave path
    AUTOSAVE_PATH = Path.home() / ".brain_v3_visual_config.json"
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = CreatureVisualConfig()
        self.current_part = BodyPart.HEAD
        self.current_stage = AgeStage.ADULT
        self.sprite_cache: Dict[str, QPixmap] = {}  # Cache loaded pixmaps
        
        self._setup_ui()
        self._load_autosave()  # Load last saved config automatically
    
    def _load_autosave(self):
        """Load the last saved configuration if it exists."""
        if self.AUTOSAVE_PATH.exists():
            try:
                self.config = CreatureVisualConfig.load(str(self.AUTOSAVE_PATH))
                self._update_background_preview()
                self._update_sprite_ui()
                self._update_config_info()
                print(f"Loaded visual config from {self.AUTOSAVE_PATH}")
            except Exception as e:
                print(f"Failed to load autosave: {e}")
                self._load_default_config()
        else:
            self._load_default_config()
    
    def _autosave(self):
        """Auto-save configuration after changes."""
        try:
            self.config.save(str(self.AUTOSAVE_PATH))
        except Exception as e:
            print(f"Failed to autosave: {e}")
    
    def _setup_ui(self):
        """Build the settings UI."""
        layout = QVBoxLayout(self)
        
        # Connect settings_changed to autosave
        self.settings_changed.connect(self._autosave)
        
        # Create tabs for different setting categories
        tabs = QTabWidget()
        
        # Background settings
        bg_tab = self._create_background_tab()
        tabs.addTab(bg_tab, "🖼️ Background")
        
        # Body part settings
        body_tab = self._create_body_parts_tab()
        tabs.addTab(body_tab, "🦎 Body Parts")
        
        # World object settings (NEW)
        world_tab = self._create_world_objects_tab()
        tabs.addTab(world_tab, "🌍 World Objects")
        
        # DNA color settings
        color_tab = self._create_color_settings_tab()
        tabs.addTab(color_tab, "🎨 DNA Colors")
        
        # Save/Load settings
        io_tab = self._create_io_tab()
        tabs.addTab(io_tab, "💾 Save/Load")
        
        layout.addWidget(tabs)
    
    def _create_background_tab(self) -> QWidget:
        """Create background image settings."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Current background preview
        preview_group = QGroupBox("Current Background")
        preview_layout = QVBoxLayout(preview_group)
        
        self.bg_preview = QLabel("No background selected")
        self.bg_preview.setMinimumSize(400, 200)
        self.bg_preview.setMaximumSize(600, 300)
        self.bg_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bg_preview.setStyleSheet("border: 1px solid gray; background: #222;")
        preview_layout.addWidget(self.bg_preview)
        
        self.bg_path_label = QLabel("Path: None")
        preview_layout.addWidget(self.bg_path_label)
        
        layout.addWidget(preview_group)
        
        # Background controls
        control_layout = QHBoxLayout()
        
        select_btn = QPushButton("📁 Select Image...")
        select_btn.clicked.connect(self._select_background)
        control_layout.addWidget(select_btn)
        
        clear_btn = QPushButton("🗑️ Clear Background")
        clear_btn.clicked.connect(self._clear_background)
        control_layout.addWidget(clear_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        layout.addStretch()
        return widget
    
    def _create_body_parts_tab(self) -> QWidget:
        """Create body part sprite configuration."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Left side: part selection
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Part selector
        part_group = QGroupBox("Body Part")
        part_layout = QVBoxLayout(part_group)
        
        self.part_list = QListWidget()
        for part in BodyPart.ALL_PARTS:
            item = QListWidgetItem(self._part_display_name(part))
            item.setData(Qt.ItemDataRole.UserRole, part)
            self.part_list.addItem(item)
        self.part_list.currentItemChanged.connect(self._on_part_selected)
        part_layout.addWidget(self.part_list)
        
        left_layout.addWidget(part_group)
        
        # Age stage selector
        stage_group = QGroupBox("Age Stage")
        stage_layout = QVBoxLayout(stage_group)
        
        self.stage_list = QListWidget()
        for stage in AgeStage.ALL_STAGES:
            item = QListWidgetItem(self._stage_display_name(stage))
            item.setData(Qt.ItemDataRole.UserRole, stage)
            self.stage_list.addItem(item)
        self.stage_list.currentItemChanged.connect(self._on_stage_selected)
        stage_layout.addWidget(self.stage_list)
        
        left_layout.addWidget(stage_group)
        
        layout.addWidget(left_widget)
        
        # Right side: sprite configuration
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Sprite preview
        sprite_group = QGroupBox("Sprite Configuration")
        sprite_layout = QGridLayout(sprite_group)
        
        self.sprite_preview = QLabel("No image")
        self.sprite_preview.setMinimumSize(100, 100)
        self.sprite_preview.setMaximumSize(150, 150)
        self.sprite_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sprite_preview.setStyleSheet("border: 1px solid gray; background: #333;")
        sprite_layout.addWidget(self.sprite_preview, 0, 0, 4, 1)
        
        # Image list (multiple variants)
        sprite_layout.addWidget(QLabel("Image Variants:"), 0, 1)
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(80)
        self.image_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.image_list.currentRowChanged.connect(self._on_image_selected)
        sprite_layout.addWidget(self.image_list, 0, 2, 2, 1)
        
        # Image list buttons
        img_btn_layout = QVBoxLayout()
        add_img_btn = QPushButton("➕")
        add_img_btn.setMaximumWidth(35)
        add_img_btn.setToolTip("Add image variant")
        add_img_btn.clicked.connect(self._add_sprite_image)
        img_btn_layout.addWidget(add_img_btn)
        
        remove_img_btn = QPushButton("➖")
        remove_img_btn.setMaximumWidth(35)
        remove_img_btn.setToolTip("Remove selected image")
        remove_img_btn.clicked.connect(self._remove_sprite_image)
        img_btn_layout.addWidget(remove_img_btn)
        
        img_btn_layout.addStretch()
        sprite_layout.addLayout(img_btn_layout, 0, 3, 2, 1)
        
        # DNA gene mapping
        sprite_layout.addWidget(QLabel("DNA Gene:"), 2, 1)
        self.dna_gene_combo = QComboBox()
        self.dna_gene_combo.addItems([
            "color_hue", "color_saturation", "body_size", 
            "limb_count", "muscle_strength", "movement_speed",
            "pattern_type", "vision_range"
        ])
        self.dna_gene_combo.currentTextChanged.connect(self._on_dna_gene_changed)
        sprite_layout.addWidget(self.dna_gene_combo, 2, 2, 1, 2)
        
        # Info label
        sprite_layout.addWidget(QLabel(
            "DNA gene selects which image variant is used.\n"
            "Add multiple images and they'll be chosen by DNA."
        ), 3, 1, 1, 3)
        
        # Offset controls
        sprite_layout.addWidget(QLabel("Offset X:"), 4, 1)
        self.offset_x_spin = QSpinBox()
        self.offset_x_spin.setRange(-100, 100)
        self.offset_x_spin.valueChanged.connect(self._on_offset_changed)
        sprite_layout.addWidget(self.offset_x_spin, 4, 2)
        
        sprite_layout.addWidget(QLabel("Offset Y:"), 5, 1)
        self.offset_y_spin = QSpinBox()
        self.offset_y_spin.setRange(-100, 100)
        self.offset_y_spin.valueChanged.connect(self._on_offset_changed)
        sprite_layout.addWidget(self.offset_y_spin, 5, 2)
        
        # Scale
        sprite_layout.addWidget(QLabel("Scale:"), 6, 1)
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(10, 300)
        self.scale_spin.setSuffix("%")
        self.scale_spin.setValue(100)
        self.scale_spin.valueChanged.connect(self._on_scale_changed)
        sprite_layout.addWidget(self.scale_spin, 6, 2)
        
        # Z-order
        sprite_layout.addWidget(QLabel("Draw Order:"), 7, 1)
        self.z_order_spin = QSpinBox()
        self.z_order_spin.setRange(-10, 10)
        self.z_order_spin.valueChanged.connect(self._on_z_order_changed)
        sprite_layout.addWidget(self.z_order_spin, 7, 2)
        
        right_layout.addWidget(sprite_group)
        
        # Color modifiers
        color_group = QGroupBox("DNA Color Application")
        color_layout = QVBoxLayout(color_group)
        
        self.apply_hue_check = QCheckBox("Apply DNA Hue")
        self.apply_hue_check.setChecked(True)
        self.apply_hue_check.stateChanged.connect(self._on_color_check_changed)
        color_layout.addWidget(self.apply_hue_check)
        
        self.apply_sat_check = QCheckBox("Apply DNA Saturation")
        self.apply_sat_check.setChecked(True)
        self.apply_sat_check.stateChanged.connect(self._on_color_check_changed)
        color_layout.addWidget(self.apply_sat_check)
        
        self.apply_bright_check = QCheckBox("Apply DNA Brightness")
        self.apply_bright_check.setChecked(True)
        self.apply_bright_check.stateChanged.connect(self._on_color_check_changed)
        color_layout.addWidget(self.apply_bright_check)
        
        right_layout.addWidget(color_group)
        
        # Clear sprite button
        clear_sprite_btn = QPushButton("🗑️ Clear All Images")
        clear_sprite_btn.clicked.connect(self._clear_sprite)
        right_layout.addWidget(clear_sprite_btn)
        
        right_layout.addStretch()
        layout.addWidget(right_widget)
        
        return widget
    
    def _create_color_settings_tab(self) -> QWidget:
        """Create color preview and testing settings."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Preview with DNA values
        preview_group = QGroupBox("DNA Color Preview")
        preview_layout = QGridLayout(preview_group)
        
        # Hue slider
        preview_layout.addWidget(QLabel("Hue:"), 0, 0)
        self.preview_hue = QSlider(Qt.Orientation.Horizontal)
        self.preview_hue.setRange(0, 100)
        self.preview_hue.setValue(50)
        self.preview_hue.valueChanged.connect(self._update_color_preview)
        preview_layout.addWidget(self.preview_hue, 0, 1)
        self.hue_label = QLabel("0.50")
        preview_layout.addWidget(self.hue_label, 0, 2)
        
        # Saturation slider
        preview_layout.addWidget(QLabel("Saturation:"), 1, 0)
        self.preview_sat = QSlider(Qt.Orientation.Horizontal)
        self.preview_sat.setRange(0, 100)
        self.preview_sat.setValue(70)
        self.preview_sat.valueChanged.connect(self._update_color_preview)
        preview_layout.addWidget(self.preview_sat, 1, 1)
        self.sat_label = QLabel("0.70")
        preview_layout.addWidget(self.sat_label, 1, 2)
        
        # Brightness slider
        preview_layout.addWidget(QLabel("Brightness:"), 2, 0)
        self.preview_bright = QSlider(Qt.Orientation.Horizontal)
        self.preview_bright.setRange(0, 100)
        self.preview_bright.setValue(80)
        self.preview_bright.valueChanged.connect(self._update_color_preview)
        preview_layout.addWidget(self.preview_bright, 2, 1)
        self.bright_label = QLabel("0.80")
        preview_layout.addWidget(self.bright_label, 2, 2)
        
        # Color preview box
        self.color_preview_box = QLabel()
        self.color_preview_box.setMinimumSize(100, 100)
        self.color_preview_box.setStyleSheet("background: rgb(128, 128, 128); border: 2px solid black;")
        preview_layout.addWidget(self.color_preview_box, 0, 3, 3, 1)
        
        layout.addWidget(preview_group)
        
        # Explanation
        info_group = QGroupBox("How DNA Colors Work")
        info_layout = QVBoxLayout(info_group)
        info_layout.addWidget(QLabel(
            "Each creature's DNA contains color genes that affect how their\n"
            "body part sprites are rendered:\n\n"
            "• Hue (0-1): Shifts colors around the color wheel\n"
            "  0.0 = Red → 0.33 = Green → 0.66 = Blue → 1.0 = Red\n\n"
            "• Saturation (0-1): Color intensity\n"
            "  0.0 = Grayscale, 1.0 = Full color\n\n"
            "• Brightness (0-1): Light/dark\n"
            "  0.0 = Black, 1.0 = Full brightness\n\n"
            "Body parts can individually enable/disable these effects."
        ))
        layout.addWidget(info_group)
        
        layout.addStretch()
        self._update_color_preview()
        return widget
    
    def _create_world_objects_tab(self) -> QWidget:
        """Create world objects image settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Object type selector
        type_group = QGroupBox("Select Object Type")
        type_layout = QHBoxLayout(type_group)
        
        self.object_type_combo = QComboBox()
        self.object_type_combo.addItems([
            "💧 Water", "🔥 Hazard/Fire", "🍎 Food", 
            "🪨 Ground", "🏠 Shelter"
        ])
        self.object_type_combo.currentIndexChanged.connect(self._on_object_type_changed)
        type_layout.addWidget(QLabel("Object Type:"))
        type_layout.addWidget(self.object_type_combo)
        layout.addWidget(type_group)
        
        # Preview and image list
        preview_group = QGroupBox("Object Images")
        preview_layout = QHBoxLayout(preview_group)
        
        # Image list
        list_layout = QVBoxLayout()
        self.object_image_list = QListWidget()
        self.object_image_list.setMaximumHeight(150)
        self.object_image_list.currentRowChanged.connect(self._on_object_image_selected)
        list_layout.addWidget(QLabel("Image variants/frames:"))
        list_layout.addWidget(self.object_image_list)
        
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("➕ Add Image")
        add_btn.clicked.connect(self._add_object_image)
        btn_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("➖ Remove")
        remove_btn.clicked.connect(self._remove_object_image)
        btn_layout.addWidget(remove_btn)
        list_layout.addLayout(btn_layout)
        
        preview_layout.addLayout(list_layout)
        
        # Preview
        preview_v_layout = QVBoxLayout()
        self.object_preview = QLabel("No image")
        self.object_preview.setMinimumSize(150, 150)
        self.object_preview.setMaximumSize(200, 200)
        self.object_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.object_preview.setStyleSheet("border: 1px solid gray; background: #333;")
        preview_v_layout.addWidget(QLabel("Preview:"))
        preview_v_layout.addWidget(self.object_preview)
        preview_layout.addLayout(preview_v_layout)
        
        layout.addWidget(preview_group)
        
        # Settings
        settings_group = QGroupBox("Object Settings")
        settings_layout = QGridLayout(settings_group)
        
        self.tile_mode_check = QCheckBox("Tile Mode (repeat image)")
        self.tile_mode_check.setChecked(True)
        self.tile_mode_check.stateChanged.connect(self._on_object_settings_changed)
        settings_layout.addWidget(self.tile_mode_check, 0, 0, 1, 2)
        
        settings_layout.addWidget(QLabel("Animation Speed (sec):"), 1, 0)
        self.anim_speed_spin = QSpinBox()
        self.anim_speed_spin.setRange(1, 100)
        self.anim_speed_spin.setValue(20)
        self.anim_speed_spin.valueChanged.connect(self._on_object_settings_changed)
        settings_layout.addWidget(self.anim_speed_spin, 1, 1)
        
        settings_layout.addWidget(QLabel("Scale:"), 2, 0)
        self.object_scale_spin = QSpinBox()
        self.object_scale_spin.setRange(10, 500)
        self.object_scale_spin.setValue(100)
        self.object_scale_spin.setSuffix("%")
        self.object_scale_spin.valueChanged.connect(self._on_object_settings_changed)
        settings_layout.addWidget(self.object_scale_spin, 2, 1)
        
        layout.addWidget(settings_group)
        
        # Help text
        help_label = QLabel(
            "💡 Add images for world objects. Multiple images = animation.\n"
            "Water: pools, fountains. Hazard: fire, lava. Food: berries, plants."
        )
        help_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(help_label)
        
        layout.addStretch()
        
        # Initialize current selection
        self.current_object_type = WorldObjectType.WATER
        self._update_object_ui()
        
        return widget
    
    def _get_current_object_type(self) -> str:
        """Get the WorldObjectType for current combo selection."""
        idx = self.object_type_combo.currentIndex()
        types = [WorldObjectType.WATER, WorldObjectType.HAZARD, WorldObjectType.FOOD,
                 WorldObjectType.GROUND, WorldObjectType.SHELTER]
        return types[idx] if idx < len(types) else WorldObjectType.WATER
    
    def _on_object_type_changed(self, index: int):
        """Handle object type combo change."""
        self.current_object_type = self._get_current_object_type()
        self._update_object_ui()
    
    def _update_object_ui(self):
        """Update UI to show current object type's settings."""
        config = self.config.get_world_object(self.current_object_type)
        if not config:
            config = WorldObjectConfig(object_type=self.current_object_type)
            self.config.set_world_object(self.current_object_type, config)
        
        # Update image list
        self.object_image_list.clear()
        for path in config.image_paths:
            self.object_image_list.addItem(Path(path).name)
        
        # Update settings (block signals to avoid triggering save during load)
        self.tile_mode_check.blockSignals(True)
        self.anim_speed_spin.blockSignals(True)
        self.object_scale_spin.blockSignals(True)
        
        self.tile_mode_check.setChecked(config.tile_mode)
        self.anim_speed_spin.setValue(int(config.animation_speed * 100))
        self.object_scale_spin.setValue(int(config.scale * 100))
        
        self.tile_mode_check.blockSignals(False)
        self.anim_speed_spin.blockSignals(False)
        self.object_scale_spin.blockSignals(False)
        
        # Update preview
        self._update_object_preview()
    
    def _update_object_preview(self):
        """Update the object preview image."""
        config = self.config.get_world_object(self.current_object_type)
        if config and config.image_paths:
            idx = self.object_image_list.currentRow()
            if idx < 0:
                idx = 0
            if idx < len(config.image_paths):
                pixmap = QPixmap(config.image_paths[idx])
                if not pixmap.isNull():
                    self.object_preview.setPixmap(
                        pixmap.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio)
                    )
                    return
        self.object_preview.setText("No image")
    
    def _on_object_image_selected(self, row: int):
        """Handle image selection in list."""
        self._update_object_preview()
    
    def _add_object_image(self):
        """Add an image to current object type."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Object Image", "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp);;All Files (*)"
        )
        if file_path:
            config = self.config.get_world_object(self.current_object_type)
            config.add_image(file_path)
            self._update_object_ui()
            self.settings_changed.emit()
    
    def _remove_object_image(self):
        """Remove selected image from current object type."""
        config = self.config.get_world_object(self.current_object_type)
        idx = self.object_image_list.currentRow()
        if config and 0 <= idx < len(config.image_paths):
            config.remove_image(config.image_paths[idx])
            self._update_object_ui()
            self.settings_changed.emit()
    
    def _on_object_settings_changed(self):
        """Handle object settings changes."""
        config = self.config.get_world_object(self.current_object_type)
        if config:
            config.tile_mode = self.tile_mode_check.isChecked()
            config.animation_speed = self.anim_speed_spin.value() / 100.0
            config.scale = self.object_scale_spin.value() / 100.0
            self.settings_changed.emit()

    def _create_io_tab(self) -> QWidget:
        """Create save/load settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Save/Load group
        io_group = QGroupBox("Configuration Files")
        io_layout = QVBoxLayout(io_group)
        
        save_btn = QPushButton("💾 Save Settings to File...")
        save_btn.clicked.connect(self._save_config_to_file)
        io_layout.addWidget(save_btn)
        
        load_btn = QPushButton("📂 Load Settings from File...")
        load_btn.clicked.connect(self._load_config_from_file)
        io_layout.addWidget(load_btn)
        
        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.clicked.connect(self._reset_to_defaults)
        io_layout.addWidget(reset_btn)
        
        layout.addWidget(io_group)
        
        # Info
        info_group = QGroupBox("Configuration Info")
        info_layout = QVBoxLayout(info_group)
        
        self.config_info = QLabel("No configuration loaded")
        info_layout.addWidget(self.config_info)
        
        layout.addWidget(info_group)
        
        layout.addStretch()
        return widget
    
    def _part_display_name(self, part: str) -> str:
        """Get display name for body part."""
        names = {
            BodyPart.FULL_BODY: "🎨 Full Body (overrides parts)",
            BodyPart.HEAD: "🧠 Head",
            BodyPart.TORSO: "👕 Torso",
            BodyPart.LEFT_ARM: "💪 Left Arm",
            BodyPart.RIGHT_ARM: "💪 Right Arm",
            BodyPart.LEFT_LEG: "🦵 Left Leg",
            BodyPart.RIGHT_LEG: "🦵 Right Leg",
            BodyPart.TAIL: "〰️ Tail",
            BodyPart.EYES: "👁️ Eyes",
            BodyPart.MOUTH: "👄 Mouth",
        }
        return names.get(part, part)
    
    def _stage_display_name(self, stage: str) -> str:
        """Get display name for age stage."""
        names = {
            AgeStage.BABY: "👶 Baby (0-20%)",
            AgeStage.JUVENILE: "🧒 Juvenile (20-40%)",
            AgeStage.ADULT: "🧑 Adult (40-70%)",
            AgeStage.ELDER: "👴 Elder (70-100%)",
        }
        return names.get(stage, stage)
    
    def _load_default_config(self):
        """Load default configuration."""
        # Try to load from default path
        config_path = Path(__file__).parent.parent / "creature_config.json"
        if config_path.exists():
            try:
                self.config = CreatureVisualConfig.load(str(config_path))
            except Exception as e:
                print(f"Failed to load config: {e}")
                self.config = CreatureVisualConfig()
        
        # Load background from images folder if exists
        bg_path = Path(__file__).parent.parent / "images" / "BackgroundForest.png"
        if bg_path.exists() and not self.config.background_path:
            self.config.background_path = str(bg_path)
        
        self._update_background_preview()
        self._update_config_info()
    
    def _select_background(self):
        """Open file dialog to select background image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Background Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if path:
            self.config.background_path = path
            self._update_background_preview()
            self.settings_changed.emit()
    
    def _clear_background(self):
        """Clear background image."""
        self.config.background_path = None
        self._update_background_preview()
        self.settings_changed.emit()
    
    def _update_background_preview(self):
        """Update background preview image."""
        if self.config.background_path and Path(self.config.background_path).exists():
            pixmap = QPixmap(self.config.background_path)
            scaled = pixmap.scaled(
                self.bg_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.bg_preview.setPixmap(scaled)
            self.bg_path_label.setText(f"Path: {self.config.background_path}")
        else:
            self.bg_preview.clear()
            self.bg_preview.setText("No background selected")
            self.bg_path_label.setText("Path: None")
    
    def _on_part_selected(self, item):
        """Handle body part selection."""
        if item:
            self.current_part = item.data(Qt.ItemDataRole.UserRole)
            self._update_sprite_ui()
    
    def _on_stage_selected(self, item):
        """Handle age stage selection."""
        if item:
            self.current_stage = item.data(Qt.ItemDataRole.UserRole)
            self._update_sprite_ui()
    
    def _update_sprite_ui(self):
        """Update sprite configuration UI for current selection."""
        sprite = self.config.get_sprite(self.current_part, 
                                         AgeStage.ALL_STAGES.index(self.current_stage) * 0.25)
        
        # Update image list
        self.image_list.clear()
        if sprite and sprite.image_paths:
            for path in sprite.image_paths:
                filename = Path(path).name if path else "Unknown"
                self.image_list.addItem(filename)
            self.image_list.setCurrentRow(0)
        
        if sprite:
            self.dna_gene_combo.setCurrentText(sprite.dna_gene)
            self.offset_x_spin.setValue(sprite.offset_x)
            self.offset_y_spin.setValue(sprite.offset_y)
            self.scale_spin.setValue(int(sprite.scale * 100))
            self.z_order_spin.setValue(sprite.z_order)
            self.apply_hue_check.setChecked(sprite.apply_hue)
            self.apply_sat_check.setChecked(sprite.apply_saturation)
            self.apply_bright_check.setChecked(sprite.apply_brightness)
            
            # Update preview with first image
            if sprite.image_paths and Path(sprite.image_paths[0]).exists():
                pixmap = QPixmap(sprite.image_paths[0])
                scaled = pixmap.scaled(
                    self.sprite_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.sprite_preview.setPixmap(scaled)
            else:
                self.sprite_preview.clear()
                self.sprite_preview.setText("No image")
        else:
            self.sprite_preview.clear()
            self.sprite_preview.setText("No image")
    
    def _on_image_selected(self, row: int):
        """Handle image selection in list - update preview."""
        sprite = self.config.get_sprite(self.current_part,
                                         AgeStage.ALL_STAGES.index(self.current_stage) * 0.25)
        if sprite and sprite.image_paths and 0 <= row < len(sprite.image_paths):
            path = sprite.image_paths[row]
            if Path(path).exists():
                pixmap = QPixmap(path)
                scaled = pixmap.scaled(
                    self.sprite_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.sprite_preview.setPixmap(scaled)
    
    def _get_current_sprite(self) -> BodyPartSprite:
        """Get or create the current sprite."""
        sprite = self.config.get_sprite(self.current_part,
                                         AgeStage.ALL_STAGES.index(self.current_stage) * 0.25)
        if not sprite:
            sprite = BodyPartSprite(part=self.current_part, stage=self.current_stage)
            self.config.set_sprite(self.current_part, self.current_stage, sprite)
        return sprite
    
    def _add_sprite_image(self):
        """Open file dialog to add a sprite image to the list."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Sprite Image(s)", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if paths:
            sprite = self._get_current_sprite()
            for path in paths:
                sprite.add_image(path)
            self._update_sprite_ui()
            self.settings_changed.emit()
    
    def _remove_sprite_image(self):
        """Remove selected image from the list."""
        current_row = self.image_list.currentRow()
        sprite = self._get_current_sprite()
        if sprite.image_paths and 0 <= current_row < len(sprite.image_paths):
            sprite.image_paths.pop(current_row)
            self._update_sprite_ui()
            self.settings_changed.emit()
    
    def _on_dna_gene_changed(self, gene: str):
        """Handle DNA gene selection change."""
        sprite = self._get_current_sprite()
        sprite.dna_gene = gene
        self.settings_changed.emit()
    
    def _on_offset_changed(self):
        """Handle offset change."""
        sprite = self._get_current_sprite()
        sprite.offset_x = self.offset_x_spin.value()
        sprite.offset_y = self.offset_y_spin.value()
        self.settings_changed.emit()
    
    def _on_scale_changed(self, value: int):
        """Handle scale change."""
        sprite = self._get_current_sprite()
        sprite.scale = value / 100.0
        self.settings_changed.emit()
    
    def _on_z_order_changed(self, value: int):
        """Handle z-order change."""
        sprite = self._get_current_sprite()
        sprite.z_order = value
        self.settings_changed.emit()
    
    def _on_color_check_changed(self):
        """Handle color application checkbox changes."""
        sprite = self._get_current_sprite()
        sprite.apply_hue = self.apply_hue_check.isChecked()
        sprite.apply_saturation = self.apply_sat_check.isChecked()
        sprite.apply_brightness = self.apply_bright_check.isChecked()
        self.settings_changed.emit()
    
    def _clear_sprite(self):
        """Clear all sprite images."""
        sprite = self._get_current_sprite()
        sprite.image_paths.clear()
        self._update_sprite_ui()
        self.settings_changed.emit()
    
    def _update_color_preview(self):
        """Update color preview box."""
        import colorsys
        
        hue = self.preview_hue.value() / 100
        sat = self.preview_sat.value() / 100
        bright = self.preview_bright.value() / 100
        
        self.hue_label.setText(f"{hue:.2f}")
        self.sat_label.setText(f"{sat:.2f}")
        self.bright_label.setText(f"{bright:.2f}")
        
        r, g, b = colorsys.hsv_to_rgb(hue, sat, bright)
        self.color_preview_box.setStyleSheet(
            f"background: rgb({int(r*255)}, {int(g*255)}, {int(b*255)}); "
            f"border: 2px solid black;"
        )
    
    def _save_config_to_file(self):
        """Save configuration to a file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "creature_config.json",
            "JSON Files (*.json)"
        )
        if path:
            try:
                self.config.save(path)
                self._update_config_info()
            except Exception as e:
                print(f"Failed to save: {e}")
    
    def _load_config_from_file(self):
        """Load configuration from a file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "",
            "JSON Files (*.json)"
        )
        if path:
            try:
                self.config = CreatureVisualConfig.load(path)
                self._update_background_preview()
                self._update_sprite_ui()
                self._update_config_info()
                self.settings_changed.emit()
            except Exception as e:
                print(f"Failed to load: {e}")
    
    def _reset_to_defaults(self):
        """Reset to default configuration."""
        self.config = CreatureVisualConfig()
        self._load_default_config()
        self._update_sprite_ui()
        self.settings_changed.emit()
    
    def _update_config_info(self):
        """Update configuration info display."""
        parts_with_sprites = 0
        total_images = 0
        for part in BodyPart.ALL_PARTS:
            for stage in AgeStage.ALL_STAGES:
                sprite = self.config.get_sprite(part, AgeStage.ALL_STAGES.index(stage) * 0.25)
                if sprite and sprite.image_paths:
                    parts_with_sprites += 1
                    total_images += len(sprite.image_paths)
        
        total = len(BodyPart.ALL_PARTS) * len(AgeStage.ALL_STAGES)
        
        info = f"Body parts configured: {parts_with_sprites}/{total}\n"
        info += f"Total image variants: {total_images}\n"
        info += f"Background: {'Set' if self.config.background_path else 'None'}"
        self.config_info.setText(info)
    
    def get_config(self) -> CreatureVisualConfig:
        """Get current visual configuration."""
        return self.config
    
    def get_background_pixmap(self) -> Optional[QPixmap]:
        """Get background pixmap if set."""
        if self.config.background_path and Path(self.config.background_path).exists():
            return QPixmap(self.config.background_path)
        return None
