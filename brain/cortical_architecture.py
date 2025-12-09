"""
Advanced Cortical Architecture

Implements:
- Hierarchical cortical topology with proper feedforward/feedback
- Topographic mapping (retinotopic-like organization)
- Cortical columns (minicolumns → macrocolumns → areas)
- Layer-specific connectivity (L4 input, L2/3 integration, L5/6 output)
- Lateral inhibition with proper surround suppression
- Feedback connections for predictive processing

Key insight: The cortex is organized hierarchically with distinct
feedforward (driving) and feedback (modulatory) pathways.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import math


class CorticalLayer(Enum):
    """Cortical layers with distinct functions"""
    L1 = 1    # Molecular layer - feedback targets
    L2_3 = 2  # Superficial - corticocortical output
    L4 = 4    # Granular - thalamic input
    L5 = 5    # Deep - subcortical output
    L6 = 6    # Deep - thalamic feedback


class CorticalArea(Enum):
    """Cortical areas in hierarchy"""
    V1 = "V1"           # Primary visual (level 1)
    V2 = "V2"           # Secondary visual (level 2)
    V4 = "V4"           # Color/form (level 3)
    IT = "IT"           # Inferotemporal - object recognition (level 4)
    A1 = "A1"           # Primary auditory (level 1)
    A2 = "A2"           # Secondary auditory (level 2)
    WERNICKE = "WA"     # Language comprehension (level 3)
    S1 = "S1"           # Primary somatosensory (level 1)
    M1 = "M1"           # Primary motor (level 3)
    PFC = "PFC"         # Prefrontal cortex (level 5)
    DLPFC = "DLPFC"     # Dorsolateral PFC - working memory (level 5)
    OFC = "OFC"         # Orbitofrontal - value (level 4)
    ACC = "ACC"         # Anterior cingulate - conflict monitoring (level 4)
    HIPPOCAMPUS = "HC"  # Memory formation (special)
    AMYGDALA = "AMG"    # Emotion (special)


# Hierarchy levels (lower = closer to sensory, higher = more abstract)
AREA_HIERARCHY = {
    CorticalArea.V1: 1, CorticalArea.A1: 1, CorticalArea.S1: 1,
    CorticalArea.V2: 2, CorticalArea.A2: 2,
    CorticalArea.V4: 3, CorticalArea.WERNICKE: 3, CorticalArea.M1: 3,
    CorticalArea.IT: 4, CorticalArea.OFC: 4, CorticalArea.ACC: 4,
    CorticalArea.PFC: 5, CorticalArea.DLPFC: 5,
    CorticalArea.HIPPOCAMPUS: 3, CorticalArea.AMYGDALA: 2,
}


@dataclass
class CorticalPosition:
    """Position within cortical sheet"""
    area: CorticalArea
    layer: CorticalLayer
    x: float  # Position within area (0-1)
    y: float  # Position within area (0-1)
    column_id: int = 0  # Which column this belongs to
    
    def distance_to(self, other: 'CorticalPosition') -> float:
        """Euclidean distance in cortical sheet"""
        if self.area != other.area:
            return float('inf')  # Different areas
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Minicolumn:
    """
    Cortical minicolumn (~100 neurons in 6 layers)
    
    Minicolumns are the fundamental processing unit.
    They implement canonical microcircuit computations.
    """
    id: int
    area: CorticalArea
    position: Tuple[float, float]  # x, y in area
    
    # Neurons per layer (simplified: 1 representative per layer)
    layer_activity: Dict[CorticalLayer, float] = field(default_factory=dict)
    layer_neurons: Dict[CorticalLayer, List[str]] = field(default_factory=dict)
    
    # Column state
    active: bool = False
    activity_level: float = 0.0
    
    # Lateral connections (inhibitory)
    inhibits: List[int] = field(default_factory=list)  # Column IDs we inhibit
    inhibited_by: List[int] = field(default_factory=list)
    
    # Feedforward/feedback connections
    ff_inputs: List[Tuple[int, float]] = field(default_factory=list)  # (column_id, weight)
    fb_inputs: List[Tuple[int, float]] = field(default_factory=list)
    ff_outputs: List[int] = field(default_factory=list)
    fb_outputs: List[int] = field(default_factory=list)
    
    # Prediction
    prediction: float = 0.0
    prediction_error: float = 0.0
    
    def __post_init__(self):
        for layer in CorticalLayer:
            self.layer_activity[layer] = 0.0
            self.layer_neurons[layer] = []
    
    def receive_input(self, layer: CorticalLayer, strength: float):
        """Receive input at specific layer"""
        self.layer_activity[layer] = min(1.0, self.layer_activity[layer] + strength)
    
    def compute_output(self) -> Tuple[float, float]:
        """
        Compute column output
        Returns: (feedforward_output, feedback_output)
        """
        # L4 receives feedforward, L2/3 sends feedforward
        ff_out = self.layer_activity[CorticalLayer.L2_3]
        
        # L5/6 send feedback
        fb_out = (self.layer_activity[CorticalLayer.L5] + 
                  self.layer_activity[CorticalLayer.L6]) / 2
        
        return ff_out, fb_out
    
    def update(self, dt: float) -> None:
        """Update column state"""
        # Canonical microcircuit computation
        # L4 → L2/3 (feedforward pathway)
        l4_input = self.layer_activity[CorticalLayer.L4]
        self.layer_activity[CorticalLayer.L2_3] = (
            0.8 * self.layer_activity[CorticalLayer.L2_3] + 
            0.2 * l4_input
        )
        
        # L2/3 → L5 (output pathway)
        l23_activity = self.layer_activity[CorticalLayer.L2_3]
        self.layer_activity[CorticalLayer.L5] = (
            0.8 * self.layer_activity[CorticalLayer.L5] +
            0.2 * l23_activity
        )
        
        # L5 → L6 (motor output & thalamic feedback)
        self.layer_activity[CorticalLayer.L6] = (
            0.9 * self.layer_activity[CorticalLayer.L6] +
            0.1 * self.layer_activity[CorticalLayer.L5]
        )
        
        # L1 receives feedback - modulates L2/3
        l1_fb = self.layer_activity[CorticalLayer.L1]
        if l1_fb > 0.3:  # Feedback enhances response
            self.layer_activity[CorticalLayer.L2_3] *= (1.0 + l1_fb * 0.3)
        
        # Overall activity
        self.activity_level = np.mean([v for v in self.layer_activity.values()])
        self.active = self.activity_level > 0.3
        
        # Decay
        for layer in CorticalLayer:
            self.layer_activity[layer] *= (1.0 - 0.1 * dt)


@dataclass
class CorticalAreaInfo:
    """Information about a cortical area"""
    area: CorticalArea
    hierarchy_level: int
    size: Tuple[int, int]  # Grid size (columns x columns)
    
    # Connectivity
    feedforward_targets: List[CorticalArea] = field(default_factory=list)
    feedback_targets: List[CorticalArea] = field(default_factory=list)
    lateral_inhibition_radius: float = 0.2
    
    # Processing characteristics
    time_scale: float = 1.0  # Relative time scale (higher = slower)
    sparsity: float = 0.02  # Target active fraction


class TopographicMap:
    """
    Topographic (retinotopic-like) mapping between areas
    
    Preserves spatial relationships during feedforward projection.
    """
    
    def __init__(self, source_area: CorticalArea, target_area: CorticalArea):
        self.source = source_area
        self.target = target_area
        
        # Mapping parameters
        self.magnification = 1.0  # Cortical magnification factor
        self.convergence = 2.0   # How many source columns → one target
        self.scatter = 0.1       # Random jitter in mapping
    
    def map_position(self, source_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Map position from source to target area"""
        x, y = source_pos
        
        # Apply magnification (center gets more cortical area)
        center_dist = math.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        mag = self.magnification * (1.0 + 0.5 * (0.5 - center_dist))
        
        # Apply scatter
        x_target = x / self.convergence + np.random.normal(0, self.scatter)
        y_target = y / self.convergence + np.random.normal(0, self.scatter)
        
        return (max(0, min(1, x_target)), max(0, min(1, y_target)))
    
    def get_receptive_field(
        self, 
        target_pos: Tuple[float, float], 
        rf_size: float = 0.2
    ) -> List[Tuple[float, float]]:
        """Get source positions that feed into target position"""
        # Inverse mapping with convergence
        positions = []
        for dx in np.linspace(-rf_size, rf_size, 5):
            for dy in np.linspace(-rf_size, rf_size, 5):
                dist = math.sqrt(dx**2 + dy**2)
                if dist <= rf_size:
                    x = target_pos[0] * self.convergence + dx
                    y = target_pos[1] * self.convergence + dy
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        positions.append((x, y))
        return positions


class LateralInhibition:
    """
    Implements center-surround lateral inhibition
    
    Creates competition between nearby columns,
    implementing winner-take-all dynamics.
    """
    
    def __init__(self, radius: float = 0.2, strength: float = 0.5):
        self.radius = radius
        self.strength = strength
        
        # Mexican hat profile (excitatory center, inhibitory surround)
        self.center_radius = radius * 0.3
        self.surround_radius = radius
        
    def compute_inhibition(
        self,
        columns: List[Minicolumn],
        active_column_id: int
    ) -> Dict[int, float]:
        """
        Compute inhibition from active column to others
        Returns dict of column_id -> inhibition_amount
        """
        active_col = None
        for col in columns:
            if col.id == active_column_id:
                active_col = col
                break
        
        if active_col is None:
            return {}
        
        inhibitions = {}
        for col in columns:
            if col.id == active_column_id:
                continue
            if col.area != active_col.area:
                continue
            
            # Distance in cortical sheet
            dx = col.position[0] - active_col.position[0]
            dy = col.position[1] - active_col.position[1]
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist <= self.surround_radius:
                # Mexican hat: center excitation, surround inhibition
                if dist <= self.center_radius:
                    # Slight excitation (not used here, pure inhibition)
                    pass
                else:
                    # Surround inhibition
                    norm_dist = (dist - self.center_radius) / (self.surround_radius - self.center_radius)
                    inhibition = self.strength * active_col.activity_level * (1.0 - norm_dist)
                    inhibitions[col.id] = inhibition
        
        return inhibitions


class HierarchicalCortex:
    """
    Complete hierarchical cortical architecture
    
    Implements:
    - Multiple cortical areas with hierarchy
    - Feedforward driving connections
    - Feedback modulatory connections
    - Topographic mapping
    - Lateral inhibition
    - Predictive processing
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        self.columns: Dict[int, Minicolumn] = {}
        self.areas: Dict[CorticalArea, CorticalAreaInfo] = {}
        self.area_columns: Dict[CorticalArea, List[int]] = {}
        
        self.topographic_maps: Dict[Tuple[CorticalArea, CorticalArea], TopographicMap] = {}
        self.lateral_inhibition: Dict[CorticalArea, LateralInhibition] = {}
        
        # Global state
        self.current_time: float = 0.0
        self.next_column_id: int = 0
        
        # Prediction state
        self.area_predictions: Dict[CorticalArea, np.ndarray] = {}
        self.prediction_errors: Dict[CorticalArea, np.ndarray] = {}
        
        # Initialize architecture
        self._initialize_areas(config)
        self._initialize_connectivity()
    
    def _initialize_areas(self, config: Dict) -> None:
        """Initialize cortical areas with columns"""
        
        # Area specifications
        area_specs = {
            CorticalArea.V1: {'size': (8, 8), 'sparsity': 0.05},
            CorticalArea.V2: {'size': (6, 6), 'sparsity': 0.04},
            CorticalArea.V4: {'size': (4, 4), 'sparsity': 0.03},
            CorticalArea.IT: {'size': (4, 4), 'sparsity': 0.02},
            CorticalArea.A1: {'size': (6, 6), 'sparsity': 0.05},
            CorticalArea.A2: {'size': (4, 4), 'sparsity': 0.03},
            CorticalArea.WERNICKE: {'size': (4, 4), 'sparsity': 0.02},
            CorticalArea.PFC: {'size': (6, 6), 'sparsity': 0.02},
            CorticalArea.DLPFC: {'size': (4, 4), 'sparsity': 0.02},
            CorticalArea.HIPPOCAMPUS: {'size': (4, 4), 'sparsity': 0.05},
            CorticalArea.AMYGDALA: {'size': (3, 3), 'sparsity': 0.10},
            CorticalArea.M1: {'size': (4, 4), 'sparsity': 0.03},
        }
        
        # Feedforward hierarchy
        ff_connections = {
            CorticalArea.V1: [CorticalArea.V2],
            CorticalArea.V2: [CorticalArea.V4],
            CorticalArea.V4: [CorticalArea.IT, CorticalArea.PFC],
            CorticalArea.IT: [CorticalArea.PFC, CorticalArea.HIPPOCAMPUS],
            CorticalArea.A1: [CorticalArea.A2],
            CorticalArea.A2: [CorticalArea.WERNICKE],
            CorticalArea.WERNICKE: [CorticalArea.PFC, CorticalArea.DLPFC],
            CorticalArea.PFC: [CorticalArea.DLPFC, CorticalArea.M1],
            CorticalArea.DLPFC: [CorticalArea.M1],
            CorticalArea.HIPPOCAMPUS: [CorticalArea.PFC],
            CorticalArea.AMYGDALA: [CorticalArea.PFC, CorticalArea.HIPPOCAMPUS],
        }
        
        for area, spec in area_specs.items():
            size = spec['size']
            hierarchy_level = AREA_HIERARCHY.get(area, 3)
            
            # Time scale increases with hierarchy
            time_scale = 1.0 + (hierarchy_level - 1) * 0.5
            
            area_info = CorticalAreaInfo(
                area=area,
                hierarchy_level=hierarchy_level,
                size=size,
                feedforward_targets=ff_connections.get(area, []),
                lateral_inhibition_radius=0.3,
                time_scale=time_scale,
                sparsity=spec['sparsity'],
            )
            self.areas[area] = area_info
            self.area_columns[area] = []
            
            # Create columns in grid
            for i in range(size[0]):
                for j in range(size[1]):
                    x = (i + 0.5) / size[0]
                    y = (j + 0.5) / size[1]
                    
                    column = Minicolumn(
                        id=self.next_column_id,
                        area=area,
                        position=(x, y),
                    )
                    self.columns[column.id] = column
                    self.area_columns[area].append(column.id)
                    self.next_column_id += 1
            
            # Initialize lateral inhibition for this area
            self.lateral_inhibition[area] = LateralInhibition(
                radius=area_info.lateral_inhibition_radius,
                strength=0.5
            )
            
            # Initialize prediction arrays
            n_columns = len(self.area_columns[area])
            self.area_predictions[area] = np.zeros(n_columns)
            self.prediction_errors[area] = np.zeros(n_columns)
    
    def _initialize_connectivity(self) -> None:
        """Set up feedforward and feedback connections"""
        
        for area, area_info in self.areas.items():
            # Feedforward connections
            for target_area in area_info.feedforward_targets:
                if target_area not in self.areas:
                    continue
                
                # Create topographic map
                topo_map = TopographicMap(area, target_area)
                self.topographic_maps[(area, target_area)] = topo_map
                
                # Connect columns based on topography
                source_columns = [self.columns[cid] for cid in self.area_columns[area]]
                target_columns = [self.columns[cid] for cid in self.area_columns[target_area]]
                
                for source_col in source_columns:
                    target_pos = topo_map.map_position(source_col.position)
                    
                    # Find nearest target columns (convergent connectivity)
                    for target_col in target_columns:
                        dist = math.sqrt(
                            (target_col.position[0] - target_pos[0])**2 +
                            (target_col.position[1] - target_pos[1])**2
                        )
                        if dist < 0.4:  # Receptive field size
                            weight = max(0.1, 1.0 - dist * 2)
                            source_col.ff_outputs.append(target_col.id)
                            target_col.ff_inputs.append((source_col.id, weight))
                
                # Also set up feedback connections (inverse)
                target_info = self.areas[target_area]
                if area not in target_info.feedback_targets:
                    target_info.feedback_targets.append(area)
                
                for target_col in target_columns:
                    # Feedback is more diffuse
                    rf = topo_map.get_receptive_field(target_col.position, rf_size=0.3)
                    for source_col in source_columns:
                        for rf_pos in rf:
                            dist = math.sqrt(
                                (source_col.position[0] - rf_pos[0])**2 +
                                (source_col.position[1] - rf_pos[1])**2
                            )
                            if dist < 0.2:
                                weight = 0.3 * max(0.1, 1.0 - dist * 3)
                                target_col.fb_outputs.append(source_col.id)
                                source_col.fb_inputs.append((target_col.id, weight))
                                break
            
            # Lateral inhibition connections within area
            area_cols = [self.columns[cid] for cid in self.area_columns[area]]
            lat_inhib = self.lateral_inhibition[area]
            
            for col in area_cols:
                for other_col in area_cols:
                    if col.id == other_col.id:
                        continue
                    dist = math.sqrt(
                        (col.position[0] - other_col.position[0])**2 +
                        (col.position[1] - other_col.position[1])**2
                    )
                    if dist < lat_inhib.surround_radius:
                        col.inhibits.append(other_col.id)
                        other_col.inhibited_by.append(col.id)
    
    def inject_input(
        self, 
        area: CorticalArea, 
        pattern: np.ndarray,
        layer: CorticalLayer = CorticalLayer.L4
    ) -> None:
        """Inject input pattern into an area"""
        column_ids = self.area_columns.get(area, [])
        if len(column_ids) == 0:
            return
        
        # Reshape pattern to match column grid
        pattern_flat = pattern.flatten()
        
        for i, col_id in enumerate(column_ids):
            if i < len(pattern_flat):
                strength = pattern_flat[i]
                self.columns[col_id].receive_input(layer, strength)
    
    def step(self, dt: float) -> Dict[str, any]:
        """
        Run one simulation step
        
        Processing order:
        1. Feedforward sweep (bottom-up)
        2. Lateral inhibition (within areas)
        3. Feedback sweep (top-down)
        4. Prediction error computation
        5. Update all columns
        """
        self.current_time += dt
        results = {'spikes': [], 'prediction_errors': {}}
        
        # Sort areas by hierarchy level (bottom-up for feedforward)
        sorted_areas = sorted(self.areas.keys(), key=lambda a: AREA_HIERARCHY.get(a, 3))
        
        # 1. Feedforward sweep
        for area in sorted_areas:
            area_info = self.areas[area]
            
            # Scale dt by area's time constant
            effective_dt = dt / area_info.time_scale
            
            for col_id in self.area_columns[area]:
                col = self.columns[col_id]
                
                # Accumulate feedforward input to L4
                ff_input = 0.0
                for source_id, weight in col.ff_inputs:
                    source_col = self.columns.get(source_id)
                    if source_col:
                        ff_out, _ = source_col.compute_output()
                        ff_input += ff_out * weight
                
                if ff_input > 0:
                    col.receive_input(CorticalLayer.L4, ff_input * 0.5)
        
        # 2. Lateral inhibition (k-winners-take-all per area)
        for area in self.areas:
            area_cols = [self.columns[cid] for cid in self.area_columns[area]]
            lat_inhib = self.lateral_inhibition[area]
            sparsity = self.areas[area].sparsity
            
            # Find top-k active columns (k-winners-take-all)
            activities = [(col.id, col.activity_level) for col in area_cols]
            activities.sort(key=lambda x: x[1], reverse=True)
            
            k = max(1, int(len(area_cols) * sparsity))
            winners = set(act[0] for act in activities[:k])
            
            # Apply lateral inhibition
            for col in area_cols:
                if col.id in winners:
                    # Winner inhibits neighbors
                    inhibitions = lat_inhib.compute_inhibition(area_cols, col.id)
                    for target_id, inhibition in inhibitions.items():
                        target_col = self.columns.get(target_id)
                        if target_col:
                            for layer in CorticalLayer:
                                target_col.layer_activity[layer] *= (1.0 - inhibition)
                else:
                    # Non-winner gets suppressed
                    for layer in CorticalLayer:
                        col.layer_activity[layer] *= 0.5
        
        # 3. Feedback sweep (top-down)
        for area in reversed(sorted_areas):
            for col_id in self.area_columns[area]:
                col = self.columns[col_id]
                
                # Accumulate feedback input to L1
                fb_input = 0.0
                for source_id, weight in col.fb_inputs:
                    source_col = self.columns.get(source_id)
                    if source_col:
                        _, fb_out = source_col.compute_output()
                        fb_input += fb_out * weight
                
                if fb_input > 0:
                    col.receive_input(CorticalLayer.L1, fb_input * 0.3)
                    # Store as prediction
                    col.prediction = fb_input
        
        # 4. Compute prediction errors
        for area in self.areas:
            errors = []
            for i, col_id in enumerate(self.area_columns[area]):
                col = self.columns[col_id]
                
                # Prediction error = actual - predicted
                actual = col.layer_activity[CorticalLayer.L4]
                predicted = col.prediction
                error = actual - predicted
                
                col.prediction_error = error
                errors.append(abs(error))
            
            self.prediction_errors[area] = np.array(errors)
            results['prediction_errors'][area.value] = np.mean(errors)
        
        # 5. Update all columns
        for col in self.columns.values():
            area_info = self.areas[col.area]
            effective_dt = dt / area_info.time_scale
            col.update(effective_dt)
            
            if col.active:
                results['spikes'].append(col.id)
        
        return results
    
    def get_area_activity(self, area: CorticalArea) -> np.ndarray:
        """Get activity pattern for an area"""
        column_ids = self.area_columns.get(area, [])
        activities = np.array([self.columns[cid].activity_level for cid in column_ids])
        return activities
    
    def get_sparse_representation(self, area: CorticalArea) -> Set[int]:
        """Get set of active column IDs (sparse code)"""
        active = set()
        for col_id in self.area_columns.get(area, []):
            if self.columns[col_id].active:
                active.add(col_id)
        return active
    
    def get_output(self) -> np.ndarray:
        """Get output from executive areas (M1, PFC)"""
        output_areas = [CorticalArea.M1, CorticalArea.PFC, CorticalArea.DLPFC]
        outputs = []
        for area in output_areas:
            if area in self.area_columns:
                outputs.extend([self.columns[cid].activity_level 
                              for cid in self.area_columns[area]])
        return np.array(outputs)
    
    def get_stats(self) -> Dict:
        """Get cortex statistics"""
        stats = {
            'total_columns': len(self.columns),
            'areas': {},
            'total_active': sum(1 for c in self.columns.values() if c.active),
        }
        
        for area in self.areas:
            cols = [self.columns[cid] for cid in self.area_columns[area]]
            stats['areas'][area.value] = {
                'columns': len(cols),
                'active': sum(1 for c in cols if c.active),
                'avg_activity': np.mean([c.activity_level for c in cols]),
                'avg_pred_error': np.mean(self.prediction_errors.get(area, [0])),
            }
        
        return stats
