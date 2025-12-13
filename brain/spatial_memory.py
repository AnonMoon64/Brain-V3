
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PlaceCell:
    id: str
    x_center: float
    y_center: float
    activation: float = 0.0
    
    # Synaptic weights to value neurons (The Memory)
    # Positive = attraction (food/water), Negative = repulsion (danger)
    # These effectively map: Place -> Value
    synapses: Dict[str, float] = None

    def __post_init__(self):
        if self.synapses is None:
            self.synapses = {
                'food': 0.0,
                'water': 0.0,
                'hazard': 0.0,
                'comfort': 0.0
            }

class SpatialMemorySystem:
    """
    Implements Structural Spatial Memory via Place Cells.
    
    Philosophy: Memory is Structure.
    - We do not store coordinates of food.
    - We store synaptic weights between Place Cells and Value Neurons.
    - Recall is activation, not lookup.
    """
    
    def __init__(self, world_width: float, world_height: float, grid_size: int = 8):
        self.world_width = world_width
        self.world_height = world_height
        self.grid_size = grid_size
        
        # Create structural grid of Place Cells
        self.place_cells: List[PlaceCell] = []
        self._init_place_cells()
        
        # State tracking
        self.current_place_idx: int = -1
        self.last_place_idx: int = -1
        
        # Neural parameters
        self.learning_rate = 0.1
        self.decay_rate = 0.001  # Slow forgetting
        
    def _init_place_cells(self):
        """Create the grid structure of neurons."""
        cols = int(self.world_width / self.grid_size) + 1
        rows = int(self.world_height / self.grid_size) + 1
        
        for r in range(rows):
            for c in range(cols):
                # Center of this zone
                cx = c * self.grid_size + self.grid_size / 2
                cy = r * self.grid_size + self.grid_size / 2
                
                cell = PlaceCell(
                    id=f"PC_{c}_{r}",
                    x_center=cx,
                    y_center=cy
                )
                self.place_cells.append(cell)
                
    def get_cell_index(self, x: float, y: float) -> int:
        """Map physical coordinate to structural unit (Place Cell)."""
        c = int(x / self.grid_size)
        r = int(y / self.grid_size)
        
        cols = int(self.world_width / self.grid_size) + 1
        idx = r * cols + c
        
        # Safety bounds
        return max(0, min(len(self.place_cells) - 1, idx))

    def update(self, x: float, y: float, sensory_experiences: Dict[str, float], dt: float):
        """
        Process Structural Memory Update.
        
        1. Activate current Place Cell (Where am I?)
        2. Strengthen synapses if experiencing value (Hebbian Learning)
        3. Decay unused synapses (Forgetting)
        """
        # 1. Activate Place Cell
        idx = self.get_cell_index(x, y)
        self.current_place_idx = idx
        
        current_cell = self.place_cells[idx]
        current_cell.activation = 1.0  # Firing!
        
        # 2. Hebbian Learning: Wired together if fired together
        # If I am here (Place Cell Firing) AND experiencing value (Value Neuron Firing),
        # strengthen the connection.
        
        # Experience thresholds
        if sensory_experiences.get('energy_gain', 0) > 0.1:
            # Found food here -> Strengthen connection to 'food' concept
            current_cell.synapses['food'] = min(1.0, current_cell.synapses['food'] + self.learning_rate)
            
        if sensory_experiences.get('pain', 0) > 0.2:
            # Danger here -> Strengthen connection to 'hazard' concept
            current_cell.synapses['hazard'] = min(1.0, current_cell.synapses['hazard'] + self.learning_rate * 2)
            
        # 3. Structural Decay (unused memories fade)
        for cell in self.place_cells:
            for k in cell.synapses:
                if cell.synapses[k] > 0.01:
                    cell.synapses[k] -= self.decay_rate * dt

    def get_navigation_gradient(self, drives: Dict[str, float]) -> Optional[Tuple[float, float]]:
        """
        Recall: Activation of navigational intention.
        
        Inactivates place cells based on current drives + synaptic weights.
        Returns vector to most attractive active place cell.
        """
        max_attraction = -999.0
        target_cell = None
        
        # Drive mapping
        hunger = drives.get('hunger', 0)
        fear = drives.get('fear', 0)
        
        # Scan structure for resonance
        for cell in self.place_cells:
            # Skip if empty memory (optimization, structurally equivalent to no firing)
            if sum(cell.synapses.values()) < 0.01:
                continue
                
            # Compute attraction score (Neuron Activation)
            # Excitation from Hunger drive -> Food Synapse -> Place Cell
            score = 0.0
            score += cell.synapses['food'] * hunger
            
            # Inhibition from Fear drive -> Hazard Synapse -> Place Cell
            score -= cell.synapses['hazard'] * fear * 1.5  # Fear is stronger
            
            # Distance inhibition (Travel cost)
            # We don't want to go to the other side of the world for a snack
            # Simple Manhattan distance penalty
            current_pos = self.place_cells[self.current_place_idx]
            dist = abs(cell.x_center - current_pos.x_center) + abs(cell.y_center - current_pos.y_center)
            score -= dist * 0.001
            
            if score > max_attraction:
                max_attraction = score
                target_cell = cell
                
        # Return vector to target if attraction is strong enough
        if target_cell and max_attraction > 0.1:
            return (target_cell.x_center, target_cell.y_center)
            
        return None
