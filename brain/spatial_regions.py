"""
Spatial Brain Regions (UPGRADE 3)

Real brains aren't random connection soup - they have REGIONS:
- Visual cortex (V1, V2, V4, IT)
- Motor cortex (M1, premotor)
- Reward circuitry (VTA, nucleus accumbens)
- Navigation (hippocampus, entorhinal cortex, grid cells)
- Limbic (amygdala, hypothalamus, emotional tagging)

Regions have:
1. Local dense connectivity (within-region)
2. Sparse long-range projections (between-region)
3. Specific computational roles
4. Different neuromodulator sensitivity

Still evolves, but with regional BIAS.
Prevents "everything connects to everything" mess.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum


class BrainRegion(Enum):
    """Major brain regions."""
    VISUAL = "visual"                   # Sensory input processing
    MOTOR = "motor"                     # Action output
    REWARD = "reward"                   # Value / motivation
    NAVIGATION = "navigation"           # Spatial representation
    LIMBIC = "limbic"                   # Emotion / valence
    ASSOCIATION = "association"         # Integration / abstract thought
    WORKING_MEMORY = "working_memory"   # Short-term holding
    LANGUAGE = "language"               # Symbolic communication


@dataclass
class RegionParams:
    """Parameters defining a brain region's properties."""
    # === BASIC INFO ===
    region_type: BrainRegion
    n_neurons: int
    
    # === CONNECTIVITY ===
    # Within-region connection density (0-1)
    local_density: float = 0.2
    
    # Allowed target regions for projections
    target_regions: Set[BrainRegion] = field(default_factory=set)
    
    # Projection density to each target (0-1)
    projection_densities: Dict[BrainRegion, float] = field(default_factory=dict)
    
    # === COMPUTATIONAL PROPERTIES ===
    # Sparsity level (fraction of active neurons)
    sparsity: float = 0.02
    
    # Time constant (fast = sensory, slow = limbic)
    tau: float = 0.01
    
    # === NEUROMODULATOR SENSITIVITY ===
    # How much each neuromodulator affects this region (0-1)
    dopamine_sensitivity: float = 0.5
    serotonin_sensitivity: float = 0.5
    norepinephrine_sensitivity: float = 0.5
    acetylcholine_sensitivity: float = 0.5
    
    # === PLASTICITY ===
    # Learning rate multiplier
    plasticity_rate: float = 1.0
    
    @classmethod
    def for_region_type(cls, region_type: BrainRegion, n_neurons: int) -> 'RegionParams':
        """Get default parameters for a specific region type."""
        
        if region_type == BrainRegion.VISUAL:
            return cls(
                region_type=region_type,
                n_neurons=n_neurons,
                local_density=0.3,  # Dense local processing
                target_regions={BrainRegion.ASSOCIATION, BrainRegion.NAVIGATION, BrainRegion.LIMBIC},
                projection_densities={
                    BrainRegion.ASSOCIATION: 0.05,
                    BrainRegion.NAVIGATION: 0.03,
                    BrainRegion.LIMBIC: 0.02
                },
                sparsity=0.02,
                tau=0.005,  # Fast (10ms)
                acetylcholine_sensitivity=0.8,  # Attention modulation
                plasticity_rate=0.8
            )
        
        elif region_type == BrainRegion.MOTOR:
            return cls(
                region_type=region_type,
                n_neurons=n_neurons,
                local_density=0.25,
                target_regions={BrainRegion.MOTOR},  # Motor loops
                projection_densities={BrainRegion.MOTOR: 0.1},
                sparsity=0.03,
                tau=0.01,  # Medium (20ms)
                dopamine_sensitivity=0.9,  # Reward-based learning
                plasticity_rate=1.2
            )
        
        elif region_type == BrainRegion.REWARD:
            return cls(
                region_type=region_type,
                n_neurons=n_neurons,
                local_density=0.15,
                target_regions={BrainRegion.MOTOR, BrainRegion.ASSOCIATION, BrainRegion.LIMBIC},
                projection_densities={
                    BrainRegion.MOTOR: 0.08,
                    BrainRegion.ASSOCIATION: 0.06,
                    BrainRegion.LIMBIC: 0.05
                },
                sparsity=0.01,  # Very sparse (dopamine neurons)
                tau=0.05,  # Slow (100ms)
                dopamine_sensitivity=0.3,  # Self-modulated
                plasticity_rate=0.5
            )
        
        elif region_type == BrainRegion.NAVIGATION:
            return cls(
                region_type=region_type,
                n_neurons=n_neurons,
                local_density=0.4,  # Very dense (grid cells, place cells)
                target_regions={BrainRegion.MOTOR, BrainRegion.ASSOCIATION, BrainRegion.REWARD},
                projection_densities={
                    BrainRegion.MOTOR: 0.05,
                    BrainRegion.ASSOCIATION: 0.07,
                    BrainRegion.REWARD: 0.03
                },
                sparsity=0.04,  # More active (spatial representation)
                tau=0.02,
                acetylcholine_sensitivity=0.7,
                plasticity_rate=1.5  # Fast spatial learning
            )
        
        elif region_type == BrainRegion.LIMBIC:
            return cls(
                region_type=region_type,
                n_neurons=n_neurons,
                local_density=0.2,
                target_regions={BrainRegion.REWARD, BrainRegion.ASSOCIATION, BrainRegion.MOTOR},
                projection_densities={
                    BrainRegion.REWARD: 0.08,
                    BrainRegion.ASSOCIATION: 0.06,
                    BrainRegion.MOTOR: 0.04
                },
                sparsity=0.03,
                tau=0.1,  # Very slow (emotional persistence)
                serotonin_sensitivity=0.9,  # Mood regulation
                norepinephrine_sensitivity=0.8,  # Arousal/stress
                plasticity_rate=0.6
            )
        
        elif region_type == BrainRegion.ASSOCIATION:
            return cls(
                region_type=region_type,
                n_neurons=n_neurons,
                local_density=0.2,
                target_regions={BrainRegion.MOTOR, BrainRegion.WORKING_MEMORY, BrainRegion.LANGUAGE},
                projection_densities={
                    BrainRegion.MOTOR: 0.06,
                    BrainRegion.WORKING_MEMORY: 0.08,
                    BrainRegion.LANGUAGE: 0.05
                },
                sparsity=0.02,
                tau=0.03,
                acetylcholine_sensitivity=0.7,
                dopamine_sensitivity=0.6,
                plasticity_rate=1.0
            )
        
        elif region_type == BrainRegion.WORKING_MEMORY:
            return cls(
                region_type=region_type,
                n_neurons=n_neurons,
                local_density=0.3,  # Recurrent loops
                target_regions={BrainRegion.ASSOCIATION, BrainRegion.MOTOR},
                projection_densities={
                    BrainRegion.ASSOCIATION: 0.07,
                    BrainRegion.MOTOR: 0.05
                },
                sparsity=0.03,
                tau=0.05,  # Persistent activity
                dopamine_sensitivity=0.7,  # WM gating
                plasticity_rate=0.8
            )
        
        elif region_type == BrainRegion.LANGUAGE:
            return cls(
                region_type=region_type,
                n_neurons=n_neurons,
                local_density=0.25,
                target_regions={BrainRegion.MOTOR, BrainRegion.ASSOCIATION},
                projection_densities={
                    BrainRegion.MOTOR: 0.06,
                    BrainRegion.ASSOCIATION: 0.08
                },
                sparsity=0.02,
                tau=0.02,
                acetylcholine_sensitivity=0.8,
                plasticity_rate=1.1
            )
        
        else:
            # Default region
            return cls(
                region_type=region_type,
                n_neurons=n_neurons,
                local_density=0.2,
                target_regions=set(),
                sparsity=0.02,
                tau=0.01
            )


class SpatialBrainRegions:
    """
    Manages spatial organization of brain into functional regions.
    
    Key features:
    - Each region has specific properties and connectivity patterns
    - Within-region connections are dense
    - Between-region connections are sparse and structured
    - Different neuromodulator sensitivity per region
    - Neurons can migrate between regions (rare)
    """
    
    def __init__(self, total_neurons: int):
        """
        Initialize spatial brain organization.
        
        Args:
            total_neurons: Total number of neurons to partition
        """
        self.total_neurons = total_neurons
        
        # Define regions and their sizes
        # Allocate neurons proportionally to "biological" ratios
        self.regions: Dict[BrainRegion, RegionParams] = {}
        self._initialize_regions()
        
        # Neuron assignments (neuron_id → region)
        self.neuron_to_region: Dict[int, BrainRegion] = {}
        self._assign_neurons_to_regions()
        
        # Connection matrices (sparse representation)
        # (from_neuron_id, to_neuron_id) → weight
        self.connections: Dict[Tuple[int, int], float] = {}
        
        # Regional activity tracking
        self.region_activity: Dict[BrainRegion, float] = {
            region: 0.0 for region in BrainRegion
        }
    
    def _initialize_regions(self):
        """Create regions with default parameters."""
        # Allocate neurons by approximate biological ratios
        allocations = {
            BrainRegion.VISUAL: 0.20,           # 20%
            BrainRegion.MOTOR: 0.15,            # 15%
            BrainRegion.ASSOCIATION: 0.25,      # 25% (largest - prefrontal cortex)
            BrainRegion.NAVIGATION: 0.10,       # 10%
            BrainRegion.REWARD: 0.05,           # 5% (small but critical)
            BrainRegion.LIMBIC: 0.08,           # 8%
            BrainRegion.WORKING_MEMORY: 0.10,   # 10%
            BrainRegion.LANGUAGE: 0.07          # 7%
        }
        
        for region_type, fraction in allocations.items():
            n_neurons = max(10, int(self.total_neurons * fraction))
            params = RegionParams.for_region_type(region_type, n_neurons)
            self.regions[region_type] = params
    
    def _assign_neurons_to_regions(self):
        """Assign each neuron to a region."""
        neuron_id = 0
        for region_type, params in self.regions.items():
            for _ in range(params.n_neurons):
                self.neuron_to_region[neuron_id] = region_type
                neuron_id += 1
    
    def get_region_for_neuron(self, neuron_id: int) -> Optional[BrainRegion]:
        """Get the region a neuron belongs to."""
        return self.neuron_to_region.get(neuron_id)
    
    def get_neurons_in_region(self, region: BrainRegion) -> List[int]:
        """Get all neuron IDs in a specific region."""
        return [nid for nid, reg in self.neuron_to_region.items() if reg == region]
    
    def should_connect(self, from_neuron: int, to_neuron: int) -> bool:
        """
        Determine if two neurons should be connected based on regional rules.
        
        Args:
            from_neuron: Source neuron ID
            to_neuron: Target neuron ID
        
        Returns:
            True if connection is anatomically plausible
        """
        from_region = self.get_region_for_neuron(from_neuron)
        to_region = self.get_region_for_neuron(to_neuron)
        
        if from_region is None or to_region is None:
            return False
        
        from_params = self.regions[from_region]
        
        if from_region == to_region:
            # Within-region connection - use local density
            return np.random.random() < from_params.local_density
        
        else:
            # Between-region connection - check if allowed
            if to_region not in from_params.target_regions:
                return False
            
            # Use projection density
            density = from_params.projection_densities.get(to_region, 0.0)
            return np.random.random() < density
    
    def get_connection_weight_bias(
        self,
        from_neuron: int,
        to_neuron: int,
        base_weight: float
    ) -> float:
        """
        Get region-specific weight bias.
        
        Some regions naturally have stronger/weaker connections.
        
        Args:
            from_neuron: Source neuron
            to_neuron: Target neuron
            base_weight: Base weight before regional scaling
        
        Returns:
            Scaled weight
        """
        from_region = self.get_region_for_neuron(from_neuron)
        to_region = self.get_region_for_neuron(to_neuron)
        
        if from_region is None or to_region is None:
            return base_weight
        
        # Same region - stronger connections
        if from_region == to_region:
            return base_weight * 1.5
        
        # Cross-region - moderate
        return base_weight * 1.0
    
    def get_neuromodulator_sensitivity(
        self,
        neuron_id: int,
        modulator: str
    ) -> float:
        """
        Get how sensitive a neuron is to a specific neuromodulator.
        
        Args:
            neuron_id: Neuron ID
            modulator: Neuromodulator name ('dopamine', 'serotonin', etc.)
        
        Returns:
            Sensitivity multiplier (0-1)
        """
        region = self.get_region_for_neuron(neuron_id)
        if region is None:
            return 0.5  # Default
        
        params = self.regions[region]
        
        if modulator == 'dopamine':
            return params.dopamine_sensitivity
        elif modulator == 'serotonin':
            return params.serotonin_sensitivity
        elif modulator == 'norepinephrine':
            return params.norepinephrine_sensitivity
        elif modulator == 'acetylcholine':
            return params.acetylcholine_sensitivity
        else:
            return 0.5
    
    def get_plasticity_rate(self, neuron_id: int) -> float:
        """Get region-specific learning rate."""
        region = self.get_region_for_neuron(neuron_id)
        if region is None:
            return 1.0
        return self.regions[region].plasticity_rate
    
    def update_region_activity(self, neuron_activities: np.ndarray):
        """
        Update activity levels for each region.
        
        Args:
            neuron_activities: Array of neuron activation values
        """
        # Compute mean activity per region
        for region in BrainRegion:
            neurons_in_region = self.get_neurons_in_region(region)
            if neurons_in_region:
                activities = [neuron_activities[nid] for nid in neurons_in_region 
                             if nid < len(neuron_activities)]
                if activities:
                    self.region_activity[region] = np.mean(activities)
                else:
                    self.region_activity[region] = 0.0
            else:
                self.region_activity[region] = 0.0
    
    def get_region_stats(self) -> Dict[str, Dict[str, any]]:
        """Get statistics for all regions."""
        stats = {}
        for region, params in self.regions.items():
            neurons = self.get_neurons_in_region(region)
            stats[region.value] = {
                'n_neurons': len(neurons),
                'activity': self.region_activity[region],
                'sparsity': params.sparsity,
                'tau': params.tau,
                'local_density': params.local_density,
                'dopamine_sensitivity': params.dopamine_sensitivity,
                'serotonin_sensitivity': params.serotonin_sensitivity,
                'plasticity_rate': params.plasticity_rate,
                'target_regions': [r.value for r in params.target_regions]
            }
        return stats
    
    def migrate_neuron(self, neuron_id: int, new_region: BrainRegion):
        """
        Migrate a neuron to a different region (rare event).
        
        This simulates neuronal migration during development or
        structural plasticity in adults.
        """
        old_region = self.get_region_for_neuron(neuron_id)
        if old_region is not None:
            # Update region neuron counts
            self.regions[old_region].n_neurons -= 1
            self.regions[new_region].n_neurons += 1
        
        # Update assignment
        self.neuron_to_region[neuron_id] = new_region
