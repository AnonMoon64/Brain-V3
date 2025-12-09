"""
Integrated Brain System - Mouse-Level Complexity
=================================================

This module integrates all advanced neural systems into a unified brain:
- Kinetic neuromodulation with receptor binding dynamics
- Hierarchical cortical architecture (6 layers, columns, minicolumns)
- Neuron-specific metabolism and homeostatic plasticity
- Sparse representations with k-winners-take-all
- Event-driven simulation for efficiency
- Reservoir computing layer for complex dynamics
- Hierarchical time scales (sensory to memory)
- Neural language decoder for coherent output

Target: ~1M effective neurons (sparse), mouse-level complexity on laptop hardware
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
import time

# Import all subsystems using local module names
from .neuromodulation import (
    KineticNeuromodulationSystem as NeuromodulationSystem, 
    ThreeFactorLearning, 
    ModulatorType as NeuromodulatorType
)
from .cortical_architecture import (
    HierarchicalCortex, Minicolumn, TopographicMap
)
from .metabolism import MetabolicNetwork, SynapticScaling, IntrinsicPlasticity
from .signal_processing import RobustInputPipeline as SignalProcessor
from .language_decoder import NeuralLanguageDecoder
from .sparse_network import SparseNetwork as SparseNeuralNetwork, EventDrivenSimulator
from .gpu_acceleration import GPUNeuralNetwork as GPUAccelerator
from .reservoir import HybridReservoir
from .hierarchical_time import HierarchicalTimeManager, TemporalScale


class BrainRegion(Enum):
    """Major brain regions with different processing characteristics."""
    SENSORY = "sensory"          # Primary sensory cortex
    MOTOR = "motor"              # Primary motor cortex
    ASSOCIATION = "association"  # Higher association areas
    PREFRONTAL = "prefrontal"    # Executive function
    HIPPOCAMPAL = "hippocampal"  # Memory consolidation
    THALAMUS = "thalamus"        # Relay and gating
    BASAL_GANGLIA = "basal_ganglia"  # Action selection
    CEREBELLUM = "cerebellum"    # Timing and coordination


@dataclass
class BrainConfig:
    """Configuration for the integrated brain."""
    # Scale parameters
    neurons_per_column: int = 100
    columns_per_region: int = 100
    num_regions: int = 8
    
    # Sparsity and efficiency
    target_sparsity: float = 0.02  # 2% active neurons
    use_gpu: bool = False
    use_quantization: bool = True
    
    # Reservoir parameters
    reservoir_size: int = 5000
    spectral_radius: float = 0.9
    reservoir_sparsity: float = 0.1
    
    # Time scales (in seconds)
    sensory_dt: float = 0.001   # 1ms
    fast_dt: float = 0.01       # 10ms
    slow_dt: float = 0.1        # 100ms
    memory_dt: float = 1.0      # 1s
    
    # Language output
    vocabulary_size: int = 10000
    embedding_dim: int = 300
    beam_width: int = 5
    
    # Neuromodulation
    dopamine_baseline: float = 0.5
    acetylcholine_baseline: float = 0.5
    norepinephrine_baseline: float = 0.3
    serotonin_baseline: float = 0.5
    
    # Metabolism
    base_atp_production: float = 100.0
    firing_cost: float = 1.0
    maintenance_cost: float = 0.1
    
    # Learning
    learning_rate: float = 0.01
    stdp_window: float = 0.02  # 20ms
    eligibility_decay: float = 0.99
    
    # Dynamic neurogenesis settings (proof of concept mode)
    dynamic_neurons: bool = True  # Enable neuron creation/death
    initial_neurons_fraction: float = 0.1  # Start with 10% of max neurons
    max_neurons_per_region: int = 10000  # Maximum neurons per region
    neurogenesis_rate: float = 0.2  # Probability of creating neuron on novelty (was 0.05)
    pruning_threshold: float = 0.001  # Neurons with activity below this die
    pruning_interval: int = 100  # Steps between pruning
    
    def total_neurons(self) -> int:
        """Calculate total neuron count."""
        return self.neurons_per_column * self.columns_per_region * self.num_regions


@dataclass
class BrainState:
    """Current state of the integrated brain."""
    # Activity
    global_activity: float = 0.0
    region_activity: Dict[BrainRegion, float] = field(default_factory=dict)
    
    # Neuromodulators
    dopamine_level: float = 0.5
    acetylcholine_level: float = 0.5
    norepinephrine_level: float = 0.3
    serotonin_level: float = 0.5
    
    # Metabolic
    total_energy: float = 100.0
    energy_deficit: bool = False
    
    # Temporal
    current_time: float = 0.0
    simulation_step: int = 0
    
    # Output
    last_output: str = ""
    output_confidence: float = 0.0
    
    # Dynamic neurogenesis tracking
    neurons_created: int = 0
    neurons_pruned: int = 0
    total_active_neurons: int = 0
    novelty_level: float = 0.0  # How novel is current input
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'global_activity': self.global_activity,
            'region_activity': {r.value: v for r, v in self.region_activity.items()},
            'dopamine': self.dopamine_level,
            'acetylcholine': self.acetylcholine_level,
            'norepinephrine': self.norepinephrine_level,
            'serotonin': self.serotonin_level,
            'energy': self.total_energy,
            'time': self.current_time,
            'step': self.simulation_step,
            'output': self.last_output,
            'confidence': self.output_confidence,
            'neurons_created': self.neurons_created,
            'neurons_pruned': self.neurons_pruned,
            'total_active_neurons': self.total_active_neurons,
            'novelty': self.novelty_level
        }


@dataclass
class SimpleColumn:
    """Simple cortical column representation."""
    column_id: str
    position: Tuple[int, int, int]
    neurons_per_minicolumn: int = 10


class RegionalProcessor:
    """
    Processor for a single brain region with dynamic neurogenesis.
    
    Key concept: Neurons spawn when encountering novel patterns,
    and die when unused. Memory IS structure.
    """
    
    def __init__(
        self,
        region: BrainRegion,
        config: BrainConfig,
        time_scale: Optional[TemporalScale] = None
    ):
        self.region = region
        self.config = config
        self.time_scale = time_scale or TemporalScale("default", 0.001, 1000)
        
        # Create simple column representations for this region
        self.columns: List[SimpleColumn] = []
        for i in range(config.columns_per_region):
            col = SimpleColumn(
                column_id=f"{region.value}_{i}",
                position=(i % 10, i // 10, 0),  # 10x10 grid
                neurons_per_minicolumn=config.neurons_per_column // 10
            )
            self.columns.append(col)
        
        # Dynamic neurogenesis: start with fraction of max neurons
        if config.dynamic_neurons:
            self.max_neurons = config.max_neurons_per_region
            initial_neurons = max(100, int(self.max_neurons * config.initial_neurons_fraction))
        else:
            initial_neurons = config.neurons_per_column * config.columns_per_region
            self.max_neurons = initial_neurons
        
        self.num_neurons = initial_neurons
        
        # Weight matrix grows dynamically
        self.input_weights = np.random.randn(initial_neurons, config.embedding_dim) * 0.1
        self.k_winners = max(1, int(initial_neurons * config.target_sparsity))
        
        # State arrays (can grow)
        self.activity = np.zeros(initial_neurons)
        self.membrane_potential = np.zeros(initial_neurons)
        self.last_spike_time = np.full(initial_neurons, -np.inf)
        
        # Track neuron usage for pruning
        self.neuron_usage = np.zeros(initial_neurons)  # Cumulative activity
        self.neuron_age = np.zeros(initial_neurons, dtype=int)  # Steps since creation
        
        # Statistics
        self.neurons_created = 0
        self.neurons_pruned = 0
        
        # Inter-column connectivity
        self._setup_lateral_connections()
    
    def _setup_lateral_connections(self):
        """Setup lateral inhibition and excitation between columns."""
        n_cols = len(self.columns)
        self.lateral_weights = np.zeros((n_cols, n_cols))
        
        for i in range(n_cols):
            for j in range(n_cols):
                if i == j:
                    continue
                # Distance-based connectivity
                pos_i = np.array(self.columns[i].position)
                pos_j = np.array(self.columns[j].position)
                dist = np.linalg.norm(pos_i - pos_j)
                
                # Mexican hat: excitation nearby, inhibition further
                if dist < 2:
                    self.lateral_weights[i, j] = 0.1 * np.exp(-dist)
                else:
                    self.lateral_weights[i, j] = -0.05 * np.exp(-(dist - 2) / 3)
    
    def _k_winners_take_all(self, activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply k-winners-take-all sparsity."""
        k = min(self.k_winners, len(activations))
        if k >= len(activations):
            return activations, np.arange(len(activations))
        
        # Find top k indices
        indices = np.argpartition(activations, -k)[-k:]
        
        # Create sparse output
        sparse_output = np.zeros_like(activations)
        sparse_output[indices] = activations[indices]
        
        return sparse_output, indices
    
    def create_neurons(self, num_new: int, input_pattern: np.ndarray) -> int:
        """
        Create new neurons tuned to the input pattern (neurogenesis).
        Called when encountering novel patterns.
        
        Returns number of neurons actually created.
        """
        if not self.config.dynamic_neurons:
            return 0
            
        # Check capacity
        available = self.max_neurons - self.num_neurons
        num_new = min(num_new, available)
        
        if num_new <= 0:
            return 0
        
        # Create neurons tuned to current input pattern (with noise)
        new_weights = np.tile(input_pattern, (num_new, 1)) * 0.3
        new_weights += np.random.randn(num_new, self.config.embedding_dim) * 0.05
        
        # Expand arrays
        self.input_weights = np.vstack([self.input_weights, new_weights])
        self.activity = np.concatenate([self.activity, np.zeros(num_new)])
        self.membrane_potential = np.concatenate([self.membrane_potential, np.zeros(num_new)])
        self.last_spike_time = np.concatenate([self.last_spike_time, np.full(num_new, -np.inf)])
        self.neuron_usage = np.concatenate([self.neuron_usage, np.zeros(num_new)])
        self.neuron_age = np.concatenate([self.neuron_age, np.zeros(num_new, dtype=int)])
        
        self.num_neurons += num_new
        self.neurons_created += num_new
        
        # Update k for sparse coding
        self.k_winners = max(1, int(self.num_neurons * self.config.target_sparsity))
        
        return num_new
    
    def prune_neurons(self) -> int:
        """
        Remove neurons with low activity (synaptic pruning).
        Called periodically to clean up unused neurons.
        
        Returns number of neurons pruned.
        """
        if not self.config.dynamic_neurons:
            return 0
            
        if self.num_neurons <= 100:  # Keep minimum neurons
            return 0
        
        # Calculate average usage per age
        avg_usage = self.neuron_usage / (self.neuron_age + 1)
        
        # Find neurons below threshold (but protect young neurons)
        old_enough = self.neuron_age > 50  # Don't prune neurons younger than 50 steps
        below_threshold = avg_usage < self.config.pruning_threshold
        to_prune = old_enough & below_threshold
        
        # Keep at least 100 neurons
        keep_indices = np.where(~to_prune)[0]
        if len(keep_indices) < 100:
            # Keep the top 100 by usage
            keep_indices = np.argsort(avg_usage)[-100:]
            to_prune = np.ones(self.num_neurons, dtype=bool)
            to_prune[keep_indices] = False
        
        num_pruned = int(np.sum(to_prune))
        
        if num_pruned == 0:
            return 0
        
        # Remove pruned neurons
        self.input_weights = self.input_weights[~to_prune]
        self.activity = self.activity[~to_prune]
        self.membrane_potential = self.membrane_potential[~to_prune]
        self.last_spike_time = self.last_spike_time[~to_prune]
        self.neuron_usage = self.neuron_usage[~to_prune]
        self.neuron_age = self.neuron_age[~to_prune]
        
        self.num_neurons = len(self.activity)
        self.neurons_pruned += num_pruned
        
        # Update k for sparse coding
        self.k_winners = max(1, int(self.num_neurons * self.config.target_sparsity))
        
        return num_pruned
    
    def compute_novelty(self, input_activity: np.ndarray) -> float:
        """
        Compute how novel the input is (low match to existing neurons = high novelty).
        """
        if self.num_neurons == 0:
            return 1.0
            
        # Compute similarity to best matching neurons
        similarities = np.dot(self.input_weights, input_activity)
        max_similarity = np.max(similarities) if len(similarities) > 0 else 0
        
        # Normalize (assuming inputs are roughly unit norm)
        norm = np.linalg.norm(input_activity) + 1e-8
        max_similarity /= norm
        
        # Convert to novelty (1 = very novel, 0 = familiar)
        novelty = 1.0 - np.clip(max_similarity, 0, 1)
        return novelty
    
    def process(
        self,
        input_activity: np.ndarray,
        neuromodulator_levels: Dict[str, float],
        dt: float
    ) -> np.ndarray:
        """Process input through cortical columns with sparse coding."""
        # Apply neuromodulation to gain
        gain = 1.0 + 0.5 * neuromodulator_levels.get('acetylcholine', 0.5)
        
        # Ensure input is right size
        if len(input_activity) < self.config.embedding_dim:
            padded = np.zeros(self.config.embedding_dim)
            padded[:len(input_activity)] = input_activity
            input_activity = padded
        elif len(input_activity) > self.config.embedding_dim:
            input_activity = input_activity[:self.config.embedding_dim]
        
        # Compute activations
        raw_activations = np.dot(self.input_weights, input_activity * gain)
        raw_activations = np.maximum(0, raw_activations)  # ReLU
        
        # Apply k-winners-take-all
        sparse_output, indices = self._k_winners_take_all(raw_activations)
        
        # Update membrane potentials (simplified LIF)
        leak = np.exp(-dt / 0.02)  # 20ms membrane time constant
        self.membrane_potential *= leak
        
        # Add input current
        self.membrane_potential += sparse_output * dt * 10  # Scale factor
        
        # Threshold and spike
        threshold = 1.0
        spikes = self.membrane_potential > threshold
        self.membrane_potential[spikes] = 0.0  # Reset
        self.activity = spikes.astype(float)
        
        # Track neuron usage for pruning decisions
        self.neuron_usage += self.activity
        self.neuron_age += 1
        
        return self.activity
    
    def process_with_neurogenesis(
        self,
        input_activity: np.ndarray,
        neuromodulator_levels: Dict[str, float],
        dt: float,
        step: int
    ) -> Tuple[np.ndarray, int, int]:
        """
        Process input with dynamic neurogenesis.
        
        Returns:
            activity: Neural activity pattern
            neurons_created: Number of neurons created this step
            neurons_pruned: Number of neurons pruned this step
        """
        neurons_created = 0
        neurons_pruned = 0
        
        # Ensure input is right size
        if len(input_activity) < self.config.embedding_dim:
            padded = np.zeros(self.config.embedding_dim)
            padded[:len(input_activity)] = input_activity
            input_activity = padded
        elif len(input_activity) > self.config.embedding_dim:
            input_activity = input_activity[:self.config.embedding_dim]
        
        # Check novelty - should we create new neurons?
        if self.config.dynamic_neurons:
            novelty = self.compute_novelty(input_activity)
            
            # High novelty + dopamine promotes neurogenesis
            dopamine = neuromodulator_levels.get('dopamine', 0.5)
            cortisol = neuromodulator_levels.get('cortisol', 0.3)
            
            # Neurogenesis probability: high novelty + high dopamine + low cortisol
            # Surprise triggers neuron birth (like original brain.py grow() method)
            neurogenesis_prob = novelty * (0.5 + dopamine) * (1 - cortisol * 0.3)
            
            if novelty > 0.3 and np.random.random() < neurogenesis_prob * self.config.neurogenesis_rate:
                # Create 1-5 neurons tuned to this novel pattern
                num_new = np.random.randint(1, 6)
                neurons_created = self.create_neurons(num_new, input_activity)
            
            # Periodic pruning
            if step > 0 and step % self.config.pruning_interval == 0:
                neurons_pruned = self.prune_neurons()
        
        # Regular processing
        activity = self.process(input_activity, neuromodulator_levels, dt)
        
        return activity, neurons_created, neurons_pruned
    
    def get_column_activities(self) -> np.ndarray:
        """Get mean activity per column."""
        neurons_per_col = self.config.neurons_per_column
        activities = np.zeros(len(self.columns))
        for i in range(len(self.columns)):
            start = i * neurons_per_col
            end = min(start + neurons_per_col, self.num_neurons)
            if start < self.num_neurons:
                activities[i] = np.mean(self.activity[start:end])
        return activities


class IntegratedBrain:
    """
    Unified brain system integrating all advanced neural mechanisms.
    
    Architecture:
    - Multiple brain regions with cortical column organization
    - Kinetic neuromodulation affecting learning and gain
    - Sparse, event-driven processing for efficiency
    - Reservoir layer for rich temporal dynamics
    - Hierarchical time scales for multi-rate processing
    - Neural language decoder for coherent text output
    
    Dynamic Neurogenesis:
    - Neurons spawn when encountering novel patterns
    - Neurons die when unused or causing conflict
    - Memory IS structure - the network topology encodes memories
    """
    
    def __init__(self, config: Optional[BrainConfig] = None):
        self.config = config or BrainConfig()
        self.state = BrainState()
        
        if self.config.dynamic_neurons:
            initial_neurons = int(self.config.max_neurons_per_region * 
                                  self.config.initial_neurons_fraction * 
                                  self.config.num_regions)
            print(f"Initializing Dynamic Brain with {initial_neurons:,} initial neurons...")
            print(f"  (Can grow up to {self.config.max_neurons_per_region * self.config.num_regions:,} neurons)")
        else:
            print(f"Initializing Integrated Brain with {self.config.total_neurons():,} neurons...")
        
        # Initialize subsystems
        self._init_time_manager()
        self._init_regions()
        self._init_neuromodulation()
        self._init_metabolism()
        self._init_reservoir()
        self._init_signal_processing()
        self._init_language_decoder()
        self._init_gpu_accelerator()
        
        # Inter-region connectivity
        self._setup_region_connections()
        
        # Count actual neurons after dynamic initialization
        actual_neurons = sum(proc.num_neurons for proc in self.regions.values())
        self.state.total_active_neurons = actual_neurons
        
        print(f"Brain initialization complete. Active neurons: {actual_neurons:,}")
    
    def _init_time_manager(self):
        """Initialize hierarchical time manager."""
        self.time_manager = HierarchicalTimeManager()
        
        # Add time scales for different processing levels
        self.time_manager.add_scale(TemporalScale(
            name="sensory",
            base_dt=self.config.sensory_dt,
            update_frequency=1000  # 1000 Hz
        ))
        self.time_manager.add_scale(TemporalScale(
            name="fast",
            base_dt=self.config.fast_dt,
            update_frequency=100  # 100 Hz
        ))
        self.time_manager.add_scale(TemporalScale(
            name="slow",
            base_dt=self.config.slow_dt,
            update_frequency=10  # 10 Hz
        ))
        self.time_manager.add_scale(TemporalScale(
            name="memory",
            base_dt=self.config.memory_dt,
            update_frequency=1  # 1 Hz
        ))
    
    def _init_regions(self):
        """Initialize brain regions with cortical processors."""
        self.regions: Dict[BrainRegion, RegionalProcessor] = {}
        
        # Map regions to time scales
        region_scales = {
            BrainRegion.SENSORY: "sensory",
            BrainRegion.MOTOR: "sensory",
            BrainRegion.THALAMUS: "sensory",
            BrainRegion.ASSOCIATION: "fast",
            BrainRegion.PREFRONTAL: "slow",
            BrainRegion.BASAL_GANGLIA: "fast",
            BrainRegion.HIPPOCAMPAL: "memory",
            BrainRegion.CEREBELLUM: "fast"
        }
        
        for region in BrainRegion:
            scale_name = region_scales.get(region, "fast")
            scale = self.time_manager.get_scale(scale_name)
            self.regions[region] = RegionalProcessor(region, self.config, scale)
            self.state.region_activity[region] = 0.0
    
    def _init_neuromodulation(self):
        """Initialize neuromodulation system."""
        self.neuromod = NeuromodulationSystem()
        
        # Set baseline levels (these methods may not exist, use try/except)
        try:
            self.neuromod.set_baseline(
                NeuromodulatorType.DOPAMINE,
                self.config.dopamine_baseline
            )
            self.neuromod.set_baseline(
                NeuromodulatorType.ACETYLCHOLINE,
                self.config.acetylcholine_baseline
            )
            self.neuromod.set_baseline(
                NeuromodulatorType.NOREPINEPHRINE,
                self.config.norepinephrine_baseline
            )
            self.neuromod.set_baseline(
                NeuromodulatorType.SEROTONIN,
                self.config.serotonin_baseline
            )
        except (AttributeError, TypeError):
            pass  # Method may not exist in all versions
        
        # Three-factor learning rule
        self.three_factor = ThreeFactorLearning(
            num_synapses=self.config.total_neurons() * 100,  # Sparse connectivity
            learning_rate=self.config.learning_rate,
            eligibility_decay=self.config.eligibility_decay
        )
    
    def _init_metabolism(self):
        """Initialize metabolic network."""
        # Create neuron IDs
        neuron_ids = [f"n_{i}" for i in range(min(1000, self.config.total_neurons()))]
        self.metabolism = MetabolicNetwork(neuron_ids=neuron_ids)
        
        # Homeostatic mechanisms
        self.synaptic_scaling = SynapticScaling(
            num_neurons=self.config.total_neurons(),
            target_rate=0.02  # Match sparsity
        )
        
        self.intrinsic_plasticity = IntrinsicPlasticity(
            num_neurons=self.config.total_neurons(),
            target_rate=0.02
        )
    
    def _init_reservoir(self):
        """Initialize reservoir computing layer."""
        self.reservoir = HybridReservoir(
            input_dim=self.config.embedding_dim,
            reservoir_size=self.config.reservoir_size,
            output_dim=self.config.embedding_dim,
            spectral_radius=self.config.spectral_radius,
            sparsity=self.config.reservoir_sparsity
        )
    
    def _init_signal_processing(self):
        """Initialize signal processing pipeline."""
        self.signal_processor = SignalProcessor(
            input_dim=self.config.embedding_dim
        )
    
    def _init_language_decoder(self):
        """Initialize language decoder for output."""
        self.language_decoder = NeuralLanguageDecoder(
            vocabulary_size=self.config.vocabulary_size,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.reservoir_size,
            beam_width=self.config.beam_width
        )
    
    def _init_gpu_accelerator(self):
        """Initialize GPU accelerator if available."""
        # Simple GPU state tracking (actual acceleration handled per-operation)
        self.use_gpu = self.config.use_gpu
        self.use_quantization = self.config.use_quantization
    
    def _setup_region_connections(self):
        """Setup connectivity between brain regions."""
        # Feedforward pathway: sensory -> association -> prefrontal
        # Feedback pathway: prefrontal -> association -> sensory
        # Thalamic relay: thalamus <-> all cortical areas
        # Basal ganglia loop: cortex -> BG -> thalamus -> cortex
        # Hippocampal loop: association <-> hippocampus
        # Cerebellar loop: motor <-> cerebellum
        
        self.region_connections: Dict[Tuple[BrainRegion, BrainRegion], float] = {
            # Feedforward
            (BrainRegion.SENSORY, BrainRegion.ASSOCIATION): 0.3,
            (BrainRegion.ASSOCIATION, BrainRegion.PREFRONTAL): 0.2,
            
            # Feedback
            (BrainRegion.PREFRONTAL, BrainRegion.ASSOCIATION): 0.15,
            (BrainRegion.ASSOCIATION, BrainRegion.SENSORY): 0.1,
            
            # Thalamic relay
            (BrainRegion.THALAMUS, BrainRegion.SENSORY): 0.4,
            (BrainRegion.THALAMUS, BrainRegion.MOTOR): 0.3,
            (BrainRegion.SENSORY, BrainRegion.THALAMUS): 0.2,
            
            # Basal ganglia
            (BrainRegion.PREFRONTAL, BrainRegion.BASAL_GANGLIA): 0.25,
            (BrainRegion.BASAL_GANGLIA, BrainRegion.THALAMUS): 0.2,
            
            # Hippocampal
            (BrainRegion.ASSOCIATION, BrainRegion.HIPPOCAMPAL): 0.2,
            (BrainRegion.HIPPOCAMPAL, BrainRegion.PREFRONTAL): 0.15,
            
            # Cerebellar
            (BrainRegion.MOTOR, BrainRegion.CEREBELLUM): 0.3,
            (BrainRegion.CEREBELLUM, BrainRegion.MOTOR): 0.25,
        }

    def extract_input_signals(self, text: str, network_activity: Dict[str, float]) -> tuple:
        """
        Extract reward and arousal signals from text input and network state.

        Primary driver: Internal network activity (what the brain experiences)
        Secondary modulator: Keyword-based affect detection (linguistic context)

        Args:
            text: Input text to analyze
            network_activity: Dict with 'activation_spread', 'pattern_stability', 'novelty'

        Returns:
            (reward, arousal) tuple, both in range appropriate for modulation
        """
        # === 1. PRIMARY: Internal network-driven signals ===

        # Activation spread: More neurons firing = higher arousal
        activation_spread = network_activity.get('activation_spread', 0.5)
        arousal_base = activation_spread * 2.0  # Scale to [0, 2] before clipping

        # Pattern stability: Good match to existing patterns = reward (prediction success)
        pattern_stability = network_activity.get('pattern_stability', 0.0)
        reward_base = pattern_stability * 2.0 - 1.0  # Scale to [-1, 1]

        # Novelty: High novelty increases arousal (alertness to new patterns)
        novelty = network_activity.get('novelty', 0.0)
        arousal_base += novelty * 0.5

        # === 2. SECONDARY: Keyword-based affect modulation ===

        text_lower = text.lower()
        valence_shift = 0.0
        arousal_shift = 0.0

        # Sentiment concepts from language_decoder (valence, arousal)
        sentiment_concepts = {
            'greeting': (['hello', 'hi', 'hey', 'greetings'], 0.3, 0.4),
            'farewell': (['goodbye', 'bye', 'see you', 'farewell'], 0.1, 0.3),
            'affirmation': (['yes', 'correct', 'right', 'indeed', 'true'], 0.4, 0.3),
            'negation': (['no', 'not', 'never', 'wrong', 'false'], -0.2, 0.4),
            'question': (['what', 'why', 'how', 'when', 'where', 'who'], 0.0, 0.5),
            'gratitude': (['thanks', 'thank you', 'grateful', 'appreciate'], 0.6, 0.4),
            'help': (['help', 'assist', 'support', 'aid'], 0.2, 0.5),
            'understanding': (['understand', 'see', 'know', 'realize', 'get it'], 0.3, 0.3),
            'confusion': (['confused', 'unclear', "don't understand", 'puzzled'], -0.2, 0.5),
            'happiness': (['happy', 'glad', 'pleased', 'delighted', 'joy', 'excited', 'love'], 0.8, 0.6),
            'sadness': (['sad', 'unhappy', 'sorry', 'regret', 'depressed'], -0.6, 0.3),
            'anger': (['angry', 'furious', 'mad', 'annoyed', 'irritated'], -0.5, 0.8),
            'fear': (['afraid', 'scared', 'worried', 'anxious', 'nervous'], -0.4, 0.7),
            'thinking': (['think', 'consider', 'ponder', 'believe', 'suppose'], 0.0, 0.4),
            'feeling': (['feel', 'sense', 'experience', 'emotion'], 0.0, 0.5),
            'curiosity': (['curious', 'interested', 'wonder', 'intrigued'], 0.4, 0.6),
        }

        # Check for keyword matches
        matches = 0
        for concept, (keywords, valence, arsl) in sentiment_concepts.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Stronger modulation weight for keywords (0.5 instead of 0.3)
                    # Keywords are the PRIMARY signal, network is SECONDARY
                    valence_shift += valence * 0.5
                    arousal_shift += (arsl - 0.5) * 0.5
                    matches += 1
                    break  # Only count each concept once

        # Average if multiple matches
        if matches > 0:
            valence_shift /= matches
            arousal_shift /= matches

        # === 3. COMBINE: Keywords drive when present, network fills in ===

        # If we have keyword matches, they should dominate
        # Otherwise, use network-based signals
        if matches > 0:
            # Keywords present: 70% keyword, 30% network
            reward = 0.7 * valence_shift + 0.3 * reward_base
            arousal = 0.7 * (arousal_shift + 0.5) + 0.3 * arousal_base
        else:
            # No keywords: use network signals
            reward = reward_base
            arousal = arousal_base

        # Clip to valid ranges
        reward = np.clip(reward, -1.0, 1.0)
        arousal = np.clip(arousal, 0.0, 1.0)

        return reward, arousal

    def process_input(
        self,
        text_input: str,
        reward: float = 0.0,
        arousal: float = 0.5
    ) -> str:
        """
        Process text input and generate response.
        
        Args:
            text_input: Input text to process
            reward: Reward signal for learning (-1 to 1)
            arousal: Arousal/attention level (0 to 1)
            
        Returns:
            Generated text response
        """
        # Encode input to embedding
        input_embedding = self._encode_input(text_input)
        
        # Process through signal processing
        processed_input = self.signal_processor.process(input_embedding)
        
        # Update neuromodulator levels based on context
        self._update_neuromodulators(reward, arousal)
        
        # Get current neuromodulator state (all 8 chemicals now active!)
        neuromod_levels = {
            'dopamine': self.state.dopamine_level,
            'acetylcholine': self.state.acetylcholine_level,
            'norepinephrine': self.state.norepinephrine_level,
            'serotonin': self.state.serotonin_level,
            'cortisol': getattr(self.state, 'cortisol_level', 0.3),
            'gaba': getattr(self.state, 'gaba_level', 0.4),
            'glutamate': getattr(self.state, 'glutamate_level', 0.5),
            'oxytocin': getattr(self.state, 'oxytocin_level', 0.3),
        }
        
        # Process through time scales
        outputs = self._hierarchical_process(processed_input, neuromod_levels)
        
        # Aggregate regional outputs
        aggregated = self._aggregate_outputs(outputs)
        
        # Process through reservoir for temporal integration
        reservoir_output = self.reservoir.process(aggregated)
        
        # Decode to language
        result = self.language_decoder.decode(reservoir_output)
        if isinstance(result, tuple):
            output_text, confidence = result
        else:
            output_text = result
            confidence = 0.5
        
        # Update state
        self.state.last_output = output_text
        self.state.output_confidence = float(confidence) if not isinstance(confidence, str) else 0.5
        self.state.simulation_step += 1
        
        # Apply learning
        self._apply_learning(reward)
        
        # Update metabolism
        self._update_metabolism()
        
        return output_text
    
    def _encode_input(self, text: str) -> np.ndarray:
        """Encode text input to embedding vector."""
        # Simple character-level encoding (placeholder for real embedding)
        embedding = np.zeros(self.config.embedding_dim)
        
        for i, char in enumerate(text[:self.config.embedding_dim]):
            embedding[i] = ord(char) / 256.0
        
        # Add some structure
        if len(text) > 0:
            # Word count feature
            word_count = len(text.split())
            embedding[-1] = min(word_count / 20.0, 1.0)
            
            # Character count feature
            embedding[-2] = min(len(text) / 200.0, 1.0)
        
        return embedding
    
    def _update_neuromodulators(self, reward: float, arousal: float):
        """
        Update neuromodulator levels based on reward and arousal signals.

        Now supports negative reward (punishment/aversion) which modulates:
        - Dopamine dips (reward prediction error)
        - Serotonin changes (mood)
        - Cortisol increases (stress from negative events)
        """
        dt = self.config.sensory_dt

        # === Primary Neuromodulators ===

        # Dopamine: driven by reward prediction error (supports negative reward!)
        # Positive reward → dopamine spike, Negative reward → dopamine dip
        dopamine_target = self.config.dopamine_baseline + 0.4 * reward
        self.state.dopamine_level += 0.1 * (dopamine_target - self.state.dopamine_level)

        # Acetylcholine: driven by arousal/attention
        # High arousal → more attention/alertness
        ach_target = self.config.acetylcholine_baseline + 0.3 * arousal
        self.state.acetylcholine_level += 0.1 * (ach_target - self.state.acetylcholine_level)

        # Norepinephrine: driven by arousal and surprise
        # High arousal → more alertness/vigilance
        ne_target = self.config.norepinephrine_baseline + 0.2 * arousal
        self.state.norepinephrine_level += 0.05 * (ne_target - self.state.norepinephrine_level)

        # Serotonin: affected by both arousal and reward
        # Low arousal + positive reward → high serotonin (calm, content)
        # High arousal or negative reward → low serotonin (agitated, stressed)
        serotonin_target = self.config.serotonin_baseline - 0.2 * (arousal - 0.5) + 0.2 * max(reward, 0)
        self.state.serotonin_level += 0.05 * (serotonin_target - self.state.serotonin_level)

        # === Secondary Neuromodulators (NEW!) ===
        # These were previously unused, now driven by reward/arousal

        # Cortisol: stress hormone, increases with negative reward and high arousal
        # High cortisol → aggressive pruning, defensive state
        if not hasattr(self.state, 'cortisol_level'):
            self.state.cortisol_level = 0.3  # Initialize if missing

        stress_signal = max(-reward, 0) * 0.5 + max(arousal - 0.7, 0) * 0.3
        cortisol_target = 0.3 + stress_signal  # Baseline 0.3, can spike to ~0.8
        self.state.cortisol_level += 0.08 * (cortisol_target - self.state.cortisol_level)

        # GABA: inhibitory, increases in calm/low arousal states
        # High GABA → reduced noise, stable patterns
        if not hasattr(self.state, 'gaba_level'):
            self.state.gaba_level = 0.4

        gaba_target = 0.6 - 0.3 * arousal  # High when calm
        self.state.gaba_level += 0.06 * (gaba_target - self.state.gaba_level)

        # Glutamate: excitatory, increases with high arousal and novelty
        # High glutamate → more plasticity, faster learning
        if not hasattr(self.state, 'glutamate_level'):
            self.state.glutamate_level = 0.5

        glutamate_target = 0.5 + 0.3 * arousal
        self.state.glutamate_level += 0.08 * (glutamate_target - self.state.glutamate_level)

        # Oxytocin: bonding/trust, increases with positive reward
        # High oxytocin → strengthen connections, social learning
        if not hasattr(self.state, 'oxytocin_level'):
            self.state.oxytocin_level = 0.3

        oxytocin_target = 0.3 + 0.4 * max(reward, 0)  # Only positive reward
        self.state.oxytocin_level += 0.05 * (oxytocin_target - self.state.oxytocin_level)

        # Clamp all to valid range [0, 1]
        self.state.dopamine_level = np.clip(self.state.dopamine_level, 0, 1)
        self.state.acetylcholine_level = np.clip(self.state.acetylcholine_level, 0, 1)
        self.state.norepinephrine_level = np.clip(self.state.norepinephrine_level, 0, 1)
        self.state.serotonin_level = np.clip(self.state.serotonin_level, 0, 1)
        self.state.cortisol_level = np.clip(self.state.cortisol_level, 0, 1)
        self.state.gaba_level = np.clip(self.state.gaba_level, 0, 1)
        self.state.glutamate_level = np.clip(self.state.glutamate_level, 0, 1)
        self.state.oxytocin_level = np.clip(self.state.oxytocin_level, 0, 1)
    
    def _hierarchical_process(
        self,
        input_data: np.ndarray,
        neuromod_levels: Dict[str, float]
    ) -> Dict[BrainRegion, np.ndarray]:
        """Process through hierarchical time scales with dynamic neurogenesis."""
        outputs = {}
        
        # Determine which scales need updating this step
        scales_to_update = self.time_manager.get_active_scales(
            self.state.current_time
        )
        
        # Track neurogenesis across regions
        total_neurons_created = 0
        total_neurons_pruned = 0
        
        # Process each region according to its time scale
        for region, processor in self.regions.items():
            scale_name = processor.time_scale.name
            
            if scale_name in scales_to_update:
                dt = processor.time_scale.base_dt
                
                # Get input for this region (from connected regions)
                region_input = self._get_region_input(region, input_data, outputs)
                
                # Process through cortical columns with neurogenesis
                if self.config.dynamic_neurons:
                    output, created, pruned = processor.process_with_neurogenesis(
                        region_input, 
                        neuromod_levels, 
                        dt,
                        self.state.simulation_step
                    )
                    total_neurons_created += created
                    total_neurons_pruned += pruned
                else:
                    output = processor.process(region_input, neuromod_levels, dt)
                
                outputs[region] = output
                
                # Update region activity in state
                self.state.region_activity[region] = float(np.mean(output))
        
        # Update neurogenesis statistics
        self.state.neurons_created += total_neurons_created
        self.state.neurons_pruned += total_neurons_pruned
        self.state.total_active_neurons = sum(proc.num_neurons for proc in self.regions.values())
        
        # Compute novelty from outputs
        if outputs:
            avg_activity = np.mean([np.mean(np.abs(o)) for o in outputs.values()])
            self.state.novelty_level = 1.0 - min(1.0, avg_activity * 10)
        
        # Update global time
        self.state.current_time += self.config.sensory_dt
        
        return outputs
    
    def _get_region_input(
        self,
        region: BrainRegion,
        external_input: np.ndarray,
        current_outputs: Dict[BrainRegion, np.ndarray]
    ) -> np.ndarray:
        """Get input for a region from external and internal sources."""
        # Start with external input (for sensory regions)
        if region in [BrainRegion.SENSORY, BrainRegion.THALAMUS]:
            result = external_input.copy()
        else:
            result = np.zeros(self.config.embedding_dim)
        
        # Add input from connected regions
        for (source, target), weight in self.region_connections.items():
            if target == region and source in current_outputs:
                source_output = current_outputs[source]
                # Project to embedding dimension
                if len(source_output) > self.config.embedding_dim:
                    projected = source_output[:self.config.embedding_dim]
                else:
                    projected = np.zeros(self.config.embedding_dim)
                    projected[:len(source_output)] = source_output
                result += weight * projected
        
        return result
    
    def _aggregate_outputs(
        self,
        outputs: Dict[BrainRegion, np.ndarray]
    ) -> np.ndarray:
        """Aggregate regional outputs for decoder input."""
        # Weight by region importance for language
        weights = {
            BrainRegion.PREFRONTAL: 0.3,      # Executive/planning
            BrainRegion.ASSOCIATION: 0.25,     # Integration
            BrainRegion.HIPPOCAMPAL: 0.2,      # Memory
            BrainRegion.SENSORY: 0.1,          # Perception
            BrainRegion.MOTOR: 0.05,           # Motor planning (speech)
            BrainRegion.THALAMUS: 0.05,        # Relay
            BrainRegion.BASAL_GANGLIA: 0.03,   # Action selection
            BrainRegion.CEREBELLUM: 0.02       # Coordination
        }
        
        result = np.zeros(self.config.embedding_dim)
        
        for region, output in outputs.items():
            weight = weights.get(region, 0.1)
            # Project to embedding dimension
            if len(output) > self.config.embedding_dim:
                projected = output[:self.config.embedding_dim]
            else:
                projected = np.zeros(self.config.embedding_dim)
                projected[:len(output)] = output
            result += weight * projected
        
        # Update global activity
        self.state.global_activity = float(np.mean(np.abs(result)))
        
        return result
    
    def _apply_learning(self, reward: float):
        """Apply three-factor learning with neuromodulation."""
        # Get dopamine modulation for learning
        dopamine = self.state.dopamine_level
        
        # Modulate learning rate by dopamine
        effective_lr = self.config.learning_rate * (0.5 + dopamine)
        
        # Update eligibility traces
        self.three_factor.update_eligibility(
            decay=self.config.eligibility_decay
        )
        
        # Apply learning if there's significant reward
        if abs(reward) > 0.1:
            self.three_factor.apply_learning(
                reward=reward,
                dopamine=dopamine,
                learning_rate=effective_lr
            )
    
    def _update_metabolism(self):
        """Update metabolic state."""
        # Collect activity across regions
        neuron_activities = {}
        spikes = []
        
        for region, proc in self.regions.items():
            for i, activity in enumerate(proc.activity):
                nid = f"{region.value}_{i}"
                neuron_activities[nid] = float(activity)
                if activity > 0.5:
                    spikes.append(nid)
        
        # Update metabolic network (use simpler step if update fails)
        try:
            self.metabolism.update(
                neuron_activities=neuron_activities,
                spikes=spikes,
                dt=self.config.sensory_dt * 1000  # Convert to ms
            )
        except Exception:
            # Fallback to simple step
            self.metabolism.step(self.config.sensory_dt)
        
        # Update state
        self.state.total_energy = self.metabolism.total_atp
        self.state.energy_deficit = self.metabolism.total_atp < 50.0
        
        # Apply homeostatic plasticity if energy is low
        if self.state.energy_deficit:
            self.synaptic_scaling.scale_down(factor=0.99)
    
    def step(self, dt: Optional[float] = None) -> BrainState:
        """
        Advance simulation by one time step.
        
        Args:
            dt: Time step (uses sensory_dt if not provided)
            
        Returns:
            Current brain state
        """
        if dt is None:
            dt = self.config.sensory_dt
        
        # Update time
        self.state.current_time += dt
        self.state.simulation_step += 1
        
        # Update neuromodulator dynamics
        self.neuromod.update(dt)
        
        # Update metabolism
        self.metabolism.step(dt)
        
        # Update homeostatic mechanisms
        self.synaptic_scaling.update(dt)
        self.intrinsic_plasticity.update(dt)
        
        return self.state
    
    def get_state(self) -> BrainState:
        """Get current brain state."""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about brain activity."""
        return {
            'state': self.state.to_dict(),
            'config': {
                'total_neurons': self.config.total_neurons(),
                'target_sparsity': self.config.target_sparsity,
                'reservoir_size': self.config.reservoir_size,
                'vocabulary_size': self.config.vocabulary_size
            },
            'regions': {
                region.value: {
                    'activity': float(np.mean(proc.activity)),
                    'active_neurons': int(np.sum(proc.activity > 0)),
                    'columns': len(proc.columns)
                }
                for region, proc in self.regions.items()
            },
            'metabolism': {
                'total_atp': self.metabolism.total_atp,
                'deficit': self.state.energy_deficit
            },
            'neuromodulation': {
                'dopamine': self.state.dopamine_level,
                'acetylcholine': self.state.acetylcholine_level,
                'norepinephrine': self.state.norepinephrine_level,
                'serotonin': self.state.serotonin_level
            }
        }
    
    def save_state(self, filepath: str):
        """Save brain state to file."""
        import json
        state_dict = {
            'state': self.state.to_dict(),
            'config': {k: v for k, v in self.config.__dict__.items() 
                      if not callable(v)},
        }
        # Save as JSON for compatibility
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2, default=str)
    
    def load_state(self, filepath: str):
        """Load brain state from file."""
        import json
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        # Restore state (simplified)
        for key, value in state_dict.get('state', {}).items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
    
    # =========================================================================
    # ChemicalBrain-Compatible Interface (for GUI)
    # =========================================================================
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process input text - ChemicalBrain compatible interface.

        Args:
            text: Input text to process

        Returns:
            Dict with response and metadata
        """
        # Compute network activity metrics BEFORE processing
        # This allows us to extract reward/arousal from what the brain is experiencing
        total_neurons = sum(proc.num_neurons for proc in self.regions.values())
        active_neurons = sum(np.sum(proc.activity > 0) for proc in self.regions.values())
        activation_spread = active_neurons / max(total_neurons, 1)

        # Compute novelty for this input (before processing)
        input_embedding = self._encode_input(text)
        processed_input = self.signal_processor.process(input_embedding)

        # Measure pattern stability (how well it matches existing patterns)
        # Use the sensory region's novelty computation as inverse of stability
        sensory_proc = self.regions.get(BrainRegion.SENSORY)
        if sensory_proc:
            novelty = sensory_proc.compute_novelty(processed_input)
            pattern_stability = 1.0 - novelty  # High novelty = low stability
        else:
            novelty = 0.5
            pattern_stability = 0.5

        # Create network activity dict for signal extraction
        network_activity = {
            'activation_spread': activation_spread,
            'pattern_stability': pattern_stability,
            'novelty': novelty,
        }

        # Extract reward and arousal from input + network state
        reward, arousal = self.extract_input_signals(text, network_activity)

        # Now process with the extracted signals
        response = self.process_input(text, reward=reward, arousal=arousal)
        
        # Determine mood from neuromodulator levels
        mood = self._compute_mood()
        
        # Compute growth stats (neurons born/died this interaction)
        total_neurons = sum(proc.num_neurons for proc in self.regions.values())
        
        # Extract concepts from input (simple word extraction)
        words = text.lower().split()
        concepts = [w.strip('.,!?;:"\'()[]{}') for w in words if len(w) > 3][:5]
        
        return {
            'response': response,
            'mood': mood,
            'confidence': self.state.output_confidence,
            'dopamine': self.state.dopamine_level,
            'serotonin': self.state.serotonin_level,
            'energy': self.state.total_energy,
            'growth_stats': {
                'neurons_born': self.state.neurons_created,
                'neurons_died': self.state.neurons_pruned,
                'synapses_formed': 0,
                'synapses_pruned': 0,
            },
            'network_stats': {
                'total_neurons': total_neurons,
                'total_synapses': total_neurons * 50,
                'global_energy': self.state.total_energy / 100.0,
            },
            'surprise': self.state.novelty_level,
            'concepts_detected': concepts if concepts else ['general'],
        }
    
    def train(self, input_text: str, target_output: str) -> Dict[str, Any]:
        """
        Train the brain on an input-output pair.
        ChemicalBrain compatible interface.
        
        Args:
            input_text: User input
            target_output: Expected response
            
        Returns:
            Training result dictionary
        """
        # Encode both texts
        input_embedding = self._encode_input(input_text)
        target_embedding = self._encode_input(target_output)
        
        # Process input first to activate network
        _ = self.process_input(input_text)
        
        # Train language decoder on this pair
        if hasattr(self.language_decoder, 'train'):
            self.language_decoder.train(input_text, target_output, reward=1.0)
        
        # Apply positive reward signal for learning
        self._apply_learning(reward=0.5)
        
        # Dopamine burst for successful training
        self.state.dopamine_level = min(1.0, self.state.dopamine_level + 0.1)
        
        mood = self._compute_mood()
        
        return {
            'mood': mood,
            'trained': True,
            'input': input_text,
            'output': target_output
        }
    
    def _compute_mood(self) -> str:
        """Compute mood from neuromodulator levels."""
        da = self.state.dopamine_level
        se = self.state.serotonin_level
        ne = self.state.norepinephrine_level
        ach = self.state.acetylcholine_level
        
        if da > 0.7 and se > 0.5:
            return "happy"
        elif da > 0.7:
            return "excited"
        elif ne > 0.7 and ach > 0.6:
            return "focused"
        elif ne > 0.7:
            return "alert"
        elif se > 0.7:
            return "calm"
        elif da < 0.3 and se < 0.3:
            return "tired"
        elif ne > 0.6 and se < 0.4:
            return "anxious"
        elif ach > 0.7:
            return "curious"
        else:
            return "neutral"
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for GUI dashboard display.
        ChemicalBrain compatible interface.
        """
        stats = self.get_stats()
        
        # Count neurons across all regions (dynamic count)
        total_neurons = sum(proc.num_neurons for proc in self.regions.values())
        neurons_born = sum(proc.neurons_created for proc in self.regions.values())
        neurons_died = sum(proc.neurons_pruned for proc in self.regions.values())
        
        # Count active neurons across all regions
        active_neurons = sum(
            int(np.sum(proc.activity > 0))
            for proc in self.regions.values()
        )
        
        # Get all 10 chemical levels from neuromodulation system
        chem_levels = self.neuromod.get_all_levels()
        
        # Compute synapse stats
        total_synapses = sum(proc.num_neurons * 50 for proc in self.regions.values())
        
        return {
            'chemicals': {
                # Primary chemicals (actively modulated by input signals)
                'dopamine': chem_levels.get('dopamine', self.state.dopamine_level),
                'serotonin': chem_levels.get('serotonin', self.state.serotonin_level),
                'norepinephrine': chem_levels.get('norepinephrine', self.state.norepinephrine_level),
                'acetylcholine': chem_levels.get('acetylcholine', self.state.acetylcholine_level),

                # Secondary chemicals (now actively modulated!)
                'cortisol': chem_levels.get('cortisol', getattr(self.state, 'cortisol_level', 0.3)),
                'gaba': chem_levels.get('gaba', getattr(self.state, 'gaba_level', 0.4)),
                'glutamate': chem_levels.get('glutamate', getattr(self.state, 'glutamate_level', 0.5)),
                'oxytocin': chem_levels.get('oxytocin', getattr(self.state, 'oxytocin_level', 0.3)),

                # Tertiary chemicals (from neuromod system only)
                'endorphin': chem_levels.get('endorphin', 0.3),
                'adrenaline': chem_levels.get('adrenaline', 0.2),
            },
            'neurons': {
                'total': total_neurons,
                'hidden': int(total_neurons * 0.7),
                'sensory': int(total_neurons * 0.15),
                'output': int(total_neurons * 0.15),
                'active': active_neurons,
                'spikes': self.state.simulation_step * 100,
                'born': neurons_born + self.state.neurons_created,
                'died': neurons_died + self.state.neurons_pruned,
            },
            'synapses': {
                'total': total_synapses,
                'active': active_neurons * 50,
                'density': self.config.target_sparsity,
            },
            'network': {
                'total_neurons': total_neurons,
                'total_synapses': total_synapses,
                'total_spikes': self.state.simulation_step * 100,
                'connection_density': self.config.target_sparsity,
            },
            'personality': self._compute_personality(),
            'mood': self._compute_mood(),
            'energy': self.state.total_energy / 100.0,
            'memory': {
                'total': self.config.reservoir_size,
                'encoded': self.state.simulation_step,
                'forgotten': 0,
                'avg_strength': 0.5,
            },
            'sleep_stage': 'awake',
            'regions': {
                region.value: {
                    'activity': float(np.mean(proc.activity)),
                    'active_neurons': int(np.sum(proc.activity > 0)),
                }
                for region, proc in self.regions.items()
            },
            'interactions': self.state.simulation_step,
            'training_count': 0,
            'learning': {
                'growth_rate': self.config.neurogenesis_rate,
                'pruning_rate': self.config.pruning_threshold,
                'plasticity': self.config.learning_rate,
                'consolidation': 0.5,
            },
        }
    
    def _compute_personality(self) -> Dict[str, float]:
        """Compute personality traits from brain state."""
        # Derive personality from neuromodulator baselines
        return {
            'openness': 0.5 + 0.3 * (self.state.dopamine_level - 0.5),
            'conscientiousness': 0.5 + 0.3 * (self.state.serotonin_level - 0.5),
            'extraversion': 0.5 + 0.3 * (self.state.norepinephrine_level - 0.5),
            'agreeableness': 0.5 + 0.3 * (self.state.serotonin_level - 0.5),
            'neuroticism': 0.5 - 0.3 * (self.state.serotonin_level - 0.5),
        }
    
    def save(self, filepath: str) -> None:
        """Save brain state - ChemicalBrain compatible interface."""
        from .persistence import BrainPersistence
        persistence = BrainPersistence(enable_auto_save=False)
        persistence.save(self, filepath)
    
    def load(self, filepath: str) -> None:
        """Load brain state - ChemicalBrain compatible interface."""
        from .persistence import BrainPersistence
        persistence = BrainPersistence(enable_auto_save=False)
        loaded = persistence.load(filepath)
        # Copy state from loaded brain
        if hasattr(loaded, 'state'):
            self.state = loaded.state
        if hasattr(loaded, 'regions'):
            self.regions = loaded.regions


def create_brain(
    scale: str = "small",
    use_gpu: bool = False
) -> IntegratedBrain:
    """
    Factory function to create brain with preset configurations.
    
    Args:
        scale: "small", "medium", or "large"
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Configured IntegratedBrain instance
    """
    configs = {
        "small": BrainConfig(
            neurons_per_column=50,
            columns_per_region=50,
            reservoir_size=1000,
            vocabulary_size=5000
        ),
        "medium": BrainConfig(
            neurons_per_column=100,
            columns_per_region=100,
            reservoir_size=5000,
            vocabulary_size=10000
        ),
        "large": BrainConfig(
            neurons_per_column=200,
            columns_per_region=200,
            reservoir_size=10000,
            vocabulary_size=20000
        )
    }
    
    config = configs.get(scale, configs["medium"])
    config.use_gpu = use_gpu
    
    return IntegratedBrain(config)


# Example usage
if __name__ == "__main__":
    # Create a small brain for testing
    brain = create_brain("small", use_gpu=False)
    
    print("\n=== Brain Statistics ===")
    stats = brain.get_stats()
    print(f"Total neurons: {stats['config']['total_neurons']:,}")
    print(f"Reservoir size: {stats['config']['reservoir_size']}")
    print(f"Vocabulary: {stats['config']['vocabulary_size']} words")
    
    # Test processing
    print("\n=== Processing Test ===")
    response = brain.process_input(
        "Hello, how are you today?",
        reward=0.0,
        arousal=0.6
    )
    print(f"Response: {response}")
    print(f"Confidence: {brain.state.output_confidence:.3f}")
    
    # Show neuromodulator levels
    print("\n=== Neuromodulator Levels ===")
    print(f"Dopamine: {brain.state.dopamine_level:.3f}")
    print(f"Acetylcholine: {brain.state.acetylcholine_level:.3f}")
    print(f"Norepinephrine: {brain.state.norepinephrine_level:.3f}")
    print(f"Serotonin: {brain.state.serotonin_level:.3f}")
    
    # Show regional activity
    print("\n=== Regional Activity ===")
    for region, activity in brain.state.region_activity.items():
        print(f"{region.value}: {activity:.4f}")
