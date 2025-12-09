"""
Three-System Brain Architecture
================================

This implements the concentrated intelligence architecture where everything
slots into one of three interacting subsystems:

SYSTEM 1 - Sparse Cortical Engine
---------------------------------
Representation + Perception + Prediction
- HTM-like SDR + cortical microcircuits + k-winners + lateral inhibition
- Handles: sensory input → sparse codes, columns building patterns,
  predictive coding, feedforward/feedback paths, minicolumn competition,
  error signals (prediction mismatch)
- This is the structure that actually thinks in symbols and categories
- All "concepts," "patterns," "features" live here

SYSTEM 2 - Dynamic Recurrent Core (Reservoir)
---------------------------------------------
Memory + Imagination + Sequence Modeling
- The liquid brain - chaotic dynamical heart
- Handles: working memory, multi-step reasoning, temporal abstraction,
  generating internal patterns, "dreaming"/replay, rich state transitions
- Reservoir computing is what most brain simulators are missing
- This is the temporal glue

SYSTEM 3 - Neuromodulated Learning System
-----------------------------------------
Motivation + Plasticity + Value Assignment
- This is where the "soul" lives
- Handles: dopamine/serotonin/ACh/NE kinetics, three-factor learning,
  receptor activation → plasticity modulation, novelty detection,
  neurogenesis, pruning decisions, emotional valence tagging,
  long-term synaptic drift, metabolic gating
- Governs WHEN cortex and reservoir change, not computation itself

THE CLEAN INTERACTION LOOP:
1. Sensory Input → Sparse Cortical Engine (creates sparse patterns)
2. Cortical Output → Reservoir (transforms to temporal trajectories)
3. Reservoir Output → Cortex (predictions, imagined futures, memory)
4. Neuromodulatory System Monitors Everything and decides:
   - When to strengthen cortical patterns
   - When to modify reservoir dynamics
   - When to grow/prune neurons
   - When to shift value/emotion associations
5. Language Decoder sits ON TOP of Cortex (not as a system - as interface)

This is your replacement for backprop.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# Import existing modules
from .cortical_architecture import HierarchicalCortex, Minicolumn, TopographicMap
from .sparse_network import SparseNetwork, EventDrivenSimulator, KWinnersNetwork, SDRMemory
from .signal_processing import RobustInputPipeline as SignalProcessor
from .reservoir import HybridReservoir, EchoStateReservoir
from .hierarchical_time import HierarchicalTimeManager, TemporalScale, NestedOscillator
from .neuromodulation import (
    KineticNeuromodulationSystem,
    ModulatorType,
    CrossModulatorMatrix,
    EpigeneticLearningModifiers,
    TemporalCreditAssignment,
    ErrorBargainingSystem,
    HomeostaticController
)
from .metabolism import MetabolicNetwork, SynapticScaling, IntrinsicPlasticity


# =============================================================================
# SYSTEM 1: SPARSE CORTICAL ENGINE
# =============================================================================

class SparseCorticalEngine:
    """
    SYSTEM 1: Representation + Perception + Prediction

    The backbone of cognition. This is the structure that actually thinks
    in symbols and categories. All concepts, patterns, and features live here.

    Components:
    - HTM-like sparse distributed representations (SDR)
    - Cortical microcircuits with k-winners-take-all
    - Lateral inhibition for competition
    - Predictive coding with feedforward/feedback paths
    - Minicolumn competition
    - Error signals (prediction mismatch)

    Behaves like: neocortex + HTM + early transformer attention
    """

    def __init__(
        self,
        input_dim: int = 300,
        num_columns: int = 200,
        cells_per_column: int = 32,
        sparsity: float = 0.02,
        num_cortical_areas: int = 6
    ):
        """
        Initialize the Sparse Cortical Engine.

        Args:
            input_dim: Dimension of input embeddings
            num_columns: Number of minicolumns
            cells_per_column: Cells per minicolumn
            sparsity: Target sparsity (fraction of active columns)
            num_cortical_areas: Number of hierarchical cortical areas
        """
        self.input_dim = input_dim
        self.num_columns = num_columns
        self.cells_per_column = cells_per_column
        self.sparsity = sparsity
        self.num_cortical_areas = num_cortical_areas

        # K-winners for sparse activation
        self.k_winners = max(1, int(num_columns * sparsity))

        # Signal processing pipeline
        self.signal_processor = SignalProcessor(input_dim=input_dim)

        # SDR Memory for pattern storage
        self.sdr_memory = SDRMemory(
            n_bits=num_columns * cells_per_column,
            active_bits=self.k_winners * cells_per_column // 4,
            sparsity=sparsity
        )

        # Feedforward weights (input → columns)
        self.ff_weights = np.random.randn(num_columns, input_dim) * 0.1

        # Lateral inhibition weights (column → column)
        self.lateral_weights = self._init_lateral_weights()

        # Feedback weights (higher areas → lower areas)
        self.fb_weights = np.random.randn(num_columns, num_columns) * 0.05

        # Predictive state (what we expect next)
        self.prediction = np.zeros(num_columns)

        # Current activation (sparse)
        self.activation = np.zeros(num_columns)

        # Prediction error
        self.prediction_error = np.zeros(num_columns)

        # Column duty cycles (for boosting)
        self.duty_cycles = np.ones(num_columns) * sparsity
        self.duty_cycle_alpha = 0.001

        # Hierarchical cortical representation
        self.cortical_states = [np.zeros(num_columns) for _ in range(num_cortical_areas)]

        # Refractory periods
        self.last_spike_time = np.full(num_columns, -np.inf)
        self.refractory_period = 0.002  # 2ms

    def _init_lateral_weights(self) -> np.ndarray:
        """Initialize Mexican-hat lateral inhibition weights."""
        weights = np.zeros((self.num_columns, self.num_columns))

        # Create spatial positions for columns (1D for simplicity)
        positions = np.arange(self.num_columns)

        for i in range(self.num_columns):
            for j in range(self.num_columns):
                if i == j:
                    continue
                dist = abs(positions[i] - positions[j])
                # Mexican hat: excitation nearby, inhibition further
                if dist < 5:
                    weights[i, j] = 0.1 * np.exp(-dist / 2)
                else:
                    weights[i, j] = -0.05 * np.exp(-(dist - 5) / 10)

        return weights

    def process(
        self,
        input_data: np.ndarray,
        current_time: float,
        learning_enabled: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Process input through the cortical engine.

        Args:
            input_data: Input embedding vector
            current_time: Current simulation time
            learning_enabled: Whether to update weights

        Returns:
            Dict with activation, prediction, error, and cortical states
        """
        # 1. Signal processing (normalize, handle noise)
        processed = self.signal_processor.process(input_data)

        # Ensure correct dimension
        if len(processed) < self.input_dim:
            padded = np.zeros(self.input_dim)
            padded[:len(processed)] = processed
            processed = padded
        elif len(processed) > self.input_dim:
            processed = processed[:self.input_dim]

        # 2. Feedforward activation
        raw_activation = np.dot(self.ff_weights, processed)

        # 3. Add lateral interactions (competition)
        lateral_input = np.dot(self.lateral_weights, self.activation)
        raw_activation += lateral_input * 0.3

        # 4. Add predictive feedback boost
        # Columns that were predicted get a boost
        raw_activation += self.prediction * 0.2

        # 5. Apply boosting for underactive columns
        boost = np.exp((self.sparsity - self.duty_cycles) / self.sparsity)
        raw_activation *= boost

        # 6. Apply refractory period
        time_since_spike = current_time - self.last_spike_time
        refractory_mask = time_since_spike > self.refractory_period
        raw_activation *= refractory_mask

        # 7. K-winners-take-all
        new_activation = self._k_winners_take_all(raw_activation)

        # 8. Compute prediction error
        self.prediction_error = new_activation - self.prediction
        error_magnitude = np.mean(np.abs(self.prediction_error))

        # 9. Update activation
        self.activation = new_activation

        # 10. Update spike times
        spiking = new_activation > 0.5
        self.last_spike_time[spiking] = current_time

        # 11. Update duty cycles
        self.duty_cycles = (
            (1 - self.duty_cycle_alpha) * self.duty_cycles +
            self.duty_cycle_alpha * (new_activation > 0)
        )

        # 12. Generate prediction for next step
        # Prediction is based on current state through recurrent connections
        self.prediction = np.tanh(
            np.dot(self.fb_weights, new_activation) +
            np.dot(self.ff_weights, processed) * 0.3
        )
        self.prediction = np.clip(self.prediction, 0, 1)

        # 13. Propagate through cortical hierarchy
        self._update_hierarchy(new_activation)

        # 14. Learning (if enabled)
        if learning_enabled and error_magnitude > 0.1:
            self._update_weights(processed, new_activation, error_magnitude)

        return {
            'activation': new_activation,
            'prediction': self.prediction,
            'error': self.prediction_error,
            'error_magnitude': error_magnitude,
            'cortical_states': self.cortical_states.copy(),
            'active_columns': np.sum(new_activation > 0),
            'sparsity': np.mean(new_activation > 0)
        }

    def _k_winners_take_all(self, activations: np.ndarray) -> np.ndarray:
        """Apply k-winners-take-all sparsity."""
        k = min(self.k_winners, len(activations))

        # Find top k indices
        indices = np.argpartition(activations, -k)[-k:]

        # Create sparse output
        sparse_output = np.zeros_like(activations)
        sparse_output[indices] = np.maximum(0, activations[indices])

        # Normalize
        if np.max(sparse_output) > 0:
            sparse_output /= np.max(sparse_output)

        return sparse_output

    def _update_hierarchy(self, activation: np.ndarray):
        """Update hierarchical cortical states."""
        # Layer 0 = input layer
        self.cortical_states[0] = activation.copy()

        # Each subsequent layer integrates from below
        for i in range(1, self.num_cortical_areas):
            # Simple temporal integration with leaky accumulation
            alpha = 0.1 / (i + 1)  # Slower at higher levels
            self.cortical_states[i] = (
                (1 - alpha) * self.cortical_states[i] +
                alpha * self.cortical_states[i-1]
            )

    def _update_weights(
        self,
        input_data: np.ndarray,
        activation: np.ndarray,
        error_magnitude: float
    ):
        """Update weights based on Hebbian learning modulated by error."""
        learning_rate = 0.001 * error_magnitude

        # Feedforward: strengthen active column → active input connections
        active_mask = activation > 0.5
        for i in np.where(active_mask)[0]:
            self.ff_weights[i] += learning_rate * (input_data - self.ff_weights[i])

        # Feedback: strengthen prediction → activation connections
        pred_active = self.prediction > 0.3
        act_active = activation > 0.3
        if np.any(pred_active) and np.any(act_active):
            outer = np.outer(act_active, pred_active)
            self.fb_weights += learning_rate * 0.5 * outer
            self.fb_weights = np.clip(self.fb_weights, -1, 1)

    def get_representation(self) -> np.ndarray:
        """Get current sparse representation."""
        return self.activation.copy()

    def get_prediction_confidence(self) -> float:
        """Get confidence based on prediction error."""
        return 1.0 - np.clip(np.mean(np.abs(self.prediction_error)), 0, 1)

    def get_state(self) -> Dict[str, Any]:
        """Get complete cortical state."""
        return {
            'activation': self.activation.copy(),
            'prediction': self.prediction.copy(),
            'prediction_error': self.prediction_error.copy(),
            'duty_cycles': self.duty_cycles.copy(),
            'cortical_states': [s.copy() for s in self.cortical_states],
            'sparsity': np.mean(self.activation > 0),
            'active_columns': int(np.sum(self.activation > 0))
        }


# =============================================================================
# SYSTEM 2: DYNAMIC RECURRENT CORE (RESERVOIR)
# =============================================================================

class DynamicRecurrentCore:
    """
    SYSTEM 2: Memory + Imagination + Sequence Modeling

    The liquid brain - chaotic dynamical heart. Handles working memory,
    multi-step reasoning, temporal abstraction, generating internal patterns,
    "dreaming"/replay, and rich state transitions.

    Components:
    - Echo State Network (ESN) for fast dynamics
    - Liquid State Machine (LSM) concepts for spiking compatibility
    - Hierarchical time scales (gamma → delta oscillations)
    - Slow/fast processing loops

    Behaves like: hippocampus + prefrontal loops + basal ganglia gating
    """

    def __init__(
        self,
        input_dim: int = 200,
        reservoir_size: int = 2000,
        output_dim: int = 300,
        spectral_radius: float = 0.95,
        sparsity: float = 0.1,
        leak_rate: float = 0.3
    ):
        """
        Initialize the Dynamic Recurrent Core.

        Args:
            input_dim: Input dimension (from cortex)
            reservoir_size: Number of reservoir neurons
            output_dim: Output dimension
            spectral_radius: Controls chaos (< 1 for stability)
            sparsity: Connection sparsity
            leak_rate: Leaky integration rate
        """
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate

        # Initialize reservoir
        self._init_reservoir()

        # Hierarchical time manager
        self.time_manager = HierarchicalTimeManager()
        self._setup_time_scales()

        # Working memory buffer
        self.memory_buffer = []
        self.memory_capacity = 50

        # State history for replay
        self.state_history = []
        self.max_history = 100

        # Output weights (learned via ridge regression)
        self.output_weights = np.zeros((output_dim, reservoir_size))

        # Readout
        self.last_output = np.zeros(output_dim)

    def _init_reservoir(self):
        """Initialize reservoir weights with echo state property."""
        # Random sparse reservoir weights
        W = np.random.randn(self.reservoir_size, self.reservoir_size)

        # Apply sparsity mask
        mask = np.random.random((self.reservoir_size, self.reservoir_size)) < self.sparsity
        W *= mask

        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W *= self.spectral_radius / current_radius

        self.W_reservoir = W

        # Input weights
        self.W_input = np.random.randn(self.reservoir_size, self.input_dim) * 0.5

        # Feedback weights
        self.W_feedback = np.random.randn(self.reservoir_size, self.output_dim) * 0.1

        # Reservoir state
        self.state = np.zeros(self.reservoir_size)

        # Fast and slow reservoir states (for multi-timescale)
        self.fast_state = np.zeros(self.reservoir_size)
        self.slow_state = np.zeros(self.reservoir_size)

    def _setup_time_scales(self):
        """Setup hierarchical time scales for oscillations."""
        # Gamma (30-100Hz) - fast sensory
        self.time_manager.add_scale(TemporalScale(
            name="gamma",
            base_dt=0.01,
            update_frequency=50
        ))

        # Beta (12-30Hz) - motor/attention
        self.time_manager.add_scale(TemporalScale(
            name="beta",
            base_dt=0.04,
            update_frequency=20
        ))

        # Theta (4-8Hz) - memory/navigation
        self.time_manager.add_scale(TemporalScale(
            name="theta",
            base_dt=0.15,
            update_frequency=6
        ))

        # Delta (0.5-4Hz) - slow integration
        self.time_manager.add_scale(TemporalScale(
            name="delta",
            base_dt=0.5,
            update_frequency=2
        ))

    def process(
        self,
        cortical_input: np.ndarray,
        modulation: float = 1.0,
        dt: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """
        Process cortical input through reservoir.

        Args:
            cortical_input: Input from cortex (System 1)
            modulation: Neuromodulatory gain (from System 3)
            dt: Time step

        Returns:
            Dict with reservoir state, output, and temporal features
        """
        # Ensure correct input dimension
        if len(cortical_input) < self.input_dim:
            padded = np.zeros(self.input_dim)
            padded[:len(cortical_input)] = cortical_input
            cortical_input = padded
        elif len(cortical_input) > self.input_dim:
            cortical_input = cortical_input[:self.input_dim]

        # 1. Input projection
        input_drive = np.dot(self.W_input, cortical_input)

        # 2. Feedback from output
        feedback_drive = np.dot(self.W_feedback, self.last_output)

        # 3. Reservoir dynamics (leaky ESN)
        pre_activation = (
            np.dot(self.W_reservoir, self.state) +
            input_drive +
            feedback_drive * 0.3
        )

        # 4. Apply nonlinearity with modulation
        new_state = np.tanh(pre_activation * modulation)

        # 5. Leaky integration
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * new_state

        # 6. Update fast/slow states (multi-timescale)
        self.fast_state = 0.7 * self.fast_state + 0.3 * self.state
        self.slow_state = 0.95 * self.slow_state + 0.05 * self.state

        # 7. Compute output
        output = np.dot(self.output_weights, self.state)
        self.last_output = output

        # 8. Store in working memory
        self._update_memory(self.state.copy())

        # 9. Store in history for replay
        self.state_history.append(self.state.copy())
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)

        # 10. Extract temporal features
        temporal_features = self._extract_temporal_features()

        return {
            'state': self.state.copy(),
            'output': output,
            'fast_state': self.fast_state.copy(),
            'slow_state': self.slow_state.copy(),
            'temporal_features': temporal_features,
            'memory_load': len(self.memory_buffer) / self.memory_capacity,
            'state_norm': np.linalg.norm(self.state)
        }

    def _update_memory(self, state: np.ndarray):
        """Update working memory buffer."""
        self.memory_buffer.append(state)
        if len(self.memory_buffer) > self.memory_capacity:
            self.memory_buffer.pop(0)

    def _extract_temporal_features(self) -> Dict[str, float]:
        """Extract temporal features from state history."""
        if len(self.state_history) < 10:
            return {'variance': 0, 'trend': 0, 'complexity': 0}

        recent = np.array(self.state_history[-10:])

        # Variance (activity level)
        variance = np.mean(np.var(recent, axis=0))

        # Trend (direction of change)
        trend = np.mean(recent[-1] - recent[0])

        # Complexity (entropy proxy)
        complexity = np.std(np.diff(recent, axis=0))

        return {
            'variance': variance,
            'trend': trend,
            'complexity': complexity
        }

    def train_readout(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        ridge_alpha: float = 1e-4
    ):
        """
        Train output weights using ridge regression.

        Args:
            states: Reservoir states (n_samples, reservoir_size)
            targets: Target outputs (n_samples, output_dim)
            ridge_alpha: Regularization strength
        """
        # Ridge regression: W = (X^T X + alpha I)^-1 X^T Y
        n_samples = states.shape[0]

        XtX = np.dot(states.T, states)
        XtY = np.dot(states.T, targets)

        reg = ridge_alpha * np.eye(self.reservoir_size)

        self.output_weights = np.linalg.solve(XtX + reg, XtY).T

    def replay(self, steps: int = 10) -> List[np.ndarray]:
        """
        Replay recent states (dreaming/consolidation).

        Args:
            steps: Number of replay steps

        Returns:
            List of replayed outputs
        """
        if len(self.state_history) < steps:
            return []

        outputs = []
        for state in self.state_history[-steps:]:
            output = np.dot(self.output_weights, state)
            outputs.append(output)

        return outputs

    def imagine(self, seed: np.ndarray, steps: int = 10) -> List[np.ndarray]:
        """
        Generate imagined trajectory from seed.

        Args:
            seed: Initial state
            steps: Number of imagination steps

        Returns:
            List of imagined outputs
        """
        state = seed.copy() if len(seed) == self.reservoir_size else self.state.copy()

        outputs = []
        for _ in range(steps):
            # Autonomous dynamics (no input)
            new_state = np.tanh(np.dot(self.W_reservoir, state))
            state = 0.8 * state + 0.2 * new_state

            output = np.dot(self.output_weights, state)
            outputs.append(output)

        return outputs

    def get_memory_content(self) -> np.ndarray:
        """Get integrated working memory content."""
        if len(self.memory_buffer) == 0:
            return np.zeros(self.reservoir_size)

        # Weighted sum of memory (recent more important)
        weights = np.linspace(0.5, 1.0, len(self.memory_buffer))
        weighted = np.average(self.memory_buffer, axis=0, weights=weights)
        return weighted

    def get_state(self) -> Dict[str, Any]:
        """Get complete reservoir state."""
        return {
            'state': self.state.copy(),
            'fast_state': self.fast_state.copy(),
            'slow_state': self.slow_state.copy(),
            'output': self.last_output.copy(),
            'memory_buffer_size': len(self.memory_buffer),
            'history_size': len(self.state_history)
        }


# =============================================================================
# SYSTEM 3: NEUROMODULATED LEARNING SYSTEM
# =============================================================================

class NeuromodulatedLearningSystem:
    """
    SYSTEM 3: Motivation + Plasticity + Value Assignment

    This is where the "soul" lives. Handles dopamine/serotonin/ACh/NE kinetics,
    three-factor learning rules, receptor activation → plasticity modulation,
    novelty detection → neurogenesis, pruning decisions, emotional valence
    tagging, long-term synaptic drift, metabolic gating.

    This governs WHEN the cortex and reservoir change, not computation itself.

    Components:
    - Enhanced neuromodulation with stochastic kinetics
    - Cross-modulator antagonism matrix
    - Epigenetic learning switches
    - Time-asymmetric plasticity
    - Cross-system error bargaining
    - Homeostatic stability (BCM)
    - Metabolism integration
    - Neurogenesis control

    Behaves like: reward system + stress system + sleep/adaptation + motivation
    """

    def __init__(
        self,
        num_neurons: int = 10000,
        enable_stochasticity: bool = True,
        enable_cross_modulator: bool = True,
        enable_epigenetics: bool = True,
        enable_temporal_credit: bool = True,
        enable_error_bargaining: bool = True,
        enable_metabolism: bool = True
    ):
        """
        Initialize the Neuromodulated Learning System.

        Args:
            num_neurons: Number of neurons to track metabolism for
            enable_stochasticity: Enable stochastic kinetics
            enable_cross_modulator: Enable cross-modulator interactions
            enable_epigenetics: Enable epigenetic switches
            enable_temporal_credit: Enable time-asymmetric plasticity
            enable_error_bargaining: Enable cross-system error bargaining
            enable_metabolism: Enable metabolic constraints
        """
        # Core neuromodulation system (enhanced)
        self.neuromod = KineticNeuromodulationSystem(
            enable_stochasticity=enable_stochasticity,
            enable_cross_modulator=enable_cross_modulator,
            enable_epigenetics=enable_epigenetics,
            enable_temporal_credit=enable_temporal_credit,
            enable_error_bargaining=enable_error_bargaining
        )

        # Feature flags
        self.enable_metabolism = enable_metabolism

        # Metabolism (simplified if disabled)
        if enable_metabolism:
            neuron_ids = [f"n_{i}" for i in range(min(1000, num_neurons))]
            self.metabolism = MetabolicNetwork(neuron_ids=neuron_ids)
            self.synaptic_scaling = SynapticScaling(
                num_neurons=num_neurons,
                target_rate=0.02
            )
            self.intrinsic_plasticity = IntrinsicPlasticity(
                num_neurons=num_neurons,
                target_rate=0.02
            )
        else:
            self.metabolism = None
            self.synaptic_scaling = None
            self.intrinsic_plasticity = None

        # Simple fatigue meter (always available, even without full metabolism)
        self.fatigue = 0.0
        self.fatigue_recovery_rate = 0.01
        self.fatigue_cost_per_spike = 0.001

        # Neurogenesis control
        self.novelty_threshold = 0.3
        self.neurogenesis_rate = 0.2
        self.pruning_threshold = 0.001

        # Track neurons created/pruned
        self.neurons_created = 0
        self.neurons_pruned = 0

        # Emotional valence memory
        self.valence_memory: Dict[str, float] = {}

        # Current state
        self.current_time = 0.0

    def update(
        self,
        dt: float,
        activations: Optional[np.ndarray] = None,
        activity_level: float = 0.1
    ) -> Dict[str, Any]:
        """
        Update the neuromodulation system.

        Args:
            dt: Time step
            activations: Current neural activations
            activity_level: Overall activity level (0-1)

        Returns:
            Dict with neuromodulation state and learning modifiers
        """
        self.current_time += dt

        # Update core neuromodulation
        neuromod_state = self.neuromod.update(dt, activations)

        # Update fatigue
        activity_cost = activity_level * self.fatigue_cost_per_spike * 100
        self.fatigue = min(1.0, self.fatigue + activity_cost * dt)
        self.fatigue = max(0.0, self.fatigue - self.fatigue_recovery_rate * dt)

        # Update metabolism if enabled
        if self.enable_metabolism and self.metabolism is not None:
            self.metabolism.step(dt)

            # Synaptic scaling based on activity
            if self.synaptic_scaling is not None:
                self.synaptic_scaling.update(dt)
            if self.intrinsic_plasticity is not None:
                self.intrinsic_plasticity.update(dt)

        # Get learning modulation
        learning_mod = self.neuromod.get_learning_modulation()

        # Apply fatigue to learning
        fatigue_factor = 1.0 - self.fatigue * 0.5
        learning_mod['ltp_modulation'] *= fatigue_factor
        learning_mod['attention'] *= fatigue_factor

        # Combine state
        state = {
            **neuromod_state,
            'fatigue': self.fatigue,
            'learning_modulation': learning_mod,
            'neurons_created': self.neurons_created,
            'neurons_pruned': self.neurons_pruned,
            'current_time': self.current_time
        }

        return state

    def should_create_neurons(
        self,
        novelty: float,
        dopamine: Optional[float] = None,
        cortisol: Optional[float] = None
    ) -> Tuple[bool, int]:
        """
        Determine if neurogenesis should occur.

        Args:
            novelty: Novelty level (0-1)
            dopamine: Current dopamine level (uses internal if None)
            cortisol: Current cortisol level (uses internal if None)

        Returns:
            (should_create, num_neurons)
        """
        if dopamine is None:
            dopamine = self.neuromod.get_level(ModulatorType.DOPAMINE)
        if cortisol is None:
            cortisol = self.neuromod.simple_chemicals.get(ModulatorType.CORTISOL, 0.3)

        # Neurogenesis probability: high novelty + high dopamine + low cortisol
        prob = novelty * (0.5 + dopamine) * (1 - cortisol * 0.3)

        if novelty > self.novelty_threshold and np.random.random() < prob * self.neurogenesis_rate:
            num_new = np.random.randint(1, 6)
            self.neurons_created += num_new
            return True, num_new

        return False, 0

    def should_prune_neurons(self, usage: np.ndarray, age: np.ndarray) -> np.ndarray:
        """
        Determine which neurons should be pruned.

        Args:
            usage: Usage counts per neuron
            age: Age in steps per neuron

        Returns:
            Boolean mask of neurons to prune
        """
        # Average usage per age
        avg_usage = usage / (age + 1)

        # Prune if old enough and below threshold
        old_enough = age > 50
        below_threshold = avg_usage < self.pruning_threshold

        to_prune = old_enough & below_threshold
        self.neurons_pruned += int(np.sum(to_prune))

        return to_prune

    def compute_learning_signal(
        self,
        cortex_error: float,
        reservoir_error: float,
        outcome: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute learning signals for cortex and reservoir.

        Uses error bargaining if enabled, otherwise simple averaging.

        Args:
            cortex_error: Prediction error from cortex
            reservoir_error: Prediction error from reservoir
            outcome: Optional reward/error signal for credit assignment

        Returns:
            Dict with cortex_learns, reservoir_learns, total_error
        """
        # Report errors to bargaining system
        self.neuromod.report_system_error('cortex', cortex_error, confidence=0.6)
        self.neuromod.report_system_error('reservoir', reservoir_error, confidence=0.5)

        # Get arbitrated learning decisions
        decisions = self.neuromod.get_learning_decisions()

        # If outcome provided, assign retrospective credit
        if outcome is not None:
            self.neuromod.assign_retrospective_credit(outcome, self.current_time)

        return {
            'cortex_learns': decisions['cortex_learns'],
            'reservoir_learns': decisions['reservoir_learns'],
            'agreed_error': decisions['agreed_error'],
            'conflict': decisions['conflict']
        }

    def tag_valence(self, concept: str, valence: float):
        """
        Tag a concept with emotional valence.

        Args:
            concept: Concept identifier
            valence: Emotional valence (-1 to 1)
        """
        # Apply emotional dampening if active
        if self.neuromod.epigenetic_modifiers is not None:
            for switch in self.neuromod.epigenetic_modifiers.switches.values():
                if switch.is_active and switch.target_parameter == 'emotional_valence_strength':
                    valence *= switch.modifier

        self.valence_memory[concept] = valence

    def get_valence(self, concept: str) -> float:
        """Get emotional valence for a concept."""
        return self.valence_memory.get(concept, 0.0)

    def report_outcome(self, success: bool, magnitude: float = 0.5):
        """Report success/failure for learning."""
        self.neuromod.report_outcome(success, magnitude)

    def get_neuromodulator_levels(self) -> Dict[str, float]:
        """Get all neuromodulator levels."""
        return self.neuromod.get_all_levels()

    def get_epigenetic_state(self) -> Dict[str, Any]:
        """Get epigenetic state."""
        if self.neuromod.epigenetic_modifiers is None:
            return {'active_switches': [], 'modifiers': {}}

        return {
            'active_switches': self.neuromod.epigenetic_modifiers.get_active_switches(),
            'modifiers': {
                name: switch.modifier
                for name, switch in self.neuromod.epigenetic_modifiers.switches.items()
            }
        }

    def get_state(self) -> Dict[str, Any]:
        """Get complete learning system state."""
        return {
            'neuromod_levels': self.get_neuromodulator_levels(),
            'fatigue': self.fatigue,
            'neurons_created': self.neurons_created,
            'neurons_pruned': self.neurons_pruned,
            'learning_modulation': self.neuromod.get_learning_modulation(),
            'epigenetic_state': self.get_epigenetic_state(),
            'current_time': self.current_time
        }


# =============================================================================
# THREE-SYSTEM BRAIN: THE UNIFIED ARCHITECTURE
# =============================================================================

class ThreeSystemBrain:
    """
    The unified three-system brain architecture.

    This orchestrates the clean interaction loop:
    1. Sensory Input → Sparse Cortical Engine
    2. Cortical Output → Reservoir
    3. Reservoir Output → Cortex (feedback)
    4. Neuromodulatory System monitors and decides learning
    5. Language Decoder consumes cortical representations

    This is the replacement for backprop - learning through neuromodulation.
    """

    def __init__(
        self,
        input_dim: int = 300,
        num_columns: int = 200,
        reservoir_size: int = 2000,
        output_dim: int = 300,
        enable_all_features: bool = True
    ):
        """
        Initialize the Three-System Brain.

        Args:
            input_dim: Input embedding dimension
            num_columns: Number of cortical columns
            reservoir_size: Reservoir size
            output_dim: Output dimension
            enable_all_features: Enable all enhanced features
        """
        print("Initializing Three-System Brain Architecture...")

        # System 1: Sparse Cortical Engine
        print("  [System 1] Sparse Cortical Engine...")
        self.cortex = SparseCorticalEngine(
            input_dim=input_dim,
            num_columns=num_columns,
            cells_per_column=32,
            sparsity=0.02
        )

        # System 2: Dynamic Recurrent Core
        print("  [System 2] Dynamic Recurrent Core...")
        self.reservoir = DynamicRecurrentCore(
            input_dim=num_columns,  # Takes cortical output
            reservoir_size=reservoir_size,
            output_dim=output_dim,
            spectral_radius=0.95,
            leak_rate=0.3
        )

        # System 3: Neuromodulated Learning System
        print("  [System 3] Neuromodulated Learning System...")
        self.learning = NeuromodulatedLearningSystem(
            num_neurons=num_columns * 32,
            enable_stochasticity=enable_all_features,
            enable_cross_modulator=enable_all_features,
            enable_epigenetics=enable_all_features,
            enable_temporal_credit=enable_all_features,
            enable_error_bargaining=enable_all_features,
            enable_metabolism=False  # Simplified metabolism
        )

        # Current state
        self.step_count = 0
        self.current_time = 0.0

        # Output cache
        self.last_cortical_output = np.zeros(num_columns)
        self.last_reservoir_output = np.zeros(output_dim)

        print("Three-System Brain initialized!")

    def process(
        self,
        input_data: np.ndarray,
        dt: float = 0.01,
        learning_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Process input through the three-system loop.

        Args:
            input_data: Input embedding
            dt: Time step
            learning_enabled: Whether learning is enabled

        Returns:
            Dict with outputs from all systems
        """
        self.step_count += 1
        self.current_time += dt

        # =====================================================================
        # STEP 1: Update neuromodulation (monitors everything)
        # =====================================================================
        learning_state = self.learning.update(
            dt=dt,
            activations=self.last_cortical_output,
            activity_level=np.mean(self.last_cortical_output > 0)
        )

        # Get modulation factors
        neuromod_levels = self.learning.get_neuromodulator_levels()
        learning_mod = learning_state['learning_modulation']

        # =====================================================================
        # STEP 2: Process through cortex (sparse patterns)
        # =====================================================================
        cortical_result = self.cortex.process(
            input_data=input_data,
            current_time=self.current_time,
            learning_enabled=learning_enabled and learning_mod.get('should_learn', True)
        )

        cortical_activation = cortical_result['activation']
        cortical_error = cortical_result['error_magnitude']

        # =====================================================================
        # STEP 3: Process through reservoir (temporal dynamics)
        # =====================================================================
        # Modulation from neuromodulators
        reservoir_modulation = 0.5 + 0.5 * neuromod_levels.get('dopamine', 0.5)

        reservoir_result = self.reservoir.process(
            cortical_input=cortical_activation,
            modulation=reservoir_modulation,
            dt=dt
        )

        reservoir_output = reservoir_result['output']
        reservoir_error = 1.0 - reservoir_result['temporal_features'].get('variance', 0.5)

        # =====================================================================
        # STEP 4: Compute learning signals via error bargaining
        # =====================================================================
        learning_signals = self.learning.compute_learning_signal(
            cortex_error=cortical_error,
            reservoir_error=reservoir_error
        )

        # =====================================================================
        # STEP 5: Check for neurogenesis
        # =====================================================================
        novelty = 1.0 - self.cortex.get_prediction_confidence()
        should_create, num_new = self.learning.should_create_neurons(novelty)

        # =====================================================================
        # STEP 6: Update caches
        # =====================================================================
        self.last_cortical_output = cortical_activation
        self.last_reservoir_output = reservoir_output

        # =====================================================================
        # STEP 7: Compile results
        # =====================================================================
        return {
            # Outputs
            'cortical_activation': cortical_activation,
            'reservoir_output': reservoir_output,
            'prediction': cortical_result['prediction'],

            # Errors
            'cortical_error': cortical_error,
            'reservoir_error': reservoir_error,
            'prediction_error': cortical_result['error'],

            # Learning
            'learning_signals': learning_signals,
            'learning_modulation': learning_mod,

            # Neuromodulation
            'neuromod_levels': neuromod_levels,
            'fatigue': learning_state['fatigue'],

            # Neurogenesis
            'novelty': novelty,
            'neurons_created': should_create and num_new or 0,

            # Metadata
            'step': self.step_count,
            'time': self.current_time,
            'sparsity': cortical_result['sparsity'],
            'active_columns': cortical_result['active_columns']
        }

    def train(
        self,
        inputs: List[np.ndarray],
        targets: List[np.ndarray],
        dt: float = 0.01
    ) -> Dict[str, float]:
        """
        Train the brain on input-target pairs.

        Uses the neuromodulation system to gate learning rather than backprop.

        Args:
            inputs: List of input embeddings
            targets: List of target outputs
            dt: Time step

        Returns:
            Training statistics
        """
        states = []
        errors = []

        for inp, target in zip(inputs, targets):
            # Process input
            result = self.process(inp, dt=dt, learning_enabled=True)
            states.append(self.reservoir.state.copy())

            # Compute error (for statistics)
            output_error = np.mean(np.abs(result['reservoir_output'][:len(target)] - target))
            errors.append(output_error)

            # Report outcome to neuromodulation
            success = output_error < 0.5
            self.learning.report_outcome(success, magnitude=1.0 - output_error)

        # Train reservoir readout
        if len(states) > 10:
            states_array = np.array(states)
            targets_array = np.array(targets)

            # Pad targets if needed
            if targets_array.shape[1] < self.reservoir.output_dim:
                padded = np.zeros((len(targets), self.reservoir.output_dim))
                padded[:, :targets_array.shape[1]] = targets_array
                targets_array = padded

            self.reservoir.train_readout(states_array, targets_array)

        return {
            'mean_error': np.mean(errors),
            'final_error': errors[-1] if errors else 0,
            'num_samples': len(inputs)
        }

    def get_representation(self) -> np.ndarray:
        """Get current brain representation."""
        # Combine cortical and reservoir representations
        cortical = self.last_cortical_output
        reservoir = self.reservoir.state

        # Concatenate (or use cortical as primary)
        return cortical

    def get_output(self) -> np.ndarray:
        """Get current output."""
        return self.last_reservoir_output

    def get_state(self) -> Dict[str, Any]:
        """Get complete brain state."""
        return {
            'cortex': self.cortex.get_state(),
            'reservoir': self.reservoir.get_state(),
            'learning': self.learning.get_state(),
            'step_count': self.step_count,
            'current_time': self.current_time
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard visualization."""
        neuromod = self.learning.get_neuromodulator_levels()
        learning_mod = self.learning.neuromod.get_learning_modulation()

        return {
            'chemicals': neuromod,
            'neurons': {
                'total': self.cortex.num_columns * self.cortex.cells_per_column,
                'active': int(np.sum(self.last_cortical_output > 0)),
                'born': self.learning.neurons_created,
                'died': self.learning.neurons_pruned
            },
            'reservoir': {
                'size': self.reservoir.reservoir_size,
                'state_norm': float(np.linalg.norm(self.reservoir.state)),
                'memory_load': self.reservoir.get_state()['memory_buffer_size']
            },
            'learning': learning_mod,
            'epigenetic': self.learning.get_epigenetic_state(),
            'errors': {
                'cortex': float(np.mean(np.abs(self.cortex.prediction_error))),
                'prediction_confidence': self.cortex.get_prediction_confidence()
            },
            'step': self.step_count,
            'time': self.current_time,
            'fatigue': self.learning.fatigue
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_three_system_brain(
    scale: str = "medium",
    enable_all_features: bool = True
) -> ThreeSystemBrain:
    """
    Factory function to create a Three-System Brain.

    Args:
        scale: "small", "medium", or "large"
        enable_all_features: Enable all enhanced features

    Returns:
        Configured ThreeSystemBrain
    """
    configs = {
        "small": {
            "input_dim": 100,
            "num_columns": 100,
            "reservoir_size": 500,
            "output_dim": 100
        },
        "medium": {
            "input_dim": 300,
            "num_columns": 200,
            "reservoir_size": 2000,
            "output_dim": 300
        },
        "large": {
            "input_dim": 500,
            "num_columns": 500,
            "reservoir_size": 5000,
            "output_dim": 500
        }
    }

    config = configs.get(scale, configs["medium"])

    return ThreeSystemBrain(
        **config,
        enable_all_features=enable_all_features
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("THREE-SYSTEM BRAIN DEMO")
    print("="*60)

    # Create brain
    brain = create_three_system_brain("small", enable_all_features=True)

    # Test processing
    print("\n--- Processing Test ---")
    for i in range(10):
        # Random input
        inp = np.random.randn(100) * 0.5

        # Process
        result = brain.process(inp)

        print(f"Step {i+1}:")
        print(f"  Active columns: {result['active_columns']}")
        print(f"  Cortical error: {result['cortical_error']:.3f}")
        print(f"  Novelty: {result['novelty']:.3f}")
        print(f"  Dopamine: {result['neuromod_levels'].get('dopamine', 0):.3f}")
        print(f"  Fatigue: {result['fatigue']:.3f}")

    # Get dashboard data
    print("\n--- Dashboard Data ---")
    dashboard = brain.get_dashboard_data()
    print(f"Total neurons: {dashboard['neurons']['total']}")
    print(f"Active neurons: {dashboard['neurons']['active']}")
    print(f"Neurons born: {dashboard['neurons']['born']}")
    print(f"Reservoir state norm: {dashboard['reservoir']['state_norm']:.3f}")
    print(f"Prediction confidence: {dashboard['errors']['prediction_confidence']:.3f}")

    print("\n" + "="*60)
    print("Three-System Brain demo complete!")
    print("="*60)
