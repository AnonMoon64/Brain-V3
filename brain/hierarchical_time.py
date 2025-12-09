"""
Hierarchical Time Scales

Implements:
- Multi-rate simulation (sensory fast, cognitive slow)
- Temporal integration across scales
- Nested oscillations (gamma → beta → theta → delta)
- Working memory through sustained activity
- Sequence learning across time scales

Key insight: Different brain regions operate at different
time scales. Sensory areas respond in milliseconds,
while prefrontal cortex integrates over seconds.
This is both biologically accurate and computationally efficient.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
from collections import deque
import math


class TimeScale(Enum):
    """Temporal scales in the brain"""
    FAST = "fast"           # ~1000 Hz - sensory processing
    MEDIUM = "medium"       # ~100 Hz - association, motor
    SLOW = "slow"           # ~10 Hz - memory, prefrontal
    VERY_SLOW = "very_slow" # ~1 Hz - consolidation


@dataclass
class TimeScaleConfig:
    """Configuration for a time scale"""
    name: TimeScale
    update_rate: float  # Hz
    dt: float = field(init=False)  # ms per update
    
    # Integration parameters
    tau: float = 20.0  # Time constant for this scale
    
    # Buffer sizes
    history_length: int = 100
    
    def __post_init__(self):
        self.dt = 1000.0 / self.update_rate


# Standard configurations
TIME_SCALE_CONFIGS = {
    TimeScale.FAST: TimeScaleConfig(
        name=TimeScale.FAST,
        update_rate=1000.0,
        tau=10.0,
        history_length=1000,
    ),
    TimeScale.MEDIUM: TimeScaleConfig(
        name=TimeScale.MEDIUM,
        update_rate=100.0,
        tau=50.0,
        history_length=500,
    ),
    TimeScale.SLOW: TimeScaleConfig(
        name=TimeScale.SLOW,
        update_rate=10.0,
        tau=200.0,
        history_length=100,
    ),
    TimeScale.VERY_SLOW: TimeScaleConfig(
        name=TimeScale.VERY_SLOW,
        update_rate=1.0,
        tau=1000.0,
        history_length=60,
    ),
}


class TemporalIntegrator:
    """
    Integrates information across time scales
    
    Faster scales feed into slower scales through
    temporal averaging and peak detection.
    """
    
    def __init__(self, dim: int, config: TimeScaleConfig):
        self.dim = dim
        self.config = config
        
        # State
        self.state = np.zeros(dim)
        self.history = deque(maxlen=config.history_length)
        
        # Integration buffers
        self.accumulator = np.zeros(dim)
        self.sample_count = 0
        
        # Peak detection
        self.peak_buffer = np.zeros(dim)
        self.peak_threshold = 0.5
        
        # Timing
        self.last_update_time = 0.0
        self.total_time = 0.0
    
    def update(self, input_val: np.ndarray, current_time: float) -> np.ndarray:
        """
        Update with new input, return integrated state
        """
        # Time since last update
        dt = current_time - self.last_update_time
        self.total_time = current_time
        
        # Accumulate input
        self.accumulator += input_val
        self.sample_count += 1
        
        # Check if time for update at this scale
        if dt >= self.config.dt:
            self._do_update()
            self.last_update_time = current_time
        
        return self.state
    
    def _do_update(self) -> None:
        """Perform temporal integration"""
        if self.sample_count == 0:
            return
        
        # Average accumulated input
        avg_input = self.accumulator / self.sample_count
        
        # Leaky integration
        alpha = self.config.dt / (self.config.tau + self.config.dt)
        self.state = (1 - alpha) * self.state + alpha * avg_input
        
        # Track peaks
        self.peak_buffer = np.maximum(self.peak_buffer * 0.9, np.abs(avg_input))
        
        # Store in history
        self.history.append(self.state.copy())
        
        # Reset accumulator
        self.accumulator = np.zeros(self.dim)
        self.sample_count = 0
    
    def get_temporal_context(self, n_steps: int = 10) -> np.ndarray:
        """Get temporal context from history"""
        if len(self.history) < n_steps:
            return np.zeros((n_steps, self.dim))
        
        recent = list(self.history)[-n_steps:]
        return np.array(recent)
    
    def get_trend(self) -> np.ndarray:
        """Get trend (derivative) of state"""
        if len(self.history) < 2:
            return np.zeros(self.dim)
        
        recent = list(self.history)[-10:]
        if len(recent) < 2:
            return np.zeros(self.dim)
        
        return (recent[-1] - recent[0]) / len(recent)


class NestedOscillator:
    """
    Nested oscillations across frequency bands
    
    Models the coupling between:
    - Gamma (30-100 Hz) - local processing, binding
    - Beta (13-30 Hz) - motor, attention
    - Theta (4-8 Hz) - memory encoding
    - Delta (0.5-4 Hz) - sleep, consolidation
    
    Faster oscillations are nested within slower ones.
    """
    
    def __init__(self):
        # Oscillator frequencies (Hz)
        self.frequencies = {
            'gamma': 40.0,
            'beta': 20.0,
            'theta': 6.0,
            'delta': 2.0,
        }
        
        # Phase of each oscillator
        self.phases = {band: 0.0 for band in self.frequencies}
        
        # Amplitude (can be modulated)
        self.amplitudes = {band: 1.0 for band in self.frequencies}
        
        # Phase-amplitude coupling strength
        self.coupling_strength = 0.3
        
        # Current time
        self.time = 0.0
    
    def step(self, dt: float) -> Dict[str, float]:
        """
        Update oscillator phases
        Returns current phase and amplitude for each band
        """
        self.time += dt
        
        output = {}
        
        for band, freq in self.frequencies.items():
            # Update phase
            self.phases[band] += 2 * np.pi * freq * (dt / 1000.0)
            self.phases[band] = self.phases[band] % (2 * np.pi)
            
            # Compute oscillation value
            osc_value = np.sin(self.phases[band]) * self.amplitudes[band]
            output[band] = float(osc_value)
        
        # Phase-amplitude coupling: gamma amplitude modulated by theta phase
        theta_phase = self.phases['theta']
        gamma_modulation = 0.5 + 0.5 * np.cos(theta_phase)  # Peak at theta trough
        output['gamma'] *= gamma_modulation
        
        # Beta modulated by theta too
        output['beta'] *= (0.7 + 0.3 * np.cos(theta_phase))
        
        return output
    
    def get_dominant_band(self) -> str:
        """Get currently dominant frequency band"""
        # Based on amplitude
        max_amp = 0
        dominant = 'gamma'
        
        for band, amp in self.amplitudes.items():
            if amp > max_amp:
                max_amp = amp
                dominant = band
        
        return dominant
    
    def modulate_amplitude(self, band: str, factor: float) -> None:
        """Modulate amplitude of a band"""
        if band in self.amplitudes:
            self.amplitudes[band] = max(0.1, min(2.0, self.amplitudes[band] * factor))
    
    def get_phase_coherence(self, band1: str, band2: str) -> float:
        """Compute phase coherence between bands"""
        if band1 not in self.phases or band2 not in self.phases:
            return 0.0
        
        phase_diff = self.phases[band1] - self.phases[band2]
        return abs(np.cos(phase_diff))


class WorkingMemoryBuffer:
    """
    Working memory through sustained activity
    
    Implements:
    - Bump attractors for continuous values
    - Discrete slots for items
    - Decay without rehearsal
    - Interference between items
    """
    
    def __init__(self, n_slots: int = 7, dim: int = 64):
        self.n_slots = n_slots
        self.dim = dim
        
        # Memory slots (each is a bump attractor)
        self.slots = [np.zeros(dim) for _ in range(n_slots)]
        self.slot_ages = [0.0 for _ in range(n_slots)]
        self.slot_strengths = [0.0 for _ in range(n_slots)]
        
        # Decay parameters
        self.decay_rate = 0.001
        self.interference_rate = 0.0001
        
        # Rehearsal
        self.rehearsal_boost = 1.5
    
    def store(self, pattern: np.ndarray) -> int:
        """
        Store pattern in working memory
        Returns slot index used
        """
        # Find least used slot (or oldest)
        slot_scores = [
            (i, self.slot_strengths[i], self.slot_ages[i])
            for i in range(self.n_slots)
        ]
        
        # Prefer empty slots, then old ones
        slot_scores.sort(key=lambda x: (x[1], -x[2]))
        slot_idx = slot_scores[0][0]
        
        # Store with some noise
        self.slots[slot_idx] = pattern.copy() + np.random.randn(self.dim) * 0.01
        self.slot_strengths[slot_idx] = 1.0
        self.slot_ages[slot_idx] = 0.0
        
        # Interference: stored pattern interferes with similar patterns
        for i in range(self.n_slots):
            if i != slot_idx and self.slot_strengths[i] > 0:
                similarity = np.dot(self.slots[i], pattern) / (
                    np.linalg.norm(self.slots[i]) * np.linalg.norm(pattern) + 1e-8
                )
                if similarity > 0.5:
                    # Similar patterns interfere
                    self.slot_strengths[i] *= (1.0 - similarity * self.interference_rate * 100)
        
        return slot_idx
    
    def retrieve(self, query: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Retrieve most similar pattern
        Returns (pattern, strength)
        """
        best_match = None
        best_score = -1
        best_strength = 0
        
        for i in range(self.n_slots):
            if self.slot_strengths[i] > 0.1:
                similarity = np.dot(self.slots[i], query) / (
                    np.linalg.norm(self.slots[i]) * np.linalg.norm(query) + 1e-8
                )
                score = similarity * self.slot_strengths[i]
                
                if score > best_score:
                    best_score = score
                    best_match = self.slots[i].copy()
                    best_strength = self.slot_strengths[i]
                    
                    # Rehearsal: accessing strengthens
                    self.slot_strengths[i] = min(1.0, self.slot_strengths[i] * self.rehearsal_boost)
                    self.slot_ages[i] = 0.0
        
        if best_match is None:
            return np.zeros(self.dim), 0.0
        
        return best_match, best_strength
    
    def update(self, dt: float) -> None:
        """Update decay and ages"""
        for i in range(self.n_slots):
            # Age increases
            self.slot_ages[i] += dt
            
            # Strength decays
            self.slot_strengths[i] *= (1.0 - self.decay_rate * dt)
            
            # Below threshold = forgotten
            if self.slot_strengths[i] < 0.05:
                self.slot_strengths[i] = 0.0
    
    def get_contents(self) -> List[Tuple[np.ndarray, float]]:
        """Get all stored items with strengths"""
        contents = []
        for i in range(self.n_slots):
            if self.slot_strengths[i] > 0.1:
                contents.append((self.slots[i].copy(), self.slot_strengths[i]))
        return contents
    
    @property
    def load(self) -> float:
        """Current memory load (0-1)"""
        return sum(1 for s in self.slot_strengths if s > 0.1) / self.n_slots


class HierarchicalTimeProcessor:
    """
    Complete hierarchical time processing system
    
    Manages multiple time scales with:
    - Sensory (fast)
    - Association (medium)  
    - Executive (slow)
    - Consolidation (very slow)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 64
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Integrators at each time scale
        self.integrators = {
            TimeScale.FAST: TemporalIntegrator(input_dim, TIME_SCALE_CONFIGS[TimeScale.FAST]),
            TimeScale.MEDIUM: TemporalIntegrator(hidden_dim, TIME_SCALE_CONFIGS[TimeScale.MEDIUM]),
            TimeScale.SLOW: TemporalIntegrator(hidden_dim, TIME_SCALE_CONFIGS[TimeScale.SLOW]),
            TimeScale.VERY_SLOW: TemporalIntegrator(output_dim, TIME_SCALE_CONFIGS[TimeScale.VERY_SLOW]),
        }
        
        # Projection matrices between scales
        self.fast_to_medium = np.random.randn(hidden_dim, input_dim) * 0.1
        self.medium_to_slow = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.slow_to_veryslow = np.random.randn(output_dim, hidden_dim) * 0.1
        
        # Feedback from slow to fast
        self.slow_to_fast = np.random.randn(input_dim, hidden_dim) * 0.05
        
        # Oscillator
        self.oscillator = NestedOscillator()
        
        # Working memory
        self.working_memory = WorkingMemoryBuffer(n_slots=7, dim=hidden_dim)
        
        # Current time
        self.current_time = 0.0
        
        # Update counters (for different rates)
        self.fast_counter = 0
        self.medium_counter = 0
        self.slow_counter = 0
    
    def step(self, input_val: np.ndarray, dt: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Process input through hierarchical time scales
        
        dt is in milliseconds (typically 1ms for finest resolution)
        """
        self.current_time += dt
        
        # Update oscillator
        osc = self.oscillator.step(dt)
        
        # Always update fast scale
        fast_state = self.integrators[TimeScale.FAST].update(input_val, self.current_time)
        
        # Modulate by gamma oscillation
        fast_state = fast_state * (0.5 + 0.5 * osc['gamma'])
        
        results = {
            'fast': fast_state,
            'oscillations': osc,
        }
        
        # Medium scale: update every 10ms (100 Hz)
        self.fast_counter += 1
        if self.fast_counter >= 10:
            self.fast_counter = 0
            
            # Project fast to medium
            medium_input = np.tanh(np.dot(self.fast_to_medium, fast_state))
            
            # Add feedback from slow scale
            slow_state = self.integrators[TimeScale.SLOW].state
            medium_input += np.tanh(np.dot(self.slow_to_fast, slow_state).T)[:self.hidden_dim] * 0.3
            
            # Modulate by beta
            medium_input *= (0.5 + 0.5 * osc['beta'])
            
            medium_state = self.integrators[TimeScale.MEDIUM].update(
                medium_input, self.current_time
            )
            results['medium'] = medium_state
            
            # Update working memory with medium-scale context
            self.working_memory.update(dt * 10)
        else:
            results['medium'] = self.integrators[TimeScale.MEDIUM].state
        
        # Slow scale: update every 100ms (10 Hz)
        self.medium_counter += 1
        if self.medium_counter >= 100:
            self.medium_counter = 0
            
            # Project medium to slow
            slow_input = np.tanh(np.dot(self.medium_to_slow, 
                                       self.integrators[TimeScale.MEDIUM].state))
            
            # Modulate by theta
            slow_input *= (0.5 + 0.5 * osc['theta'])
            
            slow_state = self.integrators[TimeScale.SLOW].update(
                slow_input, self.current_time
            )
            results['slow'] = slow_state
            
            # Maybe store in working memory if significant
            if np.linalg.norm(slow_state) > 0.5:
                self.working_memory.store(slow_state)
        else:
            results['slow'] = self.integrators[TimeScale.SLOW].state
        
        # Very slow scale: update every 1000ms (1 Hz)
        self.slow_counter += 1
        if self.slow_counter >= 1000:
            self.slow_counter = 0
            
            # Project slow to very slow
            veryslow_input = np.tanh(np.dot(
                self.slow_to_veryslow,
                self.integrators[TimeScale.SLOW].state
            ))
            
            # Modulate by delta
            veryslow_input *= (0.5 + 0.5 * osc['delta'])
            
            veryslow_state = self.integrators[TimeScale.VERY_SLOW].update(
                veryslow_input, self.current_time
            )
            results['very_slow'] = veryslow_state
        else:
            results['very_slow'] = self.integrators[TimeScale.VERY_SLOW].state
        
        # Add working memory state
        results['working_memory_load'] = self.working_memory.load
        
        return results
    
    def get_temporal_context(self, scale: TimeScale) -> np.ndarray:
        """Get temporal context at specific scale"""
        return self.integrators[scale].get_temporal_context()
    
    def retrieve_memory(self, query: np.ndarray) -> Tuple[np.ndarray, float]:
        """Retrieve from working memory"""
        return self.working_memory.retrieve(query)
    
    def get_output(self) -> np.ndarray:
        """Get current output (from slowest relevant scale)"""
        # Combine slow and very slow for output
        slow = self.integrators[TimeScale.SLOW].state
        very_slow = self.integrators[TimeScale.VERY_SLOW].state
        
        # Project to output dim
        output = np.tanh(np.dot(self.slow_to_veryslow, slow)) + very_slow * 0.5
        
        return output
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'current_time_ms': self.current_time,
            'current_time_sec': self.current_time / 1000.0,
            'dominant_oscillation': self.oscillator.get_dominant_band(),
            'working_memory_load': self.working_memory.load,
            'scale_states': {
                scale.value: float(np.mean(np.abs(integrator.state)))
                for scale, integrator in self.integrators.items()
            },
            'phase_coherence_theta_gamma': self.oscillator.get_phase_coherence('theta', 'gamma'),
        }


@dataclass
class TemporalScale:
    """Configuration for a named temporal scale (interface for IntegratedBrain)."""
    name: str
    base_dt: float  # Base time step in seconds
    update_frequency: float = 100.0  # Hz
    
    def __post_init__(self):
        self.dt_ms = self.base_dt * 1000.0
        self.period_ms = 1000.0 / self.update_frequency


class OscillatorBand:
    """Single oscillator band (gamma, beta, theta, delta)."""
    
    def __init__(self, name: str, frequency: float):
        self.name = name
        self.frequency = frequency
        self.phase = 0.0
        self.amplitude = 1.0
    
    def step(self, dt: float) -> float:
        """Update and return current oscillation value."""
        self.phase += 2 * np.pi * self.frequency * dt
        self.phase = self.phase % (2 * np.pi)
        return np.sin(self.phase) * self.amplitude


class HierarchicalTimeManager:
    """
    Manager for hierarchical time scales (interface for IntegratedBrain).
    
    Wraps the core HierarchicalTimeProcessor with a simpler API
    for adding/querying named time scales.
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        """Initialize time manager."""
        # Core processor
        self.processor = HierarchicalTimeProcessor(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        
        # Named scales registry
        self.scales: Dict[str, TemporalScale] = {}
        
        # Counters for each scale
        self.counters: Dict[str, int] = {}
        
        # Current time in seconds
        self.current_time = 0.0
        
        # Default scales
        self._setup_default_scales()
    
    def _setup_default_scales(self):
        """Setup default time scales."""
        defaults = [
            TemporalScale("sensory", 0.001, 1000),   # 1ms, 1000 Hz
            TemporalScale("fast", 0.01, 100),        # 10ms, 100 Hz
            TemporalScale("slow", 0.1, 10),          # 100ms, 10 Hz
            TemporalScale("memory", 1.0, 1),         # 1s, 1 Hz
        ]
        for scale in defaults:
            self.add_scale(scale)
    
    def add_scale(self, scale: TemporalScale):
        """Add a named time scale."""
        self.scales[scale.name] = scale
        self.counters[scale.name] = 0
    
    def get_scale(self, name: str) -> Optional[TemporalScale]:
        """Get a time scale by name."""
        return self.scales.get(name)
    
    def get_active_scales(self, current_time: float) -> List[str]:
        """
        Determine which scales should update at current time.
        Returns list of scale names that are due for update.
        """
        active = []
        
        for name, scale in self.scales.items():
            # Compute expected update times
            updates_per_sensory = 1000.0 / scale.update_frequency
            
            # Increment counter
            self.counters[name] += 1
            
            # Check if this scale should update
            if self.counters[name] >= updates_per_sensory:
                self.counters[name] = 0
                active.append(name)
        
        return active
    
    def step(self, dt: float, input_data: Optional[np.ndarray] = None) -> Dict:
        """
        Step all time scales forward.
        
        Args:
            dt: Time step in seconds
            input_data: Optional input for processing
            
        Returns:
            Dict with processing results
        """
        self.current_time += dt
        
        # Step the core processor
        if input_data is not None:
            results = self.processor.step(input_data)
        else:
            # Step with zero input
            results = self.processor.step(np.zeros(self.processor.input_dim))
        
        results['current_time'] = self.current_time
        results['active_scales'] = self.get_active_scales(self.current_time)
        
        return results
    
    def get_temporal_context(self, scale_name: str = "slow") -> np.ndarray:
        """Get temporal context at a specific scale."""
        scale_map = {
            "sensory": TimeScale.FAST,
            "fast": TimeScale.MEDIUM,
            "slow": TimeScale.SLOW,
            "memory": TimeScale.VERY_SLOW
        }
        scale = scale_map.get(scale_name, TimeScale.SLOW)
        return self.processor.get_temporal_context(scale)
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        stats = self.processor.get_stats()
        stats['registered_scales'] = list(self.scales.keys())
        stats['current_time_sec'] = self.current_time
        return stats

