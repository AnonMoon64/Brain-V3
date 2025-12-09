"""
Sparse Representations and Event-Driven Simulation

Implements:
- K-winners-take-all (sparse coding)
- Event-driven simulation (only process active neurons)
- Sparse Distributed Representations (SDR)
- ~2% active neurons at any time
- Efficient spike propagation

Key insight: The brain is extremely sparse - only ~2% of neurons
fire at any given moment. This provides massive computational
efficiency and robust representational capacity.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Callable
from collections import defaultdict
from heapq import heappush, heappop
import math


@dataclass
class SparseNeuron:
    """
    Neuron optimized for sparse, event-driven simulation
    
    Only maintains essential state, updates only when receiving input.
    """
    id: str
    
    # Membrane state
    membrane_potential: float = 0.0
    threshold: float = 1.0
    reset_value: float = -0.2
    
    # Timing
    last_spike_time: float = -1000.0
    last_input_time: float = 0.0
    refractory_until: float = 0.0
    refractory_period: float = 2.0  # ms
    
    # Leak (for when we do update)
    tau: float = 20.0  # membrane time constant
    
    # Input accumulator
    pending_input: float = 0.0
    
    # Homeostatic target
    target_rate: float = 0.02  # 2% active
    excitability: float = 1.0
    
    def receive_input(self, current: float, time: float) -> bool:
        """
        Receive input, potentially triggering spike
        Returns True if neuron spiked
        """
        # Apply leak since last input
        dt = time - self.last_input_time
        if dt > 0:
            decay = np.exp(-dt / self.tau)
            self.membrane_potential *= decay
        
        self.last_input_time = time
        
        # Check refractory
        if time < self.refractory_until:
            return False
        
        # Accumulate input (scaled by excitability for homeostasis)
        self.membrane_potential += current * self.excitability
        
        # Check threshold
        if self.membrane_potential >= self.threshold:
            self._spike(time)
            return True
        
        return False
    
    def _spike(self, time: float) -> None:
        """Generate spike"""
        self.membrane_potential = self.reset_value
        self.last_spike_time = time
        self.refractory_until = time + self.refractory_period
    
    def update_homeostasis(self, actual_rate: float, dt: float) -> None:
        """Adjust excitability for homeostatic target rate"""
        error = self.target_rate - actual_rate
        # Slow adjustment
        self.excitability += error * 0.001 * dt
        self.excitability = np.clip(self.excitability, 0.1, 10.0)


@dataclass
class SparseConnection:
    """Sparse connection with event-driven transmission"""
    source_id: str
    target_id: str
    weight: float
    delay: float = 1.0  # ms
    
    # Eligibility for learning
    eligibility: float = 0.0
    eligibility_decay: float = 0.95


@dataclass
class SpikeEvent:
    """Event in the event queue"""
    time: float
    source_id: str
    target_id: str
    weight: float
    
    def __lt__(self, other):
        return self.time < other.time


class KWinnersNetwork:
    """
    K-Winners-Take-All network
    
    Ensures only top-k neurons are active in each group,
    implementing sparse distributed representations.
    """
    
    def __init__(
        self,
        n_neurons: int,
        k: int = None,
        sparsity: float = 0.02,
        boost_strength: float = 0.1
    ):
        self.n_neurons = n_neurons
        self.k = k or max(1, int(n_neurons * sparsity))
        self.sparsity = sparsity
        self.boost_strength = boost_strength
        
        # Neurons
        self.neurons: Dict[str, SparseNeuron] = {}
        for i in range(n_neurons):
            nid = f"n_{i}"
            self.neurons[nid] = SparseNeuron(id=nid)
        
        self.neuron_ids = list(self.neurons.keys())
        
        # Duty cycle tracking (for boosting)
        self.duty_cycles = np.zeros(n_neurons)
        self.duty_alpha = 0.01  # Exponential moving average rate
        
        # Overlap tracking
        self.overlap_history: List[float] = []
    
    def compute(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute sparse output with k-winners-take-all
        """
        if len(inputs) != self.n_neurons:
            # Pad or truncate
            padded = np.zeros(self.n_neurons)
            padded[:min(len(inputs), self.n_neurons)] = inputs[:self.n_neurons]
            inputs = padded
        
        # Apply boosting based on duty cycle
        boosted = inputs * self._compute_boost_factors()
        
        # Find k winners
        winners = self._select_winners(boosted)
        
        # Create sparse output
        output = np.zeros(self.n_neurons)
        output[winners] = 1.0
        
        # Update duty cycles
        self._update_duty_cycles(winners)
        
        return output
    
    def _compute_boost_factors(self) -> np.ndarray:
        """
        Compute boost factors based on duty cycle
        
        Neurons that fire less than target get boosted.
        Neurons that fire more get suppressed.
        """
        target_duty = self.sparsity
        
        # Boost = exp((target - actual) * boost_strength)
        boost = np.exp(
            (target_duty - self.duty_cycles) * self.boost_strength / target_duty
        )
        
        return boost
    
    def _select_winners(self, activities: np.ndarray) -> np.ndarray:
        """Select top-k neurons as winners"""
        # Add small noise to break ties
        noisy = activities + np.random.randn(len(activities)) * 1e-6
        
        # Get indices of top k
        winners = np.argsort(noisy)[-self.k:]
        
        return winners
    
    def _update_duty_cycles(self, winners: np.ndarray) -> None:
        """Update duty cycle estimates"""
        active = np.zeros(self.n_neurons)
        active[winners] = 1.0
        
        # Exponential moving average
        self.duty_cycles = (
            (1 - self.duty_alpha) * self.duty_cycles + 
            self.duty_alpha * active
        )
    
    def compute_overlap(self, sdr1: np.ndarray, sdr2: np.ndarray) -> float:
        """Compute overlap between two SDRs"""
        active1 = sdr1 > 0.5
        active2 = sdr2 > 0.5
        
        if not np.any(active1) or not np.any(active2):
            return 0.0
        
        intersection = np.sum(active1 & active2)
        union = np.sum(active1 | active2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_active_indices(self, sdr: np.ndarray) -> List[int]:
        """Get indices of active neurons in SDR"""
        return list(np.where(sdr > 0.5)[0])


class EventDrivenSimulator:
    """
    Event-driven neural simulation
    
    Only processes neurons that receive spikes,
    dramatically reducing computation when activity is sparse.
    """
    
    def __init__(self, n_neurons: int, sparsity: float = 0.02):
        self.n_neurons = n_neurons
        self.sparsity = sparsity
        
        # Neurons
        self.neurons: Dict[str, SparseNeuron] = {}
        for i in range(n_neurons):
            nid = f"n_{i}"
            self.neurons[nid] = SparseNeuron(
                id=nid,
                target_rate=sparsity
            )
        
        # Connections (sparse storage)
        self.connections: Dict[str, List[SparseConnection]] = defaultdict(list)
        
        # Event queue (priority queue ordered by time)
        self.event_queue: List[SpikeEvent] = []
        
        # Current time
        self.current_time: float = 0.0
        
        # Spike history
        self.spike_history: List[Tuple[float, str]] = []
        self.max_history = 10000
        
        # Statistics
        self.spikes_processed = 0
        self.events_processed = 0
    
    def add_connection(
        self, 
        source_id: str, 
        target_id: str, 
        weight: float,
        delay: float = 1.0
    ) -> None:
        """Add connection between neurons"""
        conn = SparseConnection(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            delay=delay
        )
        self.connections[source_id].append(conn)
    
    def inject_current(
        self, 
        neuron_id: str, 
        current: float,
        time: float = None
    ) -> bool:
        """
        Inject current into a neuron
        Returns True if neuron spiked
        """
        if time is None:
            time = self.current_time
        
        if neuron_id not in self.neurons:
            return False
        
        neuron = self.neurons[neuron_id]
        spiked = neuron.receive_input(current, time)
        
        if spiked:
            self._schedule_spike_effects(neuron_id, time)
            self.spikes_processed += 1
        
        return spiked
    
    def _schedule_spike_effects(self, source_id: str, spike_time: float) -> None:
        """Schedule events for spike propagation"""
        for conn in self.connections[source_id]:
            event = SpikeEvent(
                time=spike_time + conn.delay,
                source_id=source_id,
                target_id=conn.target_id,
                weight=conn.weight
            )
            heappush(self.event_queue, event)
        
        # Record spike
        self.spike_history.append((spike_time, source_id))
        if len(self.spike_history) > self.max_history:
            self.spike_history.pop(0)
    
    def process_events_until(self, end_time: float) -> List[str]:
        """
        Process all events until end_time
        Returns list of neurons that spiked
        """
        spiked_neurons = []
        
        while self.event_queue and self.event_queue[0].time <= end_time:
            event = heappop(self.event_queue)
            self.events_processed += 1
            
            # Process this event
            if event.target_id in self.neurons:
                neuron = self.neurons[event.target_id]
                spiked = neuron.receive_input(event.weight, event.time)
                
                if spiked:
                    self._schedule_spike_effects(event.target_id, event.time)
                    spiked_neurons.append(event.target_id)
                    self.spikes_processed += 1
        
        self.current_time = end_time
        return spiked_neurons
    
    def step(self, dt: float, external_input: Dict[str, float] = None) -> Dict:
        """
        Advance simulation by dt milliseconds
        """
        external_input = external_input or {}
        end_time = self.current_time + dt
        
        # Apply external input
        external_spikes = []
        for nid, current in external_input.items():
            if self.inject_current(nid, current, self.current_time):
                external_spikes.append(nid)
        
        # Process internal events
        internal_spikes = self.process_events_until(end_time)
        
        # Compute sparsity
        all_spikes = set(external_spikes + internal_spikes)
        current_sparsity = len(all_spikes) / self.n_neurons if self.n_neurons > 0 else 0
        
        return {
            'time': self.current_time,
            'spikes': list(all_spikes),
            'spike_count': len(all_spikes),
            'sparsity': current_sparsity,
            'events_in_queue': len(self.event_queue),
        }
    
    def get_firing_rates(self, window: float = 100.0) -> Dict[str, float]:
        """Compute firing rates over recent window (Hz)"""
        rates = {}
        cutoff = self.current_time - window
        
        recent_spikes = defaultdict(int)
        for time, nid in self.spike_history:
            if time > cutoff:
                recent_spikes[nid] += 1
        
        for nid in self.neurons:
            count = recent_spikes[nid]
            rates[nid] = count / (window / 1000.0)  # Convert to Hz
        
        return rates
    
    def update_homeostasis(self, window: float = 1000.0) -> None:
        """Update homeostatic parameters based on firing rates"""
        rates = self.get_firing_rates(window)
        
        for nid, neuron in self.neurons.items():
            rate = rates.get(nid, 0.0)
            neuron.update_homeostasis(rate, dt=1.0)
    
    def get_stats(self) -> Dict:
        """Get simulation statistics"""
        return {
            'current_time': self.current_time,
            'n_neurons': self.n_neurons,
            'n_connections': sum(len(conns) for conns in self.connections.values()),
            'spikes_processed': self.spikes_processed,
            'events_processed': self.events_processed,
            'events_pending': len(self.event_queue),
            'spike_history_length': len(self.spike_history),
        }


class SDRMemory:
    """
    Sparse Distributed Representation memory
    
    Stores patterns as SDRs and retrieves by similarity.
    Uses union/intersection operations for pattern matching.
    """
    
    def __init__(self, n_bits: int, active_bits: int = None, sparsity: float = 0.02):
        self.n_bits = n_bits
        self.active_bits = active_bits or max(1, int(n_bits * sparsity))
        self.sparsity = sparsity
        
        # Stored patterns
        self.patterns: Dict[str, Set[int]] = {}
        
        # Pattern metadata
        self.metadata: Dict[str, dict] = {}
        
        # Access tracking
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, float] = {}
    
    def encode(self, values: np.ndarray) -> Set[int]:
        """Encode values to SDR"""
        if len(values) > self.n_bits:
            values = values[:self.n_bits]
        elif len(values) < self.n_bits:
            padded = np.zeros(self.n_bits)
            padded[:len(values)] = values
            values = padded
        
        # Select top-k as active bits
        indices = np.argsort(values)[-self.active_bits:]
        return set(indices.tolist())
    
    def store(self, key: str, pattern: Set[int], metadata: dict = None) -> None:
        """Store pattern with key"""
        self.patterns[key] = pattern
        self.metadata[key] = metadata or {}
        self.last_access[key] = 0.0
    
    def retrieve(self, query: Set[int], threshold: float = 0.3) -> List[Tuple[str, float]]:
        """
        Retrieve patterns similar to query
        Returns list of (key, similarity) sorted by similarity
        """
        results = []
        
        for key, pattern in self.patterns.items():
            sim = self._overlap(query, pattern)
            if sim >= threshold:
                results.append((key, sim))
                self.access_counts[key] += 1
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _overlap(self, sdr1: Set[int], sdr2: Set[int]) -> float:
        """Compute overlap between SDRs"""
        if not sdr1 or not sdr2:
            return 0.0
        
        intersection = len(sdr1 & sdr2)
        min_size = min(len(sdr1), len(sdr2))
        
        return intersection / min_size if min_size > 0 else 0.0
    
    def union(self, patterns: List[Set[int]]) -> Set[int]:
        """Compute union of patterns"""
        if not patterns:
            return set()
        result = set()
        for p in patterns:
            result |= p
        return result
    
    def intersection(self, patterns: List[Set[int]]) -> Set[int]:
        """Compute intersection of patterns"""
        if not patterns:
            return set()
        result = patterns[0].copy()
        for p in patterns[1:]:
            result &= p
        return result
    
    def get_capacity(self) -> float:
        """
        Estimate remaining capacity
        
        SDR memory capacity is approximately:
        C = n! / (k! * (n-k)!) where n=bits, k=active
        """
        from math import factorial, log10
        
        n = self.n_bits
        k = self.active_bits
        
        # Log of combinations
        log_capacity = (
            sum(log10(i) for i in range(n-k+1, n+1)) -
            sum(log10(i) for i in range(1, k+1))
        )
        
        used_fraction = len(self.patterns) / (10 ** log_capacity)
        return 1.0 - min(1.0, used_fraction)


class SparseNetwork:
    """
    Complete sparse neural network
    
    Combines:
    - K-winners-take-all
    - Event-driven simulation  
    - SDR memory
    - Homeostatic regulation
    """
    
    def __init__(
        self,
        n_neurons: int,
        sparsity: float = 0.02,
        connection_density: float = 0.1
    ):
        self.n_neurons = n_neurons
        self.sparsity = sparsity
        
        # Components
        self.kwta = KWinnersNetwork(n_neurons, sparsity=sparsity)
        self.simulator = EventDrivenSimulator(n_neurons, sparsity=sparsity)
        self.memory = SDRMemory(n_neurons, sparsity=sparsity)
        
        # Initialize random connections
        self._init_connections(connection_density)
    
    def _init_connections(self, density: float) -> None:
        """Initialize sparse random connections"""
        n_connections = int(self.n_neurons * self.n_neurons * density)
        
        for _ in range(n_connections):
            source = f"n_{np.random.randint(self.n_neurons)}"
            target = f"n_{np.random.randint(self.n_neurons)}"
            
            if source != target:
                weight = np.random.randn() * 0.5
                delay = 1.0 + np.random.exponential(2.0)
                self.simulator.add_connection(source, target, weight, delay)
    
    def encode_input(self, values: np.ndarray) -> np.ndarray:
        """Encode input through k-winners-take-all"""
        return self.kwta.compute(values)
    
    def step(self, input_values: np.ndarray, dt: float) -> Dict:
        """Process one timestep"""
        # Sparse encode input
        sparse_input = self.encode_input(input_values)
        
        # Convert to neuron currents
        active_indices = self.kwta.get_active_indices(sparse_input)
        external_input = {
            f"n_{i}": 5.0  # Strong input to active neurons
            for i in active_indices
        }
        
        # Run event-driven simulation
        result = self.simulator.step(dt, external_input)
        
        # Get output as SDR
        output_sdr = set(
            int(nid.split('_')[1]) 
            for nid in result['spikes']
        )
        
        result['sdr'] = output_sdr
        result['sdr_size'] = len(output_sdr)
        
        return result
    
    def get_stats(self) -> Dict:
        """Get network statistics"""
        return {
            'kwta': {
                'k': self.kwta.k,
                'avg_duty_cycle': float(np.mean(self.kwta.duty_cycles)),
            },
            'simulator': self.simulator.get_stats(),
            'memory': {
                'stored_patterns': len(self.memory.patterns),
                'capacity': self.memory.get_capacity(),
            },
        }
