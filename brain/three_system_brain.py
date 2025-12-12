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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import time
import os

# Core module imports (6 files total including this one)
from .signal_processing import RobustInputPipeline as SignalProcessor
from .hierarchical_time import HierarchicalTimeManager, TemporalScale
from .neuromodulation import KineticNeuromodulationSystem, ModulatorType
from .language_decoder import NeuralLanguageDecoder
from .consolidation import (
    ChemicalTaggingSystem, ConsolidationEngine, InheritedMemoryInstaller,
    SleepManager, MarkerType, create_consolidation_system
)


# =============================================================================
# SDR MEMORY (Inlined from sparse_network.py)
# =============================================================================

class SDRMemory:
    """
    Sparse Distributed Representation memory.
    
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
        """Encode values to SDR."""
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
        """Store pattern with key."""
        self.patterns[key] = pattern
        self.metadata[key] = metadata or {}
        self.last_access[key] = 0.0
    
    def retrieve(self, query: Set[int], threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Retrieve patterns similar to query (optimized)."""
        if not self.patterns:
            return []
        
        results = []
        query_list = list(query)
        query_size = len(query)
        
        if query_size == 0:
            return []
        
        # For small pattern stores, use direct comparison
        for key, pattern in self.patterns.items():
            pattern_size = len(pattern)
            if pattern_size == 0:
                continue
                
            intersection = len(query & pattern)
            min_size = min(query_size, pattern_size)
            sim = intersection / min_size
            
            if sim >= threshold:
                results.append((key, sim))
                self.access_counts[key] += 1
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _overlap(self, sdr1: Set[int], sdr2: Set[int]) -> float:
        """Compute overlap between SDRs."""
        if not sdr1 or not sdr2:
            return 0.0
        
        intersection = len(sdr1 & sdr2)
        min_size = min(len(sdr1), len(sdr2))
        
        return intersection / min_size if min_size > 0 else 0.0


# =============================================================================
# SPARSE CONTAINERS (COO Format) - Potato Optimization #1
# =============================================================================

class SparseVector:
    """
    Coordinate list (COO) sparse vector.
    Replaces dense numpy arrays for SDR and activation patterns.
    Memory: O(k) where k = active elements instead of O(n).
    """
    __slots__ = ['size', 'indices', 'values', '_cache_dense']
    
    def __init__(self, size: int, indices: List[int] = None, values: List[float] = None):
        self.size = size
        self.indices = indices or []
        self.values = values or []
        self._cache_dense = None
    
    @classmethod
    def from_dense(cls, arr: List[float], threshold: float = 0.001) -> 'SparseVector':
        """Convert dense array to sparse, keeping only values above threshold."""
        indices = []
        values = []
        for i, v in enumerate(arr):
            if abs(v) > threshold:
                indices.append(i)
                values.append(v)
        return cls(len(arr), indices, values)
    
    def to_dense(self) -> List[float]:
        """Convert to dense list (cached)."""
        if self._cache_dense is None:
            self._cache_dense = [0.0] * self.size
            for i, v in zip(self.indices, self.values):
                self._cache_dense[i] = v
        return self._cache_dense
    
    def dot(self, other: 'SparseVector') -> float:
        """Sparse dot product - O(k1 + k2) instead of O(n)."""
        # Build index map for smaller vector
        if len(self.indices) < len(other.indices):
            small, large = self, other
        else:
            small, large = other, self
        
        small_map = {i: v for i, v in zip(small.indices, small.values)}
        result = 0.0
        for i, v in zip(large.indices, large.values):
            if i in small_map:
                result += v * small_map[i]
        return result
    
    def add(self, other: 'SparseVector', alpha: float = 1.0) -> 'SparseVector':
        """Sparse addition: self + alpha * other."""
        combined = dict(zip(self.indices, self.values))
        for i, v in zip(other.indices, other.values):
            combined[i] = combined.get(i, 0.0) + alpha * v
        
        # Filter near-zero values
        indices = [i for i, v in combined.items() if abs(v) > 0.001]
        values = [combined[i] for i in indices]
        return SparseVector(self.size, indices, values)
    
    def k_winners(self, k: int) -> 'SparseVector':
        """Keep only top-k values."""
        if len(self.values) <= k:
            return SparseVector(self.size, self.indices.copy(), self.values.copy())
        
        # Sort by value and keep top k
        pairs = sorted(zip(self.values, self.indices), reverse=True)[:k]
        return SparseVector(self.size, [p[1] for p in pairs], [p[0] for p in pairs])
    
    def sparsity(self) -> float:
        """Fraction of non-zero elements."""
        return len(self.indices) / self.size if self.size > 0 else 0.0
    
    def invalidate_cache(self):
        """Call after modifying indices/values."""
        self._cache_dense = None


class SparseMatrix:
    """
    Sparse matrix in COO format for weight matrices.
    Memory: O(nnz) instead of O(n*m).
    """
    __slots__ = ['rows', 'cols', 'row_indices', 'col_indices', 'values', '_row_map']
    
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.row_indices: List[int] = []
        self.col_indices: List[int] = []
        self.values: List[float] = []
        self._row_map: Dict[int, List[int]] = {}  # row -> list of positions
    
    def set(self, row: int, col: int, value: float):
        """Set a value (overwrites if exists)."""
        # Check if exists
        if row in self._row_map:
            for pos in self._row_map[row]:
                if self.col_indices[pos] == col:
                    self.values[pos] = value
                    return
        
        # Add new entry
        pos = len(self.row_indices)
        self.row_indices.append(row)
        self.col_indices.append(col)
        self.values.append(value)
        
        if row not in self._row_map:
            self._row_map[row] = []
        self._row_map[row].append(pos)
    
    def get(self, row: int, col: int, default: float = 0.0) -> float:
        """Get value at position."""
        if row in self._row_map:
            for pos in self._row_map[row]:
                if self.col_indices[pos] == col:
                    return self.values[pos]
        return default
    
    def dot_vector(self, vec: SparseVector) -> SparseVector:
        """Matrix-vector multiplication: M @ v."""
        result = defaultdict(float)
        vec_map = {i: v for i, v in zip(vec.indices, vec.values)}
        
        for row, col, val in zip(self.row_indices, self.col_indices, self.values):
            if col in vec_map:
                result[row] += val * vec_map[col]
        
        indices = list(result.keys())
        values = [result[i] for i in indices]
        return SparseVector(self.rows, indices, values)
    
    def density(self) -> float:
        """Fraction of non-zero elements."""
        total = self.rows * self.cols
        return len(self.values) / total if total > 0 else 0.0


# =============================================================================
# GLOBAL EVENT QUEUE - Potato Optimization #2
# =============================================================================

@dataclass
class Event:
    """A scheduled event in the brain."""
    time: float
    event_type: str  # 'gamma', 'beta', 'theta', 'delta', 'spike', 'modulator'
    target: str  # Which system: 'cortex', 'reservoir', 'learning'
    data: Any = None
    priority: int = 0  # Higher = process first at same time


class EventQueue:
    """
    Single global event queue replacing 4 temporal layers.
    Implements dynamic timestep scaling for efficiency.
    """
    
    def __init__(self):
        self.events: List[Event] = []
        self.current_time: float = 0.0
        self.processed_count: int = 0
        
        # Dynamic timestep parameters
        self.min_dt: float = 0.001  # 1ms minimum
        self.max_dt: float = 0.1    # 100ms maximum
        self.current_dt: float = 0.01
        
        # Event type base intervals
        self.intervals = {
            'gamma': 0.01,   # 100Hz
            'beta': 0.04,    # 25Hz
            'theta': 0.15,   # ~7Hz
            'delta': 0.5,    # 2Hz
            'spike': 0.001,  # 1000Hz max
            'modulator': 0.1 # 10Hz
        }
    
    def schedule(self, event: Event):
        """Add event to queue (maintains sorted order by time)."""
        # Binary search insert
        lo, hi = 0, len(self.events)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.events[mid].time < event.time:
                lo = mid + 1
            elif self.events[mid].time == event.time and self.events[mid].priority >= event.priority:
                lo = mid + 1
            else:
                hi = mid
        self.events.insert(lo, event)
    
    def schedule_recurring(self, event_type: str, target: str, start_time: float = 0.0):
        """Schedule a recurring event based on its type interval."""
        interval = self.intervals.get(event_type, 0.01)
        self.schedule(Event(
            time=start_time + interval,
            event_type=event_type,
            target=target,
            priority=1 if event_type == 'gamma' else 0
        ))
    
    def pop_due(self, up_to_time: float) -> List[Event]:
        """Get all events due up to given time."""
        due = []
        while self.events and self.events[0].time <= up_to_time:
            due.append(self.events.pop(0))
            self.processed_count += 1
        return due
    
    def peek_next_time(self) -> Optional[float]:
        """Get time of next event without removing it."""
        return self.events[0].time if self.events else None
    
    def advance(self, dt: float = None) -> Tuple[float, List[Event]]:
        """
        Advance time and return due events.
        Uses dynamic timestep if dt not provided.
        """
        if dt is None:
            # Dynamic timestep: jump to next event or max_dt
            next_time = self.peek_next_time()
            if next_time is not None:
                dt = min(next_time - self.current_time, self.max_dt)
            else:
                dt = self.max_dt
            dt = max(dt, self.min_dt)
        
        self.current_time += dt
        self.current_dt = dt
        due = self.pop_due(self.current_time)
        return dt, due
    
    def clear(self):
        """Clear all pending events."""
        self.events.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        type_counts = defaultdict(int)
        for e in self.events:
            type_counts[e.event_type] += 1
        
        return {
            'pending': len(self.events),
            'processed': self.processed_count,
            'current_time': self.current_time,
            'current_dt': self.current_dt,
            'by_type': dict(type_counts)
        }


# =============================================================================
# BINARY RESERVOIR + LOOKUP TABLE - Potato Optimization #3
# =============================================================================

class BinaryReservoir:
    """
    Binary spike reservoir with lookup table for tanh approximation.
    Much faster than dense matrix operations.
    """
    
    # Pre-computed tanh lookup table (256 entries)
    TANH_LUT = [0.0] * 256
    for i in range(256):
        x = (i - 128) / 32.0  # Maps 0-255 to roughly -4 to 4
        TANH_LUT[i] = (1.0 - 2.0 / (1.0 + 2.718281828 ** (2 * x))) if abs(x) < 10 else (1.0 if x > 0 else -1.0)
    
    def __init__(self, size: int, sparsity: float = 0.1, spectral_radius: float = 0.95):
        self.size = size
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        
        # Binary state (0 or 1)
        self.state: List[int] = [0] * size
        
        # Pre-generated random transition rules
        # For each neuron: list of (source_indices, weights) pairs
        self.transitions: List[List[Tuple[int, int]]] = []
        self._init_transitions()
        
        # Threshold for spiking
        self.threshold: float = 0.5
        
        # Leak (probability of turning off without input)
        self.leak: float = 0.1
        
        # State history for output
        self.state_history: List[List[int]] = []
        self.max_history: int = 20
    
    def _init_transitions(self):
        """Pre-generate sparse random transitions."""
        import random
        random.seed(42)  # Reproducible
        
        n_connections = int(self.size * self.sparsity)
        
        for i in range(self.size):
            connections = []
            sources = random.sample(range(self.size), min(n_connections, self.size))
            for src in sources:
                # Weight as signed integer (-128 to 127 scaled)
                weight = random.randint(-64, 64)
                connections.append((src, weight))
            self.transitions.append(connections)
    
    def _fast_tanh(self, x: float) -> float:
        """Lookup table tanh approximation."""
        # Map x to 0-255 range
        idx = int((x + 4.0) * 32)
        idx = max(0, min(255, idx))
        return self.TANH_LUT[idx]
    
    def step(self, input_spikes: List[int]) -> List[int]:
        """
        Process one timestep with binary spikes.
        
        Args:
            input_spikes: List of input neuron indices that spiked
            
        Returns:
            List of output neuron indices that spiked
        """
        import random
        
        new_state = [0] * self.size
        input_set = set(input_spikes)
        
        for i in range(self.size):
            # Compute activation from transitions
            activation = 0
            for src, weight in self.transitions[i]:
                if self.state[src] == 1:
                    activation += weight
            
            # Add input contribution
            if i in input_set:
                activation += 64  # Strong input
            
            # Apply leak
            if self.state[i] == 1 and random.random() < self.leak:
                activation -= 32
            
            # Threshold + stochasticity
            prob = self._fast_tanh(activation / 64.0)
            if (prob + 1) / 2 > random.random():
                new_state[i] = 1
        
        self.state = new_state
        
        # Store history
        output_spikes = [i for i, s in enumerate(self.state) if s == 1]
        self.state_history.append(output_spikes)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        return output_spikes
    
    def get_rate_code(self) -> List[float]:
        """Convert recent spike history to rate code."""
        if not self.state_history:
            return [0.0] * self.size
        
        rates = [0.0] * self.size
        for spikes in self.state_history:
            for i in spikes:
                rates[i] += 1.0
        
        n = len(self.state_history)
        return [r / n for r in rates]
    
    def stats(self) -> Dict[str, float]:
        """Get reservoir statistics."""
        active = sum(self.state)
        return {
            'active_neurons': active,
            'sparsity': active / self.size if self.size > 0 else 0,
            'history_length': len(self.state_history)
        }


# =============================================================================
# LSH HASH LATTICE - Potato Optimization #4
# =============================================================================

class LSHHashLattice:
    """
    Locality-Sensitive Hashing for fast semantic similarity.
    Replaces cosine similarity with bitwise distance.
    100x faster with ~80% accuracy.
    """
    
    def __init__(self, dim: int, num_hashes: int = 64, num_tables: int = 4):
        """
        Args:
            dim: Embedding dimension
            num_hashes: Bits per hash (more = more accurate, slower)
            num_tables: Number of hash tables (more = better recall)
        """
        self.dim = dim
        self.num_hashes = num_hashes
        self.num_tables = num_tables
        
        # Random hyperplanes for hashing (one set per table)
        import random
        random.seed(42)
        self.hyperplanes: List[List[List[float]]] = []
        for _ in range(num_tables):
            table_planes = []
            for _ in range(num_hashes):
                # Random unit vector
                plane = [random.gauss(0, 1) for _ in range(dim)]
                norm = sum(x*x for x in plane) ** 0.5
                plane = [x / norm for x in plane]
                table_planes.append(plane)
            self.hyperplanes.append(table_planes)
        
        # Hash tables: table_idx -> {hash -> [word_ids]}
        self.tables: List[Dict[int, List[int]]] = [{} for _ in range(num_tables)]
        
        # Word storage: id -> (word, embedding)
        self.words: Dict[int, Tuple[str, List[float]]] = {}
        self.next_id: int = 0
    
    def _hash_vector(self, vec: List[float], table_idx: int) -> int:
        """Compute hash for a vector using random hyperplanes."""
        h = 0
        for i, plane in enumerate(self.hyperplanes[table_idx]):
            # Dot product
            dot = sum(v * p for v, p in zip(vec, plane))
            if dot > 0:
                h |= (1 << i)
        return h
    
    def _hamming_distance(self, h1: int, h2: int) -> int:
        """Count differing bits between two hashes."""
        x = h1 ^ h2
        count = 0
        while x:
            count += x & 1
            x >>= 1
        return count
    
    def add(self, word: str, embedding: List[float]) -> int:
        """Add a word with its embedding."""
        word_id = self.next_id
        self.next_id += 1
        
        self.words[word_id] = (word, embedding)
        
        # Add to all hash tables
        for t in range(self.num_tables):
            h = self._hash_vector(embedding, t)
            if h not in self.tables[t]:
                self.tables[t][h] = []
            self.tables[t][h].append(word_id)
        
        return word_id
    
    def query(self, embedding: List[float], k: int = 5, max_distance: int = 10) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors.
        
        Returns:
            List of (word, similarity) pairs, sorted by similarity
        """
        candidates = set()
        query_hashes = [self._hash_vector(embedding, t) for t in range(self.num_tables)]
        
        # Collect candidates from all tables
        for t in range(self.num_tables):
            qh = query_hashes[t]
            
            # Exact match
            if qh in self.tables[t]:
                candidates.update(self.tables[t][qh])
            
            # Also check nearby buckets (within Hamming distance)
            for h, ids in self.tables[t].items():
                if self._hamming_distance(qh, h) <= max_distance:
                    candidates.update(ids)
        
        # Compute actual distances for candidates
        results = []
        for word_id in candidates:
            word, stored_emb = self.words[word_id]
            
            # Approximate similarity from hash distance
            total_dist = sum(
                self._hamming_distance(query_hashes[t], self._hash_vector(stored_emb, t))
                for t in range(self.num_tables)
            )
            avg_dist = total_dist / (self.num_tables * self.num_hashes)
            similarity = 1.0 - avg_dist  # Convert distance to similarity
            
            results.append((word, similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def stats(self) -> Dict[str, Any]:
        """Get LSH statistics."""
        bucket_sizes = []
        for t in self.tables:
            bucket_sizes.extend(len(ids) for ids in t.values())
        
        return {
            'num_words': len(self.words),
            'num_tables': self.num_tables,
            'num_hashes': self.num_hashes,
            'avg_bucket_size': sum(bucket_sizes) / len(bucket_sizes) if bucket_sizes else 0,
            'total_buckets': sum(len(t) for t in self.tables)
        }


# =============================================================================
# RECIPROCAL SELF-COMPRESSION - Advanced Feature #1
# =============================================================================

class SelfCompressionEngine:
    """
    Reciprocal self-compression: the brain continuously compresses its own
    internal state into new representational primitives, then reuses those
    as new neurons/concepts.
    
    - Cortex invents new columns
    - Reservoir invents new dynamical modes
    - Semantic layer invents new concept embeddings
    
    This creates a living, compressive restructuring loop.
    Neurogenesis with PURPOSE, not RNG.
    """
    
    def __init__(self, pattern_dim: int = 100, max_primitives: int = 1000):
        self.pattern_dim = pattern_dim
        self.max_primitives = max_primitives
        
        # Learned primitives (prototypes)
        self.cortical_primitives: List[List[float]] = []
        self.reservoir_modes: List[List[float]] = []
        self.semantic_concepts: Dict[str, List[float]] = {}
        
        # Compression statistics
        self.compression_ratio: float = 1.0
        self.bits_saved: int = 0
        
        # Activation history for detecting recurring patterns
        self.activation_buffer: List[List[float]] = []
        self.buffer_size: int = 100
        
        # Novelty threshold for creating new primitives
        self.novelty_threshold: float = 0.3
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Fast cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def _find_best_match(self, pattern: List[float], primitives: List[List[float]]) -> Tuple[int, float]:
        """Find best matching primitive for a pattern."""
        if not primitives:
            return -1, 0.0
        
        best_idx = -1
        best_sim = -1.0
        for i, prim in enumerate(primitives):
            sim = self._cosine_similarity(pattern, prim)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        
        return best_idx, best_sim
    
    def compress_cortical(self, activation: List[float]) -> Tuple[int, float, bool]:
        """
        Compress cortical activation pattern.
        
        Returns:
            (primitive_id, reconstruction_error, created_new)
        """
        # Normalize
        norm = sum(x * x for x in activation) ** 0.5
        if norm > 0:
            activation = [x / norm for x in activation]
        
        # Find best matching primitive
        idx, similarity = self._find_best_match(activation, self.cortical_primitives)
        
        created_new = False
        
        if similarity < (1 - self.novelty_threshold):
            # Pattern is novel - create new primitive
            if len(self.cortical_primitives) < self.max_primitives:
                self.cortical_primitives.append(activation.copy())
                idx = len(self.cortical_primitives) - 1
                similarity = 1.0
                created_new = True
                self.bits_saved += self.pattern_dim  # We now represent this with just an index
            else:
                # Replace least-used primitive (simple LRU)
                idx = 0  # TODO: Track usage
                self.cortical_primitives[idx] = activation.copy()
                created_new = True
        else:
            # Update primitive with running average (online learning)
            alpha = 0.1
            prim = self.cortical_primitives[idx]
            for i in range(min(len(prim), len(activation))):
                prim[i] = (1 - alpha) * prim[i] + alpha * activation[i]
        
        reconstruction_error = 1.0 - similarity
        return idx, reconstruction_error, created_new
    
    def compress_reservoir(self, trajectory: List[List[float]]) -> Tuple[int, float, bool]:
        """
        Compress reservoir trajectory into a dynamical mode.
        
        A "mode" is a prototypical trajectory pattern.
        """
        if not trajectory:
            return -1, 1.0, False
        
        # Compute trajectory signature (mean + variance pattern)
        n = len(trajectory)
        dim = len(trajectory[0]) if trajectory else 0
        
        mean = [sum(t[i] for t in trajectory) / n for i in range(dim)]
        variance = [sum((t[i] - mean[i])**2 for t in trajectory) / n for i in range(dim)]
        
        # Signature = concatenate mean and variance
        signature = mean + variance
        
        # Find matching mode
        idx, similarity = self._find_best_match(signature, self.reservoir_modes)
        
        created_new = False
        
        if similarity < (1 - self.novelty_threshold):
            if len(self.reservoir_modes) < self.max_primitives:
                self.reservoir_modes.append(signature)
                idx = len(self.reservoir_modes) - 1
                created_new = True
        
        return idx, 1.0 - similarity, created_new

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            'compression_ratio': self.compression_ratio,
            'bits_saved': self.bits_saved,
            'num_cortical_primitives': len(self.cortical_primitives),
            'num_reservoir_modes': len(self.reservoir_modes),
            'num_concepts': len(self.semantic_concepts)
        }

    
    def compress_semantic(self, text: str, embedding: List[float]) -> Tuple[str, float, bool]:
        """
        Compress semantic content into concept embeddings.
        Returns concept key, compression error, and whether new concept created.
        """
        # Check similarity with existing concepts
        best_concept = None
        best_sim = 0.0
        
        for key, emb in self.semantic_concepts.items():
            sim = self._cosine_similarity(embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_concept = key
        
        created_new = False
        
        if best_sim < (1 - self.novelty_threshold):
            # Create new concept
            # Use first few words as key
            key = "_".join(text.split()[:3])[:20]
            self.semantic_concepts[key] = embedding.copy()
            best_concept = key
            created_new = True
        
        return best_concept, 1.0 - best_sim, created_new
    


    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get structural snapshot for inheritance.
        
        Returns:
            Dict containing primitives and modes.
        """
        return {
            'cortical_primitives': [p.copy() for p in self.cortical_primitives],
            'reservoir_modes': [m.copy() for m in self.reservoir_modes],
            'semantic_concepts': {k: v.copy() for k, v in self.semantic_concepts.items()}
        }
    
    def load_snapshot(self, snapshot: Dict[str, Any]):
        """
        Load structural snapshot (e.g. from parent).
        
        Args:
            snapshot: Dict from get_snapshot()
        """
        if not snapshot:
            return
            
        # Copy primitives - limit to max
        if 'cortical_primitives' in snapshot:
            self.cortical_primitives = [p.copy() for p in snapshot['cortical_primitives'][:self.max_primitives]]
            
        if 'reservoir_modes' in snapshot:
            self.reservoir_modes = [m.copy() for m in snapshot['reservoir_modes'][:self.max_primitives]]
            
        if 'semantic_concepts' in snapshot:
            # Merge concepts
            for k, v in snapshot['semantic_concepts'].items():
                if len(self.semantic_concepts) < self.max_primitives:
                    self.semantic_concepts[k] = v.copy()


# =============================================================================
# EMERGENT MOTOR LOOP - Advanced Feature #2
# =============================================================================

class InternalBody:
    """
    Emergent motor loop: a synthetic internal body that creates a closed
    perception-action loop without physical limbs.
    
    Components:
    - Internal muscles: vectors that the brain can "move"
    - Internal sensations: noise patterns the body generates
    - Internal rewards: pleasure/pain from body states
    
    Intelligence without embodiment is rare; simulated embodiment
    inside a cognitive engine is almost unheard of.
    """
    
    def __init__(self, num_muscles: int = 8, num_sensors: int = 16):
        self.num_muscles = num_muscles
        self.num_sensors = num_sensors
        
        # Muscle state (effort being exerted)
        self.muscle_activation: List[float] = [0.0] * num_muscles
        
        # Muscle fatigue
        self.muscle_fatigue: List[float] = [0.0] * num_muscles
        
        # Proprioceptive state (where muscles "are")
        self.proprioception: List[float] = [0.5] * num_muscles
        
        # Internal sensations (interoception)
        self.sensations: List[float] = [0.0] * num_sensors
        
        # Body state variables
        self.energy: float = 1.0
        self.arousal: float = 0.5
        self.comfort: float = 0.7
        
        # Homeostatic targets
        self.target_energy: float = 0.8
        self.target_arousal: float = 0.5
        self.target_comfort: float = 0.8
        
        # Internal reward signal
        self.reward: float = 0.0
        
        # History for learning
        self.action_history: List[List[float]] = []
        self.reward_history: List[float] = []
    
    def act(self, motor_commands: List[float]) -> Dict[str, Any]:
        """
        Execute motor commands and update body state.
        
        Args:
            motor_commands: Desired muscle activations (0-1)
            
        Returns:
            Dict with sensations, reward, and body state
        """
        import random
        
        # Clip commands
        commands = [max(0, min(1, c)) for c in motor_commands[:self.num_muscles]]
        while len(commands) < self.num_muscles:
            commands.append(0.0)
        
        # Apply fatigue - fatigued muscles respond less
        actual_activation = []
        for i, cmd in enumerate(commands):
            fatigue_factor = 1.0 - self.muscle_fatigue[i] * 0.8
            actual = cmd * fatigue_factor
            actual_activation.append(actual)
            self.muscle_activation[i] = actual
        
        # Update fatigue (muscles tire with use, recover with rest)
        for i in range(self.num_muscles):
            if self.muscle_activation[i] > 0.3:
                self.muscle_fatigue[i] = min(1.0, self.muscle_fatigue[i] + 0.1)
            else:
                self.muscle_fatigue[i] = max(0.0, self.muscle_fatigue[i] - 0.05)
        
        # Update proprioception (muscle position changes with activation)
        for i in range(self.num_muscles):
            self.proprioception[i] += 0.1 * (self.muscle_activation[i] - 0.5)
            self.proprioception[i] = max(0, min(1, self.proprioception[i]))
        
        # Energy cost of action
        energy_cost = sum(actual_activation) * 0.05
        self.energy = max(0, self.energy - energy_cost)
        
        # Energy recovery (baseline metabolism)
        self.energy = min(1.0, self.energy + 0.02)
        
        # Update arousal based on activity
        activity_level = sum(actual_activation) / self.num_muscles
        self.arousal = 0.9 * self.arousal + 0.1 * activity_level
        
        # Generate internal sensations
        self._update_sensations()
        
        # Compute reward (homeostatic)
        self._compute_reward()
        
        # Store history
        self.action_history.append(actual_activation.copy())
        self.reward_history.append(self.reward)
        if len(self.action_history) > 100:
            self.action_history.pop(0)
            self.reward_history.pop(0)
        
        return {
            'sensations': self.sensations.copy(),
            'proprioception': self.proprioception.copy(),
            'reward': self.reward,
            'energy': self.energy,
            'arousal': self.arousal,
            'comfort': self.comfort
        }
    
    def _update_sensations(self):
        """Generate internal sensations based on body state."""
        import random
        
        # Hunger sensation (low energy)
        self.sensations[0] = 1.0 - self.energy
        
        # Fatigue sensation
        self.sensations[1] = sum(self.muscle_fatigue) / self.num_muscles
        
        # Arousal sensation
        self.sensations[2] = self.arousal
        
        # Comfort sensation
        self.sensations[3] = self.comfort
        
        # Proprioceptive feedback (summarized)
        self.sensations[4] = sum(self.proprioception) / self.num_muscles
        self.sensations[5] = max(self.proprioception) - min(self.proprioception)  # Spread
        
        # Random interoceptive noise (heartbeat, breathing, etc.)
        for i in range(6, self.num_sensors):
            self.sensations[i] = 0.8 * self.sensations[i] + 0.2 * random.random()
    
    def _compute_reward(self):
        """Compute reward based on homeostatic state."""
        # Reward for being near homeostatic targets
        energy_error = abs(self.energy - self.target_energy)
        arousal_error = abs(self.arousal - self.target_arousal)
        comfort_error = abs(self.comfort - self.target_comfort)
        
        total_error = energy_error + arousal_error + comfort_error
        
        # Reward is inverse of error
        self.reward = 1.0 - total_error / 3.0
        
        # Penalty for extreme fatigue
        if max(self.muscle_fatigue) > 0.8:
            self.reward -= 0.2
        
        # Bonus for movement variety (exploration)
        if self.action_history and len(self.action_history) > 5:
            recent = self.action_history[-5:]
            variance = sum(
                sum((a - b)**2 for a, b in zip(recent[i], recent[i+1]))
                for i in range(len(recent)-1)
            )
            self.reward += 0.1 * min(1.0, variance)
        
        self.reward = max(-1, min(1, self.reward))
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete body state."""
        return {
            'muscle_activation': self.muscle_activation.copy(),
            'muscle_fatigue': self.muscle_fatigue.copy(),
            'proprioception': self.proprioception.copy(),
            'sensations': self.sensations.copy(),
            'energy': self.energy,
            'arousal': self.arousal,
            'comfort': self.comfort,
            'reward': self.reward
        }
    
    def reset(self):
        """Reset body to resting state."""
        self.muscle_activation = [0.0] * self.num_muscles
        self.muscle_fatigue = [0.0] * self.num_muscles
        self.proprioception = [0.5] * self.num_muscles
        self.energy = 1.0
        self.arousal = 0.5
        self.comfort = 0.7


# =============================================================================
# TIER 4: TOOL USE EMERGENCE
# =============================================================================

class ToolUseSystem:
    """
    Tool Use Emergence: Creatures learn to use objects as extensions of their body.
    
    Key principles:
    - Tools are represented as extensions of body schema (proprioceptive incorporation)
    - Success with a tool strengthens the neural pathway connecting action→tool→outcome
    - Objects become "transparent" to the user when mastered (like a hammer becomes part of the arm)
    - Failure weakens tool-action associations
    
    This is NOT hardcoded tool use - it emerges from motor-sensory loop learning.
    """
    
    def __init__(self, max_tools: int = 20):
        self.max_tools = max_tools
        
        # Tool representations: each tool has an embedding that can fuse with body schema
        self.tool_embeddings: Dict[str, np.ndarray] = {}  # tool_id -> embedding (64-dim)
        self.tool_familiarity: Dict[str, float] = {}  # How well we know each tool
        self.tool_success_history: Dict[str, List[float]] = {}  # Recent success rates
        
        # Body schema extension - when holding a tool, proprioception extends
        self.current_tool: Optional[str] = None
        self.tool_incorporation_strength: float = 0.0  # How "part of body" the tool feels
        
        # Motor-tool binding: which motor patterns work with which tools
        self.motor_tool_associations: Dict[str, np.ndarray] = {}  # tool_id -> motor pattern
        
        # Learning parameters
        self.incorporation_rate = 0.1  # How fast tools become "part of body"
        self.forgetting_rate = 0.01   # How fast unused tool skills decay
        
    def encounter_object(self, object_id: str, object_features: np.ndarray) -> Dict[str, Any]:
        """
        Encounter a new or known object that could be used as a tool.
        
        Args:
            object_id: Unique identifier for the object
            object_features: Feature vector describing the object
            
        Returns:
            Recognition info and suggested motor patterns
        """
        # Initialize if new tool
        if object_id not in self.tool_embeddings:
            if len(self.tool_embeddings) >= self.max_tools:
                # Remove least familiar tool
                if self.tool_familiarity:
                    weakest = min(self.tool_familiarity, key=self.tool_familiarity.get)
                    del self.tool_embeddings[weakest]
                    del self.tool_familiarity[weakest]
                    if weakest in self.tool_success_history:
                        del self.tool_success_history[weakest]
                    if weakest in self.motor_tool_associations:
                        del self.motor_tool_associations[weakest]
            
            # Create embedding (normalized)
            embedding = object_features[:64] if len(object_features) >= 64 else np.zeros(64)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            self.tool_embeddings[object_id] = embedding
            self.tool_familiarity[object_id] = 0.0
            self.tool_success_history[object_id] = []
            self.motor_tool_associations[object_id] = np.random.randn(8) * 0.1  # Random motor pattern
        
        familiarity = self.tool_familiarity[object_id]
        
        return {
            'is_known': familiarity > 0.1,
            'familiarity': familiarity,
            'suggested_motor_pattern': self.motor_tool_associations[object_id].copy(),
            'success_rate': np.mean(self.tool_success_history[object_id][-10:]) if self.tool_success_history[object_id] else 0.0
        }
    
    def grasp_tool(self, object_id: str) -> bool:
        """
        Attempt to grasp a tool, extending body schema.
        
        Returns: True if tool was grasped
        """
        if object_id not in self.tool_embeddings:
            return False
            
        self.current_tool = object_id
        self.tool_incorporation_strength = self.tool_familiarity.get(object_id, 0.0) * 0.5
        
        return True
    
    def release_tool(self):
        """Release currently held tool."""
        self.current_tool = None
        self.tool_incorporation_strength = 0.0
    
    def use_tool(self, motor_command: np.ndarray, outcome_success: float) -> Dict[str, Any]:
        """
        Use current tool with motor command and learn from outcome.
        
        Args:
            motor_command: 8-dim motor activation pattern
            outcome_success: 0-1 success of the action
            
        Returns:
            Learning result
        """
        if self.current_tool is None:
            return {'error': 'no_tool_held'}
        
        tool_id = self.current_tool
        
        # Record success
        self.tool_success_history[tool_id].append(outcome_success)
        if len(self.tool_success_history[tool_id]) > 100:
            self.tool_success_history[tool_id].pop(0)
        
        # Update familiarity based on use
        delta_familiarity = self.incorporation_rate * (outcome_success - 0.3)  # Baseline at 0.3
        self.tool_familiarity[tool_id] = np.clip(
            self.tool_familiarity[tool_id] + delta_familiarity, 0.0, 1.0
        )
        
        # Update motor-tool association (what motor patterns work with this tool)
        # Hebbian-ish: if success, strengthen this motor pattern for this tool
        learning_signal = (outcome_success - 0.5) * 2  # -1 to +1
        self.motor_tool_associations[tool_id] += 0.1 * learning_signal * motor_command[:8]
        self.motor_tool_associations[tool_id] = np.clip(
            self.motor_tool_associations[tool_id], -1, 1
        )
        
        # Update incorporation strength (tool becomes more "part of body")
        if outcome_success > 0.5:
            self.tool_incorporation_strength = min(
                1.0, self.tool_incorporation_strength + self.incorporation_rate
            )
        
        return {
            'familiarity': self.tool_familiarity[tool_id],
            'incorporation': self.tool_incorporation_strength,
            'success_rate': np.mean(self.tool_success_history[tool_id][-10:])
        }
    
    def get_extended_proprioception(self, base_proprioception: List[float]) -> List[float]:
        """
        Get proprioception extended by current tool.
        
        When holding a familiar tool, proprioception extends to include the tool.
        """
        if self.current_tool is None or self.tool_incorporation_strength < 0.1:
            return base_proprioception
        
        # Tool extends proprioception - add virtual "limb" positions
        tool_embedding = self.tool_embeddings[self.current_tool][:len(base_proprioception)]
        
        # Blend tool embedding with proprioception based on incorporation
        extended = []
        for i, p in enumerate(base_proprioception):
            if i < len(tool_embedding):
                # More incorporated = more the tool feels like part of body
                blended = p + self.tool_incorporation_strength * tool_embedding[i] * 0.3
                extended.append(np.clip(blended, 0, 1))
            else:
                extended.append(p)
        
        return extended
    
    def decay_unused_tools(self):
        """Decay familiarity with unused tools over time."""
        for tool_id in list(self.tool_familiarity.keys()):
            if tool_id != self.current_tool:
                self.tool_familiarity[tool_id] = max(
                    0.0, self.tool_familiarity[tool_id] - self.forgetting_rate
                )
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete tool use system state."""
        return {
            'current_tool': self.current_tool,
            'incorporation_strength': self.tool_incorporation_strength,
            'known_tools': list(self.tool_embeddings.keys()),
            'tool_familiarity': dict(self.tool_familiarity),
            'num_tools': len(self.tool_embeddings)
        }


# =============================================================================
# TIER 4: ABSTRACT REASONING
# =============================================================================

class AbstractReasoningSystem:
    """
    Abstract Reasoning: Pattern recognition and analogical thinking.
    
    Key principles:
    - Patterns are discovered through compression (SelfCompressionEngine)
    - Analogies map relationships between pattern domains
    - Reasoning emerges from reservoir dynamics, not symbolic rules
    - "If A:B :: C:?" solved by finding D that completes relationship
    
    This creates genuine abstraction through structure, not hardcoded logic.
    """
    
    def __init__(self, pattern_dim: int = 64, max_concepts: int = 100):
        self.pattern_dim = pattern_dim
        self.max_concepts = max_concepts
        
        # Abstract concepts - discovered patterns
        self.concepts: Dict[str, np.ndarray] = {}  # concept_id -> embedding
        self.concept_relations: Dict[Tuple[str, str], np.ndarray] = {}  # (A,B) -> relationship vector
        self.concept_usage: Dict[str, int] = {}  # How often each concept is activated
        
        # Analogy history - successful analogies strengthen pathways
        self.analogy_history: List[Dict[str, Any]] = []
        
        # Pattern detection state
        self.active_patterns: List[np.ndarray] = []
        self.pattern_buffer_size = 50
        
    def observe_pattern(self, activation: np.ndarray, context: str = "") -> Dict[str, Any]:
        """
        Observe an activation pattern and potentially discover concepts.
        
        Args:
            activation: Neural activation pattern
            context: Optional context string
            
        Returns:
            Recognition and concept discovery info
        """
        # Normalize to pattern_dim
        if len(activation) > self.pattern_dim:
            pattern = activation[:self.pattern_dim]
        else:
            pattern = np.zeros(self.pattern_dim)
            pattern[:len(activation)] = activation
        pattern = pattern / (np.linalg.norm(pattern) + 1e-8)
        
        # Add to buffer
        self.active_patterns.append(pattern)
        if len(self.active_patterns) > self.pattern_buffer_size:
            self.active_patterns.pop(0)
        
        # Check similarity to known concepts
        best_match = None
        best_similarity = 0.0
        
        for concept_id, concept_vec in self.concepts.items():
            similarity = float(np.dot(pattern, concept_vec))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = concept_id
        
        result = {
            'recognized': best_similarity > 0.7,
            'best_match': best_match,
            'similarity': best_similarity,
            'new_concept': False
        }
        
        # If not recognized, potentially create new concept
        if best_similarity < 0.5 and len(self.concepts) < self.max_concepts:
            # Check if this pattern recurs in buffer
            recurring = self._check_pattern_recurrence(pattern)
            if recurring > 2:  # Pattern appeared 3+ times
                concept_id = f"concept_{len(self.concepts)}"
                self.concepts[concept_id] = pattern.copy()
                self.concept_usage[concept_id] = recurring
                result['new_concept'] = True
                result['created_concept'] = concept_id
        
        # Update usage of matched concept
        if best_match and best_similarity > 0.7:
            self.concept_usage[best_match] = self.concept_usage.get(best_match, 0) + 1
        
        return result
    
    def _check_pattern_recurrence(self, pattern: np.ndarray, threshold: float = 0.8) -> int:
        """Count how many times this pattern appears in buffer."""
        count = 0
        for p in self.active_patterns:
            if np.dot(pattern, p) > threshold:
                count += 1
        return count
    
    def learn_relationship(self, concept_a: str, concept_b: str, relationship_vec: Optional[np.ndarray] = None) -> bool:
        """
        Learn a relationship between two concepts.
        
        If relationship_vec is None, compute it as B - A (vector difference).
        """
        if concept_a not in self.concepts or concept_b not in self.concepts:
            return False
        
        if relationship_vec is None:
            # Relationship is the transformation from A to B
            relationship_vec = self.concepts[concept_b] - self.concepts[concept_a]
        
        self.concept_relations[(concept_a, concept_b)] = relationship_vec.copy()
        return True
    
    def solve_analogy(self, concept_a: str, concept_b: str, concept_c: str) -> Tuple[Optional[str], float]:
        """
        Solve: A:B :: C:?
        
        Find the concept D such that the relationship A→B matches C→D.
        
        Returns:
            (concept_d, confidence) or (None, 0.0) if no solution found
        """
        if concept_a not in self.concepts or concept_b not in self.concepts or concept_c not in self.concepts:
            return None, 0.0
        
        # Get or compute the A→B relationship
        if (concept_a, concept_b) in self.concept_relations:
            relationship = self.concept_relations[(concept_a, concept_b)]
        else:
            relationship = self.concepts[concept_b] - self.concepts[concept_a]
        
        # Apply relationship to C to predict D
        predicted_d = self.concepts[concept_c] + relationship
        predicted_d = predicted_d / (np.linalg.norm(predicted_d) + 1e-8)
        
        # Find closest concept to predicted_d
        best_match = None
        best_similarity = 0.0
        
        for concept_id, concept_vec in self.concepts.items():
            if concept_id == concept_c:
                continue  # D shouldn't be C
            similarity = float(np.dot(predicted_d, concept_vec))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = concept_id
        
        # Record successful analogies
        if best_match and best_similarity > 0.6:
            self.analogy_history.append({
                'A': concept_a, 'B': concept_b, 
                'C': concept_c, 'D': best_match,
                'confidence': best_similarity
            })
            if len(self.analogy_history) > 100:
                self.analogy_history.pop(0)
        
        return best_match, best_similarity
    
    def abstract_rule_detection(self) -> List[Dict[str, Any]]:
        """
        Detect abstract rules from relationship patterns.
        
        Looks for relationships that appear multiple times across different concepts.
        """
        if len(self.concept_relations) < 3:
            return []
        
        # Cluster relationships
        relationships = list(self.concept_relations.values())
        detected_rules = []
        
        # Simple clustering: find similar relationships
        for i, rel_i in enumerate(relationships):
            similar_count = 0
            for j, rel_j in enumerate(relationships):
                if i != j:
                    similarity = np.dot(rel_i, rel_j) / (np.linalg.norm(rel_i) * np.linalg.norm(rel_j) + 1e-8)
                    if similarity > 0.7:
                        similar_count += 1
            
            if similar_count >= 2:  # This relationship pattern appears 3+ times
                rule_name = f"rule_{len(detected_rules)}"
                detected_rules.append({
                    'rule_id': rule_name,
                    'relationship_vector': rel_i.copy(),
                    'occurrences': similar_count + 1
                })
        
        return detected_rules
    
    def prune_unused_concepts(self, min_usage: int = 2):
        """Remove concepts that are rarely used."""
        to_remove = [c for c, count in self.concept_usage.items() if count < min_usage]
        for c in to_remove:
            del self.concepts[c]
            del self.concept_usage[c]
            # Remove relationships involving this concept
            self.concept_relations = {
                k: v for k, v in self.concept_relations.items()
                if c not in k
            }
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete abstract reasoning state."""
        return {
            'num_concepts': len(self.concepts),
            'num_relationships': len(self.concept_relations),
            'num_rules': len(self.abstract_rule_detection()),
            'concept_usage': dict(self.concept_usage),
            'recent_analogies': self.analogy_history[-5:]
        }


# =============================================================================
# TIER 4: SOCIAL STRUCTURES
# =============================================================================

class SocialStructureSystem:
    """
    Social Structures: Emergent hierarchy, cooperation, and resource sharing.
    
    Key principles:
    - Social rank emerges from interaction outcomes, not assignment
    - Cooperation happens when mutual benefit is predicted
    - Resource sharing influenced by oxytocin levels
    - Groups form naturally through repeated positive interactions
    
    Individual brains don't store group structure - it's distributed across interactions.
    """
    
    def __init__(self, max_relationships: int = 50):
        self.max_relationships = max_relationships
        
        # Relationship memory: how we relate to other agents
        # Values: affinity (-1 to +1), dominance (-1 to +1), trust (0 to 1)
        self.relationships: Dict[str, Dict[str, float]] = {}
        
        # Interaction history per agent
        self.interaction_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Group memberships (emergent)
        self.groups: Dict[str, Set[str]] = {}  # group_id -> set of agent_ids
        self.my_groups: Set[str] = set()
        
        # Resource sharing state
        self.shared_resources: Dict[str, float] = {}  # resource_id -> amount shared
        self.received_resources: Dict[str, float] = {}  # resource_id -> amount received
        
        # Cooperation prediction model (simple linear)
        self.cooperation_weights = np.random.randn(5) * 0.1  # Features: affinity, dominance, trust, oxytocin, history
        
    def observe_agent(self, agent_id: str, features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Observe another agent, initializing relationship if new.
        
        Args:
            agent_id: Unique identifier for the other agent
            features: Optional feature vector describing the agent
            
        Returns:
            Relationship info and social recommendations
        """
        if agent_id not in self.relationships:
            if len(self.relationships) >= self.max_relationships:
                # Remove weakest relationship
                weakest = min(
                    self.relationships.keys(),
                    key=lambda a: abs(self.relationships[a].get('affinity', 0))
                )
                del self.relationships[weakest]
                if weakest in self.interaction_history:
                    del self.interaction_history[weakest]
            
            # Initialize new relationship (neutral)
            self.relationships[agent_id] = {
                'affinity': 0.0,       # Like/dislike
                'dominance': 0.0,      # Relative status (+ = I'm dominant)
                'trust': 0.3,          # How much I trust them
                'familiarity': 0.0,    # How well I know them
            }
            self.interaction_history[agent_id] = []
        
        rel = self.relationships[agent_id]
        
        # Predict cooperation likelihood
        coop_features = np.array([
            rel['affinity'],
            rel['dominance'],
            rel['trust'],
            0.5,  # Placeholder for oxytocin (will be filled in by brain)
            len(self.interaction_history.get(agent_id, [])) / 10.0
        ])
        coop_prediction = float(np.tanh(np.dot(self.cooperation_weights, coop_features)))
        
        return {
            'known': rel['familiarity'] > 0.1,
            'relationship': dict(rel),
            'cooperation_likelihood': (coop_prediction + 1) / 2,  # 0-1 scale
            'shared_groups': self._get_shared_groups(agent_id)
        }
    
    def _get_shared_groups(self, agent_id: str) -> List[str]:
        """Get groups shared with another agent."""
        shared = []
        for group_id, members in self.groups.items():
            if agent_id in members and group_id in self.my_groups:
                shared.append(group_id)
        return shared
    
    def record_interaction(
        self, 
        agent_id: str, 
        interaction_type: str,  # "cooperation", "competition", "neutral", "conflict"
        outcome: float,  # -1 to +1 (my perspective)
        their_outcome: float = 0.0  # -1 to +1 (their perspective)
    ) -> Dict[str, Any]:
        """
        Record an interaction with another agent and update relationship.
        
        Returns:
            Updated relationship state
        """
        if agent_id not in self.relationships:
            self.observe_agent(agent_id)
        
        rel = self.relationships[agent_id]
        history = self.interaction_history.setdefault(agent_id, [])
        
        # Record interaction
        interaction = {
            'type': interaction_type,
            'outcome': outcome,
            'their_outcome': their_outcome
        }
        history.append(interaction)
        if len(history) > 50:
            history.pop(0)
        
        # Update relationship based on interaction
        learning_rate = 0.2
        
        # Affinity: increases with mutual benefit, decreases with conflict
        mutual_benefit = (outcome + their_outcome) / 2
        if interaction_type == "cooperation":
            rel['affinity'] += learning_rate * (0.3 + mutual_benefit * 0.2)
        elif interaction_type == "conflict":
            rel['affinity'] -= learning_rate * 0.3
        else:
            rel['affinity'] += learning_rate * outcome * 0.1
        rel['affinity'] = np.clip(rel['affinity'], -1, 1)
        
        # Dominance: winning increases, losing decreases
        if interaction_type == "competition":
            rel['dominance'] += learning_rate * (outcome - their_outcome) * 0.3
        elif interaction_type == "conflict":
            rel['dominance'] += learning_rate * outcome * 0.2
        rel['dominance'] = np.clip(rel['dominance'], -1, 1)
        
        # Trust: builds slowly with positive outcomes, breaks quickly with betrayal
        if outcome > 0:
            rel['trust'] += learning_rate * 0.1
        elif outcome < -0.5 and interaction_type == "cooperation":
            # Betrayal - they defected in cooperation
            rel['trust'] -= learning_rate * 0.5
        elif outcome < 0:
            rel['trust'] -= learning_rate * 0.1
        rel['trust'] = np.clip(rel['trust'], 0, 1)
        
        # Familiarity always increases with interaction
        rel['familiarity'] = min(1.0, rel['familiarity'] + 0.05)
        
        # Update cooperation prediction weights (simple reinforcement)
        if interaction_type == "cooperation":
            # If cooperation succeeded, strengthen weights that predicted it
            coop_features = np.array([
                rel['affinity'], rel['dominance'], rel['trust'], 0.5, len(history) / 10.0
            ])
            error = outcome - np.tanh(np.dot(self.cooperation_weights, coop_features))
            self.cooperation_weights += 0.01 * error * coop_features
        
        return {'relationship': dict(rel)}
    
    def propose_cooperation(self, agent_id: str, resource_value: float = 0.5) -> Dict[str, Any]:
        """
        Propose cooperation with another agent.
        
        Returns predicted outcomes and whether to proceed.
        """
        if agent_id not in self.relationships:
            return {'should_cooperate': False, 'reason': 'unknown_agent'}
        
        rel = self.relationships[agent_id]
        
        # Decision based on trust and affinity
        trust_threshold = 0.3 - resource_value * 0.2  # Higher stakes = need more trust
        affinity_threshold = -0.3
        
        should_cooperate = rel['trust'] > trust_threshold and rel['affinity'] > affinity_threshold
        
        # Expected outcome
        expected_outcome = rel['trust'] * 0.5 + rel['affinity'] * 0.3
        
        return {
            'should_cooperate': should_cooperate,
            'expected_outcome': expected_outcome,
            'trust': rel['trust'],
            'affinity': rel['affinity'],
            'reason': 'trust_and_affinity' if should_cooperate else 'insufficient_trust'
        }
    
    def share_resource(self, agent_id: str, resource_id: str, amount: float, oxytocin_level: float = 0.5) -> Dict[str, Any]:
        """
        Share a resource with another agent.
        
        Sharing is modulated by oxytocin and relationship quality.
        """
        if agent_id not in self.relationships:
            self.observe_agent(agent_id)
        
        rel = self.relationships[agent_id]
        
        # Willingness to share based on oxytocin and affinity
        share_willingness = 0.3 + oxytocin_level * 0.4 + rel['affinity'] * 0.3
        share_willingness = np.clip(share_willingness, 0, 1)
        
        # Actual amount shared
        actual_amount = amount * share_willingness
        
        # Record sharing
        self.shared_resources[resource_id] = self.shared_resources.get(resource_id, 0) + actual_amount
        
        # Sharing builds affinity and trust
        rel['affinity'] += 0.05 * actual_amount
        rel['trust'] += 0.02 * actual_amount
        rel['affinity'] = np.clip(rel['affinity'], -1, 1)
        rel['trust'] = np.clip(rel['trust'], 0, 1)
        
        return {
            'amount_shared': actual_amount,
            'willingness': share_willingness,
            'relationship_updated': dict(rel)
        }
    
    def form_group(self, group_id: str, initial_members: List[str]) -> bool:
        """Form or join a group with other agents."""
        if group_id not in self.groups:
            self.groups[group_id] = set()
        
        self.groups[group_id].update(initial_members)
        self.my_groups.add(group_id)
        
        # Being in a group increases affinity between members
        for member in initial_members:
            if member in self.relationships:
                self.relationships[member]['affinity'] += 0.1
                self.relationships[member]['affinity'] = min(1.0, self.relationships[member]['affinity'])
        
        return True
    
    def get_social_hierarchy(self) -> List[Tuple[str, float]]:
        """
        Get perceived social hierarchy based on dominance relationships.
        
        Returns list of (agent_id, relative_rank) sorted by rank.
        """
        if not self.relationships:
            return [('self', 0.0)]
        
        # Compute relative ranks
        my_rank = 0.0  # Self is reference point
        ranks = [('self', my_rank)]
        
        for agent_id, rel in self.relationships.items():
            # Their rank relative to me (negative dominance means they're higher)
            their_rank = -rel['dominance']
            ranks.append((agent_id, their_rank))
        
        # Sort by rank (higher rank = more dominant)
        ranks.sort(key=lambda x: x[1], reverse=True)
        return ranks
    
    def decay_relationships(self):
        """Decay relationships that aren't maintained."""
        for agent_id in list(self.relationships.keys()):
            rel = self.relationships[agent_id]
            # Affinity decays toward neutral
            rel['affinity'] *= 0.99
            # Trust decays slowly
            rel['trust'] = max(0.2, rel['trust'] * 0.995)  # Min baseline trust
            # Dominance is more stable
            rel['dominance'] *= 0.999
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete social structure state."""
        return {
            'num_relationships': len(self.relationships),
            'num_groups': len(self.my_groups),
            'hierarchy': self.get_social_hierarchy()[:5],
            'avg_affinity': np.mean([r['affinity'] for r in self.relationships.values()]) if self.relationships else 0,
            'avg_trust': np.mean([r['trust'] for r in self.relationships.values()]) if self.relationships else 0.3,
            'total_shared': sum(self.shared_resources.values()),
            'total_received': sum(self.received_resources.values())
        }


# =============================================================================
# BRAIN CONFIGURATION
# =============================================================================

@dataclass
class BrainConfig:
    """
    Configuration for the Three-System Brain.
    
    All hyperparameters are exposed here for easy experimentation.
    Use create_brain(scale, **overrides) to customize any parameter.
    """
    # ==========================================================================
    # SCALE PARAMETERS
    # ==========================================================================
    input_dim: int = 300
    num_columns: int = 200
    cells_per_column: int = 32
    reservoir_size: int = 2000
    output_dim: int = 300
    
    # ==========================================================================
    # SPARSITY & CORTEX PARAMETERS
    # ==========================================================================
    target_sparsity: float = 0.02  # 2% active neurons (k-winners)
    lateral_inhibition_strength: float = 0.3  # Mexican-hat lateral weights
    predictive_boost: float = 0.2  # Boost for predicted columns
    duty_cycle_alpha: float = 0.001  # Duty cycle learning rate
    
    # ==========================================================================
    # RESERVOIR PARAMETERS
    # ==========================================================================
    spectral_radius: float = 0.95  # Controls chaos (< 1 for stability)
    reservoir_sparsity: float = 0.1  # Connection sparsity
    leak_rate: float = 0.3  # Leaky integration rate
    
    # ==========================================================================
    # LANGUAGE DECODER
    # ==========================================================================
    vocabulary_size: int = 10000
    embedding_dim: int = 300
    
    # ==========================================================================
    # STRUCTURAL PLASTICITY (NEUROGENESIS/PRUNING)
    # ==========================================================================
    dynamic_neurons: bool = True
    max_neurons: int = 100000
    neurogenesis_rate: float = 0.5  # Probability of spawning when triggered (was 0.2)
    pruning_threshold: float = 0.01  # Activity threshold for pruning (was 0.001)
    min_columns: int = 10  # Never prune below this
    
    # ==========================================================================
    # NEUROMODULATOR BASELINES
    # ==========================================================================
    dopamine_baseline: float = 0.5
    serotonin_baseline: float = 0.5
    norepinephrine_baseline: float = 0.3
    acetylcholine_baseline: float = 0.5
    cortisol_baseline: float = 0.3
    gaba_baseline: float = 0.4
    glutamate_baseline: float = 0.5
    oxytocin_baseline: float = 0.3
    
    # ==========================================================================
    # NEUROMODULATOR DECAY RATES (per timestep)
    # ==========================================================================
    dopamine_decay: float = 0.05  # How fast DA returns to baseline
    serotonin_decay: float = 0.03  # Slower decay for mood stability
    norepinephrine_decay: float = 0.08  # Faster for arousal
    acetylcholine_decay: float = 0.06  # Medium for attention
    
    # ==========================================================================
    # SYSTEM GLUE PARAMETERS (cross-system modulation)
    # ==========================================================================
    ach_sparsity_gain: float = 0.6  # How much ACh affects sparsity
    da_gain_effect: float = 0.4  # How much DA affects activation gain
    serotonin_plasticity_effect: float = 0.4  # How much 5-HT stabilizes learning
    confidence_dopamine_feedback: float = 0.4  # Decoder confidence → DA
    
    # ==========================================================================
    # DREAMING / REPLAY CONFIGURATION
    # ==========================================================================
    dream_enabled: bool = True
    dream_steps: int = 5
    dream_serotonin_threshold: float = 0.45
    dream_fatigue_threshold: float = 0.7
    dream_rewire_count: int = 50
    
    # ==========================================================================
    # PERFORMANCE TUNING
    # ==========================================================================
    use_fp16_reservoir: bool = False  # Use float16 for reservoir weights
    reservoir_update_fraction: float = 1.0  # Fraction of reservoir to update each step
    
    def total_neurons(self) -> int:
        """Calculate total neuron count."""
        return self.num_columns * self.cells_per_column


@dataclass 
class BrainState:
    """Current state of the brain."""
    # Activity
    global_activity: float = 0.0
    
    # Neuromodulators (mirror for compatibility)
    dopamine_level: float = 0.5
    serotonin_level: float = 0.5
    norepinephrine_level: float = 0.3
    acetylcholine_level: float = 0.5
    cortisol_level: float = 0.3
    gaba_level: float = 0.4
    glutamate_level: float = 0.5
    oxytocin_level: float = 0.3
    
    # Energy
    total_energy: float = 100.0
    
    # Temporal
    current_time: float = 0.0
    simulation_step: int = 0
    
    # Output
    last_output: str = ""
    output_confidence: float = 0.0
    
    # Neurogenesis tracking
    neurons_created: int = 0
    neurons_pruned: int = 0
    novelty_level: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'global_activity': self.global_activity,
            'dopamine': self.dopamine_level,
            'serotonin': self.serotonin_level,
            'norepinephrine': self.norepinephrine_level,
            'acetylcholine': self.acetylcholine_level,
            'cortisol': self.cortisol_level,
            'gaba': self.gaba_level,
            'glutamate': self.glutamate_level,
            'oxytocin': self.oxytocin_level,
            'energy': self.total_energy,
            'time': self.current_time,
            'step': self.simulation_step,
            'output': self.last_output,
            'confidence': self.output_confidence,
            'neurons_created': self.neurons_created,
            'neurons_pruned': self.neurons_pruned,
            'novelty': self.novelty_level
        }# =============================================================================
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
        # Usage and age tracking for structural plasticity
        self.usage_counts = np.zeros(num_columns, dtype=int)
        self.ages = np.zeros(num_columns, dtype=int)

        # Hierarchical cortical representation
        self.cortical_states = [np.zeros(num_columns) for _ in range(num_cortical_areas)]

        # Refractory periods
        self.last_spike_time = np.full(num_columns, -np.inf)
        self.refractory_period = 0.002  # 2ms

    def _init_lateral_weights(self) -> np.ndarray:
        """Initialize Mexican-hat lateral inhibition weights (vectorized)."""
        # Create distance matrix
        positions = np.arange(self.num_columns)
        dist_matrix = np.abs(positions[:, np.newaxis] - positions[np.newaxis, :])
        
        # Mexican hat: excitation nearby, inhibition further
        # Close range (dist < 5): positive weights
        close_mask = dist_matrix < 5
        far_mask = dist_matrix >= 5
        
        weights = np.zeros((self.num_columns, self.num_columns))
        weights[close_mask] = 0.1 * np.exp(-dist_matrix[close_mask] / 2)
        weights[far_mask] = -0.05 * np.exp(-(dist_matrix[far_mask] - 5) / 10)
        
        # Zero diagonal (no self-connection)
        np.fill_diagonal(weights, 0)
        
        return weights

    def process(
        self,
        input_data: np.ndarray,
        current_time: float,
        learning_enabled: bool = True,
        modulation: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Process input through the cortical engine.

        Args:
            input_data: Input embedding vector
            current_time: Current simulation time
            learning_enabled: Whether to update weights
            modulation: Dict of neuromodulator influences:
                - 'sparsity_mod': multiplier for sparsity (higher = sparser)
                - 'gain_mod': overall gain multiplier
                - 'plasticity_mod': learning rate multiplier

        Returns:
            Dict with activation, prediction, error, and cortical states
        """
        # Apply neuromodulation effects
        mod = modulation or {}
        sparsity_mod = mod.get('sparsity_mod', 1.0)
        gain_mod = mod.get('gain_mod', 1.0)
        plasticity_mod = mod.get('plasticity_mod', 1.0)
        
        # Modulated sparsity (acetylcholine increases sparsity/focus)
        effective_sparsity = self.sparsity * sparsity_mod
        effective_k_winners = max(1, int(self.num_columns * effective_sparsity))
        
        # Safety: validate all arrays match num_columns to prevent broadcast errors
        n = self.num_columns
        def _ensure_size(arr, size, default=0.0):
            if len(arr) == size:
                return arr
            elif len(arr) < size:
                return np.concatenate([arr, np.full(size - len(arr), default)])
            else:
                return arr[:size]
        
        self.prediction = _ensure_size(self.prediction, n)
        self.activation = _ensure_size(self.activation, n)
        self.prediction_error = _ensure_size(self.prediction_error, n)
        self.duty_cycles = _ensure_size(self.duty_cycles, n, self.sparsity)
        self.last_spike_time = _ensure_size(self.last_spike_time, n, -np.inf)
        if len(self.usage_counts) != n:
            self.usage_counts = _ensure_size(self.usage_counts, n, 0).astype(int)
        if len(self.ages) != n:
            self.ages = _ensure_size(self.ages, n, 0).astype(int)
        
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

        # 6.5. Apply gain modulation from neuromodulators
        raw_activation *= gain_mod

        # 7. K-winners-take-all (with modulated sparsity)
        new_activation = self._k_winners_take_all(raw_activation, k_override=effective_k_winners)

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

        # 11.5 Update usage counts and ages
        active_mask = new_activation > 0
        # Increment age for all, reset age for active
        self.ages = self.ages + 1
        self.ages[active_mask] = 0
        # Increment usage counts for active columns
        for i in np.where(active_mask)[0]:
            self.usage_counts[i] += 1

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

    def _k_winners_take_all(self, activations: np.ndarray, k_override: Optional[int] = None) -> np.ndarray:
        """Apply k-winners-take-all sparsity with optional dynamic k."""
        k = k_override if k_override is not None else self.k_winners
        k = max(1, min(k, len(activations)))

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
        error_magnitude: float,
        plasticity_mod: float = 1.0
    ):
        """Update weights based on Hebbian learning modulated by error and plasticity."""
        learning_rate = 0.001 * error_magnitude * plasticity_mod

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
        """
        Get confidence based on prediction error.
        
        Note: Since activation is sparse (2% non-zero), we measure prediction
        accuracy on active neurons only, not the full array mean.
        """
        # Find active neurons (nonzero activation or prediction)
        active_mask = (self.activation > 0.01) | (self.prediction > 0.01)
        
        if np.sum(active_mask) < 2:
            # Very few active neurons - high novelty/low confidence
            return 0.3
            
        # Measure error only on active region
        active_error = np.abs(self.prediction_error[active_mask])
        
        # Also count how many predicted neurons didn't fire (missed predictions)
        predicted = self.prediction > 0.1
        actually_active = self.activation > 0.01
        misses = np.sum(predicted & ~actually_active)
        miss_rate = misses / max(1, np.sum(predicted))
        
        # Combine mean error with miss rate
        error_rate = np.mean(active_error) if len(active_error) > 0 else 0.5
        combined_error = 0.7 * error_rate + 0.3 * miss_rate
        
        return float(1.0 - np.clip(combined_error, 0, 1))

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
        leak_rate: float = 0.3,
        use_fp16: bool = False,
        update_fraction: float = 1.0
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
            use_fp16: Use float16 for weights (memory optimization)
            update_fraction: Fraction of neurons to update each step (0-1)
        """
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.use_fp16 = use_fp16
        self.update_fraction = update_fraction
        self.dtype = np.float16 if use_fp16 else np.float32

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
        W = np.random.randn(self.reservoir_size, self.reservoir_size).astype(np.float32)

        # Apply sparsity mask
        mask = np.random.random((self.reservoir_size, self.reservoir_size)) < self.sparsity
        W *= mask

        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W *= self.spectral_radius / current_radius

        # Optionally convert to fp16 for memory savings
        self.W_reservoir = W.astype(self.dtype)

        # Input weights
        self.W_input = (np.random.randn(self.reservoir_size, self.input_dim) * 0.5).astype(self.dtype)

        # Feedback weights
        self.W_feedback = (np.random.randn(self.reservoir_size, self.output_dim) * 0.1).astype(self.dtype)

        # Reservoir state (always float32 for numerical stability)
        self.state = np.zeros(self.reservoir_size, dtype=np.float32)

        # Fast and slow reservoir states (for multi-timescale)
        self.fast_state = np.zeros(self.reservoir_size, dtype=np.float32)
        self.slow_state = np.zeros(self.reservoir_size, dtype=np.float32)
        
        # Precompute update mask for partial updates
        if self.update_fraction < 1.0:
            self._update_mask = np.random.random(self.reservoir_size) < self.update_fraction
        else:
            self._update_mask = None

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
        new_state = np.tanh(pre_activation * modulation).astype(np.float32)

        # 5. Leaky integration (with optional partial update mask)
        if self._update_mask is not None:
            # Only update a fraction of neurons (performance optimization)
            update = (1 - self.leak_rate) * self.state + self.leak_rate * new_state
            self.state = np.where(self._update_mask, update, self.state)
            # Rotate mask for next step
            self._update_mask = np.roll(self._update_mask, int(self.reservoir_size * 0.1))
        else:
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
        # Handle None seed
        if seed is None:
            state = self.state.copy()
        elif len(seed) == self.reservoir_size:
            state = seed.copy()
        else:
            state = self.state.copy()

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

        # Simple fatigue meter (replaces full metabolism system)
        self.fatigue = 0.0
        self.fatigue_recovery_rate = 0.01
        self.fatigue_cost_per_spike = 0.001
        
        # Synaptic scaling (simplified inline)
        self._synaptic_scale = 1.0
        self._target_rate = 0.02
        self._num_neurons = num_neurons

        # Neurogenesis control
        self.novelty_threshold = 0.15  # Lowered from 0.2 for more neurogenesis
        self.neurogenesis_rate = 0.7   # Increased from 0.5
        self.pruning_threshold = 0.002  # Slightly higher for more aggressive pruning

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

        # Simple synaptic scaling (homeostatic)
        if self.enable_metabolism:
            # Scale synapses to maintain target rate
            rate_error = self._target_rate - activity_level
            self._synaptic_scale += 0.001 * rate_error
            self._synaptic_scale = np.clip(self._synaptic_scale, 0.5, 2.0)

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
        # TUNED: More aggressive for visible learning
        prob = novelty * (0.7 + dopamine) * (1 - cortisol * 0.2)

        if novelty > self.novelty_threshold and np.random.random() < prob * self.neurogenesis_rate:
            num_new = np.random.randint(1, 4)  # 1-3 new columns
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
        config: Optional[BrainConfig] = None,
        input_dim: int = 300,
        num_columns: int = 200,
        reservoir_size: int = 2000,
        output_dim: int = 300,
        enable_all_features: bool = True
    ):
        """
        Initialize the Three-System Brain.

        Args:
            config: BrainConfig instance (overrides other params if provided)
            input_dim: Input embedding dimension
            num_columns: Number of cortical columns
            reservoir_size: Reservoir size
            output_dim: Output dimension
            enable_all_features: Enable all enhanced features
        """
        print("Initializing Three-System Brain Architecture...")

        # Use config if provided, otherwise create from params
        if config is not None:
            self.config = config
            input_dim = config.input_dim
            num_columns = config.num_columns
            reservoir_size = config.reservoir_size
            output_dim = config.output_dim
        else:
            self.config = BrainConfig(
                input_dim=input_dim,
                num_columns=num_columns,
                cells_per_column=32,
                reservoir_size=reservoir_size,
                output_dim=output_dim
            )
        
        # Initialize state
        self.state = BrainState()

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
            spectral_radius=config.spectral_radius,
            sparsity=config.reservoir_sparsity,
            leak_rate=config.leak_rate,
            use_fp16=config.use_fp16_reservoir,
            update_fraction=config.reservoir_update_fraction
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

        # Signal processor for input normalization
        print("  [Interface] Signal Processor...")
        self.signal_processor = SignalProcessor(input_dim=input_dim)
        
        # Language decoder for text output
        print("  [Interface] Language Decoder...")
        self.language_decoder = NeuralLanguageDecoder(
            vocabulary_size=self.config.vocabulary_size,
            embedding_dim=output_dim,
            input_dim=output_dim
        )
        
        # =====================================================================
        # NEW FEATURES: Potato Optimizations + Advanced Capabilities
        # =====================================================================
        
        # Global event queue (replaces 4 temporal layers)
        print("  [Optimization] Global Event Queue...")
        self.event_queue = EventQueue()
        self._init_recurring_events()
        
        # Self-compression engine (reciprocal self-restructuring)
        print("  [Advanced] Self-Compression Engine...")
        self.compression = SelfCompressionEngine(
            pattern_dim=num_columns,
            max_primitives=500
        )
        
        # Internal body (emergent motor loop)
        print("  [Advanced] Internal Body (Motor Loop)...")
        self.body = InternalBody(
            num_muscles=8,
            num_sensors=16
        )
        
        # LSH for fast semantic similarity
        print("  [Optimization] LSH Hash Lattice...")
        self.lsh = LSHHashLattice(
            dim=output_dim,
            num_hashes=32,
            num_tables=4
        )
        
        # Binary reservoir (optional, for potato mode)
        self.binary_reservoir = None  # Lazy init if needed

        # =====================================================================
        # NSM: Neural Sleep & Memory Consolidation System
        # =====================================================================
        print("  [NSM] Neural Consolidation System...")
        
        # Chemical tagging for synapses (cortex feedforward weights)
        ff_shape = self.cortex.ff_weights.shape
        self.cortex_tagging = ChemicalTaggingSystem(ff_shape)
        
        # Consolidation engine for sleep processing
        self.consolidation_engine = ConsolidationEngine()
        
        # Memory installer for inheritance
        self.memory_installer = InheritedMemoryInstaller()
        
        # Sleep manager
        self.sleep_manager = SleepManager()
        
        # Track active synapses for tagging
        self.last_active_synapses = np.zeros(ff_shape, dtype=bool)
        
        # Creature age (for plasticity calculation)
        self.creature_age = 0.0  # 0-1 normalized
        
        # =====================================================================
        # TIER 2: Predictive Minds - Pain Prediction System
        # =====================================================================
        print("  [TIER 2] Predictive Pain Avoidance...")
        
        # Pain prediction: learns to anticipate pain from sensory patterns
        # Maps cortical activation patterns to predicted pain level
        self.pain_predictor_weights = np.zeros(num_columns)  # Linear predictor
        self.pain_prediction_history = []  # (cortical_pattern, actual_pain) pairs
        self.max_pain_history = 100
        self.pain_prediction_learning_rate = 0.1
        self.last_pain_prediction = 0.0
        self.predicted_pain_threshold = 0.3  # Threshold to trigger avoidance

        # =====================================================================
        # TIER 4: Higher Cognition Systems
        # =====================================================================
        print("  [TIER 4] Tool Use System...")
        self.tool_system = ToolUseSystem(max_tools=20)
        
        print("  [TIER 4] Abstract Reasoning...")
        self.reasoning_system = AbstractReasoningSystem(
            pattern_dim=min(64, num_columns),
            max_concepts=100
        )
        
        print("  [TIER 4] Social Structures...")
        self.social_system = SocialStructureSystem(max_relationships=50)

        # Current state
        self.step_count = 0
        self.current_time = 0.0

        # Output cache
        self.last_cortical_output = np.zeros(num_columns)
        self.last_reservoir_output = np.zeros(output_dim)
        
        # Persistence (lazy init)
        self._persistence = None

        print("Three-System Brain initialized!")
    
    def _init_recurring_events(self):
        """Initialize recurring events in the global event queue."""
        # Schedule initial events for each timescale
        self.event_queue.schedule_recurring('gamma', 'cortex', 0.0)
        self.event_queue.schedule_recurring('beta', 'reservoir', 0.0)
        self.event_queue.schedule_recurring('theta', 'learning', 0.0)
        self.event_queue.schedule_recurring('delta', 'consolidation', 0.0)
        self.event_queue.schedule_recurring('modulator', 'learning', 0.0)

    def _add_columns(self, num_new: int) -> None:
        """Safely add cortical columns (very small structural plasticity).

        This expands weight matrices and internal arrays with small random
        initial values so runtime shapes remain consistent.
        """
        if num_new <= 0:
            return

        # Cap to max neurons allowed
        current_cols = self.cortex.num_columns
        max_cols = max(1, int(self.config.max_neurons // self.cortex.cells_per_column))
        new_total = min(max_cols, current_cols + num_new)
        actually_added = new_total - current_cols
        if actually_added <= 0:
            return

        # Expand feedforward weights (new rows)
        inp_dim = self.cortex.input_dim
        extra_ff = np.random.randn(actually_added, inp_dim) * 0.05
        self.cortex.ff_weights = np.vstack([self.cortex.ff_weights, extra_ff])

        # Expand lateral weights (add rows and cols)
        old_lat = self.cortex.lateral_weights
        n_old = old_lat.shape[0]
        new_size = n_old + actually_added
        new_lat = np.zeros((new_size, new_size))
        new_lat[:n_old, :n_old] = old_lat
        # Fill new connections with small values
        new_lat[n_old:, :] = np.random.randn(actually_added, new_size) * 0.01
        new_lat[:, n_old:] = np.random.randn(new_size, actually_added) * 0.01
        self.cortex.lateral_weights = new_lat

        # Expand feedback weights (must be square: num_columns × num_columns)
        old_fb = self.cortex.fb_weights
        n_old_fb = old_fb.shape[0]
        new_fb = np.zeros((new_total, new_total))
        # Copy old weights into top-left
        new_fb[:n_old_fb, :n_old_fb] = old_fb[:n_old_fb, :n_old_fb] if old_fb.shape[1] >= n_old_fb else old_fb
        # Initialize new connections with small random values
        new_fb[n_old_fb:, :] = np.random.randn(actually_added, new_total) * 0.01
        new_fb[:, n_old_fb:] = np.random.randn(new_total, actually_added) * 0.01
        self.cortex.fb_weights = new_fb

        # Expand prediction / activation arrays
        self.cortex.prediction = np.concatenate([self.cortex.prediction, np.zeros(actually_added)])
        self.cortex.activation = np.concatenate([self.cortex.activation, np.zeros(actually_added)])
        self.cortex.prediction_error = np.concatenate([self.cortex.prediction_error, np.zeros(actually_added)])
        self.cortex.duty_cycles = np.concatenate([self.cortex.duty_cycles, np.ones(actually_added) * self.cortex.sparsity])
        
        # Expand usage/age tracking arrays
        self.cortex.usage_counts = np.concatenate([self.cortex.usage_counts, np.zeros(actually_added, dtype=int)])
        self.cortex.ages = np.concatenate([self.cortex.ages, np.zeros(actually_added, dtype=int)])
        
        # Expand refractory period tracking
        self.cortex.last_spike_time = np.concatenate([self.cortex.last_spike_time, np.full(actually_added, -np.inf)])

        # Expand cortical states for hierarchy
        for i in range(len(self.cortex.cortical_states)):
            self.cortex.cortical_states[i] = np.concatenate([self.cortex.cortical_states[i], np.zeros(actually_added)])

        # Update counters
        self.cortex.num_columns = new_total
        self.cortex.k_winners = max(1, int(self.cortex.num_columns * self.cortex.sparsity))
        self.config.num_columns = new_total
        self.state.neurons_created += actually_added * self.cortex.cells_per_column
        self.learning.neurons_created += actually_added * self.cortex.cells_per_column

    def _prune_columns(self, num_remove: int = 0, indices: list = None) -> None:
        """Prune cortical columns either by count (tail) or by specific indices.

        Args:
            num_remove: number of columns to remove (used when indices is None)
            indices: list of column indices to remove (preferred)

        This function carefully slices all related weight matrices and state
        arrays to remove the given columns. It keeps a minimum number of
        columns to avoid destabilizing the cortex.
        """
        current_cols = self.cortex.num_columns

        # Build removal mask
        if indices is not None and len(indices) > 0:
            # sanitize indices
            idx = sorted(set(i for i in indices if 0 <= i < current_cols))
            remove = len(idx)
            if remove == 0:
                return
        else:
            # tail-prune fallback
            remove = min(num_remove, max(0, current_cols - 10))
            if remove <= 0:
                return
            idx = list(range(current_cols - remove, current_cols))

        # Enforce conservative limit
        remove = min(remove, max(0, current_cols - 10))
        if remove <= 0:
            return

        # Compute keep indices
        keep_idx = [i for i in range(current_cols) if i not in set(idx)]
        keep = len(keep_idx)

        # Helper to slice rows/cols
        try:
            self.cortex.ff_weights = self.cortex.ff_weights[keep_idx, :]
        except Exception:
            # fallback: reinitialize smaller ff_weights
            self.cortex.ff_weights = np.random.randn(keep, self.cortex.input_dim) * 0.05

        try:
            self.cortex.lateral_weights = self.cortex.lateral_weights[np.ix_(keep_idx, keep_idx)]
        except Exception:
            self.cortex.lateral_weights = np.zeros((keep, keep))

        try:
            self.cortex.fb_weights = self.cortex.fb_weights[np.ix_(keep_idx, keep_idx)]
        except Exception:
            self.cortex.fb_weights = np.random.randn(keep, keep) * 0.05

        # Slice 1D arrays
        def _safe_slice(arr, kidx):
            try:
                return arr[kidx].copy()
            except Exception:
                return np.zeros(len(kidx))

        self.cortex.prediction = _safe_slice(self.cortex.prediction, keep_idx)
        self.cortex.activation = _safe_slice(self.cortex.activation, keep_idx)
        self.cortex.prediction_error = _safe_slice(self.cortex.prediction_error, keep_idx)
        self.cortex.duty_cycles = _safe_slice(self.cortex.duty_cycles, keep_idx)

        # Usage and age arrays
        try:
            self.cortex.usage_counts = _safe_slice(self.cortex.usage_counts, keep_idx).astype(int)
            self.cortex.ages = _safe_slice(self.cortex.ages, keep_idx).astype(int)
        except Exception:
            # initialize defaults
            self.cortex.usage_counts = np.zeros(keep, dtype=int)
            self.cortex.ages = np.zeros(keep, dtype=int)
        
        # Refractory period tracking
        try:
            self.cortex.last_spike_time = _safe_slice(self.cortex.last_spike_time, keep_idx)
        except Exception:
            self.cortex.last_spike_time = np.full(keep, -np.inf)

        # Cortical states (hierarchy)
        for i in range(len(self.cortex.cortical_states)):
            try:
                self.cortex.cortical_states[i] = _safe_slice(self.cortex.cortical_states[i], keep_idx)
            except Exception:
                self.cortex.cortical_states[i] = np.zeros(keep)

        # Update counters and shapes
        self.cortex.num_columns = keep
        self.cortex.k_winners = max(1, int(self.cortex.num_columns * self.cortex.sparsity))
        self.config.num_columns = keep

        # Update prune counters (neurons pruned = columns removed * cells per column)
        pruned_neurons = remove * self.cortex.cells_per_column
        self.state.neurons_pruned += pruned_neurons
        self.learning.neurons_pruned += pruned_neurons

    def _reservoir_driven_dream(self, steps: int = 5, imagine_if_empty: bool = True) -> Dict[str, Any]:
        """Run a short replay/imagination cycle and consolidate into cortex.

        - Uses recent reservoir states to replay outputs.
        - Feeds replayed outputs back into cortex as weak inputs to consolidate.
        - Optionally perturbs reservoir connectivity slightly (safe rewiring).
        Returns summary dict with counts.
        """
        summary = {'replayed': 0, 'imagined': 0, 'rewired': 0}

        # Try replay first
        outputs = self.reservoir.replay(steps=steps)
        if not outputs and imagine_if_empty:
            seed = self.reservoir.get_memory_content()
            outputs = self.reservoir.imagine(seed=seed, steps=steps)
            summary['imagined'] = len(outputs)
        else:
            summary['replayed'] = len(outputs)

        # Consolidate each output into cortex with weak learning
        for out in outputs:
            # Build cortex-sized input (truncate or pad)
            inp_dim = self.cortex.input_dim
            dream_input = np.zeros(inp_dim)
            if hasattr(out, '__len__'):
                L = min(len(out), inp_dim)
                dream_input[:L] = out[:L]

            # Small timestep during dreaming
            try:
                self.cortex.process(dream_input, current_time=self.current_time + 0.001, learning_enabled=True)
            except Exception:
                try:
                    self.cortex.process(dream_input, current_time=self.current_time, learning_enabled=False)
                except Exception:
                    pass

        # Gentle reservoir rewiring: small gaussian noise to a few weights
        try:
            size = self.reservoir.reservoir_size
            if size > 20:
                n_mod = min(self.config.dream_rewire_count, size)
                i_idx = np.random.randint(0, size, n_mod)
                j_idx = np.random.randint(0, size, n_mod)
                for i, j in zip(i_idx, j_idx):
                    self.reservoir.W_reservoir[i, j] += np.random.randn() * 0.001
                summary['rewired'] = int(n_mod)
        except Exception:
            pass

        return summary
    
    def _process_events(self, dt: float) -> Dict[str, List[Event]]:
        """Process due events from the global queue."""
        actual_dt, due_events = self.event_queue.advance(dt)
        
        # Categorize events by target
        events_by_target: Dict[str, List[Event]] = defaultdict(list)
        for event in due_events:
            events_by_target[event.target].append(event)
            
            # Reschedule recurring events
            if event.event_type in ['gamma', 'beta', 'theta', 'delta', 'modulator']:
                self.event_queue.schedule_recurring(
                    event.event_type, 
                    event.target, 
                    self.event_queue.current_time
                )
        
        return events_by_target

    def _encode_input(self, text: str) -> np.ndarray:
        """Encode text input to embedding vector."""
        embedding = np.zeros(self.config.input_dim)
        
        for i, char in enumerate(text[:self.config.input_dim]):
            embedding[i] = ord(char) / 256.0
        
        # Add structure features
        if len(text) > 0:
            word_count = len(text.split())
            embedding[-1] = min(word_count / 20.0, 1.0)
            embedding[-2] = min(len(text) / 200.0, 1.0)
        
        return embedding
    
    def _extract_reward_arousal(self, text: str, network_activity: Dict) -> Tuple[float, float]:
        """Extract reward and arousal signals from text and network state."""
        # Sentiment-based reward extraction
        positive_words = {'good', 'great', 'love', 'happy', 'yes', 'thank', 'amazing', 
                         'wonderful', 'excellent', 'perfect', 'awesome', 'nice'}
        negative_words = {'bad', 'hate', 'no', 'sad', 'angry', 'terrible', 'awful',
                         'horrible', 'wrong', 'fail', 'stupid', 'worst'}
        
        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        
        reward = (pos_count - neg_count) * 0.2
        reward = np.clip(reward, -1.0, 1.0)
        
        # Arousal from punctuation and novelty
        arousal = 0.5
        if '!' in text:
            arousal += 0.2
        if '?' in text:
            arousal += 0.1
        if text.isupper() and len(text) > 3:
            arousal += 0.3
            
        # Incorporate network novelty
        novelty = network_activity.get('novelty', 0.5)
        arousal = 0.7 * arousal + 0.3 * novelty
        
        return float(np.clip(reward, -1, 1)), float(np.clip(arousal, 0, 1))

    def process_raw(
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
        # STEP 0: Process events from global queue
        # =====================================================================
        events_by_target = self._process_events(dt)

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
        # SYSTEM GLUE: Compute neuromodulator → cortex/reservoir modulation
        # Uses config parameters for easy tuning
        # =====================================================================
        # Acetylcholine → focus/sparsity (high ACh = sparser, more focused)
        ach = neuromod_levels.get('acetylcholine', 0.5)
        ach_effect = self.config.ach_sparsity_gain
        sparsity_mod = (1.0 - ach_effect/2) + ach_effect * ach  # Configurable range
        
        # Dopamine → gain (high DA = stronger activations)
        da = neuromod_levels.get('dopamine', 0.5)
        da_effect = self.config.da_gain_effect
        gain_mod = (1.0 - da_effect/2) + da_effect * da  # Configurable range
        
        # Serotonin → plasticity (high 5-HT = more stable, less plastic)
        se = neuromod_levels.get('serotonin', 0.5)
        se_effect = self.config.serotonin_plasticity_effect
        plasticity_mod = (1.0 + se_effect/2) - se_effect * se  # Inverse relationship
        
        cortex_modulation = {
            'sparsity_mod': sparsity_mod,
            'gain_mod': gain_mod,
            'plasticity_mod': plasticity_mod
        }

        # =====================================================================
        # STEP 2: Process through cortex (sparse patterns) with modulation
        # =====================================================================
        cortical_result = self.cortex.process(
            input_data=input_data,
            current_time=self.current_time,
            learning_enabled=learning_enabled and learning_mod.get('should_learn', True),
            modulation=cortex_modulation
        )

        cortical_activation = cortical_result['activation']
        cortical_error = cortical_result['error_magnitude']

        # =====================================================================
        # STEP 2.5: Self-compression (reciprocal restructuring)
        # =====================================================================
        compression_result = self.compression.compress_cortical(
            list(cortical_activation)
        )
        primitive_id, compression_error, created_primitive = compression_result

        # =====================================================================
        # STEP 3: Process through reservoir (temporal dynamics)
        # =====================================================================
        # Reservoir modulation: DA increases gain, NE increases responsiveness
        ne = neuromod_levels.get('norepinephrine', 0.5)
        reservoir_modulation = 0.6 + 0.3 * da + 0.2 * ne  # Combined modulation

        reservoir_result = self.reservoir.process(
            cortical_input=cortical_activation,
            modulation=reservoir_modulation,
            dt=dt
        )

        reservoir_output = reservoir_result['output']
        reservoir_error = 1.0 - reservoir_result['temporal_features'].get('variance', 0.5)
        
        # =====================================================================
        # STEP 3.5: Motor loop (internal body interaction)
        # =====================================================================
        # Generate motor commands from reservoir output
        motor_commands = reservoir_output[:self.body.num_muscles].tolist()
        body_result = self.body.act(motor_commands)
        
        # Body reward feeds back into learning
        body_reward = body_result['reward']

        # =====================================================================
        # STEP 4: Compute learning signals via error bargaining
        # =====================================================================
        learning_signals = self.learning.compute_learning_signal(
            cortex_error=cortical_error,
            reservoir_error=reservoir_error,
            outcome=body_reward * 0.3  # Body state contributes to learning
        )

        # =====================================================================
        # STEP 4.5: Cross-system negotiation for output selection
        # If cortex and reservoir disagree, neuromodulators arbitrate which
        # system's representation should be used for downstream decoding.
        # learning_signals provides 'cortex_learns' and 'reservoir_learns'
        # which we use as soft weights; if conflict is high, do winner-take-all.
        cortex_weight = float(learning_signals.get('cortex_learns', 0.5))
        reservoir_weight = float(learning_signals.get('reservoir_learns', 0.5))
        conflict = float(learning_signals.get('conflict', 0.0))

        # Normalise weights
        s = cortex_weight + reservoir_weight
        if s <= 0:
            cortex_w = 0.5
            reservoir_w = 0.5
        else:
            cortex_w = cortex_weight / s
            reservoir_w = reservoir_weight / s

        # If conflict high, pick hard winner based on weights and neuromodulators
        winner = 'mixed'
        if conflict > 0.3:
            if cortex_w > reservoir_w:
                winner = 'cortex'
                integrated_output = cortical_result.get('prediction', reservoir_output)
            else:
                winner = 'reservoir'
                integrated_output = reservoir_output
        else:
            # Soft combination
            try:
                # Ensure same shape: if cortical prediction shorter, pad
                cort = np.asarray(cortical_result.get('prediction', np.zeros_like(reservoir_output)))
                res = np.asarray(reservoir_output)
                if cort.shape != res.shape:
                    # pad smaller to match larger
                    if cort.size < res.size:
                        pad = np.zeros(res.size - cort.size)
                        cort = np.concatenate([cort, pad])
                    else:
                        pad = np.zeros(cort.size - res.size)
                        res = np.concatenate([res, pad])
                integrated_output = cort * cortex_w + res * reservoir_w
            except Exception:
                integrated_output = reservoir_output

        # Expose arbitration info
        arbitration = {
            'cortex_weight': float(cortex_w),
            'reservoir_weight': float(reservoir_w),
            'conflict': float(conflict),
            'winner': winner
        }

        # Cache last arbitration for dashboard
        try:
            self.last_arbitration = arbitration
            # store a small summary of integrated output (norm)
            self.last_integrated_norm = float(np.linalg.norm(integrated_output)) if hasattr(integrated_output, '__len__') else float(integrated_output)
        except Exception:
            self.last_arbitration = None
            self.last_integrated_norm = 0.0

        # =====================================================================
        # Surprise detection -> phasic neuromodulator spikes
        # High prediction error should trigger dopamine + norepinephrine bursts
        # =====================================================================
        surprise = max(float(cortical_error), float(reservoir_error), float(learning_signals.get('agreed_error', 0.0)))
        phasic_info = {'dopamine': 0.0, 'norepinephrine': 0.0, 'surprise': surprise}
        try:
            surprise_threshold = 0.35
            if surprise > surprise_threshold:
                # Scale magnitude between 0..1
                mag = (surprise - surprise_threshold) / (1.0 - surprise_threshold)
                # Dopamine responds to unexpected reward/prediction error (smaller)
                da_mag = 0.3 * mag
                ne_mag = 0.6 * mag
                # Trigger phasic bursts via neuromodulation subsystem
                da_released = 0.0
                ne_released = 0.0
                try:
                    da_released = float(self.learning.neuromod.trigger_phasic_release(ModulatorType.DOPAMINE, da_mag, event_type='surprise'))
                except Exception:
                    da_released = 0.0
                try:
                    ne_released = float(self.learning.neuromod.trigger_phasic_release(ModulatorType.NOREPINEPHRINE, ne_mag, event_type='surprise'))
                except Exception:
                    ne_released = 0.0

                phasic_info['dopamine'] = da_released
                phasic_info['norepinephrine'] = ne_released

                # Optionally report outcome to learning system (surprise as a form of error)
                try:
                    self.learning.report_outcome(success=(surprise < 0.5), magnitude=mag)
                except Exception:
                    pass
        except Exception:
            pass

        # =====================================================================
        # STEP 5: Check for neurogenesis (compression-guided)
        # =====================================================================
        novelty = 1.0 - self.cortex.get_prediction_confidence()
        
        # Compression error also indicates novelty
        combined_novelty = 0.7 * novelty + 0.3 * compression_error
        should_create, num_new = self.learning.should_create_neurons(combined_novelty)

        # Perform safe structural changes if recommended
        if should_create and num_new > 0:
            try:
                self._add_columns(num_new)
            except Exception:
                pass

        # Pruning decisions (conservative): use per-column usage and age
        try:
            usage = np.array(self.cortex.usage_counts, dtype=float)
            age = np.array(self.cortex.ages, dtype=float)
            prune_mask = self.learning.should_prune_neurons(usage, age)
            # If many neurons flagged, prune a small number conservatively
            if isinstance(prune_mask, np.ndarray) and prune_mask.any():
                # Count flagged columns
                flagged = int(np.sum(prune_mask))
                n_prune = int(flagged / (self.cortex.cells_per_column or 1))
                # prune only a few to avoid instability
                n_prune = min(max(0, n_prune), 5)
                if n_prune > 0:
                    try:
                        # choose lowest-usage columns rather than arbitrary slice when possible
                        if len(usage) > n_prune:
                            # pick indices with smallest usage that are flagged
                            flagged_idx = [i for i, v in enumerate(prune_mask) if v]
                            flagged_usage = [(i, usage[i]) for i in flagged_idx]
                            flagged_usage.sort(key=lambda x: x[1])
                            cols_to_prune = [int(i) for i, _ in flagged_usage[:n_prune]]
                            # perform targeted prune
                            self._prune_columns(indices=cols_to_prune)
                        else:
                            self._prune_columns(n_prune)
                    except Exception:
                        pass
        except Exception:
            pass

        # If consolidation-time events occurred, run reservoir-driven dreaming/consolidation
        dream_summary = {}
        try:
            if events_by_target.get('consolidation'):
                neu = neuromod_levels
                # Favor dreaming when serotonin is reasonably high and low fatigue
                if self.config.dream_enabled and float(neu.get('serotonin', 0.5)) > self.config.dream_serotonin_threshold and learning_state.get('fatigue', 0.0) < self.config.dream_fatigue_threshold:
                    dream_summary = self._reservoir_driven_dream(steps=self.config.dream_steps)
                    # persist for dashboard
                    try:
                        self.last_dream_summary = dream_summary
                    except Exception:
                        self.last_dream_summary = dream_summary
        except Exception:
            dream_summary = {}

        # =====================================================================
        # STEP 6: Update caches
        # =====================================================================
        self.last_cortical_output = cortical_activation
        self.last_reservoir_output = reservoir_output
        
        # TIER 4: Update higher cognition systems
        try:
            self.tier4_update()
        except Exception:
            pass  # Fail silently to not break core processing

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
            
            # Compression
            'compression': {
                'primitive_id': primitive_id,
                'error': compression_error,
                'created_new': created_primitive,
                'stats': self.compression.get_stats()
            },
            
            # Motor loop
            'body': body_result,
            # Phasic neuromodulator activity (surprise-driven)
            'phasic': phasic_info,
            # Arbitration between systems
            'arbitration': arbitration,
            'integrated_output': integrated_output,
            
            # Event queue
            'events_processed': sum(len(e) for e in events_by_target.values()),

            # Dreaming/replay summary (if any)
            'dream_summary': dream_summary,

            # Metadata
            'step': self.step_count,
            'time': self.current_time,
            'sparsity': cortical_result['sparsity'],
            'active_columns': cortical_result['active_columns']
        }

    def train_raw(
        self,
        inputs: List[np.ndarray],
        targets: List[np.ndarray],
        dt: float = 0.01
    ) -> Dict[str, float]:
        """
        Train the brain on input-target pairs (low-level numpy API).

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

    def get_structural_snapshot(self) -> Dict[str, Any]:
        """
        Get structural memory snapshot for inheritance.
        
        Extracts "Instinct Packets":
        1. Learned primitives (SelfCompressionEngine)
        2. Key concept embeddings
        3. Cortical feedforward weights (neural topology)
        4. Lateral inhibition patterns
        5. Reservoir structure (optional, for advanced inheritance)
        6. Pain predictor weights (TIER 2: Predictive Minds)
        """
        snapshot = {
            'version': '2.1',
            'compression_engine': {},
            'evolved_modules': 0,
            # Neural topology (NSM inheritance)
            'ff_weights': self.cortex.ff_weights.copy(),
            'lateral_weights': self.cortex.lateral_weights.copy(),
            'fb_weights': self.cortex.fb_weights.copy(),
            # Stability info (which connections are consolidated)
            'stability': self.cortex_tagging.stability.copy() if hasattr(self, 'cortex_tagging') else None,
            # Reservoir (optional - heavier)
            'reservoir_W': self.reservoir.W_reservoir.copy() if hasattr(self.reservoir, 'W_reservoir') else None,
            # TIER 2: Pain predictor weights (learned danger associations)
            'pain_predictor': self.pain_predictor_weights.copy() if hasattr(self, 'pain_predictor_weights') else None,
        }
        
        # 1. Structural Primitives (The "Alphabet" of thought)
        if hasattr(self, 'compression'):
            snapshot['compression_engine'] = self.compression.get_snapshot()
        elif hasattr(self, 'compression_engine'):
            snapshot['compression_engine'] = self.compression_engine.get_snapshot()
            
        return snapshot
        
    def load_structural_snapshot(self, snapshot: Dict[str, Any], mutation_rate: float = 0.1):
        """
        Load structural memory from snapshot (Inheritance).
        
        Args:
            snapshot: Structure from parent(s)
            mutation_rate: Rate of random mutations during inheritance (0-1)
        """
        if not snapshot:
            return
        
        version = snapshot.get('version', '1.0')
        
        # NSM v2.0: Full neural topology inheritance
        if version >= '2.0' and 'ff_weights' in snapshot:
            parent_ff = snapshot['ff_weights']
            parent_lateral = snapshot.get('lateral_weights')
            parent_fb = snapshot.get('fb_weights')
            parent_stability = snapshot.get('stability')
            
            # Use memory installer for proper inheritance with mutation
            if hasattr(self, 'memory_installer') and hasattr(self, 'cortex_tagging'):
                # Install inherited patterns at reduced strength
                self.cortex.ff_weights = self.memory_installer.install_inherited_patterns(
                    self.cortex.ff_weights,
                    parent_ff,
                    self.cortex_tagging,
                    mutation_rate=mutation_rate
                )
                
                # Also inherit lateral weights with mutation
                if parent_lateral is not None:
                    inherited_lateral = parent_lateral * 0.3  # Reduced strength
                    mutation_mask = np.random.random(inherited_lateral.shape) < mutation_rate
                    inherited_lateral[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * 0.1
                    
                    # Blend into child
                    blend_mask = np.abs(self.cortex.lateral_weights) < 0.01
                    self.cortex.lateral_weights[blend_mask] = inherited_lateral[blend_mask]
                
                # Copy stability info if available
                if parent_stability is not None and hasattr(self, 'cortex_tagging'):
                    # Inherited stability is reduced (child needs to prove the connections)
                    inherited_stability = parent_stability * 0.5
                    self.cortex_tagging.stability = np.maximum(
                        self.cortex_tagging.stability,
                        inherited_stability[:self.cortex_tagging.shape[0], :self.cortex_tagging.shape[1]]
                    )
                
                print(f"[NSM] Inherited neural topology: "
                      f"{np.sum(np.abs(self.cortex.ff_weights) > 0.01)} active synapses")
            else:
                # Fallback: direct copy at reduced strength
                blend_factor = 0.3
                self.cortex.ff_weights = (
                    self.cortex.ff_weights * (1 - blend_factor) +
                    parent_ff * blend_factor
                )
        
        # TIER 2: Inherit pain predictor weights (learned danger associations)
        if 'pain_predictor' in snapshot and snapshot['pain_predictor'] is not None:
            parent_predictor = snapshot['pain_predictor']
            if hasattr(self, 'pain_predictor_weights') and len(parent_predictor) == len(self.pain_predictor_weights):
                # Inherit at reduced strength with mutation
                inherited_predictor = parent_predictor * 0.5  # 50% strength
                mutation_mask = np.random.random(len(inherited_predictor)) < mutation_rate
                inherited_predictor[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * 0.1
                
                # Blend with child's (likely zero) predictor
                self.pain_predictor_weights = (
                    self.pain_predictor_weights * 0.2 + 
                    inherited_predictor * 0.8
                )
                print(f"[TIER 2] Inherited pain predictor: "
                      f"mean weight = {np.mean(np.abs(self.pain_predictor_weights)):.3f}")
        
        # 1. Load Primitives (v1.0 and v2.0)
        compression = getattr(self, 'compression', None) or getattr(self, 'compression_engine', None)
        if 'compression_engine' in snapshot and compression:
            compression.load_snapshot(snapshot['compression_engine'])
            
        # Log inheritance
        n_primitives = len(snapshot.get('compression_engine', {}).get('cortical_primitives', []))
        print(f"[NSM] Brain received structural memory: {n_primitives} primitives")

    def train(self, input_text: str, expected_response: str, 
               reward: float = 0.8) -> Dict[str, Any]:
        """
        Train the brain on a text input-output pair (chatbot compatible).
        
        Args:
            input_text: User input text
            expected_response: Desired output response
            reward: Reward signal for reinforcement (default 0.8 for positive training)
            
        Returns:
            Dict with training result and metadata
        """
        # Process input with high reward to reinforce pattern
        result = self.process(input_text, reward=reward, arousal=0.6)
        
        # Additionally train the language decoder on this pair
        self.language_decoder.train(input_text, expected_response, reward=reward)
        
        # Report success to neuromodulation
        self.learning.report_outcome(success=True, magnitude=reward)
        
        # Track training
        self.state.training_samples = getattr(self.state, 'training_samples', 0) + 1
        
        return {
            'mood': result.get('mood', 'neutral'),
            'response': result.get('response', ''),
            'confidence': result.get('confidence', 0.5),
            'training_samples': getattr(self.state, 'training_samples', 1)
        }

    def process(self, text: str, reward: Optional[float] = None, 
                arousal: Optional[float] = None) -> Dict[str, Any]:
        """
        Process text input - chatbot compatible interface.
        
        Args:
            text: Input text to process
            reward: Optional explicit reward signal (-1 to 1)
            arousal: Optional explicit arousal signal (0 to 1)
            
        Returns:
            Dict with response and metadata
        """
        # Encode input
        input_embedding = self._encode_input(text)
        processed_input = self.signal_processor.process(input_embedding)
        
        # Compute novelty for reward/arousal extraction
        network_activity = {
            'novelty': 1.0 - self.cortex.get_prediction_confidence(),
            'activation_spread': np.mean(self.last_cortical_output > 0)
        }
        
        # Extract or use provided reward/arousal
        if reward is None or arousal is None:
            extracted_reward, extracted_arousal = self._extract_reward_arousal(text, network_activity)
            if reward is None:
                reward = extracted_reward
            if arousal is None:
                arousal = extracted_arousal
        
        # Update neuromodulators based on reward/arousal
        self._update_neuromodulators(reward, arousal)
        
        # Process through the three systems
        result = self.process_raw(processed_input, learning_enabled=True)
        
        # Decode language output
        # Prefer integrated arbitration output when available
        decode_input = result.get('integrated_output', result.get('reservoir_output'))
        output_text, confidence = self.language_decoder.decode(
            decode_input,
            mood=self._compute_mood()
        )
        
        # Update state tracking
        self.state.simulation_step = self.step_count
        self.state.current_time = self.current_time
        self.state.last_output = output_text
        self.state.output_confidence = confidence
        self.state.novelty_level = result['novelty']
        self.state.neurons_created += result.get('neurons_created', 0)
        
        # =====================================================================
        # SYSTEM GLUE: Decoder confidence → Dopamine feedback
        # High confidence = successful response = dopamine boost
        # Low confidence = uncertainty = slight dopamine dip
        # =====================================================================
        confidence_reward = (confidence - 0.5) * 0.4  # Range: -0.2 to +0.2
        self.state.dopamine_level += confidence_reward * 0.1
        self.state.dopamine_level = np.clip(self.state.dopamine_level, 0, 1)
        
        # Also report to neuromodulation system for learning gating
        if confidence > 0.6:
            self.learning.report_outcome(success=True, magnitude=confidence)
        elif confidence < 0.3:
            self.learning.report_outcome(success=False, magnitude=1.0 - confidence)
        
        # Train the decoder on this interaction (weak learning)
        if len(text) > 2 and len(output_text) > 2:
            self.language_decoder.train(text, output_text, reward=0.1)
        
        # Return chatbot-compatible result
        return {
            'response': output_text,
            'confidence': confidence,
            'mood': self._compute_mood(),
            'dopamine': result['neuromod_levels'].get('dopamine', 0.5),
            'novelty': result['novelty'],
            'structure_change': {
                'neurons_created': result.get('neurons_created', 0),
                'neurons_pruned': 0
            },
            'phasic': result.get('phasic', {}),
            'arbitration': result.get('arbitration', {})
        }
    
    def _update_neuromodulators(self, reward: float, arousal: float):
        """Update brain state neuromodulator levels based on reward/arousal."""
        # Dopamine: reward-driven
        dopamine_target = self.config.dopamine_baseline + 0.4 * reward
        self.state.dopamine_level += 0.1 * (dopamine_target - self.state.dopamine_level)
        
        # Acetylcholine: arousal-driven
        ach_target = self.config.acetylcholine_baseline + 0.3 * arousal
        self.state.acetylcholine_level += 0.1 * (ach_target - self.state.acetylcholine_level)
        
        # Norepinephrine: arousal-driven
        ne_target = self.config.norepinephrine_baseline + 0.2 * arousal
        self.state.norepinephrine_level += 0.05 * (ne_target - self.state.norepinephrine_level)
        
        # Serotonin: calm + positive reward
        serotonin_target = self.config.serotonin_baseline - 0.2 * (arousal - 0.5) + 0.2 * max(reward, 0)
        self.state.serotonin_level += 0.05 * (serotonin_target - self.state.serotonin_level)
        
        # Cortisol: stress (negative reward + high arousal)
        stress_signal = max(-reward, 0) * 0.5 + max(arousal - 0.7, 0) * 0.3
        cortisol_target = 0.3 + stress_signal
        self.state.cortisol_level += 0.08 * (cortisol_target - self.state.cortisol_level)
        
        # GABA: inhibitory, high when calm
        gaba_target = 0.6 - 0.3 * arousal
        self.state.gaba_level += 0.06 * (gaba_target - self.state.gaba_level)
        
        # Glutamate: excitatory, high with arousal
        glutamate_target = 0.5 + 0.3 * arousal
        self.state.glutamate_level += 0.08 * (glutamate_target - self.state.glutamate_level)
        
        # Oxytocin: bonding, positive reward only
        oxytocin_target = 0.3 + 0.4 * max(reward, 0)
        self.state.oxytocin_level += 0.05 * (oxytocin_target - self.state.oxytocin_level)
        
        # Clamp all to [0, 1]
        for attr in ['dopamine_level', 'acetylcholine_level', 'norepinephrine_level',
                     'serotonin_level', 'cortisol_level', 'gaba_level', 
                     'glutamate_level', 'oxytocin_level']:
            setattr(self.state, attr, np.clip(getattr(self.state, attr), 0, 1))
    
    def _compute_mood(self) -> str:
        """Compute current mood from neuromodulator levels."""
        da = self.state.dopamine_level
        se = self.state.serotonin_level
        ne = self.state.norepinephrine_level
        co = self.state.cortisol_level
        
        if co > 0.7:
            return "stressed"
        elif da > 0.7 and ne > 0.5:
            return "excited"
        elif da > 0.6 and se > 0.5:
            return "happy"
        elif se > 0.6 and ne < 0.4:
            return "calm"
        elif ne > 0.6:
            return "alert"
        elif da > 0.5:
            return "curious"
        elif se < 0.3:
            return "anxious"
        else:
            return "neutral"
    
    def _compute_personality(self) -> Dict[str, float]:
        """Compute personality traits from brain structure."""
        # Derive from cortex/reservoir characteristics
        cortex_activity = np.mean(self.last_cortical_output > 0)
        reservoir_complexity = np.std(self.reservoir.state) if hasattr(self.reservoir, 'state') else 0.5
        
        return {
            'openness': float(np.clip(0.5 + cortex_activity, 0, 1)),
            'curiosity': float(np.clip(self.state.novelty_level, 0, 1)),
            'stability': float(np.clip(self.state.serotonin_level, 0, 1)),
            'energy': float(np.clip(self.state.dopamine_level, 0, 1)),
            'focus': float(np.clip(self.state.acetylcholine_level, 0, 1))
        }

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
            'global_activity': float(np.mean(self.last_cortical_output)),
            'sparsity': float(np.mean(self.last_cortical_output > 0)),
            'neuromodulation': self.learning.get_neuromodulator_levels()
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard visualization - chatbot compatible."""
        neuromod = self.learning.get_neuromodulator_levels()
        learning_mod = self.learning.neuromod.get_learning_modulation()
        
        total_neurons = self.cortex.num_columns * self.cortex.cells_per_column
        active_neurons = int(np.sum(self.last_cortical_output > 0))
        total_synapses = total_neurons * 50  # Estimated

        return {
            'chemicals': {
                'dopamine': neuromod.get('dopamine', self.state.dopamine_level),
                'serotonin': neuromod.get('serotonin', self.state.serotonin_level),
                'norepinephrine': neuromod.get('norepinephrine', self.state.norepinephrine_level),
                'acetylcholine': neuromod.get('acetylcholine', self.state.acetylcholine_level),
                'cortisol': neuromod.get('cortisol', self.state.cortisol_level),
                'gaba': neuromod.get('gaba', self.state.gaba_level),
                'glutamate': neuromod.get('glutamate', self.state.glutamate_level),
                'oxytocin': neuromod.get('oxytocin', self.state.oxytocin_level),
                'endorphin': neuromod.get('endorphin', 0.3),
                'adrenaline': neuromod.get('adrenaline', 0.2),
            },
            'neurons': {
                'total': total_neurons,
                'hidden': int(total_neurons * 0.7),
                'sensory': int(total_neurons * 0.15),
                'output': int(total_neurons * 0.15),
                'active': active_neurons,
                'spikes': self.step_count * 100,
                'born': self.learning.neurons_created + self.state.neurons_created,
                'died': self.learning.neurons_pruned + self.state.neurons_pruned,
            },
            'synapses': {
                'total': total_synapses,
                'active': active_neurons * 50,
                'density': self.config.target_sparsity,
            },
            'network': {
                'total_neurons': total_neurons,
                'total_synapses': total_synapses,
                'total_spikes': self.step_count * 100,
                'connection_density': self.config.target_sparsity,
            },
            'personality': self._compute_personality(),
            'mood': self._compute_mood(),
            'energy': self.body.energy,  # From internal body
            'memory': {
                'total': self.config.reservoir_size,
                'encoded': self.step_count,
                'forgotten': getattr(self, 'memory_forgotten', 0),
                'avg_strength': getattr(self, 'memory_avg_strength', 0.0),
            },
            'regions': {
                'cortex': {
                    'activity': float(np.mean(self.last_cortical_output)),
                    'active_neurons': active_neurons
                },
                'reservoir': {
                    'activity': float(np.linalg.norm(self.reservoir.state)) if hasattr(self.reservoir, 'state') else 0,
                    'active_neurons': int(self.config.reservoir_size * 0.1)
                }
            },
            'reservoir': {
                'size': self.reservoir.reservoir_size,
                'state_norm': float(np.linalg.norm(self.reservoir.state)) if hasattr(self.reservoir, 'state') else 0,
                'memory_load': len(self.reservoir.memory_buffer)
            },
            'arbitration': getattr(self, 'last_arbitration', {}),
            'integrated_output_norm': getattr(self, 'last_integrated_norm', 0.0),
            'learning': {
                'growth_rate': learning_mod.get('growth_rate', 0.1) if isinstance(learning_mod, dict) else 0.1,
                'pruning_rate': learning_mod.get('pruning_rate', 0.05) if isinstance(learning_mod, dict) else 0.05,
                'plasticity': learning_mod.get('plasticity', 0.5) if isinstance(learning_mod, dict) else 0.5,
                'reinforcement_rate': learning_mod.get('reinforcement_rate', 0.5) if isinstance(learning_mod, dict) else 0.5,
                'attention': learning_mod.get('attention', 0.5) if isinstance(learning_mod, dict) else 0.5,
                'stability': learning_mod.get('stability', 0.5) if isinstance(learning_mod, dict) else 0.5,
            },
            'epigenetic': self.learning.get_epigenetic_state(),
            'errors': {
                'cortex': float(np.mean(np.abs(self.cortex.prediction_error))),
                'prediction_confidence': self.cortex.get_prediction_confidence()
            },
            # New features
            'compression': self.compression.get_stats(),
            'body': self.body.get_state(),
            'event_queue': self.event_queue.stats(),
            'lsh': self.lsh.stats(),
            'dream_summary': getattr(self, 'last_dream_summary', {}),
            'plasticity': {
                'avg_column_age': float(np.mean(self.cortex.ages)) if hasattr(self.cortex, 'ages') else 0.0,
                'low_usage_columns': int(np.sum(np.array(getattr(self.cortex, 'usage_counts', [])) < 2))
            },
            # NSM: Sleep and consolidation stats
            'nsm': {
                'fatigue': self.sleep_manager.fatigue if hasattr(self, 'sleep_manager') else 0.0,
                'is_sleeping': self.sleep_manager.is_sleeping if hasattr(self, 'sleep_manager') else False,
                'consolidation_count': self.consolidation_engine.consolidation_count if hasattr(self, 'consolidation_engine') else 0,
                'total_strengthened': self.consolidation_engine.total_strengthened if hasattr(self, 'consolidation_engine') else 0,
                'total_pruned': self.consolidation_engine.total_pruned if hasattr(self, 'consolidation_engine') else 0,
                'pending_experiences': len(self.cortex_tagging.experiences) if hasattr(self, 'cortex_tagging') else 0,
            },
            # New simple counters for GUI
            'interactions': getattr(self, 'interaction_count', getattr(self.state, 'simulation_step', self.step_count)),
            'training_count': getattr(self, 'training_count', 0),
            # TIER 4: Higher Cognition Stats
            'tier4': {
                'tool_use': self.tool_system.get_state() if hasattr(self, 'tool_system') else {},
                'reasoning': self.reasoning_system.get_state() if hasattr(self, 'reasoning_system') else {},
                'social': self.social_system.get_state() if hasattr(self, 'social_system') else {},
            },
            # Metadata
            'step': self.step_count,
            'time': self.current_time,
            'fatigue': getattr(self.learning, 'fatigue', 0.0)
        }
    
    # =========================================================================
    # NSM: Neural Sleep & Memory Consolidation Methods
    # =========================================================================
    
    def tag_reward(self, reward: float, context: str = ""):
        """
        Tag currently active synapses with a reward signal.
        
        Positive reward → dopamine marker (strengthen during sleep)
        Negative reward → cortisol marker (weaken during sleep)
        
        Args:
            reward: Reward signal (-1 to 1)
            context: Optional description of what caused the reward
        """
        if not hasattr(self, 'cortex_tagging'):
            return
        
        # Build active synapse mask from cortical activation
        active_columns = self.last_cortical_output > 0.1
        
        # Create activation mask for ff_weights (which columns were activated)
        active_mask = np.zeros(self.cortex.ff_weights.shape, dtype=bool)
        for i, is_active in enumerate(active_columns):
            if is_active and i < active_mask.shape[0]:
                # Mark all input synapses to this column
                active_mask[i, :] = True
        
        # Store for next tag
        self.last_active_synapses = active_mask
        
        # Tag based on reward sign
        strength = abs(reward)
        if reward > 0.1:
            self.cortex_tagging.tag_positive(active_mask, strength=strength, context=context)
        elif reward < -0.1:
            self.cortex_tagging.tag_negative(active_mask, strength=strength, context=context)
    
    def accumulate_fatigue(self, dt: float, cortisol_level: float = 0.0):
        """
        Accumulate neural fatigue from activity.
        
        High activity + high cortisol = faster fatigue.
        """
        if not hasattr(self, 'sleep_manager'):
            return
        
        active_neurons = np.sum(self.last_cortical_output > 0.1)
        self.sleep_manager.accumulate_fatigue(
            dt=dt,
            active_neurons=active_neurons,
            cortisol_level=cortisol_level,
            activity_level=self.body.arousal if hasattr(self.body, 'arousal') else 0.5
        )
    
    def should_sleep(self) -> bool:
        """Check if brain needs sleep for consolidation."""
        if not hasattr(self, 'sleep_manager'):
            return False
        return self.sleep_manager.should_sleep()
    
    def enter_sleep(self):
        """Enter sleep mode for consolidation."""
        if hasattr(self, 'sleep_manager'):
            self.sleep_manager.enter_sleep()
    
    def is_sleeping(self) -> bool:
        """Check if currently sleeping."""
        return hasattr(self, 'sleep_manager') and self.sleep_manager.is_sleeping
    
    def sleep_consolidation(self, sleep_duration: float = 1.0) -> Dict[str, Any]:
        """
        Perform sleep-based memory consolidation.
        
        This is the core NSM process:
        1. Replay tagged experiences
        2. Strengthen positive (dopamine) pathways
        3. Weaken negative (cortisol) pathways
        4. Prune unreinforced synapses
        5. Stabilize consolidated connections
        
        Args:
            sleep_duration: How long to simulate sleep
            
        Returns:
            Dict with consolidation stats
        """
        if not hasattr(self, 'consolidation_engine') or not hasattr(self, 'cortex_tagging'):
            return {'error': 'NSM not initialized'}
        
        # Enter sleep if not already
        if hasattr(self, 'sleep_manager') and not self.sleep_manager.is_sleeping:
            self.sleep_manager.enter_sleep()
        
        # IMPORTANT: Resize tagging to match current weight shape (may have changed due to pruning/neurogenesis)
        current_shape = self.cortex.ff_weights.shape
        if hasattr(self.cortex_tagging, 'resize'):
            self.cortex_tagging.resize(current_shape)
        
        # Consolidate cortical weights
        self.cortex.ff_weights, stats = self.consolidation_engine.consolidate(
            weights=self.cortex.ff_weights,
            tagging=self.cortex_tagging,
            sleep_duration=sleep_duration,
            age=self.creature_age,
            replay_rate=5.0
        )
        
        # Run dream/replay through reservoir
        dream_summary = self._reservoir_driven_dream(steps=5)
        
        # Update sleep manager
        if hasattr(self, 'sleep_manager'):
            self.sleep_manager.sleep_tick(sleep_duration)
            self.sleep_manager.wake_up()
        
        return {
            'strengthened': stats['strengthened'],
            'weakened': stats['weakened'],
            'pruned': stats['pruned'],
            'replayed': stats['replayed'],
            'dream': dream_summary
        }
    
    def interrupt_sleep(self):
        """Interrupt sleep - lose consolidation progress."""
        if hasattr(self, 'sleep_manager'):
            self.sleep_manager.interrupt_sleep()
    
    def get_fatigue(self) -> float:
        """Get current fatigue level."""
        if hasattr(self, 'sleep_manager'):
            return self.sleep_manager.fatigue
        return 0.0
    
    def set_creature_age(self, age: float):
        """Set creature age for plasticity calculations (0-1 normalized)."""
        self.creature_age = np.clip(age, 0.0, 1.0)
    
    def set_metabolic_plasticity(self, metabolic_rate: float):
        """
        Adjust brain plasticity based on metabolic rate.
        
        High metabolism → faster learning, faster forgetting
        Low metabolism → slower learning, better retention
        
        This implements the TIER 1: Metabolic Evolution feature:
        metabolic rate affects how the brain learns and consolidates.
        
        Args:
            metabolic_rate: Creature's metabolic rate (typically 0.5 - 2.0)
        """
        # Normalize to 0-2 range (1.0 is baseline)
        normalized = np.clip(metabolic_rate, 0.5, 2.0)
        
        # High metabolism: more plasticity during wake, faster marker decay
        # Low metabolism: less plasticity but better retention
        
        if hasattr(self, 'consolidation_engine'):
            # Adjust consolidation strength based on metabolism
            # High metabolism = stronger consolidation effect
            self.consolidation_engine.POSITIVE_STRENGTHEN = 0.2 * normalized
            self.consolidation_engine.NEGATIVE_WEAKEN = 0.2 * normalized
        
        if hasattr(self, 'cortex_tagging'):
            # High metabolism = faster marker decay (need to act on memories sooner)
            # Use custom decay rate: high metabolism = 0.95, low = 0.99
            decay_rate = 1.0 - (0.05 * normalized)  # 0.95 - 0.99
            self.cortex_tagging.decay_rate = decay_rate
        
        if hasattr(self, 'sleep_manager'):
            # High metabolism = need more sleep for same fatigue
            # (more neural activity = more fatigue per tick)
            self.sleep_manager.BASE_FATIGUE_RATE = 0.01 * normalized
            self.sleep_manager.NEURAL_FATIGUE_RATE = 0.001 * normalized
    
    # =========================================================================
    # TIER 2: Predictive Minds - Pain Prediction Methods
    # =========================================================================
    
    def predict_pain(self) -> float:
        """
        Predict pain based on current cortical activation.
        
        Uses learned associations between sensory patterns and subsequent pain.
        
        Returns:
            Predicted pain level (0-1)
        """
        if not hasattr(self, 'pain_predictor_weights'):
            return 0.0
        
        # Compute predicted pain from cortical activation
        cortical = self.last_cortical_output
        if len(cortical) != len(self.pain_predictor_weights):
            return 0.0
        
        prediction = np.dot(self.pain_predictor_weights, cortical)
        self.last_pain_prediction = np.clip(prediction, 0.0, 1.0)
        return self.last_pain_prediction
    
    def update_pain_predictor(self, actual_pain: float):
        """
        Update pain predictor based on actual pain experienced.
        
        This is called after pain is experienced to train the predictor.
        Uses simple delta learning rule.
        
        Args:
            actual_pain: Actual pain level experienced (0-1)
        """
        if not hasattr(self, 'pain_predictor_weights'):
            return
        
        cortical = self.last_cortical_output
        if len(cortical) != len(self.pain_predictor_weights):
            return
        
        # Delta rule: weights += learning_rate * (actual - predicted) * input
        prediction_error = actual_pain - self.last_pain_prediction
        
        # Only learn from significant errors
        if abs(prediction_error) > 0.05:
            # Scale learning by cortical activation (active columns learn more)
            update = self.pain_prediction_learning_rate * prediction_error * cortical
            self.pain_predictor_weights += update
            
            # Regularization: keep weights bounded
            self.pain_predictor_weights = np.clip(self.pain_predictor_weights, -2.0, 2.0)
        
        # Store for history (for analysis)
        if len(self.pain_prediction_history) >= self.max_pain_history:
            self.pain_prediction_history.pop(0)
        self.pain_prediction_history.append((
            self.last_pain_prediction,
            actual_pain,
            prediction_error
        ))
    
    def should_avoid(self) -> bool:
        """
        Check if current sensory state predicts pain.
        
        Returns:
            True if predicted pain exceeds avoidance threshold
        """
        prediction = self.predict_pain()
        return prediction > self.predicted_pain_threshold
    
    def get_pain_prediction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about pain prediction performance.
        
        Returns:
            Dict with prediction accuracy and other stats
        """
        if not self.pain_prediction_history:
            return {'accuracy': 0.0, 'samples': 0}
        
        # Calculate mean absolute error
        errors = [abs(p - a) for p, a, _ in self.pain_prediction_history]
        mae = np.mean(errors)
        
        # Calculate "accuracy" as 1 - MAE
        accuracy = max(0.0, 1.0 - mae)
        
        # Count true positives/negatives
        true_positives = sum(1 for p, a, _ in self.pain_prediction_history 
                           if p > 0.3 and a > 0.3)
        true_negatives = sum(1 for p, a, _ in self.pain_prediction_history 
                           if p <= 0.3 and a <= 0.3)
        total = len(self.pain_prediction_history)
        
        return {
            'accuracy': accuracy,
            'samples': total,
            'mae': mae,
            'true_positive_rate': true_positives / max(1, total),
            'true_negative_rate': true_negatives / max(1, total),
            'current_prediction': self.last_pain_prediction,
            'weight_magnitude': float(np.mean(np.abs(self.pain_predictor_weights)))
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Save brain state to file."""
        from .persistence import BrainPersistence
        
        if self._persistence is None:
            self._persistence = BrainPersistence()
        
        return self._persistence.save(self, filepath)
    
    def load(self, filepath: str) -> None:
        """Load brain state from file."""
        from .persistence import BrainPersistence
        
        if self._persistence is None:
            self._persistence = BrainPersistence()
        
        self._persistence.load(self, filepath)

    # =========================================================================
    # TIER 4: Tool Use Interface
    # =========================================================================
    
    def encounter_tool(self, object_id: str, object_features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Encounter an object that could be used as a tool.
        
        Args:
            object_id: Unique identifier for the object
            object_features: Optional feature vector (uses cortical activation if None)
            
        Returns:
            Recognition info and suggested motor patterns
        """
        if object_features is None:
            object_features = self.last_cortical_output
        return self.tool_system.encounter_object(object_id, object_features)
    
    def use_tool_action(self, motor_command: np.ndarray, outcome_success: float) -> Dict[str, Any]:
        """
        Use current tool and learn from outcome.
        
        Args:
            motor_command: Motor pattern used
            outcome_success: 0-1 success of the action
            
        Returns:
            Learning result with familiarity and incorporation
        """
        # Also update body with motor command
        if hasattr(self, 'body'):
            self.body.act(list(motor_command[:self.body.num_muscles]))
        return self.tool_system.use_tool(motor_command, outcome_success)
    
    def grasp_tool(self, object_id: str) -> bool:
        """Grasp a tool, extending body schema."""
        return self.tool_system.grasp_tool(object_id)
    
    def release_tool(self):
        """Release currently held tool."""
        self.tool_system.release_tool()

    # =========================================================================
    # TIER 4: Abstract Reasoning Interface
    # =========================================================================
    
    def observe_pattern(self, context: str = "") -> Dict[str, Any]:
        """
        Observe current cortical activation as a pattern for abstract reasoning.
        
        Args:
            context: Optional context string
            
        Returns:
            Recognition and concept discovery info
        """
        return self.reasoning_system.observe_pattern(self.last_cortical_output, context)
    
    def solve_analogy(self, concept_a: str, concept_b: str, concept_c: str) -> Tuple[Optional[str], float]:
        """
        Solve analogy: A is to B as C is to ?
        
        Returns:
            (answer_concept, confidence) or (None, 0.0)
        """
        return self.reasoning_system.solve_analogy(concept_a, concept_b, concept_c)
    
    def get_abstract_concepts(self) -> Dict[str, Any]:
        """Get discovered abstract concepts and rules."""
        return {
            'concepts': list(self.reasoning_system.concepts.keys()),
            'rules': self.reasoning_system.abstract_rule_detection(),
            'recent_analogies': self.reasoning_system.analogy_history[-5:]
        }

    # =========================================================================
    # TIER 4: Social Structures Interface
    # =========================================================================
    
    def observe_agent(self, agent_id: str, features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Observe another agent, initializing or updating relationship.
        
        Args:
            agent_id: Unique identifier for the other agent
            features: Optional feature vector describing the agent
            
        Returns:
            Relationship info and social recommendations
        """
        return self.social_system.observe_agent(agent_id, features)
    
    def social_interaction(
        self, 
        agent_id: str, 
        interaction_type: str,
        my_outcome: float,
        their_outcome: float = 0.0
    ) -> Dict[str, Any]:
        """
        Record a social interaction and update relationship.
        
        Args:
            agent_id: The other agent
            interaction_type: "cooperation", "competition", "neutral", "conflict"
            my_outcome: My outcome (-1 to +1)
            their_outcome: Their outcome (-1 to +1)
            
        Returns:
            Updated relationship state
        """
        return self.social_system.record_interaction(agent_id, interaction_type, my_outcome, their_outcome)
    
    def propose_cooperation(self, agent_id: str, resource_value: float = 0.5) -> Dict[str, Any]:
        """
        Should I cooperate with this agent?
        
        Returns:
            Decision and reasoning
        """
        return self.social_system.propose_cooperation(agent_id, resource_value)
    
    def share_resource(self, agent_id: str, resource_id: str, amount: float) -> Dict[str, Any]:
        """
        Share a resource with another agent (modulated by oxytocin).
        
        Args:
            agent_id: Who to share with
            resource_id: What resource
            amount: How much to share
            
        Returns:
            Sharing result
        """
        oxytocin = self.state.oxytocin_level
        return self.social_system.share_resource(agent_id, resource_id, amount, oxytocin)
    
    def get_social_hierarchy(self) -> List[Tuple[str, float]]:
        """Get perceived social hierarchy."""
        return self.social_system.get_social_hierarchy()
    
    def form_group(self, group_id: str, members: List[str]) -> bool:
        """Form or join a group."""
        return self.social_system.form_group(group_id, members)

    # =========================================================================
    # TIER 4: Integrated Processing Hook
    # =========================================================================
    
    def tier4_update(self):
        """
        Update TIER 4 systems as part of the processing loop.
        
        Called automatically during process() to maintain emergent higher cognition.
        """
        # Tool skills decay when not used
        if hasattr(self, 'tool_system'):
            self.tool_system.decay_unused_tools()
        
        # Observe current pattern for abstract reasoning
        if hasattr(self, 'reasoning_system') and np.sum(self.last_cortical_output > 0.1) > 2:
            self.reasoning_system.observe_pattern(self.last_cortical_output)
        
        # Decay social relationships over time
        if hasattr(self, 'social_system'):
            self.social_system.decay_relationships()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_three_system_brain(
    scale: str = "medium",
    enable_all_features: bool = True,
    use_gpu: bool = False  # For API compatibility
) -> ThreeSystemBrain:
    """
    Factory function to create a Three-System Brain.

    Args:
        scale: "micro", "small", "medium", or "large"
        enable_all_features: Enable all enhanced features
        use_gpu: Ignored (for API compatibility)

    Returns:
        Configured ThreeSystemBrain
    """
    configs = {
        "micro": BrainConfig(
            input_dim=50,
            num_columns=50,
            cells_per_column=16,
            reservoir_size=200,
            output_dim=50,
            vocabulary_size=2000
        ),
        "small": BrainConfig(
            input_dim=100,
            num_columns=100,
            cells_per_column=32,
            reservoir_size=500,
            output_dim=100,
            vocabulary_size=5000
        ),
        "medium": BrainConfig(
            input_dim=300,
            num_columns=200,
            cells_per_column=32,
            reservoir_size=2000,
            output_dim=300,
            vocabulary_size=10000
        ),
        "large": BrainConfig(
            input_dim=500,
            num_columns=500,
            cells_per_column=32,
            reservoir_size=5000,
            output_dim=500,
            vocabulary_size=20000
        )
    }

    config = configs.get(scale, configs["medium"])
    
    return ThreeSystemBrain(
        config=config,
        enable_all_features=enable_all_features
    )


def create_brain(
    scale: str = "small",
    use_gpu: bool = False
) -> ThreeSystemBrain:
    """
    Factory function to create a brain - backwards compatible interface.
    
    Now creates ThreeSystemBrain instead of IntegratedBrain.
    
    Args:
        scale: "micro", "small", "medium", or "large"
        use_gpu: Ignored (for API compatibility)
        
    Returns:
        Configured ThreeSystemBrain instance
    """
    return create_three_system_brain(scale=scale, use_gpu=use_gpu)


# Alias for backward compatibility
IntegratedBrain = ThreeSystemBrain


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("THREE-SYSTEM BRAIN DEMO")
    print("="*60)

    # Create brain
    brain = create_brain("small")

    # Test text processing (chatbot interface)
    print("\n--- Text Processing Test ---")
    test_inputs = [
        "Hello, how are you?",
        "I'm excited to learn!",
        "What do you think about the weather?",
    ]
    
    for text in test_inputs:
        result = brain.process(text)
        print(f"\nInput: {text}")
        print(f"Response: {result['response']}")
        print(f"Mood: {result['mood']}")
        print(f"Confidence: {result['confidence']:.3f}")

    # Get dashboard data
    print("\n--- Dashboard Data ---")
    dashboard = brain.get_dashboard_data()
    print(f"Total neurons: {dashboard['neurons']['total']}")
    print(f"Active neurons: {dashboard['neurons']['active']}")
    print(f"Mood: {dashboard['mood']}")
    print(f"Dopamine: {dashboard['chemicals']['dopamine']:.3f}")

    # Get stats
    print("\n--- Stats ---")
    stats = brain.get_stats()
    print(f"Total neurons: {stats['config']['total_neurons']}")
    print(f"Sparsity: {stats['sparsity']:.4f}")

    print("\n" + "="*60)
    print("Three-System Brain demo complete!")
    print("="*60)
