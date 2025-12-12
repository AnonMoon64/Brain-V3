"""
Neural Consolidation System (NSM)
=================================

Memory solidification through sleep and chemical tagging.

This module implements:
- Chemical tagging of synaptic pathways (dopamine/cortisol markers)
- Sleep-based replay and consolidation
- Structural pruning and stabilization
- Age-based plasticity
- Inherited memory installation

Experience creates temporary structural changes. Sleep replays those
experiences, and chemical markers determine what becomes permanent
architecture versus what gets pruned away.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum
from collections import deque


class MarkerType(Enum):
    """Type of chemical marker on a synapse/pathway."""
    NEUTRAL = 0
    POSITIVE = 1   # Dopamine-tagged (reinforce this)
    NEGATIVE = 2   # Cortisol-tagged (weaken/avoid this)


class ReplayMode(Enum):
    """Mode of hippocampal replay during sleep."""
    FORWARD = "forward"         # Replay in original order (sequential learning)
    REVERSE = "reverse"         # Replay backwards (credit assignment, TD-like)
    STOCHASTIC = "stochastic"   # Random sampling (generalization)
    COMPRESSED = "compressed"   # Time-compressed snippets (fast consolidation)


@dataclass
class SynapseMarker:
    """Chemical marker data for a synapse."""
    reinforcement_marker: float = 0.0  # 0.0-1.0 strength of marker
    marker_type: MarkerType = MarkerType.NEUTRAL
    experience_count: int = 0  # How many times activated during wake
    last_activation: float = 0.0  # Timestamp of last activation
    stability: float = 0.5  # 0-1, higher = more resistant to decay
    is_probationary: bool = False  # New synapses start probationary
    origin: str = "personal"  # "personal", "inherited", "cultural"


@dataclass
class Experience:
    """A tagged experience ready for potential consolidation."""
    pathway_indices: List[Tuple[int, int]]  # List of (row, col) synapse indices
    marker_type: MarkerType = MarkerType.NEUTRAL
    marker_strength: float = 0.0
    timestamp: float = 0.0
    context: str = ""  # Optional description


class ChemicalTaggingSystem:
    """
    Manages chemical markers on synapses.
    
    Tags pathways with dopamine (positive) or cortisol (negative)
    based on experience outcomes.
    """
    
    def __init__(self, weight_shape: Tuple[int, int]):
        """
        Initialize tagging system for a weight matrix.
        
        Args:
            weight_shape: Shape of the weight matrix to track (rows, cols)
        """
        self.shape = weight_shape
        
        # Parallel arrays for efficient storage
        self.reinforcement_markers = np.zeros(weight_shape, dtype=np.float32)
        self.marker_types = np.zeros(weight_shape, dtype=np.int8)  # MarkerType.value
        self.experience_counts = np.zeros(weight_shape, dtype=np.int32)
        self.last_activations = np.zeros(weight_shape, dtype=np.float32)
        self.stability = np.ones(weight_shape, dtype=np.float32) * 0.5
        self.is_probationary = np.zeros(weight_shape, dtype=bool)
        
        # Experience buffer for consolidation
        self.experiences: deque = deque(maxlen=1000)
        self.current_time = 0.0
    
    def tag_pathway(
        self,
        active_indices: np.ndarray,
        marker_type: MarkerType,
        marker_strength: float,
        context: str = ""
    ):
        """
        Tag currently active synapses with a chemical marker.
        
        Args:
            active_indices: Boolean mask or indices of active synapses
            marker_type: POSITIVE (dopamine) or NEGATIVE (cortisol)
            marker_strength: 0.0-1.0 strength of the marker
        """
        if isinstance(active_indices, np.ndarray) and active_indices.dtype == bool:
            # Boolean mask
            mask = active_indices
        else:
            # Convert indices to mask
            mask = np.zeros(self.shape, dtype=bool)
            if active_indices is not None and len(active_indices) > 0:
                for idx in active_indices:
                    if isinstance(idx, tuple) and len(idx) == 2:
                        mask[idx[0], idx[1]] = True
        
        # Update markers - new markers override old ones of same type
        # but stronger markers persist
        update_mask = mask & (marker_strength > self.reinforcement_markers)
        
        self.reinforcement_markers[update_mask] = marker_strength
        self.marker_types[update_mask] = marker_type.value
        self.experience_counts[mask] += 1
        self.last_activations[mask] = self.current_time
        
        # Record experience for replay
        pathway_indices = list(zip(*np.where(mask)))
        if pathway_indices:
            self.experiences.append(Experience(
                pathway_indices=pathway_indices[:100],  # Limit size
                marker_type=marker_type,
                marker_strength=marker_strength,
                timestamp=self.current_time,
                context=context
            ))
    
    def tag_positive(self, active_mask: np.ndarray, strength: float = 0.5, context: str = "reward"):
        """Tag active synapses as positive (dopamine)."""
        self.tag_pathway(active_mask, MarkerType.POSITIVE, strength, context)
    
    def tag_negative(self, active_mask: np.ndarray, strength: float = 0.5, context: str = "pain"):
        """Tag active synapses as negative (cortisol)."""
        self.tag_pathway(active_mask, MarkerType.NEGATIVE, strength, context)
    
    def update_time(self, dt: float):
        """Advance internal time."""
        self.current_time += dt
    
    def get_consolidation_candidates(self, top_fraction: float = 0.4) -> List[Experience]:
        """
        Get experiences ready for consolidation, sorted by importance.
        
        Args:
            top_fraction: Only return top X% of experiences
        """
        # Sort by marker strength (most important first)
        sorted_exp = sorted(
            self.experiences,
            key=lambda e: e.marker_strength,
            reverse=True
        )
        
        # Return top fraction
        n_return = max(1, int(len(sorted_exp) * top_fraction))
        return list(sorted_exp[:n_return])
    
    def clear_markers(self):
        """Clear all chemical markers after consolidation."""
        self.reinforcement_markers.fill(0.0)
        self.marker_types.fill(0)
        self.experiences.clear()
    
    def decay_markers(self, decay_rate: float = 0.99):
        """Slowly decay markers over time (forgetting without consolidation)."""
        self.reinforcement_markers *= decay_rate

    def resize(self, new_shape: Tuple[int, int]):
        """
        Resize tagging arrays when columns are added/removed.
        
        Args:
            new_shape: New shape (rows, cols) of the weight matrix
        """
        if new_shape == self.shape:
            return
            
        old_rows, old_cols = self.shape
        new_rows, new_cols = new_shape
        
        # Create new arrays
        new_reinforcement = np.zeros(new_shape, dtype=np.float32)
        new_marker_types = np.zeros(new_shape, dtype=np.int8)
        new_experience_counts = np.zeros(new_shape, dtype=np.int32)
        new_last_activations = np.zeros(new_shape, dtype=np.float32)
        new_stability = np.ones(new_shape, dtype=np.float32) * 0.5
        new_is_probationary = np.zeros(new_shape, dtype=bool)
        
        # Copy overlapping region
        min_rows = min(old_rows, new_rows)
        min_cols = min(old_cols, new_cols)
        
        new_reinforcement[:min_rows, :min_cols] = self.reinforcement_markers[:min_rows, :min_cols]
        new_marker_types[:min_rows, :min_cols] = self.marker_types[:min_rows, :min_cols]
        new_experience_counts[:min_rows, :min_cols] = self.experience_counts[:min_rows, :min_cols]
        new_last_activations[:min_rows, :min_cols] = self.last_activations[:min_rows, :min_cols]
        new_stability[:min_rows, :min_cols] = self.stability[:min_rows, :min_cols]
        new_is_probationary[:min_rows, :min_cols] = self.is_probationary[:min_rows, :min_cols]
        
        # Update references
        self.reinforcement_markers = new_reinforcement
        self.marker_types = new_marker_types
        self.experience_counts = new_experience_counts
        self.last_activations = new_last_activations
        self.stability = new_stability
        self.is_probationary = new_is_probationary
        self.shape = new_shape


class HippocampalReplay:
    """
    Hippocampal Replay System for memory consolidation.
    
    Implements biologically-inspired replay modes:
    - FORWARD: Sequential replay for procedural learning
    - REVERSE: Backwards replay for credit assignment (like TD learning)
    - STOCHASTIC: Random sampling for generalization
    - COMPRESSED: Time-compressed snippets for fast consolidation
    
    During sleep, experiences are replayed in different modes to:
    1. Strengthen important pathways
    2. Propagate reward signals backwards through sequences
    3. Extract generalizable patterns
    4. Compress episodic memories into semantic knowledge
    """
    
    # Replay parameters
    SNIPPET_LENGTH = 5          # Experiences per replay snippet
    COMPRESSION_RATIO = 0.3     # Time compression for COMPRESSED mode
    REVERSE_BONUS = 1.2         # Bonus for reverse replay (credit assignment)
    STOCHASTIC_SAMPLE = 0.4     # Fraction sampled in stochastic mode
    
    def __init__(self, max_episodes: int = 100):
        """
        Initialize hippocampal replay system.
        
        Args:
            max_episodes: Maximum episodes to store for replay
        """
        self.episodes: deque = deque(maxlen=max_episodes)
        self.current_episode: List[Experience] = []
        self.replay_count = 0
        self.mode_stats = {mode: 0 for mode in ReplayMode}
    
    def record_experience(self, experience: Experience):
        """Record an experience to current episode."""
        self.current_episode.append(experience)
    
    def end_episode(self, final_reward: float = 0.0):
        """
        End current episode and store for replay.
        
        Args:
            final_reward: Terminal reward (propagated in reverse replay)
        """
        if self.current_episode:
            # Tag final experience with terminal reward
            if self.current_episode:
                self.current_episode[-1].marker_strength = max(
                    self.current_episode[-1].marker_strength,
                    abs(final_reward)
                )
                if final_reward > 0:
                    self.current_episode[-1].marker_type = MarkerType.POSITIVE
                elif final_reward < 0:
                    self.current_episode[-1].marker_type = MarkerType.NEGATIVE
            
            self.episodes.append(list(self.current_episode))
            self.current_episode = []
    
    def get_replay_sequence(
        self,
        mode: ReplayMode,
        budget: int = 20
    ) -> List[Tuple[Experience, float]]:
        """
        Generate replay sequence based on mode.
        
        Args:
            mode: Replay mode (FORWARD, REVERSE, STOCHASTIC, COMPRESSED)
            budget: Maximum experiences to replay
            
        Returns:
            List of (experience, replay_weight) tuples
        """
        if not self.episodes:
            return []
        
        self.mode_stats[mode] += 1
        self.replay_count += 1
        
        if mode == ReplayMode.FORWARD:
            return self._forward_replay(budget)
        elif mode == ReplayMode.REVERSE:
            return self._reverse_replay(budget)
        elif mode == ReplayMode.STOCHASTIC:
            return self._stochastic_replay(budget)
        elif mode == ReplayMode.COMPRESSED:
            return self._compressed_replay(budget)
        
        return []
    
    def _forward_replay(self, budget: int) -> List[Tuple[Experience, float]]:
        """
        Forward replay - experiences in original order.
        Good for procedural/sequential learning.
        """
        result = []
        remaining = budget
        
        # Prioritize recent episodes
        for episode in reversed(list(self.episodes)):
            for exp in episode:
                if remaining <= 0:
                    break
                result.append((exp, 1.0))
                remaining -= 1
            if remaining <= 0:
                break
        
        return result
    
    def _reverse_replay(self, budget: int) -> List[Tuple[Experience, float]]:
        """
        Reverse replay - experiences in backwards order.
        Propagates terminal rewards backwards for credit assignment.
        Similar to TD(λ) eligibility traces.
        """
        result = []
        remaining = budget
        
        for episode in reversed(list(self.episodes)):
            # Reverse the episode order
            reversed_ep = list(reversed(episode))
            
            # Propagate reward backwards with decay
            propagated_reward = 0.0
            decay = 0.9  # Temporal discount
            
            for i, exp in enumerate(reversed_ep):
                if remaining <= 0:
                    break
                
                # Accumulate propagated reward
                propagated_reward = exp.marker_strength + decay * propagated_reward
                
                # Boost weight for reverse replay
                weight = self.REVERSE_BONUS * (1.0 + propagated_reward * 0.5)
                result.append((exp, weight))
                remaining -= 1
            
            if remaining <= 0:
                break
        
        return result
    
    def _stochastic_replay(self, budget: int) -> List[Tuple[Experience, float]]:
        """
        Stochastic replay - random sampling across episodes.
        Good for generalization and preventing overfitting.
        """
        # Flatten all experiences
        all_experiences = []
        for episode in self.episodes:
            all_experiences.extend(episode)
        
        if not all_experiences:
            return []
        
        # Random sampling
        n_sample = min(budget, int(len(all_experiences) * self.STOCHASTIC_SAMPLE))
        n_sample = max(1, n_sample)
        
        # Weight by marker strength (important experiences more likely)
        weights = np.array([exp.marker_strength + 0.1 for exp in all_experiences])
        weights = weights / weights.sum()
        
        indices = np.random.choice(
            len(all_experiences),
            size=min(n_sample, len(all_experiences)),
            replace=False,
            p=weights
        )
        
        result = [(all_experiences[i], 1.0) for i in indices]
        return result
    
    def _compressed_replay(self, budget: int) -> List[Tuple[Experience, float]]:
        """
        Compressed replay - extract key moments from episodes.
        Time-compressed snippets for fast consolidation.
        """
        result = []
        remaining = budget
        
        for episode in reversed(list(self.episodes)):
            if remaining <= 0:
                break
            
            if len(episode) <= self.SNIPPET_LENGTH:
                # Small episode - replay all
                for exp in episode:
                    result.append((exp, self.COMPRESSION_RATIO))
                    remaining -= 1
            else:
                # Extract key moments: start, peak, end
                # Plus random samples
                key_indices = {0, len(episode) - 1}  # Start and end
                
                # Find peak (highest marker strength)
                peak_idx = max(range(len(episode)), 
                              key=lambda i: episode[i].marker_strength)
                key_indices.add(peak_idx)
                
                # Add random samples to fill snippet
                while len(key_indices) < min(self.SNIPPET_LENGTH, len(episode)):
                    key_indices.add(np.random.randint(0, len(episode)))
                
                # Replay key moments with compression bonus
                for idx in sorted(key_indices):
                    if remaining <= 0:
                        break
                    result.append((episode[idx], 1.0 + self.COMPRESSION_RATIO))
                    remaining -= 1
        
        return result
    
    def sleep_replay_cycle(
        self,
        weights: np.ndarray,
        tagging: 'ChemicalTaggingSystem',
        sleep_duration: float,
        plasticity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Perform a full sleep replay cycle with all modes.
        
        Args:
            weights: Weight matrix to modify
            tagging: Chemical tagging system
            sleep_duration: Duration of sleep
            plasticity: Age-based plasticity multiplier
            
        Returns:
            Stats dict with replay results
        """
        stats = {
            'forward_replayed': 0,
            'reverse_replayed': 0,
            'stochastic_replayed': 0,
            'compressed_replayed': 0,
            'strengthened': 0,
            'weakened': 0
        }
        
        # Budget per mode based on sleep duration
        total_budget = int(sleep_duration * 10)  # 10 replays per unit time
        mode_budget = total_budget // 4
        
        # Phase 1: Forward replay (procedural consolidation)
        forward_seq = self.get_replay_sequence(ReplayMode.FORWARD, mode_budget)
        stats['forward_replayed'] = len(forward_seq)
        self._apply_replay(weights, forward_seq, plasticity, stats)
        
        # Phase 2: Reverse replay (credit assignment)
        reverse_seq = self.get_replay_sequence(ReplayMode.REVERSE, mode_budget)
        stats['reverse_replayed'] = len(reverse_seq)
        self._apply_replay(weights, reverse_seq, plasticity, stats)
        
        # Phase 3: Stochastic replay (generalization)
        stochastic_seq = self.get_replay_sequence(ReplayMode.STOCHASTIC, mode_budget)
        stats['stochastic_replayed'] = len(stochastic_seq)
        self._apply_replay(weights, stochastic_seq, plasticity * 0.5, stats)  # Lower plasticity
        
        # Phase 4: Compressed replay (semantic extraction)
        compressed_seq = self.get_replay_sequence(ReplayMode.COMPRESSED, mode_budget)
        stats['compressed_replayed'] = len(compressed_seq)
        self._apply_replay(weights, compressed_seq, plasticity * 0.7, stats)
        
        return stats
    
    def _apply_replay(
        self,
        weights: np.ndarray,
        replay_sequence: List[Tuple[Experience, float]],
        plasticity: float,
        stats: Dict[str, int]
    ):
        """Apply replay sequence to weights."""
        for exp, weight in replay_sequence:
            for row, col in exp.pathway_indices:
                if row >= weights.shape[0] or col >= weights.shape[1]:
                    continue
                
                change_amount = 0.1 * weight * plasticity
                
                if exp.marker_type == MarkerType.POSITIVE:
                    weights[row, col] = min(2.0, weights[row, col] + change_amount)
                    stats['strengthened'] += 1
                elif exp.marker_type == MarkerType.NEGATIVE:
                    weights[row, col] = max(-2.0, weights[row, col] - change_amount)
                    stats['weakened'] += 1
    
    def clear_episodes(self):
        """Clear all stored episodes after consolidation."""
        self.episodes.clear()
        self.current_episode = []


class ConsolidationEngine:
    """
    Performs sleep-based memory consolidation with hippocampal replay.
    
    During sleep:
    1. Hippocampal replay (forward/reverse/stochastic/compressed)
    2. Strengthen positive pathways
    3. Weaken negative pathways
    4. Prune unreplayed/weak synapses
    5. Stabilize consolidated pathways
    """
    
    # Consolidation parameters
    POSITIVE_STRENGTHEN = 0.2   # 20% weight increase
    NEGATIVE_WEAKEN = 0.2       # 20% weight decrease
    PRUNING_THRESHOLD = 0.1    # Weights below this get pruned
    STABILITY_THRESHOLD = 0.3  # Below this, connections decay faster
    MIN_CONNECTIONS = 3        # Neurons with fewer connections are candidates for removal
    
    # Plasticity by age (multipliers)
    PLASTICITY_JUVENILE = 2.0
    PLASTICITY_ADOLESCENT = 1.5
    PLASTICITY_ADULT = 1.0
    PLASTICITY_ELDER = 0.7
    
    def __init__(self):
        self.consolidation_count = 0
        self.total_strengthened = 0
        self.total_weakened = 0
        self.total_pruned = 0
        self.hippocampal_replay = HippocampalReplay(max_episodes=100)
    
    def record_experience(self, experience: Experience):
        """Record experience for hippocampal replay."""
        self.hippocampal_replay.record_experience(experience)
    
    def end_episode(self, final_reward: float = 0.0):
        """End current episode for hippocampal replay."""
        self.hippocampal_replay.end_episode(final_reward)
    
    def get_plasticity_multiplier(self, age: float, maturity_age: float = 0.5) -> float:
        """
        Get age-based plasticity multiplier.
        
        Args:
            age: Creature's age (0-1 normalized)
            maturity_age: Age at which creature is fully mature
        """
        if age < maturity_age * 0.3:
            return self.PLASTICITY_JUVENILE
        elif age < maturity_age * 0.7:
            return self.PLASTICITY_ADOLESCENT
        elif age < maturity_age * 1.2:
            return self.PLASTICITY_ADULT
        else:
            return self.PLASTICITY_ELDER
    
    def consolidate(
        self,
        weights: np.ndarray,
        tagging: ChemicalTaggingSystem,
        sleep_duration: float,
        age: float = 0.5,
        replay_rate: float = 5.0
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Perform consolidation on weights using hippocampal replay.
        
        Args:
            weights: The weight matrix to consolidate
            tagging: ChemicalTaggingSystem with markers
            sleep_duration: How long creature slept (affects replay budget)
            age: Creature's age for plasticity calculation
            replay_rate: Experiences replayed per unit of sleep time
            
        Returns:
            Tuple of (modified weights, stats dict)
        """
        stats = {
            'strengthened': 0,
            'weakened': 0,
            'pruned': 0,
            'replayed': 0,
            'forward_replayed': 0,
            'reverse_replayed': 0,
            'stochastic_replayed': 0,
            'compressed_replayed': 0
        }
        
        # Calculate plasticity
        plasticity = self.get_plasticity_multiplier(age)
        
        # PHASE 1: Hippocampal Replay (new multi-mode replay system)
        hippocampal_stats = self.hippocampal_replay.sleep_replay_cycle(
            weights, tagging, sleep_duration, plasticity
        )
        stats.update(hippocampal_stats)
        stats['replayed'] = (
            stats['forward_replayed'] + stats['reverse_replayed'] +
            stats['stochastic_replayed'] + stats['compressed_replayed']
        )
        
        # PHASE 2: Chemical tagging consolidation (original system)
        replay_budget = int(sleep_duration * replay_rate * plasticity)
        experiences = tagging.get_consolidation_candidates(top_fraction=0.4)
        
        replayed_indices = set()
        for i, exp in enumerate(experiences):
            if i >= replay_budget:
                break
            
            for row, col in exp.pathway_indices:
                if row >= weights.shape[0] or col >= weights.shape[1]:
                    continue
                    
                replayed_indices.add((row, col))
                
                if exp.marker_type == MarkerType.POSITIVE:
                    change = weights[row, col] * self.POSITIVE_STRENGTHEN * plasticity
                    weights[row, col] = min(2.0, weights[row, col] + change)
                    tagging.stability[row, col] = min(1.0, tagging.stability[row, col] + 0.1)
                    stats['strengthened'] += 1
                    
                elif exp.marker_type == MarkerType.NEGATIVE:
                    change = weights[row, col] * self.NEGATIVE_WEAKEN * plasticity
                    weights[row, col] = max(-2.0, weights[row, col] - abs(change))
                    stats['weakened'] += 1
                    
                    if abs(weights[row, col]) < self.PRUNING_THRESHOLD:
                        weights[row, col] = 0.0
                        stats['pruned'] += 1
        
        # PHASE 3: Prune unreplayed probationary synapses
        probationary_mask = tagging.is_probationary
        for row in range(weights.shape[0]):
            for col in range(weights.shape[1]):
                if probationary_mask[row, col] and (row, col) not in replayed_indices:
                    weights[row, col] = 0.0
                    tagging.is_probationary[row, col] = False
                    stats['pruned'] += 1
        
        # Decay unstable synapses
        unstable_mask = tagging.stability < self.STABILITY_THRESHOLD
        weights[unstable_mask] *= 0.9  # Gradual decay
        
        # Clear markers (they've been processed)
        tagging.clear_markers()
        
        # Convert replayed probationary to permanent
        for row, col in replayed_indices:
            if row < weights.shape[0] and col < weights.shape[1]:
                tagging.is_probationary[row, col] = False
        
        # Update stats
        self.consolidation_count += 1
        self.total_strengthened += stats['strengthened']
        self.total_weakened += stats['weakened']
        self.total_pruned += stats['pruned']
        
        return weights, stats
    
    def prune_isolated_neurons(
        self,
        weights: np.ndarray,
        min_connections: int = 3
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Remove neurons with too few connections.
        
        Returns:
            Tuple of (weight matrix with zeros for removed neurons, list of removed indices)
        """
        removed = []
        
        # Count connections per neuron (row)
        connections_out = np.sum(np.abs(weights) > 0.01, axis=1)
        
        # Find neurons with too few connections
        for i in range(weights.shape[0]):
            if connections_out[i] < min_connections:
                weights[i, :] = 0.0
                removed.append(i)
        
        return weights, removed


class InheritedMemoryInstaller:
    """
    Installs inherited neural patterns in offspring.
    
    Parent patterns are copied at reduced strength and marked as
    probationary. First sleep cycle consolidates inherited instincts.
    """
    
    INHERITANCE_STRENGTH = 0.3  # Inherited at 30% parent strength
    INHERITANCE_PRIORITY = 0.8  # High consolidation priority
    
    def install_inherited_patterns(
        self,
        child_weights: np.ndarray,
        parent_weights: np.ndarray,
        child_tagging: ChemicalTaggingSystem,
        mutation_rate: float = 0.1
    ) -> np.ndarray:
        """
        Install parent neural patterns in child brain.
        
        Args:
            child_weights: Child's weight matrix (will be modified)
            parent_weights: Parent's weight matrix to copy from
            child_tagging: Child's tagging system for marking probationary
            mutation_rate: Probability of mutation per synapse
            
        Returns:
            Modified child weights
        """
        # Copy parent weights at reduced strength
        inherited = parent_weights * self.INHERITANCE_STRENGTH
        
        # Apply mutations
        mutation_mask = np.random.random(inherited.shape) < mutation_rate
        mutations = np.random.randn(*inherited.shape) * 0.1
        inherited[mutation_mask] += mutations[mutation_mask]
        
        # Blend with child's existing weights (if any)
        # Inherited patterns complement but don't override
        blend_mask = np.abs(child_weights) < 0.01  # Only fill empty slots
        child_weights[blend_mask] = inherited[blend_mask]
        
        # Mark all inherited synapses as probationary
        inherited_mask = np.abs(inherited) > 0.01
        child_tagging.is_probationary[inherited_mask[:child_tagging.shape[0], :child_tagging.shape[1]]] = True
        
        # Tag for high-priority consolidation
        child_tagging.reinforcement_markers[inherited_mask[:child_tagging.shape[0], :child_tagging.shape[1]]] = self.INHERITANCE_PRIORITY
        child_tagging.marker_types[inherited_mask[:child_tagging.shape[0], :child_tagging.shape[1]]] = MarkerType.POSITIVE.value
        
        return child_weights


class SleepManager:
    """
    Manages sleep state and consolidation for a creature.
    
    Handles:
    - Fatigue accumulation
    - Sleep triggers
    - Consolidation during sleep
    - Sleep interruption
    """
    
    # Fatigue parameters
    BASE_FATIGUE_RATE = 0.01      # Per tick
    NEURAL_FATIGUE_RATE = 0.001   # Per active neuron
    STRESS_FATIGUE_MULT = 0.005   # Cortisol multiplier
    SLEEP_THRESHOLD = 10.0        # Fatigue level to trigger sleep
    
    # Sleep parameters
    SENSORY_DAMPENING = 0.9       # 90% reduction during sleep
    CONSOLIDATION_RATE = 5.0      # Experiences per sleep tick
    
    def __init__(self):
        self.fatigue = 0.0
        self.is_sleeping = False
        self.sleep_duration = 0.0
        self.consolidation_progress = 0.0
        self.sleep_interrupted = False
        self.times_interrupted = 0
    
    def accumulate_fatigue(
        self,
        dt: float,
        active_neurons: int,
        cortisol_level: float,
        activity_level: float = 0.5
    ):
        """
        Accumulate fatigue during waking hours.
        
        Args:
            dt: Time delta
            active_neurons: Number of currently active neurons
            cortisol_level: Current cortisol (stress) level
            activity_level: Physical activity level (0-1)
        """
        if self.is_sleeping:
            return
            
        fatigue_gain = (
            self.BASE_FATIGUE_RATE +
            self.NEURAL_FATIGUE_RATE * active_neurons +
            self.STRESS_FATIGUE_MULT * cortisol_level +
            0.005 * activity_level  # Physical activity adds fatigue
        ) * dt
        
        self.fatigue += fatigue_gain
    
    def should_sleep(self) -> bool:
        """Check if creature should enter sleep."""
        return self.fatigue >= self.SLEEP_THRESHOLD
    
    def enter_sleep(self):
        """Enter sleep state."""
        self.is_sleeping = True
        self.sleep_duration = 0.0
        self.consolidation_progress = 0.0
        self.sleep_interrupted = False
    
    def sleep_tick(self, dt: float) -> float:
        """
        Process one tick of sleep.
        
        Returns:
            consolidation_progress (0-1) for this tick
        """
        if not self.is_sleeping:
            return 0.0
            
        self.sleep_duration += dt
        self.consolidation_progress += dt * self.CONSOLIDATION_RATE
        
        # Fatigue decreases during sleep
        self.fatigue = max(0, self.fatigue - 0.1 * dt)
        
        return dt * self.CONSOLIDATION_RATE
    
    def interrupt_sleep(self):
        """
        Interrupt sleep - lose consolidation progress.
        """
        if self.is_sleeping:
            self.sleep_interrupted = True
            self.times_interrupted += 1
            self.fatigue *= 0.7  # Only partial rest
            self.is_sleeping = False
            self.consolidation_progress = 0.0
    
    def wake_up(self) -> bool:
        """
        Wake up naturally after consolidation.
        
        Returns:
            True if woke naturally, False if was already awake
        """
        if not self.is_sleeping:
            return False
            
        self.is_sleeping = False
        self.fatigue = 0.0
        return True
    
    def get_sensory_dampening(self) -> float:
        """Get sensory dampening factor during sleep."""
        return self.SENSORY_DAMPENING if self.is_sleeping else 0.0


# Convenience function for creating a full consolidation system
def create_consolidation_system(weight_shape: Tuple[int, int]) -> Dict[str, Any]:
    """
    Create a complete consolidation system for a brain.
    
    Args:
        weight_shape: Shape of the primary weight matrix
        
    Returns:
        Dict with all consolidation components including hippocampal replay
    """
    engine = ConsolidationEngine()
    return {
        'tagging': ChemicalTaggingSystem(weight_shape),
        'engine': engine,
        'hippocampal': engine.hippocampal_replay,  # Direct access to replay system
        'installer': InheritedMemoryInstaller(),
        'sleep': SleepManager()
    }
