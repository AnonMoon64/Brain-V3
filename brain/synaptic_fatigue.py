"""
Synaptic Fatigue and Recovery (UPGRADE 2)

Real synapses aren't fixed-strength connections. They:
1. DEPRESS after repeated use (vesicle depletion)
2. FACILITATE with rapid bursts (calcium buildup)
3. RECOVER slowly when idle

This prevents runaway oscillations and adds "thought inertia".
Fast loops tire out; slow alternations stay strong.

Key features:
- Short-term depression (STD): Synapses weaken with use
- Short-term facilitation (STF): Synapses strengthen briefly with bursts
- Recovery dynamics: Exponential recovery to baseline
- Resource-based model: Tracks available neurotransmitter vesicles

This is why you can't think the same thought 100 times/second.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum


class SynapseType(Enum):
    """
    Synaptic behavioral types.
    
    Different synapse types have different dynamics:
    - DEPRESSING: Mainly STD (cortical excitatory)
    - FACILITATING: Mainly STF (some interneurons)
    - BALANCED: Mix of both (hippocampal)
    """
    DEPRESSING = "depressing"
    FACILITATING = "facilitating"
    BALANCED = "balanced"


@dataclass
class SynapticDynamicsParams:
    """Parameters for synaptic dynamics."""
    # === SHORT-TERM DEPRESSION ===
    # Initial release probability (0-1)
    U_init: float = 0.5
    
    # Recovery time constant for resources (seconds)
    tau_rec: float = 0.8
    
    # === SHORT-TERM FACILITATION ===
    # Facilitation magnitude
    U_facil: float = 0.15
    
    # Facilitation decay time constant (seconds)
    tau_facil: float = 0.1
    
    # === RESOURCE DYNAMICS ===
    # Initial available resource (fraction 0-1)
    R_init: float = 1.0
    
    # === VESICLE REPLENISHMENT ===
    # Number of docked vesicles
    n_vesicles: int = 50
    
    # Vesicle refill rate (vesicles/second)
    refill_rate: float = 100.0
    
    @classmethod
    def for_type(cls, synapse_type: SynapseType) -> 'SynapticDynamicsParams':
        """Get parameters for specific synapse type."""
        if synapse_type == SynapseType.DEPRESSING:
            # High initial release, slow recovery
            return cls(U_init=0.6, tau_rec=0.8, U_facil=0.05, tau_facil=0.05)
        
        elif synapse_type == SynapseType.FACILITATING:
            # Low initial release, fast facilitation
            return cls(U_init=0.2, tau_rec=0.4, U_facil=0.3, tau_facil=0.2)
        
        else:  # BALANCED
            # Middle ground
            return cls(U_init=0.4, tau_rec=0.6, U_facil=0.15, tau_facil=0.1)


class DynamicSynapse:
    """
    A single synapse with short-term plasticity.
    
    Based on Tsodyks-Markram model (1997).
    
    State variables:
    - u: Current release probability (affected by facilitation)
    - R: Available resources (neurotransmitter, 0-1)
    - n_docked: Number of docked vesicles
    
    On each spike:
    1. Release amount = u * R * base_weight
    2. u increases (facilitation) and decays
    3. R decreases (depression) and recovers
    4. Vesicles are consumed and refilled
    """
    
    def __init__(
        self,
        pre_neuron_id: int,
        post_neuron_id: int,
        base_weight: float,
        synapse_type: SynapseType = SynapseType.BALANCED,
        params: Optional[SynapticDynamicsParams] = None
    ):
        """
        Initialize dynamic synapse.
        
        Args:
            pre_neuron_id: Presynaptic neuron ID
            post_neuron_id: Postsynaptic neuron ID
            base_weight: Base synaptic strength
            synapse_type: Type of synapse (determines dynamics)
            params: Custom parameters (overrides type)
        """
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.base_weight = base_weight
        self.synapse_type = synapse_type
        
        # Get parameters
        if params is None:
            self.params = SynapticDynamicsParams.for_type(synapse_type)
        else:
            self.params = params
        
        # === STATE VARIABLES ===
        # Release probability (starts at U_init)
        self.u = self.params.U_init
        
        # Available resources (starts at R_init)
        self.R = self.params.R_init
        
        # Docked vesicles
        self.n_docked = self.params.n_vesicles
        
        # Effective weight (base_weight * release_fraction)
        self.effective_weight = base_weight
        
        # === HISTORY ===
        self.last_spike_time = -np.inf
        self.total_releases = 0
        self.total_presynaptic_spikes = 0
        
        # Running average of release probability (for monitoring)
        self.avg_release_prob = self.u
    
    def process_presynaptic_spike(self, spike_time: float) -> float:
        """
        Process a presynaptic spike.
        
        Args:
            spike_time: Time of presynaptic spike
        
        Returns:
            Effective postsynaptic current (base_weight * u * R)
        """
        self.total_presynaptic_spikes += 1
        
        # Time since last spike
        dt = spike_time - self.last_spike_time if self.last_spike_time > 0 else 0.1
        dt = max(dt, 0.001)  # Minimum 1ms between spikes
        
        # === 1. UPDATE u (FACILITATION) ===
        # u decays back to U_init
        u_decay = np.exp(-dt / self.params.tau_facil)
        self.u = self.params.U_init + (self.u - self.params.U_init) * u_decay
        
        # Spike causes u to increase (facilitation)
        self.u = self.u + self.params.U_facil * (1 - self.u)
        self.u = min(self.u, 1.0)
        
        # === 2. CALCULATE RELEASE ===
        # Fraction of available resources released
        release_fraction = self.u * self.R
        
        # Check vesicle availability
        if self.n_docked > 0:
            # Successful release
            self.total_releases += 1
            
            # Consume vesicles (stochastic)
            n_released = max(1, int(self.n_docked * release_fraction))
            n_released = min(n_released, self.n_docked)
            self.n_docked -= n_released
        else:
            # No vesicles available - failure
            release_fraction = 0.0
        
        # === 3. UPDATE R (DEPRESSION) ===
        # R decays back to 1.0 (recovery)
        R_recovery = np.exp(-dt / self.params.tau_rec)
        self.R = 1.0 + (self.R - 1.0) * R_recovery
        
        # Spike causes R to decrease (depression)
        self.R = self.R * (1 - self.u)
        self.R = max(self.R, 0.0)
        
        # === 4. EFFECTIVE WEIGHT ===
        self.effective_weight = self.base_weight * release_fraction
        
        # === 5. UPDATE HISTORY ===
        self.last_spike_time = spike_time
        
        # Update running average
        self.avg_release_prob = 0.95 * self.avg_release_prob + 0.05 * release_fraction
        
        return self.effective_weight
    
    def update(self, dt: float, current_time: float):
        """
        Update synapse state (recovery during silence).
        
        Args:
            dt: Timestep in seconds
            current_time: Current simulation time
        """
        # Time since last spike
        time_since_spike = current_time - self.last_spike_time
        
        if time_since_spike > 0.001:  # Only update if not just spiked
            # === RECOVERY DYNAMICS ===
            # u decays to U_init
            u_decay = np.exp(-dt / self.params.tau_facil)
            self.u = self.params.U_init + (self.u - self.params.U_init) * u_decay
            
            # R recovers to 1.0
            R_recovery = 1.0 - (1.0 - self.R) * np.exp(-dt / self.params.tau_rec)
            self.R = R_recovery
            
            # Vesicle replenishment (Poisson process)
            expected_refill = self.params.refill_rate * dt
            n_refilled = np.random.poisson(expected_refill)
            self.n_docked = min(self.n_docked + n_refilled, self.params.n_vesicles)
            
            # Update effective weight
            self.effective_weight = self.base_weight * self.u * self.R
    
    def get_effective_weight(self) -> float:
        """Get current effective synaptic strength."""
        return self.effective_weight
    
    def get_depression_factor(self) -> float:
        """Get depression factor (0 = fully depressed, 1 = no depression)."""
        return self.R
    
    def get_facilitation_factor(self) -> float:
        """Get facilitation factor (relative to baseline U_init)."""
        return self.u / self.params.U_init if self.params.U_init > 0 else 1.0
    
    def get_state(self) -> Dict[str, float]:
        """Get full synapse state."""
        return {
            'u': self.u,
            'R': self.R,
            'n_docked': self.n_docked,
            'effective_weight': self.effective_weight,
            'base_weight': self.base_weight,
            'avg_release_prob': self.avg_release_prob,
            'total_releases': self.total_releases,
            'total_spikes': self.total_presynaptic_spikes
        }
    
    def is_fatigued(self, threshold: float = 0.5) -> bool:
        """Check if synapse is significantly fatigued."""
        return self.R < threshold or self.n_docked < self.params.n_vesicles * 0.3


class SynapticFatigueManager:
    """
    Manages short-term plasticity for all synapses in a network.
    
    Instead of static weight matrices, maintains dynamic synapses
    with fatigue and recovery.
    """
    
    def __init__(self, default_type: SynapseType = SynapseType.BALANCED):
        """
        Initialize fatigue manager.
        
        Args:
            default_type: Default synapse type for new synapses
        """
        self.default_type = default_type
        
        # Store synapses by (pre_id, post_id) key
        self.synapses: Dict[Tuple[int, int], DynamicSynapse] = {}
        
        # Cached effective weight matrix (for fast lookup)
        self.weight_matrix_cache: Optional[np.ndarray] = None
        self.cache_valid = False
    
    def add_synapse(
        self,
        pre_id: int,
        post_id: int,
        base_weight: float,
        synapse_type: Optional[SynapseType] = None
    ):
        """Add or update a synapse."""
        syn_type = synapse_type or self.default_type
        key = (pre_id, post_id)
        
        if key in self.synapses:
            # Update existing synapse weight
            self.synapses[key].base_weight = base_weight
        else:
            # Create new synapse
            self.synapses[key] = DynamicSynapse(
                pre_id, post_id, base_weight, syn_type
            )
        
        self.cache_valid = False
    
    def process_spike(self, pre_id: int, post_id: int, spike_time: float) -> float:
        """
        Process presynaptic spike.
        
        Args:
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            spike_time: Time of spike
        
        Returns:
            Effective postsynaptic current
        """
        key = (pre_id, post_id)
        if key in self.synapses:
            return self.synapses[key].process_presynaptic_spike(spike_time)
        return 0.0
    
    def update_all(self, dt: float, current_time: float):
        """Update all synapses (recovery)."""
        for synapse in self.synapses.values():
            synapse.update(dt, current_time)
        self.cache_valid = False
    
    def get_effective_weight(self, pre_id: int, post_id: int) -> float:
        """Get effective weight for specific synapse."""
        key = (pre_id, post_id)
        if key in self.synapses:
            return self.synapses[key].get_effective_weight()
        return 0.0
    
    def get_effective_weight_matrix(self, n_neurons: int) -> np.ndarray:
        """
        Get full effective weight matrix.
        
        Args:
            n_neurons: Number of neurons (matrix will be n_neurons x n_neurons)
        
        Returns:
            Weight matrix with current effective weights
        """
        W = np.zeros((n_neurons, n_neurons))
        
        for (pre_id, post_id), synapse in self.synapses.items():
            if pre_id < n_neurons and post_id < n_neurons:
                W[post_id, pre_id] = synapse.get_effective_weight()
        
        return W
    
    def get_fatigue_stats(self) -> Dict[str, any]:
        """Get statistics about synaptic fatigue."""
        if not self.synapses:
            return {
                'n_synapses': 0,
                'mean_depression': 1.0,
                'mean_facilitation': 1.0,
                'n_fatigued': 0,
                'mean_vesicles': 0
            }
        
        R_values = [s.R for s in self.synapses.values()]
        u_ratios = [s.get_facilitation_factor() for s in self.synapses.values()]
        vesicles = [s.n_docked for s in self.synapses.values()]
        fatigued = sum(1 for s in self.synapses.values() if s.is_fatigued())
        
        return {
            'n_synapses': len(self.synapses),
            'mean_depression': np.mean(R_values),
            'mean_facilitation': np.mean(u_ratios),
            'n_fatigued': fatigued,
            'fraction_fatigued': fatigued / len(self.synapses) if self.synapses else 0,
            'mean_vesicles': np.mean(vesicles)
        }
