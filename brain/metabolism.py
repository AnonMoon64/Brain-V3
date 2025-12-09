"""
Advanced Metabolism and Homeostatic Plasticity

Implements:
- Per-neuron ATP/energy metabolism
- Mitochondrial dynamics and oxidative stress
- Homeostatic plasticity (synaptic scaling, intrinsic plasticity)
- Network-wide homeostasis (E/I balance)
- Activity-dependent metabolic adaptation
- Glial support (simplified astrocyte model)

Key insight: Neurons are metabolically expensive. Energy constraints
shape computation and plasticity. Homeostatic mechanisms ensure
stability over long timescales.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import math


class MetabolicState(Enum):
    """Metabolic states affecting neural function"""
    NORMAL = "normal"
    HYPOMETABOLIC = "hypometabolic"  # Low energy, reduced firing
    HYPERMETABOLIC = "hypermetabolic"  # High energy, increased activity
    STRESSED = "stressed"  # Oxidative stress
    RECOVERY = "recovery"  # Recovering from stress


@dataclass
class MitochondrialDynamics:
    """
    Simplified mitochondrial model for ATP production
    
    ATP production depends on:
    - Substrate availability (glucose, oxygen)
    - Activity demand
    - Mitochondrial health
    """
    # ATP pool
    atp_level: float = 0.8  # Current ATP (0-1)
    atp_baseline: float = 0.8
    
    # Mitochondrial capacity
    max_production_rate: float = 0.1
    current_production: float = 0.05
    
    # Substrate availability
    glucose: float = 1.0
    oxygen: float = 1.0
    
    # Mitochondrial health
    health: float = 1.0
    damage: float = 0.0
    repair_rate: float = 0.001
    
    # Oxidative stress (byproduct of high activity)
    ros_level: float = 0.0  # Reactive oxygen species
    antioxidant_capacity: float = 0.5
    
    def produce_atp(self, activity_demand: float, dt: float) -> float:
        """
        Produce ATP based on demand and capacity
        Returns: ATP produced
        """
        # Production limited by substrates and health
        max_possible = self.max_production_rate * self.glucose * self.oxygen * self.health
        
        # Demand-driven production (up to max)
        production = min(activity_demand * 0.2, max_possible) * dt
        
        self.atp_level = min(1.0, self.atp_level + production)
        self.current_production = production / dt if dt > 0 else 0
        
        # High activity produces ROS
        if production > 0.05:
            ros_produced = production * 0.1
            self.ros_level = min(1.0, self.ros_level + ros_produced)
        
        return production
    
    def consume_atp(self, amount: float) -> float:
        """
        Consume ATP, returns actual amount consumed
        """
        consumed = min(self.atp_level, amount)
        self.atp_level -= consumed
        return consumed
    
    def update(self, dt: float) -> None:
        """Update mitochondrial state"""
        # Neutralize ROS with antioxidants
        neutralized = min(self.ros_level, self.antioxidant_capacity * dt)
        self.ros_level -= neutralized
        
        # ROS causes damage
        if self.ros_level > 0.5:
            self.damage += (self.ros_level - 0.5) * 0.01 * dt
            self.damage = min(0.9, self.damage)
        
        # Repair damage
        self.damage = max(0, self.damage - self.repair_rate * dt)
        self.health = 1.0 - self.damage
        
        # Recover ROS
        self.ros_level = max(0, self.ros_level - 0.02 * dt)
        
        # ATP homeostasis (drift toward baseline)
        self.atp_level += (self.atp_baseline - self.atp_level) * 0.01 * dt


@dataclass
class NeuronMetabolism:
    """
    Per-neuron metabolism with realistic costs
    
    Firing a spike costs ~10^9 ATP molecules
    Maintaining resting potential costs significant energy
    Synaptic transmission has metabolic costs
    """
    neuron_id: str = ""
    
    # Mitochondria
    mitochondria: MitochondrialDynamics = field(default_factory=MitochondrialDynamics)
    
    # Energy costs (relative units)
    spike_cost: float = 0.02  # Cost per spike
    maintenance_cost: float = 0.001  # Resting metabolic rate per dt
    synapse_cost: float = 0.005  # Cost per synaptic event
    plasticity_cost: float = 0.01  # Cost of weight change
    
    # Metabolic state
    state: MetabolicState = MetabolicState.NORMAL
    
    # Activity-dependent adaptation
    metabolic_demand: float = 0.5  # Running estimate of demand
    demand_history: List[float] = field(default_factory=list)
    
    # Energy-dependent thresholds
    firing_threshold_modifier: float = 1.0
    
    def can_fire(self) -> bool:
        """Check if neuron has energy to fire"""
        return self.mitochondria.atp_level > self.spike_cost * 2
    
    def fire_spike(self) -> bool:
        """Consume energy for spike, return success"""
        consumed = self.mitochondria.consume_atp(self.spike_cost)
        return consumed >= self.spike_cost * 0.8
    
    def synaptic_event(self) -> None:
        """Consume energy for synaptic transmission"""
        self.mitochondria.consume_atp(self.synapse_cost)
    
    def apply_plasticity(self) -> bool:
        """Consume energy for plasticity, return if sufficient"""
        consumed = self.mitochondria.consume_atp(self.plasticity_cost)
        return consumed >= self.plasticity_cost * 0.5
    
    def update(self, activity_level: float, dt: float) -> None:
        """Update metabolism based on activity"""
        # Maintenance cost
        self.mitochondria.consume_atp(self.maintenance_cost * dt)
        
        # Track demand
        self.demand_history.append(activity_level)
        if len(self.demand_history) > 100:
            self.demand_history.pop(0)
        self.metabolic_demand = np.mean(self.demand_history)
        
        # Produce ATP to meet demand
        self.mitochondria.produce_atp(self.metabolic_demand + 0.1, dt)
        self.mitochondria.update(dt)
        
        # Update state
        atp = self.mitochondria.atp_level
        if atp < 0.2:
            self.state = MetabolicState.HYPOMETABOLIC
            self.firing_threshold_modifier = 1.5  # Harder to fire
        elif atp > 0.9:
            self.state = MetabolicState.HYPERMETABOLIC
            self.firing_threshold_modifier = 0.8  # Easier to fire
        elif self.mitochondria.ros_level > 0.6:
            self.state = MetabolicState.STRESSED
            self.firing_threshold_modifier = 1.3
        else:
            self.state = MetabolicState.NORMAL
            self.firing_threshold_modifier = 1.0
    
    @property
    def energy(self) -> float:
        """Current energy level (0-1)"""
        return self.mitochondria.atp_level
    
    @property
    def health(self) -> float:
        """Overall metabolic health"""
        return self.mitochondria.health * (1.0 - self.mitochondria.ros_level * 0.3)


@dataclass
class HomeostaticPlasticity:
    """
    Homeostatic plasticity mechanisms
    
    Maintains stable firing rates over long timescales:
    - Synaptic scaling: multiplicative adjustment of all inputs
    - Intrinsic plasticity: adjustment of excitability
    - Sliding threshold: BCM-like metaplasticity
    """
    neuron_id: str = ""
    
    # Target firing rate
    target_rate: float = 5.0  # Hz
    
    # Synaptic scaling
    scaling_factor: float = 1.0
    scaling_tau: float = 3600.0  # Time constant (seconds) - slow!
    
    # Intrinsic excitability
    excitability: float = 1.0
    excitability_tau: float = 1800.0
    
    # Sliding threshold (BCM)
    theta_m: float = 0.5  # Modification threshold
    theta_tau: float = 900.0
    
    # Rate estimation
    spike_count: int = 0
    time_window: float = 0.0
    estimated_rate: float = 0.0
    rate_history: List[float] = field(default_factory=list)
    
    def record_spike(self) -> None:
        """Record a spike for rate estimation"""
        self.spike_count += 1
    
    def update(self, dt: float) -> Dict[str, float]:
        """
        Update homeostatic variables
        Returns changes to apply
        """
        self.time_window += dt / 1000.0  # Convert to seconds
        
        changes = {
            'scaling': 0.0,
            'excitability': 0.0,
            'theta': 0.0,
        }
        
        # Estimate rate every second
        if self.time_window >= 1.0:
            self.estimated_rate = self.spike_count / self.time_window
            self.rate_history.append(self.estimated_rate)
            if len(self.rate_history) > 100:
                self.rate_history.pop(0)
            
            self.spike_count = 0
            self.time_window = 0.0
            
            # Rate error
            rate_error = self.target_rate - self.estimated_rate
            
            # Synaptic scaling (very slow)
            scaling_change = rate_error * 0.01 / self.scaling_tau
            self.scaling_factor = max(0.1, min(10.0, self.scaling_factor + scaling_change))
            changes['scaling'] = scaling_change
            
            # Intrinsic excitability (slower)
            exc_change = rate_error * 0.02 / self.excitability_tau
            self.excitability = max(0.1, min(5.0, self.excitability + exc_change))
            changes['excitability'] = exc_change
            
            # Sliding threshold (tracks recent average)
            if len(self.rate_history) > 10:
                avg_rate = np.mean(self.rate_history[-10:])
                theta_target = avg_rate / (self.target_rate + 0.1)
                theta_change = (theta_target - self.theta_m) / self.theta_tau
                self.theta_m = max(0.1, min(2.0, self.theta_m + theta_change))
                changes['theta'] = theta_change
        
        return changes
    
    def modulate_input(self, input_current: float) -> float:
        """Apply scaling and excitability to input"""
        return input_current * self.scaling_factor * self.excitability
    
    def get_plasticity_threshold(self) -> float:
        """Get current plasticity threshold (for BCM rule)"""
        return self.theta_m


@dataclass
class SimplifiedAstrocyte:
    """
    Simplified astrocyte model for metabolic support
    
    Astrocytes:
    - Provide glucose/lactate to neurons
    - Buffer extracellular potassium
    - Release gliotransmitters
    - Regulate local blood flow
    """
    supported_neurons: List[str] = field(default_factory=list)
    
    # Metabolic support
    glycogen_store: float = 1.0  # Energy reserve
    lactate_release: float = 0.1  # Support to neurons
    
    # Potassium buffering
    extracellular_k: float = 0.0
    buffering_capacity: float = 0.5
    
    # Gliotransmission
    glutamate_uptake: float = 0.0
    atp_release: float = 0.0  # Purinergic signaling
    d_serine_release: float = 0.0  # NMDA co-agonist
    
    # Calcium waves (simplified)
    calcium_level: float = 0.1
    wave_propagating: bool = False
    
    def support_metabolism(self, neuron_demand: float) -> float:
        """
        Provide metabolic support based on demand
        Returns lactate supplied
        """
        # Activity-dependent support
        support_needed = neuron_demand * 0.3
        
        if self.glycogen_store > support_needed:
            self.glycogen_store -= support_needed
            return support_needed * self.lactate_release
        else:
            available = self.glycogen_store
            self.glycogen_store = 0
            return available * self.lactate_release * 0.5
    
    def buffer_potassium(self, k_released: float) -> float:
        """Buffer extracellular potassium, return amount buffered"""
        buffered = min(k_released, self.buffering_capacity)
        self.extracellular_k += k_released - buffered
        return buffered
    
    def uptake_glutamate(self, glutamate: float) -> None:
        """Remove glutamate from synaptic cleft"""
        self.glutamate_uptake += glutamate
        
        # High glutamate triggers calcium wave
        if self.glutamate_uptake > 0.5:
            self.calcium_level = min(1.0, self.calcium_level + glutamate * 0.2)
    
    def update(self, dt: float) -> Dict[str, float]:
        """Update astrocyte state"""
        # Replenish glycogen
        self.glycogen_store = min(1.0, self.glycogen_store + 0.01 * dt)
        
        # Clear extracellular K+
        self.extracellular_k = max(0, self.extracellular_k - 0.1 * dt)
        
        # Process glutamate
        self.glutamate_uptake = max(0, self.glutamate_uptake - 0.2 * dt)
        
        # Calcium dynamics
        if self.calcium_level > 0.5:
            self.wave_propagating = True
            # Trigger gliotransmitter release
            self.d_serine_release = self.calcium_level * 0.1
            self.atp_release = self.calcium_level * 0.05
        else:
            self.wave_propagating = False
            self.d_serine_release *= 0.9
            self.atp_release *= 0.9
        
        self.calcium_level = max(0.1, self.calcium_level - 0.05 * dt)
        
        return {
            'd_serine': self.d_serine_release,
            'atp': self.atp_release,
            'support_available': self.glycogen_store,
        }


class NetworkHomeostasis:
    """
    Network-wide homeostatic regulation
    
    Maintains:
    - E/I balance across the network
    - Total activity within bounds
    - Prevents runaway excitation or complete silence
    """
    
    def __init__(self, target_activity: float = 0.05):
        self.target_activity = target_activity  # Fraction of neurons active
        
        # Global gain control
        self.global_gain: float = 1.0
        self.gain_tau: float = 100.0  # Time constant (ms)
        
        # E/I balance
        self.ei_ratio: float = 0.8  # Target E/I ratio
        self.current_ei: float = 0.8
        
        # Activity monitoring
        self.activity_history: List[float] = []
        self.max_history: int = 100
        
        # Emergency shutoff
        self.seizure_threshold: float = 0.5  # Too much activity
        self.silent_threshold: float = 0.001  # Too little activity
        self.in_emergency: bool = False
    
    def update(
        self, 
        active_fraction: float, 
        excitatory_activity: float,
        inhibitory_activity: float,
        dt: float
    ) -> Dict[str, float]:
        """
        Update network homeostasis
        Returns modulation factors
        """
        self.activity_history.append(active_fraction)
        if len(self.activity_history) > self.max_history:
            self.activity_history.pop(0)
        
        modulation = {
            'gain': self.global_gain,
            'excitatory_scale': 1.0,
            'inhibitory_scale': 1.0,
            'emergency': False,
        }
        
        # Check for emergency conditions
        if active_fraction > self.seizure_threshold:
            self.in_emergency = True
            modulation['emergency'] = True
            modulation['gain'] = 0.1  # Massive reduction
            modulation['inhibitory_scale'] = 3.0  # Boost inhibition
            return modulation
        
        if active_fraction < self.silent_threshold:
            self.in_emergency = True
            modulation['emergency'] = True
            modulation['gain'] = 2.0  # Boost activity
            modulation['excitatory_scale'] = 1.5
            return modulation
        
        self.in_emergency = False
        
        # Normal homeostatic adjustment
        activity_error = self.target_activity - active_fraction
        gain_change = activity_error * 0.1 / self.gain_tau * dt
        self.global_gain = max(0.1, min(5.0, self.global_gain + gain_change))
        modulation['gain'] = self.global_gain
        
        # E/I balance
        if inhibitory_activity > 0.01:
            self.current_ei = excitatory_activity / inhibitory_activity
        
        ei_error = self.ei_ratio - self.current_ei
        if abs(ei_error) > 0.2:
            if ei_error > 0:  # Need more E or less I
                modulation['excitatory_scale'] = 1.0 + ei_error * 0.2
            else:  # Need more I or less E
                modulation['inhibitory_scale'] = 1.0 - ei_error * 0.2
        
        return modulation
    
    def get_stability_score(self) -> float:
        """Get network stability score (0-1, higher is more stable)"""
        if len(self.activity_history) < 10:
            return 0.5
        
        recent = self.activity_history[-10:]
        variance = np.var(recent)
        mean_activity = np.mean(recent)
        
        # Score based on variance and proximity to target
        variance_score = 1.0 / (1.0 + variance * 100)
        target_score = 1.0 - abs(mean_activity - self.target_activity) / self.target_activity
        
        return (variance_score + target_score) / 2


class MetabolicNetwork:
    """
    Complete metabolic system for neural network
    
    Combines:
    - Per-neuron metabolism
    - Homeostatic plasticity
    - Astrocytic support
    - Network homeostasis
    """
    
    def __init__(self, neuron_ids: List[str] = None):
        neuron_ids = neuron_ids or []
        
        # Per-neuron systems
        self.neuron_metabolism: Dict[str, NeuronMetabolism] = {}
        self.homeostatic: Dict[str, HomeostaticPlasticity] = {}
        
        # Astrocytes (one per ~10 neurons, simplified)
        self.astrocytes: List[SimplifiedAstrocyte] = []
        
        # Network homeostasis
        self.network_homeostasis = NetworkHomeostasis()
        
        # Initialize for given neurons
        for nid in neuron_ids:
            self.add_neuron(nid)
    
    def add_neuron(self, neuron_id: str) -> None:
        """Add metabolic tracking for a neuron"""
        self.neuron_metabolism[neuron_id] = NeuronMetabolism(neuron_id=neuron_id)
        self.homeostatic[neuron_id] = HomeostaticPlasticity(neuron_id=neuron_id)
        
        # Add astrocyte if needed (1 per 10 neurons)
        if len(self.neuron_metabolism) % 10 == 1:
            astro = SimplifiedAstrocyte()
            self.astrocytes.append(astro)
        
        # Assign to astrocyte
        if self.astrocytes:
            astro_idx = (len(self.neuron_metabolism) - 1) // 10
            astro_idx = min(astro_idx, len(self.astrocytes) - 1)
            self.astrocytes[astro_idx].supported_neurons.append(neuron_id)
    
    def can_neuron_fire(self, neuron_id: str) -> bool:
        """Check if neuron has energy to fire"""
        if neuron_id in self.neuron_metabolism:
            return self.neuron_metabolism[neuron_id].can_fire()
        return True
    
    def neuron_fired(self, neuron_id: str) -> bool:
        """Record that neuron fired, return if successful"""
        if neuron_id in self.neuron_metabolism:
            success = self.neuron_metabolism[neuron_id].fire_spike()
            if success and neuron_id in self.homeostatic:
                self.homeostatic[neuron_id].record_spike()
            return success
        return True
    
    def get_threshold_modifier(self, neuron_id: str) -> float:
        """Get firing threshold modifier based on metabolism"""
        if neuron_id in self.neuron_metabolism:
            return self.neuron_metabolism[neuron_id].firing_threshold_modifier
        return 1.0
    
    def get_input_modulation(self, neuron_id: str, input_current: float) -> float:
        """Modulate input based on homeostatic state"""
        if neuron_id in self.homeostatic:
            return self.homeostatic[neuron_id].modulate_input(input_current)
        return input_current
    
    def can_apply_plasticity(self, neuron_id: str) -> bool:
        """Check if neuron has energy for plasticity"""
        if neuron_id in self.neuron_metabolism:
            return self.neuron_metabolism[neuron_id].apply_plasticity()
        return True
    
    def update(
        self,
        neuron_activities: Dict[str, float],
        spikes: List[str],
        dt: float
    ) -> Dict[str, any]:
        """
        Update all metabolic systems
        
        Args:
            neuron_activities: activity level per neuron
            spikes: list of neuron IDs that spiked
            dt: timestep (ms)
        """
        results = {
            'energy_levels': {},
            'homeostatic_changes': {},
            'network_modulation': {},
        }
        
        # Update per-neuron metabolism
        total_demand = 0.0
        for nid, activity in neuron_activities.items():
            if nid in self.neuron_metabolism:
                self.neuron_metabolism[nid].update(activity, dt)
                results['energy_levels'][nid] = self.neuron_metabolism[nid].energy
                total_demand += self.neuron_metabolism[nid].metabolic_demand
        
        # Update homeostatic plasticity
        for nid in self.homeostatic:
            changes = self.homeostatic[nid].update(dt)
            if any(abs(v) > 0.001 for v in changes.values()):
                results['homeostatic_changes'][nid] = changes
        
        # Update astrocytes
        for astro in self.astrocytes:
            # Provide support based on demand of supported neurons
            local_demand = sum(
                self.neuron_metabolism.get(nid, NeuronMetabolism()).metabolic_demand
                for nid in astro.supported_neurons
            ) / max(1, len(astro.supported_neurons))
            
            support = astro.support_metabolism(local_demand)
            
            # Distribute support
            for nid in astro.supported_neurons:
                if nid in self.neuron_metabolism:
                    self.neuron_metabolism[nid].mitochondria.glucose = min(
                        1.0, 
                        self.neuron_metabolism[nid].mitochondria.glucose + support
                    )
            
            astro.update(dt)
        
        # Update network homeostasis
        n_total = len(neuron_activities)
        active_fraction = len(spikes) / max(1, n_total)
        
        # Simplified E/I calculation (assume 80% excitatory)
        n_excitatory = int(n_total * 0.8)
        exc_activity = sum(
            neuron_activities.get(nid, 0) 
            for i, nid in enumerate(neuron_activities) 
            if i < n_excitatory
        )
        inh_activity = sum(
            neuron_activities.get(nid, 0)
            for i, nid in enumerate(neuron_activities)
            if i >= n_excitatory
        )
        
        network_mod = self.network_homeostasis.update(
            active_fraction, exc_activity, inh_activity, dt
        )
        results['network_modulation'] = network_mod
        
        return results
    
    def get_stats(self) -> Dict:
        """Get metabolic statistics"""
        if not self.neuron_metabolism:
            return {}
        
        energies = [nm.energy for nm in self.neuron_metabolism.values()]
        healths = [nm.health for nm in self.neuron_metabolism.values()]
        
        states = {}
        for nm in self.neuron_metabolism.values():
            state = nm.state.value
            states[state] = states.get(state, 0) + 1
        
        return {
            'mean_energy': np.mean(energies),
            'min_energy': np.min(energies),
            'max_energy': np.max(energies),
            'mean_health': np.mean(healths),
            'metabolic_states': states,
            'network_stability': self.network_homeostasis.get_stability_score(),
            'network_gain': self.network_homeostasis.global_gain,
            'in_emergency': self.network_homeostasis.in_emergency,
            'n_astrocytes': len(self.astrocytes),
        }
    
    def get_state(self) -> Dict:
        """Get complete metabolic state for serialization"""
        return {
            'stats': self.get_stats(),
            'total_atp': self.total_atp,
            'n_neurons': len(self.neuron_metabolism)
        }
    
    @property
    def total_atp(self) -> float:
        """Get total ATP across all neurons"""
        if not self.neuron_metabolism:
            return 100.0
        return sum(nm.energy for nm in self.neuron_metabolism.values())
    
    def step(self, dt: float):
        """Simple step update (without explicit activities)"""
        # Update with zero activity
        dummy_activities = {nid: 0.0 for nid in self.neuron_metabolism}
        self.update(dummy_activities, [], dt)


class SynapticScaling:
    """
    Synaptic scaling homeostatic mechanism.
    
    Scales all synaptic weights to maintain target firing rate.
    """
    
    def __init__(
        self,
        num_neurons: int = 1000,
        target_rate: float = 0.02,
        scaling_rate: float = 0.001
    ):
        self.num_neurons = num_neurons
        self.target_rate = target_rate
        self.scaling_rate = scaling_rate
        
        # Scaling factors per neuron
        self.scaling_factors = np.ones(num_neurons)
        
        # Activity history
        self.activity_history = np.zeros(num_neurons)
        self.history_window = 1000  # samples
        self.history_count = 0
    
    def record_activity(self, activity: np.ndarray):
        """Record activity for averaging"""
        # Exponential moving average
        alpha = 1.0 / self.history_window
        self.activity_history = (1 - alpha) * self.activity_history + alpha * activity
        self.history_count += 1
    
    def update(self, dt: float):
        """Update scaling factors based on activity history"""
        if self.history_count < 10:
            return
        
        # Compute deviation from target
        deviation = self.target_rate - self.activity_history
        
        # Update scaling factors
        self.scaling_factors += self.scaling_rate * deviation
        
        # Clamp to reasonable range
        self.scaling_factors = np.clip(self.scaling_factors, 0.1, 10.0)
    
    def scale_down(self, factor: float = 0.99):
        """Scale down all factors (used in low-energy state)"""
        self.scaling_factors *= factor
    
    def scale_up(self, factor: float = 1.01):
        """Scale up all factors"""
        self.scaling_factors = np.minimum(10.0, self.scaling_factors * factor)
    
    def get_factor(self, neuron_idx: int) -> float:
        """Get scaling factor for a neuron"""
        if 0 <= neuron_idx < len(self.scaling_factors):
            return float(self.scaling_factors[neuron_idx])
        return 1.0


class IntrinsicPlasticity:
    """
    Intrinsic plasticity mechanism.
    
    Adjusts neuron excitability to maintain target firing rate.
    """
    
    def __init__(
        self,
        num_neurons: int = 1000,
        target_rate: float = 0.02,
        learning_rate: float = 0.01
    ):
        self.num_neurons = num_neurons
        self.target_rate = target_rate
        self.learning_rate = learning_rate
        
        # Excitability (threshold modifiers)
        self.excitability = np.ones(num_neurons)
        
        # Activity history
        self.activity_ema = np.zeros(num_neurons)
    
    def record_activity(self, activity: np.ndarray):
        """Record activity"""
        alpha = 0.01
        self.activity_ema = (1 - alpha) * self.activity_ema + alpha * activity
    
    def update(self, dt: float):
        """Update excitability based on activity"""
        # Neurons firing too much become less excitable
        deviation = self.activity_ema - self.target_rate
        
        # Adjust excitability inversely to activity
        self.excitability -= self.learning_rate * deviation
        
        # Clamp to reasonable range
        self.excitability = np.clip(self.excitability, 0.5, 2.0)
    
    def get_threshold_modifier(self, neuron_idx: int) -> float:
        """Get threshold modifier (inverse of excitability)"""
        if 0 <= neuron_idx < len(self.excitability):
            return float(1.0 / self.excitability[neuron_idx])
        return 1.0
