"""
Multi-Compartment Neuron Model (UPGRADE 1)

Real neurons aren't point processors - they have:
1. Dendritic tree (with NMDA spikes for non-linear integration)
2. Soma (cell body, integrates and fires)
3. Axon hillock (spike initiation zone)

This makes neurons smarter WITHOUT increasing neuron count.

Key features:
- Dendrites can fire local NMDA spikes (non-linear computation)
- Soma integrates dendritic outputs with temporal delay
- Axon hillock applies threshold and generates action potentials
- Back-propagating action potentials affect dendrites

This is how a single neuron can compute XOR without a hidden layer.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class CompartmentType(Enum):
    """Types of neuron compartments."""
    DENDRITE = "dendrite"
    SOMA = "soma"
    AXON_HILLOCK = "axon_hillock"


@dataclass
class CompartmentParams:
    """Parameters for a neuron compartment."""
    # Capacitance (pF)
    C: float = 100.0
    
    # Leak conductance (nS)
    g_leak: float = 10.0
    
    # Leak reversal potential (mV)
    E_leak: float = -70.0
    
    # NMDA spike threshold (for dendrites only, mV)
    nmda_threshold: float = -40.0
    
    # Sodium channel density (for axon hillock, nS)
    g_Na: float = 120.0
    
    # Potassium channel density (nS)
    g_K: float = 36.0
    
    # Spike threshold (for soma/axon, mV)
    spike_threshold: float = -50.0
    
    # Coupling strength to next compartment
    coupling_g: float = 5.0


class MultiCompartmentNeuron:
    """
    A neuron with three compartments:
    1. Dendritic tree (multiple branches)
    2. Soma
    3. Axon hillock
    
    Signal flow:
    Input → Dendrites (with NMDA spikes) → Soma (integration) → Axon (spike generation) → Output
    
    Back-propagating action potentials flow from axon → soma → dendrites.
    """
    
    def __init__(
        self,
        neuron_id: int,
        n_dendrites: int = 4,
        params: Optional[CompartmentParams] = None
    ):
        """
        Initialize multi-compartment neuron.
        
        Args:
            neuron_id: Unique neuron identifier
            n_dendrites: Number of dendritic branches
            params: Compartment parameters
        """
        self.neuron_id = neuron_id
        self.n_dendrites = n_dendrites
        self.params = params or CompartmentParams()
        
        # === DENDRITIC COMPARTMENTS ===
        # Each dendrite: voltage (mV), NMDA state, synaptic input
        self.dendrite_v = np.full(n_dendrites, self.params.E_leak)
        self.dendrite_nmda_active = np.zeros(n_dendrites, dtype=bool)
        self.dendrite_input = np.zeros(n_dendrites)
        
        # Dendritic spike history (for burst detection)
        self.dendrite_spike_times = [[] for _ in range(n_dendrites)]
        
        # === SOMA COMPARTMENT ===
        self.soma_v = self.params.E_leak
        self.soma_input = 0.0
        
        # === AXON HILLOCK COMPARTMENT ===
        self.axon_v = self.params.E_leak
        self.axon_spiking = False
        self.last_spike_time = -np.inf
        
        # Hodgkin-Huxley gating variables (for axon)
        self.m = 0.05  # Sodium activation
        self.h = 0.6   # Sodium inactivation
        self.n = 0.32  # Potassium activation
        
        # === OUTPUT ===
        self.spike_out = False
        self.output_current = 0.0
        
        # === BACK-PROPAGATING ACTION POTENTIAL ===
        self.bpap_strength = 0.0  # Strength of back-propagating spike
        
        # === STATISTICS ===
        self.total_spikes = 0
        self.nmda_spike_count = 0
        self.last_nmda_spike_time = -np.inf
    
    def receive_synaptic_input(self, dendrite_idx: int, current: float):
        """
        Receive synaptic input on specific dendrite.
        
        Args:
            dendrite_idx: Which dendrite receives input (0 to n_dendrites-1)
            current: Synaptic current (pA)
        """
        if 0 <= dendrite_idx < self.n_dendrites:
            self.dendrite_input[dendrite_idx] += current
    
    def _update_dendrite(self, dendrite_idx: int, dt: float, current_time: float):
        """Update single dendritic compartment."""
        v = self.dendrite_v[dendrite_idx]
        I_syn = self.dendrite_input[dendrite_idx]
        
        # Leak current
        I_leak = self.params.g_leak * (self.params.E_leak - v)
        
        # NMDA current (voltage-dependent, magnesium block)
        # NMDA channels provide non-linear amplification
        if I_syn > 0:
            # Magnesium block factor (Johnson & Ascher 1987)
            mg_block = 1.0 / (1.0 + 0.28 * np.exp(-0.062 * v))
            I_nmda = I_syn * mg_block * 0.5  # 50% NMDA contribution
        else:
            I_nmda = 0.0
        
        # AMPA current (linear, fast)
        I_ampa = I_syn * 0.5  # 50% AMPA contribution
        
        # Coupling from soma (feedback)
        I_coupling = self.params.coupling_g * (self.soma_v - v)
        
        # Back-propagating action potential current
        I_bpap = self.bpap_strength * 100  # Large transient current
        
        # Total current
        I_total = I_leak + I_ampa + I_nmda + I_coupling + I_bpap
        
        # Voltage update (Euler integration)
        dv = (I_total / self.params.C) * dt * 1000  # Convert to mV
        v_new = v + dv
        
        # Check for NMDA spike (local dendritic spike)
        nmda_spike = False
        if v_new > self.params.nmda_threshold and not self.dendrite_nmda_active[dendrite_idx]:
            # Initiate NMDA spike
            nmda_spike = True
            self.dendrite_nmda_active[dendrite_idx] = True
            self.dendrite_spike_times[dendrite_idx].append(current_time)
            self.nmda_spike_count += 1
            self.last_nmda_spike_time = current_time
            
            # NMDA spike adds extra depolarization
            v_new += 20.0  # mV boost
        
        # NMDA spike decay
        if self.dendrite_nmda_active[dendrite_idx]:
            # Decay over ~10ms
            if current_time - self.dendrite_spike_times[dendrite_idx][-1] > 0.01:
                self.dendrite_nmda_active[dendrite_idx] = False
        
        # Clamp voltage
        v_new = np.clip(v_new, -90.0, 40.0)
        
        self.dendrite_v[dendrite_idx] = v_new
    
    def _update_soma(self, dt: float):
        """Update somatic compartment."""
        v = self.soma_v
        
        # Leak current
        I_leak = self.params.g_leak * (self.params.E_leak - v)
        
        # Sum dendritic inputs (weighted by distance/attenuation)
        dendritic_currents = []
        for i, d_v in enumerate(self.dendrite_v):
            # Attenuation based on dendrite index (distal = more attenuation)
            attenuation = 1.0 - (i / self.n_dendrites) * 0.3
            I_dend = self.params.coupling_g * (d_v - v) * attenuation
            dendritic_currents.append(I_dend)
        
        I_dendrites = np.sum(dendritic_currents)
        
        # Coupling from axon hillock
        I_axon = self.params.coupling_g * (self.axon_v - v)
        
        # External soma input (rare, but possible)
        I_ext = self.soma_input
        
        # Total current
        I_total = I_leak + I_dendrites + I_axon + I_ext
        
        # Voltage update
        dv = (I_total / self.params.C) * dt * 1000
        v_new = v + dv
        v_new = np.clip(v_new, -90.0, 40.0)
        
        self.soma_v = v_new
    
    def _update_axon_hodgkin_huxley(self, dt: float, current_time: float):
        """
        Update axon hillock using Hodgkin-Huxley model.
        This is the spike initiation zone.
        """
        v = self.axon_v
        
        # Coupling from soma
        I_soma = self.params.coupling_g * (self.soma_v - v)
        
        # Hodgkin-Huxley sodium and potassium currents
        # Voltage-dependent rate functions
        alpha_m = 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10))
        beta_m = 4.0 * np.exp(-(v + 65) / 18)
        
        alpha_h = 0.07 * np.exp(-(v + 65) / 20)
        beta_h = 1.0 / (1 + np.exp(-(v + 35) / 10))
        
        alpha_n = 0.01 * (v + 55) / (1 - np.exp(-(v + 55) / 10))
        beta_n = 0.125 * np.exp(-(v + 65) / 80)
        
        # Update gating variables
        self.m += (alpha_m * (1 - self.m) - beta_m * self.m) * dt
        self.h += (alpha_h * (1 - self.h) - beta_h * self.h) * dt
        self.n += (alpha_n * (1 - self.n) - beta_n * self.n) * dt
        
        # Clamp gating variables
        self.m = np.clip(self.m, 0, 1)
        self.h = np.clip(self.h, 0, 1)
        self.n = np.clip(self.n, 0, 1)
        
        # Ionic currents
        E_Na = 50.0   # mV
        E_K = -77.0   # mV
        
        I_Na = self.params.g_Na * (self.m ** 3) * self.h * (E_Na - v)
        I_K = self.params.g_K * (self.n ** 4) * (E_K - v)
        I_leak = self.params.g_leak * (self.params.E_leak - v)
        
        # Total current
        I_total = I_soma + I_Na + I_K + I_leak
        
        # Voltage update
        dv = (I_total / self.params.C) * dt * 1000
        v_new = v + dv
        v_new = np.clip(v_new, -90.0, 100.0)
        
        # Spike detection
        spike = False
        if v_new > self.params.spike_threshold and v < self.params.spike_threshold:
            # Rising edge: spike initiated
            spike = True
            self.total_spikes += 1
            self.last_spike_time = current_time
            
            # Trigger back-propagating action potential
            self.bpap_strength = 1.0
        
        self.axon_v = v_new
        self.spike_out = spike
        
        # Set output current proportional to spike rate
        if spike:
            self.output_current = 1.0
        else:
            # Decay output current
            self.output_current *= 0.9
    
    def update(self, dt: float, current_time: float):
        """
        Update all compartments for one timestep.
        
        Args:
            dt: Timestep in seconds
            current_time: Current simulation time
        """
        # 1. Update all dendrites
        for i in range(self.n_dendrites):
            self._update_dendrite(i, dt, current_time)
        
        # 2. Update soma (integrates dendritic activity)
        self._update_soma(dt)
        
        # 3. Update axon hillock (spike generation)
        self._update_axon_hodgkin_huxley(dt, current_time)
        
        # 4. Decay back-propagating AP
        self.bpap_strength *= 0.7  # Fast decay
        
        # 5. Reset synaptic inputs for next step
        self.dendrite_input.fill(0.0)
        self.soma_input = 0.0
    
    def get_output(self) -> float:
        """Get neuron output (spike rate encoding)."""
        return self.output_current
    
    def is_spiking(self) -> bool:
        """Check if neuron is currently spiking."""
        return self.spike_out
    
    def get_state_vector(self) -> np.ndarray:
        """Get full neuron state as vector (for monitoring)."""
        state = np.concatenate([
            self.dendrite_v,
            [self.soma_v, self.axon_v],
            [self.m, self.h, self.n],
            [float(self.spike_out)]
        ])
        return state
    
    def get_stats(self) -> Dict[str, any]:
        """Get neuron statistics."""
        return {
            'total_spikes': self.total_spikes,
            'nmda_spikes': self.nmda_spike_count,
            'soma_voltage': self.soma_v,
            'axon_voltage': self.axon_v,
            'dendrite_voltages': self.dendrite_v.tolist(),
            'gating_m': self.m,
            'gating_h': self.h,
            'gating_n': self.n,
            'output_current': self.output_current
        }


class MultiCompartmentNeuronPool:
    """
    Pool of multi-compartment neurons.
    
    This replaces the simple point neuron model in SparseCorticalEngine.
    Each "neuron" is now a sophisticated computational unit.
    """
    
    def __init__(
        self,
        n_neurons: int,
        dendrites_per_neuron: int = 4,
        params: Optional[CompartmentParams] = None
    ):
        """
        Initialize neuron pool.
        
        Args:
            n_neurons: Number of neurons
            dendrites_per_neuron: Dendritic branches per neuron
            params: Shared compartment parameters
        """
        self.n_neurons = n_neurons
        self.dendrites_per_neuron = dendrites_per_neuron
        self.params = params or CompartmentParams()
        
        # Create all neurons
        self.neurons = [
            MultiCompartmentNeuron(i, dendrites_per_neuron, self.params)
            for i in range(n_neurons)
        ]
        
        # Cached outputs (for efficiency)
        self.outputs = np.zeros(n_neurons)
        self.spike_train = np.zeros(n_neurons, dtype=bool)
    
    def receive_inputs(self, inputs: np.ndarray):
        """
        Distribute inputs to neurons.
        
        Args:
            inputs: (n_neurons,) array of synaptic currents
        """
        for i, neuron in enumerate(self.neurons):
            if i < len(inputs):
                # Distribute to random dendrite
                dendrite_idx = np.random.randint(0, self.dendrites_per_neuron)
                neuron.receive_synaptic_input(dendrite_idx, inputs[i])
    
    def update(self, dt: float, current_time: float):
        """Update all neurons."""
        for i, neuron in enumerate(self.neurons):
            neuron.update(dt, current_time)
            self.outputs[i] = neuron.get_output()
            self.spike_train[i] = neuron.is_spiking()
    
    def get_outputs(self) -> np.ndarray:
        """Get all neuron outputs."""
        return self.outputs.copy()
    
    def get_spike_train(self) -> np.ndarray:
        """Get binary spike train."""
        return self.spike_train.copy()
    
    def get_pool_stats(self) -> Dict[str, any]:
        """Get statistics for entire pool."""
        total_spikes = sum(n.total_spikes for n in self.neurons)
        total_nmda_spikes = sum(n.nmda_spike_count for n in self.neurons)
        active_neurons = np.sum(self.outputs > 0.1)
        
        mean_soma_v = np.mean([n.soma_v for n in self.neurons])
        mean_axon_v = np.mean([n.axon_v for n in self.neurons])
        
        return {
            'total_spikes': total_spikes,
            'total_nmda_spikes': total_nmda_spikes,
            'active_neurons': int(active_neurons),
            'mean_soma_voltage': mean_soma_v,
            'mean_axon_voltage': mean_axon_v,
            'spike_rate': total_spikes / self.n_neurons if self.n_neurons > 0 else 0
        }
