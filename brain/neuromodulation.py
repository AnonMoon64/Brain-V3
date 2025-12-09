"""
Advanced Kinetic Neuromodulation System

This implements true biochemistry-based neuromodulation:
- Receptor binding kinetics (Michaelis-Menten dynamics)
- Three-factor learning rules (pre, post, neuromodulator)
- Tonic vs phasic neuromodulator release
- Receptor desensitization and internalization
- Second messenger cascades
- Metaplasticity (plasticity of plasticity)

Key insight: Neuromodulation isn't just scaling - it fundamentally
changes the learning rules and computational properties of circuits.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import math


class ModulatorType(Enum):
    """Types of neuromodulators with distinct dynamics"""
    # Core neuromodulators (with kinetic receptor models)
    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"
    NOREPINEPHRINE = "norepinephrine"
    ACETYLCHOLINE = "acetylcholine"
    
    # Additional neurotransmitters (simpler dynamics)
    GABA = "gaba"                   # Primary inhibitory
    GLUTAMATE = "glutamate"         # Primary excitatory
    CORTISOL = "cortisol"           # Stress hormone
    OXYTOCIN = "oxytocin"           # Social bonding
    ENDORPHIN = "endorphin"         # Pain/pleasure
    ADRENALINE = "adrenaline"       # Acute stress


class ReceptorSubtype(Enum):
    """Receptor subtypes with different kinetics"""
    # Dopamine receptors
    D1 = "D1"  # Excitatory, Gs-coupled, increases cAMP
    D2 = "D2"  # Inhibitory, Gi-coupled, decreases cAMP
    
    # Serotonin receptors
    HT5_1A = "5HT1A"  # Inhibitory
    HT5_2A = "5HT2A"  # Excitatory
    
    # Norepinephrine receptors  
    ALPHA1 = "α1"  # Excitatory
    ALPHA2 = "α2"  # Inhibitory (autoreceptor)
    BETA = "β"     # Excitatory
    
    # Acetylcholine receptors
    MUSCARINIC_M1 = "M1"  # Excitatory
    MUSCARINIC_M2 = "M2"  # Inhibitory
    NICOTINIC = "nAChR"   # Fast ionotropic
    
    # GABA receptors
    GABA_A = "GABA_A"  # Fast inhibitory (ionotropic)
    GABA_B = "GABA_B"  # Slow inhibitory (metabotropic)
    
    # Glutamate receptors
    AMPA = "AMPA"      # Fast excitatory
    NMDA = "NMDA"      # Slow excitatory, learning


@dataclass
class ReceptorKinetics:
    """
    Michaelis-Menten receptor binding kinetics
    
    dB/dt = kon * L * (Bmax - B) - koff * B
    
    where:
    - B = bound receptors
    - L = ligand (neuromodulator) concentration
    - Bmax = total receptors
    - kon = binding rate
    - koff = unbinding rate
    """
    receptor_type: ReceptorSubtype
    
    # Kinetic parameters
    bmax: float = 1.0           # Total receptor density
    bound: float = 0.0          # Currently bound fraction
    kon: float = 0.1            # Binding rate constant
    koff: float = 0.05          # Unbinding rate constant
    
    # Desensitization
    desensitized: float = 0.0   # Fraction of desensitized receptors
    desensitization_rate: float = 0.01
    resensitization_rate: float = 0.005
    
    # Internalization (longer-term downregulation)
    internalized: float = 0.0
    internalization_rate: float = 0.001
    recycling_rate: float = 0.0005
    
    # Efficacy (how much bound receptor activates downstream)
    efficacy: float = 1.0
    is_excitatory: bool = True
    
    def update(self, ligand_concentration: float, dt: float) -> float:
        """
        Update receptor binding state
        Returns: effective activation (accounting for desensitization)
        """
        # Available receptors (not internalized or desensitized)
        available = self.bmax - self.internalized - self.desensitized
        available = max(0.0, available)
        
        # Michaelis-Menten binding dynamics
        binding = self.kon * ligand_concentration * (available - self.bound)
        unbinding = self.koff * self.bound
        
        self.bound += (binding - unbinding) * dt
        self.bound = max(0.0, min(available, self.bound))
        
        # Desensitization (occurs when receptor is activated)
        if self.bound > 0.3:  # High occupancy triggers desensitization
            deactivate = self.desensitization_rate * self.bound * dt
            self.desensitized += deactivate
            self.bound -= deactivate
        
        # Resensitization
        resens = self.resensitization_rate * self.desensitized * dt
        self.desensitized -= resens
        self.desensitized = max(0.0, self.desensitized)
        
        # Internalization (prolonged high activation)
        if self.bound > 0.5:
            intern = self.internalization_rate * self.bound * dt
            self.internalized += intern
        
        # Recycling
        recycle = self.recycling_rate * self.internalized * dt
        self.internalized -= recycle
        self.internalized = max(0.0, min(0.5, self.internalized))
        
        # Effective activation
        activation = self.bound * self.efficacy
        return activation if self.is_excitatory else -activation
    
    @property
    def sensitivity(self) -> float:
        """Current receptor sensitivity (affected by desensitization/internalization)"""
        available = self.bmax - self.internalized - self.desensitized
        return max(0.1, available / self.bmax)


@dataclass
class SecondMessenger:
    """
    Second messenger cascade (cAMP, Ca2+, etc.)
    These are the actual effectors of neuromodulation.
    """
    name: str
    level: float = 0.0
    baseline: float = 0.0
    
    # Dynamics
    production_rate: float = 0.1
    degradation_rate: float = 0.05
    
    # Effects on plasticity
    ltp_modulation: float = 1.0   # How much this affects LTP
    ltd_modulation: float = 1.0   # How much this affects LTD
    
    # History for metaplasticity
    history: List[float] = field(default_factory=list)
    max_history: int = 100
    
    def update(self, receptor_activation: float, dt: float) -> None:
        """Update second messenger level based on receptor activation"""
        production = self.production_rate * max(0, receptor_activation)
        degradation = self.degradation_rate * self.level
        drift = 0.01 * (self.baseline - self.level)
        
        self.level += (production - degradation + drift) * dt
        self.level = max(0.0, min(2.0, self.level))
        
        # Track history for metaplasticity
        self.history.append(self.level)
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    @property
    def metaplasticity_factor(self) -> float:
        """
        Metaplasticity: recent history of activity affects plasticity threshold
        High recent activity -> harder to induce LTP (BCM rule)
        """
        if len(self.history) < 10:
            return 1.0
        recent_avg = np.mean(self.history[-10:])
        return 1.0 / (1.0 + recent_avg)  # Higher recent activity = lower plasticity


class ThreeFactorSTDP:
    """
    Three-factor learning rule: pre-synaptic, post-synaptic, neuromodulator
    
    The key insight from computational neuroscience:
    - STDP sets eligibility trace
    - Neuromodulator determines if eligibility converts to actual weight change
    - This solves the credit assignment problem
    """
    
    def __init__(self):
        # STDP time constants
        self.tau_plus = 20.0   # LTP time constant (ms)
        self.tau_minus = 20.0  # LTD time constant (ms)
        
        # Base learning rates
        self.a_plus = 0.005    # LTP amplitude
        self.a_minus = 0.006   # LTD amplitude (slightly stronger for stability)
        
        # Eligibility trace
        self.tau_eligibility = 1000.0  # Eligibility trace time constant (ms)
        
        # Neuromodulator influence
        self.dopamine_ltp_boost = 2.0
        self.dopamine_ltd_suppress = 0.5
        self.acetylcholine_attention = 1.5
        self.norepinephrine_consolidation = 1.3
        self.serotonin_stability = 0.8
    
    def compute_eligibility(
        self, 
        pre_spike_time: float, 
        post_spike_time: float,
        current_time: float
    ) -> Tuple[float, float]:
        """
        Compute eligibility traces for LTP and LTD
        Returns: (ltp_eligibility, ltd_eligibility)
        """
        dt = post_spike_time - pre_spike_time
        
        if dt > 0:  # Pre before post -> LTP eligible
            spike_age = current_time - max(pre_spike_time, post_spike_time)
            decay = math.exp(-spike_age / self.tau_eligibility)
            ltp = self.a_plus * math.exp(-dt / self.tau_plus) * decay
            return (ltp, 0.0)
        elif dt < 0:  # Post before pre -> LTD eligible
            spike_age = current_time - max(pre_spike_time, post_spike_time)
            decay = math.exp(-spike_age / self.tau_eligibility)
            ltd = self.a_minus * math.exp(dt / self.tau_minus) * decay
            return (0.0, ltd)
        return (0.0, 0.0)
    
    def compute_weight_change(
        self,
        ltp_eligibility: float,
        ltd_eligibility: float,
        modulators: Dict[ModulatorType, float],
        camp_level: float,
        metaplasticity: float
    ) -> float:
        """
        Convert eligibility to actual weight change based on neuromodulators
        
        Key rules:
        - Dopamine gates LTP (reward prediction error)
        - Serotonin promotes stability
        - Acetylcholine enhances attention-driven learning
        - Norepinephrine enhances consolidation
        - cAMP is the final effector
        """
        dopamine = modulators.get(ModulatorType.DOPAMINE, 0.5)
        serotonin = modulators.get(ModulatorType.SEROTONIN, 0.5)
        acetylcholine = modulators.get(ModulatorType.ACETYLCHOLINE, 0.5)
        norepinephrine = modulators.get(ModulatorType.NOREPINEPHRINE, 0.5)
        
        # Dopamine effect (reward signal)
        # High dopamine: boost LTP, suppress LTD
        # Low dopamine: suppress LTP, boost LTD
        dopamine_effect_ltp = dopamine * self.dopamine_ltp_boost
        dopamine_effect_ltd = (1.0 - dopamine) * self.dopamine_ltd_suppress + dopamine * 0.3
        
        # Acetylcholine effect (attention)
        ach_effect = 0.5 + acetylcholine * self.acetylcholine_attention
        
        # Norepinephrine effect (arousal/consolidation)
        ne_effect = 0.7 + norepinephrine * self.norepinephrine_consolidation
        
        # Serotonin effect (stability - reduces extreme changes)
        serotonin_stability = 1.0 - (serotonin - 0.5) * self.serotonin_stability
        
        # cAMP is the final common pathway
        camp_effect = 0.5 + camp_level
        
        # Compute final weight change
        ltp_change = ltp_eligibility * dopamine_effect_ltp * ach_effect * camp_effect * metaplasticity
        ltd_change = ltd_eligibility * dopamine_effect_ltd * camp_effect * metaplasticity
        
        # Serotonin modulates the total magnitude
        total_change = (ltp_change - ltd_change) * serotonin_stability * ne_effect
        
        return total_change


@dataclass
class NeuromodulatorRelease:
    """
    Models both tonic and phasic neuromodulator release
    
    Tonic: sustained baseline release (sets operating point)
    Phasic: burst release (signals specific events)
    """
    modulator_type: ModulatorType
    
    # Tonic release
    tonic_level: float = 0.5
    tonic_baseline: float = 0.5
    tonic_decay: float = 0.01
    
    # Phasic release
    phasic_level: float = 0.0
    phasic_decay: float = 0.1  # Fast decay
    
    # Autoreceptor regulation
    autoreceptor_sensitivity: float = 1.0
    
    # Release probability
    release_prob: float = 0.8
    vesicle_pool: float = 1.0
    vesicle_recovery: float = 0.05
    
    def tonic_release(self, target: float, dt: float) -> float:
        """Adjust tonic level toward target"""
        diff = target - self.tonic_level
        self.tonic_level += diff * self.tonic_decay * dt
        return self.tonic_level
    
    def phasic_burst(self, magnitude: float) -> float:
        """Trigger phasic burst release"""
        # Autoreceptor inhibition
        effective_release = magnitude * self.autoreceptor_sensitivity
        
        # Vesicle depletion
        if self.vesicle_pool > 0.1:
            released = min(effective_release, self.vesicle_pool * self.release_prob)
            self.vesicle_pool -= released * 0.3
            self.phasic_level += released
            return released
        return 0.0
    
    def update(self, dt: float) -> float:
        """Update and return total neuromodulator level"""
        # Decay phasic
        self.phasic_level *= (1.0 - self.phasic_decay * dt)
        
        # Recover vesicles
        self.vesicle_pool = min(1.0, self.vesicle_pool + self.vesicle_recovery * dt)
        
        # Autoreceptor adaptation
        total = self.tonic_level + self.phasic_level
        if total > 0.7:
            self.autoreceptor_sensitivity *= 0.99  # Reduce sensitivity
        elif total < 0.3:
            self.autoreceptor_sensitivity = min(1.5, self.autoreceptor_sensitivity * 1.01)
        
        return min(2.0, total)


class KineticNeuromodulationSystem:
    """
    Complete kinetic neuromodulation system
    
    Integrates:
    - Neuromodulator release dynamics
    - Receptor binding kinetics
    - Second messenger cascades
    - Three-factor learning rules
    - Metaplasticity
    - Chemical interactions (from original chemicals.py)
    """
    
    def __init__(self):
        # Neuromodulator release systems - core 4 with kinetic receptors
        self.release_systems: Dict[ModulatorType, NeuromodulatorRelease] = {
            ModulatorType.DOPAMINE: NeuromodulatorRelease(
                modulator_type=ModulatorType.DOPAMINE,
                tonic_baseline=0.3,
                phasic_decay=0.15  # Fast DA transients
            ),
            ModulatorType.SEROTONIN: NeuromodulatorRelease(
                modulator_type=ModulatorType.SEROTONIN,
                tonic_baseline=0.5,
                phasic_decay=0.05  # Slower 5-HT dynamics
            ),
            ModulatorType.NOREPINEPHRINE: NeuromodulatorRelease(
                modulator_type=ModulatorType.NOREPINEPHRINE,
                tonic_baseline=0.4,
                phasic_decay=0.12
            ),
            ModulatorType.ACETYLCHOLINE: NeuromodulatorRelease(
                modulator_type=ModulatorType.ACETYLCHOLINE,
                tonic_baseline=0.5,
                phasic_decay=0.2  # Very fast ACh dynamics
            ),
        }
        
        # Additional neurotransmitters with simpler dynamics
        self.simple_chemicals: Dict[ModulatorType, float] = {
            ModulatorType.GABA: 0.5,
            ModulatorType.GLUTAMATE: 0.5,
            ModulatorType.CORTISOL: 0.3,
            ModulatorType.OXYTOCIN: 0.4,
            ModulatorType.ENDORPHIN: 0.3,
            ModulatorType.ADRENALINE: 0.2,
        }
        
        # Baselines for simple chemicals
        self.simple_baselines: Dict[ModulatorType, float] = {
            ModulatorType.GABA: 0.5,
            ModulatorType.GLUTAMATE: 0.5,
            ModulatorType.CORTISOL: 0.3,
            ModulatorType.OXYTOCIN: 0.4,
            ModulatorType.ENDORPHIN: 0.3,
            ModulatorType.ADRENALINE: 0.2,
        }
        
        # Receptor populations (per modulator type)
        self.receptors: Dict[ReceptorSubtype, ReceptorKinetics] = {
            # Dopamine receptors
            ReceptorSubtype.D1: ReceptorKinetics(
                receptor_type=ReceptorSubtype.D1,
                kon=0.15, koff=0.05,
                efficacy=1.2, is_excitatory=True
            ),
            ReceptorSubtype.D2: ReceptorKinetics(
                receptor_type=ReceptorSubtype.D2,
                kon=0.2, koff=0.08,  # Higher affinity
                efficacy=0.8, is_excitatory=False
            ),
            # Serotonin receptors
            ReceptorSubtype.HT5_1A: ReceptorKinetics(
                receptor_type=ReceptorSubtype.HT5_1A,
                kon=0.1, koff=0.03,
                efficacy=0.7, is_excitatory=False
            ),
            ReceptorSubtype.HT5_2A: ReceptorKinetics(
                receptor_type=ReceptorSubtype.HT5_2A,
                kon=0.12, koff=0.04,
                efficacy=1.0, is_excitatory=True
            ),
            # Norepinephrine receptors
            ReceptorSubtype.ALPHA1: ReceptorKinetics(
                receptor_type=ReceptorSubtype.ALPHA1,
                kon=0.1, koff=0.05,
                efficacy=1.0, is_excitatory=True
            ),
            ReceptorSubtype.BETA: ReceptorKinetics(
                receptor_type=ReceptorSubtype.BETA,
                kon=0.15, koff=0.06,
                efficacy=1.1, is_excitatory=True
            ),
            # Acetylcholine receptors
            ReceptorSubtype.MUSCARINIC_M1: ReceptorKinetics(
                receptor_type=ReceptorSubtype.MUSCARINIC_M1,
                kon=0.08, koff=0.02,
                efficacy=1.0, is_excitatory=True
            ),
            ReceptorSubtype.NICOTINIC: ReceptorKinetics(
                receptor_type=ReceptorSubtype.NICOTINIC,
                kon=0.3, koff=0.2,  # Fast ionotropic
                efficacy=1.5, is_excitatory=True
            ),
            # GABA receptors
            ReceptorSubtype.GABA_A: ReceptorKinetics(
                receptor_type=ReceptorSubtype.GABA_A,
                kon=0.25, koff=0.15,  # Fast ionotropic
                efficacy=1.0, is_excitatory=False
            ),
            ReceptorSubtype.GABA_B: ReceptorKinetics(
                receptor_type=ReceptorSubtype.GABA_B,
                kon=0.1, koff=0.05,  # Slow metabotropic
                efficacy=0.8, is_excitatory=False
            ),
            # Glutamate receptors
            ReceptorSubtype.AMPA: ReceptorKinetics(
                receptor_type=ReceptorSubtype.AMPA,
                kon=0.3, koff=0.2,  # Fast
                efficacy=1.0, is_excitatory=True
            ),
            ReceptorSubtype.NMDA: ReceptorKinetics(
                receptor_type=ReceptorSubtype.NMDA,
                kon=0.1, koff=0.03,  # Slow, learning-related
                efficacy=1.5, is_excitatory=True
            ),
        }
        
        # Second messenger systems
        self.second_messengers = {
            'cAMP': SecondMessenger(
                name='cAMP',
                baseline=0.3,
                production_rate=0.15,
                degradation_rate=0.1,
                ltp_modulation=1.5,
                ltd_modulation=0.5
            ),
            'Ca2+': SecondMessenger(
                name='Ca2+',
                baseline=0.1,
                production_rate=0.2,
                degradation_rate=0.15,
                ltp_modulation=1.0,
                ltd_modulation=1.0
            ),
            'PKA': SecondMessenger(
                name='PKA',
                baseline=0.2,
                production_rate=0.1,
                degradation_rate=0.05,
                ltp_modulation=2.0,
                ltd_modulation=0.3
            ),
        }
        
        # Three-factor learning rule
        self.stdp = ThreeFactorSTDP()
        
        # Modulator-receptor mapping
        self.modulator_receptors = {
            ModulatorType.DOPAMINE: [ReceptorSubtype.D1, ReceptorSubtype.D2],
            ModulatorType.SEROTONIN: [ReceptorSubtype.HT5_1A, ReceptorSubtype.HT5_2A],
            ModulatorType.NOREPINEPHRINE: [ReceptorSubtype.ALPHA1, ReceptorSubtype.BETA],
            ModulatorType.ACETYLCHOLINE: [ReceptorSubtype.MUSCARINIC_M1, ReceptorSubtype.NICOTINIC],
            ModulatorType.GABA: [ReceptorSubtype.GABA_A, ReceptorSubtype.GABA_B],
            ModulatorType.GLUTAMATE: [ReceptorSubtype.AMPA, ReceptorSubtype.NMDA],
        }
        
        # Current levels cache
        self._current_levels: Dict[ModulatorType, float] = {}
    
    def _apply_chemical_interactions(self, dt: float) -> None:
        """
        Apply inter-chemical effects (ported from chemicals.py).
        Models how neurotransmitters affect each other.
        """
        # Get current levels
        dopamine = self.get_level(ModulatorType.DOPAMINE)
        serotonin = self.get_level(ModulatorType.SEROTONIN)
        norepinephrine = self.get_level(ModulatorType.NOREPINEPHRINE)
        gaba = self.simple_chemicals.get(ModulatorType.GABA, 0.5)
        glutamate = self.simple_chemicals.get(ModulatorType.GLUTAMATE, 0.5)
        cortisol = self.simple_chemicals.get(ModulatorType.CORTISOL, 0.3)
        oxytocin = self.simple_chemicals.get(ModulatorType.OXYTOCIN, 0.4)
        adrenaline = self.simple_chemicals.get(ModulatorType.ADRENALINE, 0.2)
        
        # === Dopamine-Serotonin Balance ===
        # High serotonin moderates dopamine (prevents mania)
        if serotonin > 0.7 and ModulatorType.DOPAMINE in self.release_systems:
            self.release_systems[ModulatorType.DOPAMINE].tonic_level *= 0.98
        
        # === Cortisol Effects (stress hormone) ===
        if cortisol > 0.6:
            # Chronic stress depletes serotonin
            if ModulatorType.SEROTONIN in self.release_systems:
                self.release_systems[ModulatorType.SEROTONIN].tonic_level *= 0.98
            # Stress inhibits oxytocin (social withdrawal)
            self.simple_chemicals[ModulatorType.OXYTOCIN] *= 0.98
            # Cortisol enhances glutamate
            self.simple_chemicals[ModulatorType.GLUTAMATE] = min(1.0, glutamate + cortisol * 0.01)
        
        # === GABA-Glutamate Balance ===
        excitation_ratio = glutamate / max(0.1, gaba)
        if excitation_ratio > 2.0:
            # Too much excitation - compensatory GABA increase
            self.simple_chemicals[ModulatorType.GABA] = min(1.0, gaba + 0.02)
        elif excitation_ratio < 0.5:
            # Too much inhibition - reduce GABA
            self.simple_chemicals[ModulatorType.GABA] = max(0.1, gaba - 0.01)
        
        # === Adrenaline Effects ===
        if adrenaline > 0.7:
            # High adrenaline increases norepinephrine
            if ModulatorType.NOREPINEPHRINE in self.release_systems:
                self.release_systems[ModulatorType.NOREPINEPHRINE].tonic_level = min(
                    1.0, self.release_systems[ModulatorType.NOREPINEPHRINE].tonic_level + 0.02
                )
            # Suppress oxytocin during fight-or-flight
            self.simple_chemicals[ModulatorType.OXYTOCIN] *= 0.95
        
        # === Homeostatic drift for simple chemicals ===
        for chem_type in self.simple_chemicals:
            current = self.simple_chemicals[chem_type]
            baseline = self.simple_baselines[chem_type]
            diff = baseline - current
            self.simple_chemicals[chem_type] += diff * 0.02 * dt
            # Clamp to valid range
            self.simple_chemicals[chem_type] = max(0.0, min(1.0, self.simple_chemicals[chem_type]))
    
    def update(self, dt: float) -> Dict[str, float]:
        """
        Update entire neuromodulation system
        Returns dict of current levels and states
        """
        state = {}
        
        # Update each neuromodulator release system
        for mod_type, release_sys in self.release_systems.items():
            level = release_sys.update(dt)
            self._current_levels[mod_type] = level
            state[f'{mod_type.value}_level'] = level
            state[f'{mod_type.value}_vesicles'] = release_sys.vesicle_pool
        
        # Update receptors based on neuromodulator levels
        total_excitation = 0.0
        total_inhibition = 0.0
        
        for mod_type, receptor_types in self.modulator_receptors.items():
            ligand_conc = self._current_levels.get(mod_type, 0.5)
            for rec_type in receptor_types:
                receptor = self.receptors[rec_type]
                activation = receptor.update(ligand_conc, dt)
                state[f'{rec_type.value}_bound'] = receptor.bound
                state[f'{rec_type.value}_sensitivity'] = receptor.sensitivity
                
                if activation > 0:
                    total_excitation += activation
                else:
                    total_inhibition += abs(activation)
        
        # Update second messengers based on receptor activation
        # D1 and beta-adrenergic increase cAMP
        d1_activation = self.receptors[ReceptorSubtype.D1].bound
        beta_activation = self.receptors[ReceptorSubtype.BETA].bound
        # D2 decreases cAMP
        d2_activation = self.receptors[ReceptorSubtype.D2].bound
        
        net_camp_drive = d1_activation + beta_activation - d2_activation * 0.5
        self.second_messengers['cAMP'].update(net_camp_drive, dt)
        
        # Muscarinic and glutamate increase Ca2+ (simplified)
        m1_activation = self.receptors[ReceptorSubtype.MUSCARINIC_M1].bound
        self.second_messengers['Ca2+'].update(m1_activation + total_excitation * 0.3, dt)
        
        # PKA follows cAMP
        self.second_messengers['PKA'].update(self.second_messengers['cAMP'].level, dt)
        
        for name, messenger in self.second_messengers.items():
            state[f'{name}_level'] = messenger.level
        
        # Apply chemical interactions
        self._apply_chemical_interactions(dt)
        
        # Add simple chemical levels to state
        for chem_type, level in self.simple_chemicals.items():
            state[f'{chem_type.value}_level'] = level
            self._current_levels[chem_type] = level
        
        return state
    
    def get_level(self, modulator: ModulatorType) -> float:
        """Get current level of any neuromodulator/neurotransmitter."""
        # Check kinetic systems first
        if modulator in self.release_systems:
            return self.release_systems[modulator].tonic_level
        # Then check simple chemicals
        if modulator in self.simple_chemicals:
            return self.simple_chemicals[modulator]
        # Default
        return 0.5
    
    def set_level(self, modulator: ModulatorType, level: float) -> None:
        """Set level of a neuromodulator/neurotransmitter."""
        level = max(0.0, min(1.0, level))
        if modulator in self.release_systems:
            self.release_systems[modulator].tonic_level = level
        elif modulator in self.simple_chemicals:
            self.simple_chemicals[modulator] = level
    
    def get_all_levels(self) -> Dict[str, float]:
        """Get all neurotransmitter levels as a dictionary."""
        levels = {}
        for mod_type in self.release_systems:
            levels[mod_type.value] = self.get_level(mod_type)
        for chem_type in self.simple_chemicals:
            levels[chem_type.value] = self.simple_chemicals[chem_type]
        return levels
    
    def trigger_phasic_release(
        self, 
        modulator: ModulatorType, 
        magnitude: float,
        event_type: str = 'generic'
    ) -> float:
        """
        Trigger phasic burst of neuromodulator
        
        Event types:
        - 'reward': dopamine burst
        - 'punishment': dopamine dip, norepinephrine burst
        - 'surprise': norepinephrine burst
        - 'social': serotonin, oxytocin-related effects
        - 'attention': acetylcholine burst
        """
        if modulator in self.release_systems:
            return self.release_systems[modulator].phasic_burst(magnitude)
        return 0.0
    
    def compute_plasticity(
        self,
        pre_spike_time: float,
        post_spike_time: float,
        current_time: float
    ) -> float:
        """
        Compute weight change using three-factor rule
        """
        # Get eligibility
        ltp_elig, ltd_elig = self.stdp.compute_eligibility(
            pre_spike_time, post_spike_time, current_time
        )
        
        # Get current modulator levels
        modulators = {mod: self._current_levels.get(mod, 0.5) 
                     for mod in ModulatorType}
        
        # Get second messenger state
        camp_level = self.second_messengers['cAMP'].level
        
        # Get metaplasticity factor
        metaplasticity = self.second_messengers['PKA'].metaplasticity_factor
        
        # Compute weight change
        dw = self.stdp.compute_weight_change(
            ltp_elig, ltd_elig, modulators, camp_level, metaplasticity
        )
        
        return dw
    
    def get_learning_modulation(self) -> Dict[str, float]:
        """
        Get current learning modulation factors
        Returns dict with LTP/LTD modulation, attention, etc.
        """
        da = self._current_levels.get(ModulatorType.DOPAMINE, 0.5)
        ne = self._current_levels.get(ModulatorType.NOREPINEPHRINE, 0.5)
        ach = self._current_levels.get(ModulatorType.ACETYLCHOLINE, 0.5)
        ht = self._current_levels.get(ModulatorType.SEROTONIN, 0.5)
        
        camp = self.second_messengers['cAMP'].level
        pka = self.second_messengers['PKA'].level
        
        return {
            'ltp_modulation': da * 1.5 + camp * 0.5 + pka,
            'ltd_modulation': (1.0 - da) * 0.5 + 0.5,
            'attention': ach * 1.5 + ne * 0.5,
            'consolidation': ne * 0.8 + ht * 0.4,
            'stability': ht * 0.6 + 0.4,
            'exploration': ne * 0.5 + (1.0 - ht) * 0.3,
            'plasticity_threshold': self.second_messengers['PKA'].metaplasticity_factor,
        }
    
    def get_state(self) -> Dict[str, float]:
        """Get complete neuromodulation state"""
        state = {}
        for mod, level in self._current_levels.items():
            state[mod.value] = level
        for name, messenger in self.second_messengers.items():
            state[f'second_messenger_{name}'] = messenger.level
        for rec_type, receptor in self.receptors.items():
            state[f'receptor_{rec_type.value}_sensitivity'] = receptor.sensitivity
        return state
    
    def set_baseline(self, modulator: 'NeuromodulatorType', value: float):
        """Set baseline level for a neuromodulator"""
        # Convert NeuromodulatorType to ModulatorType if needed
        if isinstance(modulator, NeuromodulatorType):
            mod_map = {
                NeuromodulatorType.DOPAMINE: ModulatorType.DOPAMINE,
                NeuromodulatorType.SEROTONIN: ModulatorType.SEROTONIN,
                NeuromodulatorType.NOREPINEPHRINE: ModulatorType.NOREPINEPHRINE,
                NeuromodulatorType.ACETYLCHOLINE: ModulatorType.ACETYLCHOLINE,
            }
            modulator = mod_map.get(modulator, modulator)
        
        if modulator in self._current_levels:
            self._current_levels[modulator] = value
        if modulator in self.release_systems:
            self.release_systems[modulator].tonic_level = value


# Alias for compatibility with integrated brain
NeuromodulatorType = ModulatorType


# Additional classes for the integrated brain interface

class ReceptorType(Enum):
    """Simplified receptor types for interface compatibility"""
    D1_DOPAMINE = "D1"
    D2_DOPAMINE = "D2"
    ALPHA_ADRENERGIC = "alpha"
    BETA_ADRENERGIC = "beta"
    MUSCARINIC = "muscarinic"
    SEROTONIN_5HT = "5HT"


@dataclass
class NeuromodulatorReceptor:
    """Single receptor with binding dynamics"""
    receptor_type: ReceptorType
    binding_affinity: float = 0.5
    bound_fraction: float = 0.0
    sensitivity: float = 1.0
    
    def bind(self, concentration: float, dt: float) -> float:
        """Update binding state and return activation"""
        k_on = self.binding_affinity * 10.0
        k_off = 1.0
        
        # Update bound fraction
        d_bound = k_on * concentration * (1 - self.bound_fraction) - k_off * self.bound_fraction
        self.bound_fraction += d_bound * dt
        self.bound_fraction = np.clip(self.bound_fraction, 0, 1)
        
        return self.bound_fraction * self.sensitivity


@dataclass  
class ReceptorField:
    """Collection of receptors on a neuron"""
    receptors: Dict[ReceptorType, NeuromodulatorReceptor] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.receptors:
            for rt in ReceptorType:
                self.receptors[rt] = NeuromodulatorReceptor(receptor_type=rt)


class KineticNeuromodulator:
    """Kinetic neuromodulator with receptor binding"""
    
    def __init__(self, modulator_type: NeuromodulatorType):
        self.modulator_type = modulator_type
        self.concentration = 0.5
        self.release_rate = 1.0
        self.reuptake_rate = 0.1
        self.degradation_rate = 0.01
    
    def release(self, amount: float):
        """Release neuromodulator"""
        self.concentration += amount * self.release_rate
    
    def update(self, dt: float):
        """Update concentration with reuptake and degradation"""
        self.concentration -= self.concentration * self.reuptake_rate * dt
        self.concentration -= self.concentration * self.degradation_rate * dt
        self.concentration = max(0, self.concentration)


class ThreeFactorLearning:
    """
    Three-factor Hebbian learning with neuromodulation.
    
    Weight change = f(pre, post, neuromodulator)
    """
    
    def __init__(
        self,
        num_synapses: int = 10000,
        learning_rate: float = 0.01,
        eligibility_decay: float = 0.99
    ):
        self.num_synapses = num_synapses
        self.learning_rate = learning_rate
        self.eligibility_decay = eligibility_decay
        
        # Eligibility traces
        self.eligibility = np.zeros(num_synapses)
        
        # Last pre/post activity
        self.last_pre = np.zeros(num_synapses)
        self.last_post = np.zeros(num_synapses)
    
    def update_eligibility(self, decay: Optional[float] = None):
        """Decay eligibility traces"""
        d = decay if decay is not None else self.eligibility_decay
        self.eligibility *= d
    
    def record_activity(self, pre: np.ndarray, post: np.ndarray):
        """Record pre/post activity for STDP"""
        # Compute STDP-like eligibility update
        # Positive if pre before post, negative otherwise
        stdp_update = post * self.last_pre - pre * self.last_post
        
        # Bound to synapse count
        n = min(len(stdp_update), self.num_synapses)
        self.eligibility[:n] += stdp_update[:n]
        
        self.last_pre = pre.copy()
        self.last_post = post.copy()
    
    def apply_learning(
        self,
        reward: float,
        dopamine: float,
        learning_rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply three-factor learning.
        
        Returns weight changes.
        """
        lr = learning_rate if learning_rate is not None else self.learning_rate
        
        # Modulate by dopamine (reward prediction error proxy)
        modulation = dopamine * reward
        
        # Compute weight changes
        dw = lr * modulation * self.eligibility
        
        return dw
