"""
Advanced Kinetic Neuromodulation System (Enhanced)

This implements true biochemistry-based neuromodulation with realistic dynamics:
- Receptor binding kinetics (Michaelis-Menten with stochastic noise)
- Cross-modulator antagonism matrix (DA-5HT, NE-ACh interactions)
- Three-factor learning rules (pre, post, neuromodulator)
- Time-asymmetric plasticity (retrospective credit assignment)
- Epigenetic-style learning rule modifiers (permanent switches)
- Tonic vs phasic neuromodulator release
- Receptor desensitization and internalization
- Second messenger cascades
- Metaplasticity (plasticity of plasticity)
- Cross-system error bargaining (modules negotiate prediction error)
- Homeostatic stability mechanisms (BCM, weight bounds)

Key insight: Neuromodulation isn't just scaling - it fundamentally
changes the learning rules and computational properties of circuits.

Enhanced 2024: Added biological realism through stochastic kinetics,
cross-modulator interactions, and emergent stability mechanisms.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
import math
from collections import deque


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


# =============================================================================
# STOCHASTIC PARAMETERS & CROSS-MODULATOR INTERACTIONS (NEW)
# =============================================================================

@dataclass
class StochasticKineticParams:
    """
    Configurable stochastic parameters for realistic kinetic noise.

    Real receptors don't have fixed kon/koff - they fluctuate due to:
    - Thermal noise (Brownian motion of ligands)
    - Channel noise (stochastic gating)
    - Vesicular release variability
    - Local concentration gradients
    """
    # Noise amplitudes (as fraction of base parameter)
    kon_noise: float = 0.15      # ±15% variation in binding rate
    koff_noise: float = 0.10     # ±10% variation in unbinding rate
    release_noise: float = 0.20  # ±20% variation in vesicle release

    # Temporal correlation (noise isn't white - it has memory)
    noise_tau: float = 50.0      # ms, autocorrelation time for noise

    # Channel noise (discrete stochastic gating)
    channel_noise_amplitude: float = 0.05  # Random fluctuations in bound fraction

    # Enable/disable stochasticity
    enabled: bool = True

    def sample_noise(self, base_value: float, noise_frac: float) -> float:
        """Sample noisy parameter value."""
        if not self.enabled:
            return base_value
        noise = np.random.normal(0, noise_frac * base_value)
        return max(0.001, base_value + noise)  # Never negative


@dataclass
class CrossModulatorInteraction:
    """
    Defines how one neuromodulator affects another.

    Real neuromodulators don't work in isolation:
    - Dopamine and serotonin have mutual antagonism
    - Norepinephrine enhances acetylcholine effects
    - Cortisol suppresses multiple modulators
    - GABA/Glutamate balance affects everything
    """
    source: 'ModulatorType'
    target: 'ModulatorType'

    # Interaction type: 'antagonism', 'synergy', 'gating', 'modulation'
    interaction_type: str = "antagonism"

    # Strength of interaction (-1 to 1, negative = inhibition)
    strength: float = -0.3

    # Threshold: interaction only occurs above this source level
    threshold: float = 0.4

    # Time constant for the interaction (ms)
    tau: float = 100.0

    # Nonlinear gain (Hill coefficient for sigmoid interactions)
    hill_coefficient: float = 2.0

    def compute_effect(self, source_level: float, dt: float) -> float:
        """
        Compute the effect of source modulator on target.
        Returns a multiplicative factor or additive shift.
        """
        if source_level < self.threshold:
            return 0.0

        # Sigmoid activation based on source level
        x = (source_level - self.threshold) / (1 - self.threshold + 1e-6)
        activation = x ** self.hill_coefficient / (0.5 ** self.hill_coefficient + x ** self.hill_coefficient)

        # Time-filtered effect
        effect = self.strength * activation * (1 - np.exp(-dt / self.tau))

        return effect


class CrossModulatorMatrix:
    """
    Matrix of all cross-modulator interactions.

    This is what makes the neuromodulation system behave like real biochemistry
    instead of independent dials. Key interactions from literature:

    1. DA-5HT Antagonism: High serotonin suppresses dopamine (prevents mania)
    2. DA-5HT Reciprocal: High dopamine suppresses serotonin (reward vs. stability)
    3. NE-ACh Synergy: Arousal enhances attention
    4. Cortisol Suppression: Stress hormone depletes most others
    5. GABA-Glu Balance: Inhibition-Excitation homeostasis
    6. Oxytocin-Cortisol: Social bonding reduces stress
    """

    def __init__(self):
        self.interactions: List[CrossModulatorInteraction] = []
        self._setup_default_interactions()

    def _setup_default_interactions(self):
        """Setup biologically-grounded interaction matrix."""

        # === DA-5HT Mutual Antagonism ===
        # The classic dopamine-serotonin balance
        self.interactions.append(CrossModulatorInteraction(
            source=ModulatorType.SEROTONIN,
            target=ModulatorType.DOPAMINE,
            interaction_type="antagonism",
            strength=-0.35,  # High 5HT suppresses DA
            threshold=0.6,
            tau=200.0,
            hill_coefficient=2.0
        ))
        self.interactions.append(CrossModulatorInteraction(
            source=ModulatorType.DOPAMINE,
            target=ModulatorType.SEROTONIN,
            interaction_type="antagonism",
            strength=-0.25,  # High DA suppresses 5HT (weaker)
            threshold=0.7,
            tau=300.0,
            hill_coefficient=1.5
        ))

        # === NE-ACh Synergy ===
        # Arousal enhances attention
        self.interactions.append(CrossModulatorInteraction(
            source=ModulatorType.NOREPINEPHRINE,
            target=ModulatorType.ACETYLCHOLINE,
            interaction_type="synergy",
            strength=0.2,  # NE enhances ACh
            threshold=0.5,
            tau=100.0,
            hill_coefficient=1.0
        ))

        # === Cortisol Global Suppression ===
        # Stress depletes monoamines
        for target in [ModulatorType.SEROTONIN, ModulatorType.DOPAMINE,
                       ModulatorType.ACETYLCHOLINE]:
            self.interactions.append(CrossModulatorInteraction(
                source=ModulatorType.CORTISOL,
                target=target,
                interaction_type="antagonism",
                strength=-0.4,
                threshold=0.5,
                tau=500.0,  # Slow stress effects
                hill_coefficient=1.5
            ))

        # === Oxytocin-Cortisol Antagonism ===
        # Social bonding reduces stress
        self.interactions.append(CrossModulatorInteraction(
            source=ModulatorType.OXYTOCIN,
            target=ModulatorType.CORTISOL,
            interaction_type="antagonism",
            strength=-0.3,
            threshold=0.4,
            tau=150.0,
            hill_coefficient=1.0
        ))

        # === Adrenaline-NE Coupling ===
        # Acute stress activates both
        self.interactions.append(CrossModulatorInteraction(
            source=ModulatorType.ADRENALINE,
            target=ModulatorType.NOREPINEPHRINE,
            interaction_type="synergy",
            strength=0.4,
            threshold=0.5,
            tau=50.0,  # Fast coupling
            hill_coefficient=2.0
        ))

        # === Endorphin-Pain/Reward ===
        # Endorphins enhance dopamine
        self.interactions.append(CrossModulatorInteraction(
            source=ModulatorType.ENDORPHIN,
            target=ModulatorType.DOPAMINE,
            interaction_type="synergy",
            strength=0.25,
            threshold=0.4,
            tau=200.0,
            hill_coefficient=1.5
        ))

    def compute_all_effects(
        self,
        current_levels: Dict[ModulatorType, float],
        dt: float
    ) -> Dict[ModulatorType, float]:
        """
        Compute all interaction effects on all modulators.
        Returns dict of additive adjustments to apply.
        """
        effects: Dict[ModulatorType, float] = {mod: 0.0 for mod in ModulatorType}

        for interaction in self.interactions:
            source_level = current_levels.get(interaction.source, 0.5)
            effect = interaction.compute_effect(source_level, dt)
            effects[interaction.target] += effect

        return effects

    def add_interaction(self, interaction: CrossModulatorInteraction):
        """Add a custom interaction to the matrix."""
        self.interactions.append(interaction)


# =============================================================================
# EPIGENETIC LEARNING RULE MODIFIERS (NEW)
# =============================================================================

@dataclass
class EpigeneticSwitch:
    """
    Permanent or semi-permanent modifications to learning rules.

    Unlike metaplasticity (which is temporary), epigenetic switches
    represent long-term adaptations:
    - Chronic stress → permanently reduced LTP amplitude
    - Repeated failure → reduced plasticity in specific pathways
    - High success → enhanced learning rate consolidation

    These are the "scars" and "gifts" the brain accumulates.
    """
    name: str

    # What this switch affects
    target_parameter: str  # e.g., "ltp_amplitude", "learning_rate", "stdp_window"

    # Current modification (multiplicative)
    modifier: float = 1.0

    # Activation threshold and current activation
    activation_threshold: float = 0.7
    current_activation: float = 0.0

    # Switch dynamics
    activation_rate: float = 0.001   # How fast it activates
    deactivation_rate: float = 0.0001  # How fast it reverts (very slow!)

    # Is this switch currently "on"?
    is_active: bool = False

    # History for triggering
    trigger_history: List[float] = field(default_factory=list)
    trigger_window: int = 100  # Steps to consider

    def update(self, trigger_signal: float, dt: float) -> float:
        """
        Update switch state based on trigger signal.
        Returns current modifier value.
        """
        # Track history
        self.trigger_history.append(trigger_signal)
        if len(self.trigger_history) > self.trigger_window:
            self.trigger_history.pop(0)

        # Compute sustained activation from history
        if len(self.trigger_history) >= 10:
            sustained = np.mean(self.trigger_history[-10:])
        else:
            sustained = trigger_signal

        # Update activation
        if sustained > self.activation_threshold:
            self.current_activation += self.activation_rate * dt
        else:
            self.current_activation -= self.deactivation_rate * dt

        self.current_activation = np.clip(self.current_activation, 0, 1)

        # Check if switch should flip
        if self.current_activation > 0.8 and not self.is_active:
            self.is_active = True
        elif self.current_activation < 0.1 and self.is_active:
            self.is_active = False  # Can deactivate, but very slowly

        return self.modifier if self.is_active else 1.0


class EpigeneticLearningModifiers:
    """
    Collection of epigenetic switches that permanently modify learning.

    These create "personality" and "learned dispositions" that emerge
    from experience and persist long-term.
    """

    def __init__(self):
        self.switches: Dict[str, EpigeneticSwitch] = {}
        self._setup_default_switches()

    def _setup_default_switches(self):
        """Setup biologically-inspired epigenetic switches."""

        # Chronic stress switch - reduces all plasticity
        self.switches["chronic_stress"] = EpigeneticSwitch(
            name="chronic_stress",
            target_parameter="global_plasticity",
            modifier=0.6,  # 40% reduction when active
            activation_threshold=0.7,  # High cortisol threshold
            activation_rate=0.0005,
            deactivation_rate=0.00005  # Very slow recovery
        )

        # Learned helplessness - reduces exploration
        self.switches["learned_helplessness"] = EpigeneticSwitch(
            name="learned_helplessness",
            target_parameter="exploration_rate",
            modifier=0.4,  # Severe reduction
            activation_threshold=0.8,  # Repeated failure
            activation_rate=0.001,
            deactivation_rate=0.0001
        )

        # Success consolidation - enhances learning in successful domains
        self.switches["success_consolidation"] = EpigeneticSwitch(
            name="success_consolidation",
            target_parameter="ltp_amplitude",
            modifier=1.5,  # 50% boost
            activation_threshold=0.6,  # Moderate success
            activation_rate=0.0008,
            deactivation_rate=0.0002
        )

        # Novelty seeking - enhanced by dopamine history
        self.switches["novelty_seeking"] = EpigeneticSwitch(
            name="novelty_seeking",
            target_parameter="neurogenesis_rate",
            modifier=1.3,
            activation_threshold=0.65,
            activation_rate=0.0006,
            deactivation_rate=0.0003
        )

        # Emotional dampening - from chronic high arousal
        self.switches["emotional_dampening"] = EpigeneticSwitch(
            name="emotional_dampening",
            target_parameter="emotional_valence_strength",
            modifier=0.7,
            activation_threshold=0.75,
            activation_rate=0.0004,
            deactivation_rate=0.0001
        )

    def update_all(
        self,
        signals: Dict[str, float],
        dt: float
    ) -> Dict[str, float]:
        """
        Update all switches and return current modifiers.

        Args:
            signals: Dict mapping switch names to their trigger signals
            dt: Time step

        Returns:
            Dict of target_parameter -> combined modifier
        """
        modifiers: Dict[str, float] = {}

        for name, switch in self.switches.items():
            trigger = signals.get(name, 0.0)
            mod = switch.update(trigger, dt)

            # Combine modifiers for same target (multiplicative)
            target = switch.target_parameter
            if target in modifiers:
                modifiers[target] *= mod
            else:
                modifiers[target] = mod

        return modifiers

    def get_active_switches(self) -> List[str]:
        """Return names of currently active switches."""
        return [name for name, switch in self.switches.items() if switch.is_active]

    def add_switch(self, switch: EpigeneticSwitch):
        """Add a custom epigenetic switch."""
        self.switches[switch.name] = switch


# =============================================================================
# TIME-ASYMMETRIC PLASTICITY (NEW)
# =============================================================================

class TemporalCreditAssignment:
    """
    Time-asymmetric plasticity with retrospective credit assignment.

    The brain doesn't just learn from immediate consequences - it can
    retroactively assign credit to past states when outcomes become clear.

    This is more sophisticated than simple eligibility traces:
    - Records full activation history for a time window
    - After outcomes, reweights past states using temporal attribution kernel
    - Implements something like "episodic credit assignment"

    Biologically inspired by:
    - Hippocampal replay
    - Dopaminergic teaching signals
    - Working memory maintenance
    """

    def __init__(
        self,
        history_window: float = 10.0,  # seconds
        temporal_resolution: float = 0.01,  # seconds (10ms bins)
        decay_rate: float = 0.3  # How fast credit decays into past
    ):
        self.history_window = history_window
        self.temporal_resolution = temporal_resolution
        self.decay_rate = decay_rate

        # Calculate number of history bins
        self.num_bins = int(history_window / temporal_resolution)

        # Circular buffer for activation history
        self.activation_history: deque = deque(maxlen=self.num_bins)
        self.time_stamps: deque = deque(maxlen=self.num_bins)

        # Eligibility traces (accumulated credit)
        self.eligibility_buffer: Optional[np.ndarray] = None

        # Attribution kernel (how credit is distributed in time)
        self._build_attribution_kernel()

    def _build_attribution_kernel(self):
        """
        Build the temporal attribution kernel.

        This kernel determines how much credit past states receive
        when an outcome (reward/error) occurs NOW.

        Shape: Recent past gets more credit, with nonlinear decay
        """
        t = np.arange(self.num_bins) * self.temporal_resolution

        # Exponential decay with boost for very recent
        base_kernel = np.exp(-self.decay_rate * t)

        # Add bump for recent past (last 1-2 seconds are especially important)
        recent_bump = np.exp(-((t - 0.5) ** 2) / 0.5)

        # Combine and normalize
        self.attribution_kernel = base_kernel + 0.3 * recent_bump
        self.attribution_kernel /= np.sum(self.attribution_kernel)

        # Reverse so index 0 = most recent
        self.attribution_kernel = self.attribution_kernel[::-1]

    def record_state(
        self,
        activations: np.ndarray,
        current_time: float
    ):
        """Record current activation state for later credit assignment."""
        self.activation_history.append(activations.copy())
        self.time_stamps.append(current_time)

        # Initialize eligibility buffer if needed
        if self.eligibility_buffer is None or len(self.eligibility_buffer) != len(activations):
            self.eligibility_buffer = np.zeros(len(activations))

    def assign_credit(
        self,
        outcome: float,
        current_time: float
    ) -> np.ndarray:
        """
        Retroactively assign credit to past states based on outcome.

        Args:
            outcome: Reward/error signal (-1 to 1)
            current_time: Current simulation time

        Returns:
            Credit assignment for all recorded neurons
        """
        if len(self.activation_history) == 0:
            return np.zeros(1)

        # Stack history into array
        history_array = np.array(list(self.activation_history))
        n_steps, n_neurons = history_array.shape

        # Apply attribution kernel to history
        # kernel[0] = weight for most recent, kernel[-1] = weight for oldest
        kernel_slice = self.attribution_kernel[:n_steps]
        kernel_slice = kernel_slice / (np.sum(kernel_slice) + 1e-8)  # Renormalize

        # Compute weighted sum of past activations
        weighted_history = history_array * kernel_slice[:, np.newaxis]
        credit = outcome * np.sum(weighted_history, axis=0)

        # Update eligibility buffer
        self.eligibility_buffer = 0.9 * self.eligibility_buffer + 0.1 * credit

        return credit

    def get_eligibility(self) -> np.ndarray:
        """Get current eligibility traces."""
        if self.eligibility_buffer is None:
            return np.zeros(1)
        return self.eligibility_buffer

    def clear_history(self):
        """Clear activation history (e.g., at episode boundaries)."""
        self.activation_history.clear()
        self.time_stamps.clear()
        if self.eligibility_buffer is not None:
            self.eligibility_buffer *= 0.1  # Decay but don't fully reset


# =============================================================================
# CROSS-SYSTEM ERROR BARGAINING (NEW)
# =============================================================================

class ErrorBargainingSystem:
    """
    Cross-system error bargaining: modules argue about prediction error.

    Instead of a single loss function, different brain systems can
    have different "opinions" about what went wrong and negotiate
    which internal model should change.

    This creates a form of agency and prevents any single system
    from dominating learning.

    Systems:
    - Cortex: "The prediction was wrong" (pattern mismatch)
    - Reservoir: "The temporal model was wrong" (sequence error)
    - Neuromodulators: Arbitrate based on context (reward, stress, etc.)
    """

    def __init__(self):
        # Error claims from each system
        self.cortex_error: float = 0.0
        self.reservoir_error: float = 0.0
        self.prediction_error: float = 0.0

        # Confidence/certainty of each claim
        self.cortex_confidence: float = 0.5
        self.reservoir_confidence: float = 0.5

        # History for trend detection
        self.error_history: Dict[str, deque] = {
            'cortex': deque(maxlen=50),
            'reservoir': deque(maxlen=50),
            'total': deque(maxlen=50)
        }

        # Arbitration weights (modified by neuromodulators)
        self.cortex_weight: float = 0.5
        self.reservoir_weight: float = 0.5

    def report_error(
        self,
        system: str,
        error: float,
        confidence: float = 0.5
    ):
        """
        System reports its error estimate.

        Args:
            system: 'cortex' or 'reservoir'
            error: Error magnitude (0 to 1)
            confidence: How confident is this system? (0 to 1)
        """
        if system == 'cortex':
            self.cortex_error = error
            self.cortex_confidence = confidence
            self.error_history['cortex'].append(error)
        elif system == 'reservoir':
            self.reservoir_error = error
            self.reservoir_confidence = confidence
            self.error_history['reservoir'].append(error)

    def arbitrate(
        self,
        dopamine: float,
        norepinephrine: float,
        serotonin: float,
        acetylcholine: float
    ) -> Dict[str, float]:
        """
        Neuromodulators arbitrate between competing error signals.

        Returns dict with:
        - 'cortex_learns': How much cortex should update (0-1)
        - 'reservoir_learns': How much reservoir should update (0-1)
        - 'agreed_error': Consensus error estimate
        - 'conflict': How much systems disagree (0-1)
        """
        # Compute disagreement
        conflict = abs(self.cortex_error - self.reservoir_error)

        # Neuromodulator arbitration rules:
        # - High dopamine: favor cortex (reward prediction = pattern learning)
        # - High norepinephrine: favor reservoir (temporal/arousal learning)
        # - High serotonin: reduce all learning (stability)
        # - High acetylcholine: increase all learning (attention)

        cortex_bias = dopamine * 0.3 - norepinephrine * 0.1
        reservoir_bias = norepinephrine * 0.3 - dopamine * 0.1

        # Adjust weights
        self.cortex_weight = np.clip(0.5 + cortex_bias, 0.2, 0.8)
        self.reservoir_weight = np.clip(0.5 + reservoir_bias, 0.2, 0.8)

        # Normalize
        total_weight = self.cortex_weight + self.reservoir_weight
        self.cortex_weight /= total_weight
        self.reservoir_weight /= total_weight

        # Compute agreed error (weighted by confidence and weight)
        cortex_vote = self.cortex_error * self.cortex_confidence * self.cortex_weight
        reservoir_vote = self.reservoir_error * self.reservoir_confidence * self.reservoir_weight
        agreed_error = cortex_vote + reservoir_vote

        # Learning amounts (modulated by serotonin/ACh)
        stability_factor = 1.0 - serotonin * 0.5
        attention_factor = 0.5 + acetylcholine * 0.5

        base_learning = agreed_error * stability_factor * attention_factor

        # Distribute learning based on who was more "wrong"
        if self.cortex_error > self.reservoir_error:
            cortex_learns = base_learning * (0.6 + 0.4 * self.cortex_weight)
            reservoir_learns = base_learning * (0.4 + 0.4 * self.reservoir_weight)
        else:
            cortex_learns = base_learning * (0.4 + 0.4 * self.cortex_weight)
            reservoir_learns = base_learning * (0.6 + 0.4 * self.reservoir_weight)

        self.error_history['total'].append(agreed_error)
        self.prediction_error = agreed_error

        return {
            'cortex_learns': np.clip(cortex_learns, 0, 1),
            'reservoir_learns': np.clip(reservoir_learns, 0, 1),
            'agreed_error': agreed_error,
            'conflict': conflict,
            'cortex_weight': self.cortex_weight,
            'reservoir_weight': self.reservoir_weight
        }

    def get_error_trends(self) -> Dict[str, float]:
        """Get trend information about errors over time."""
        trends = {}
        for system, history in self.error_history.items():
            if len(history) >= 10:
                recent = list(history)[-10:]
                older = list(history)[-20:-10] if len(history) >= 20 else recent
                trends[f'{system}_trend'] = np.mean(recent) - np.mean(older)
                trends[f'{system}_variance'] = np.var(recent)
            else:
                trends[f'{system}_trend'] = 0.0
                trends[f'{system}_variance'] = 0.0
        return trends


# =============================================================================
# HOMEOSTATIC STABILITY MECHANISMS (NEW)
# =============================================================================

@dataclass
class HomeostaticController:
    """
    BCM-style homeostatic plasticity controller.

    Prevents runaway learning by:
    - Maintaining target activity levels
    - Adjusting plasticity thresholds based on history
    - Implementing synaptic scaling
    - Weight normalization
    """
    # Target activity level
    target_activity: float = 0.1

    # Sliding threshold (BCM rule)
    theta: float = 0.5
    theta_tau: float = 1000.0  # Time constant for threshold adjustment (ms)

    # Activity history for threshold computation
    activity_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # Weight bounds
    weight_min: float = -2.0
    weight_max: float = 2.0

    # Synaptic scaling parameters
    scaling_rate: float = 0.001
    scaling_target: float = 1.0

    def update_threshold(self, current_activity: float, dt: float):
        """
        Update BCM sliding threshold based on activity history.

        The threshold moves toward recent average activity,
        making LTP harder when activity is high and easier when low.
        """
        self.activity_history.append(current_activity)

        if len(self.activity_history) >= 10:
            avg_activity = np.mean(list(self.activity_history)[-20:])
            # Threshold drifts toward squared average (BCM)
            target_theta = avg_activity ** 2
            self.theta += (target_theta - self.theta) * (dt / self.theta_tau)
            self.theta = np.clip(self.theta, 0.01, 1.0)

    def compute_plasticity_gate(self, activity: float) -> float:
        """
        Compute plasticity gating factor based on BCM rule.

        Returns:
        - Positive for LTP (activity > theta)
        - Negative for LTD (activity < theta)
        - Zero at threshold
        """
        return activity * (activity - self.theta)

    def apply_weight_bounds(self, weights: np.ndarray) -> np.ndarray:
        """Enforce weight bounds to prevent explosion."""
        return np.clip(weights, self.weight_min, self.weight_max)

    def compute_scaling_factor(self, current_mean_weight: float) -> float:
        """
        Compute synaptic scaling factor to maintain total synaptic strength.
        """
        if current_mean_weight < 0.01:
            return 1.0
        return self.scaling_target / current_mean_weight

    def should_learn(self, activity: float, threshold: float = 0.1) -> bool:
        """Determine if learning should occur based on activity level."""
        return activity > threshold and activity < 0.9  # Not too low, not saturated


@dataclass
class ReceptorKinetics:
    """
    Michaelis-Menten receptor binding kinetics with STOCHASTIC noise.

    dB/dt = kon * L * (Bmax - B) - koff * B + noise

    where:
    - B = bound receptors
    - L = ligand (neuromodulator) concentration
    - Bmax = total receptors
    - kon = binding rate (with noise)
    - koff = unbinding rate (with noise)

    Enhanced: Now includes channel noise, parameter variability,
    and temporally correlated fluctuations for biological realism.
    """
    receptor_type: ReceptorSubtype

    # Kinetic parameters (base values - actual values include noise)
    bmax: float = 1.0           # Total receptor density
    bound: float = 0.0          # Currently bound fraction
    kon: float = 0.1            # Binding rate constant (base)
    koff: float = 0.05          # Unbinding rate constant (base)

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

    # Stochastic parameters (NEW)
    stochastic_params: Optional[StochasticKineticParams] = None

    # Noise state (for temporally correlated noise)
    _noise_state: float = field(default=0.0, repr=False)
    _last_kon: float = field(default=0.0, repr=False)
    _last_koff: float = field(default=0.0, repr=False)

    def __post_init__(self):
        """Initialize stochastic parameters if not provided."""
        if self.stochastic_params is None:
            self.stochastic_params = StochasticKineticParams()
        self._last_kon = self.kon
        self._last_koff = self.koff

    def _get_noisy_parameters(self, dt: float) -> Tuple[float, float]:
        """
        Get noisy kon and koff values with temporal correlation.

        Instead of white noise, parameters have memory - they drift
        smoothly rather than jumping randomly each timestep.
        """
        if not self.stochastic_params.enabled:
            return self.kon, self.koff

        # Ornstein-Uhlenbeck process for temporally correlated noise
        tau = self.stochastic_params.noise_tau
        decay = np.exp(-dt / tau)

        # Update noise state
        noise_increment = np.sqrt(1 - decay**2) * np.random.normal()
        self._noise_state = decay * self._noise_state + noise_increment

        # Apply noise to parameters
        kon_noise = self.stochastic_params.kon_noise * self._noise_state
        koff_noise = self.stochastic_params.koff_noise * self._noise_state * 0.7  # Slightly different

        noisy_kon = self.kon * (1 + kon_noise)
        noisy_koff = self.koff * (1 + koff_noise)

        # Smooth transition (low-pass filter)
        alpha = 0.3
        self._last_kon = alpha * noisy_kon + (1 - alpha) * self._last_kon
        self._last_koff = alpha * noisy_koff + (1 - alpha) * self._last_koff

        return max(0.001, self._last_kon), max(0.001, self._last_koff)

    def _apply_channel_noise(self, bound_fraction: float) -> float:
        """
        Apply discrete channel noise to bound fraction.

        Real ion channels open/close stochastically, creating
        fluctuations in the "effective" bound fraction.
        """
        if not self.stochastic_params.enabled:
            return bound_fraction

        noise = np.random.normal(0, self.stochastic_params.channel_noise_amplitude)
        return np.clip(bound_fraction + noise, 0.0, 1.0)

    def update(
        self,
        ligand_concentration: float,
        dt: float,
        stochastic: bool = True
    ) -> float:
        """
        Update receptor binding state with stochastic dynamics.

        Args:
            ligand_concentration: Current ligand level (0-1)
            dt: Time step in ms
            stochastic: Whether to apply stochastic noise

        Returns:
            Effective activation (accounting for desensitization and noise)
        """
        # Get (potentially noisy) kinetic parameters
        if stochastic and self.stochastic_params and self.stochastic_params.enabled:
            kon_eff, koff_eff = self._get_noisy_parameters(dt)
        else:
            kon_eff, koff_eff = self.kon, self.koff

        # Available receptors (not internalized or desensitized)
        available = self.bmax - self.internalized - self.desensitized
        available = max(0.0, available)

        # Michaelis-Menten binding dynamics with noisy parameters
        binding = kon_eff * ligand_concentration * (available - self.bound)
        unbinding = koff_eff * self.bound

        self.bound += (binding - unbinding) * dt
        self.bound = max(0.0, min(available, self.bound))

        # Apply channel noise if enabled
        if stochastic:
            effective_bound = self._apply_channel_noise(self.bound)
        else:
            effective_bound = self.bound

        # Desensitization (occurs when receptor is activated)
        # Threshold also has some variability
        desens_threshold = 0.3 + (np.random.normal() * 0.05 if stochastic else 0)
        if self.bound > desens_threshold:
            deactivate = self.desensitization_rate * self.bound * dt
            self.desensitized += deactivate
            self.bound -= deactivate

        # Resensitization
        resens = self.resensitization_rate * self.desensitized * dt
        self.desensitized -= resens
        self.desensitized = max(0.0, self.desensitized)

        # Internalization (prolonged high activation)
        intern_threshold = 0.5 + (np.random.normal() * 0.05 if stochastic else 0)
        if self.bound > intern_threshold:
            intern = self.internalization_rate * self.bound * dt
            self.internalized += intern

        # Recycling
        recycle = self.recycling_rate * self.internalized * dt
        self.internalized -= recycle
        self.internalized = max(0.0, min(0.5, self.internalized))

        # Effective activation (using noisy bound for output)
        activation = effective_bound * self.efficacy
        return activation if self.is_excitatory else -activation

    @property
    def sensitivity(self) -> float:
        """Current receptor sensitivity (affected by desensitization/internalization)"""
        available = self.bmax - self.internalized - self.desensitized
        return max(0.1, available / self.bmax)

    def get_state(self) -> Dict[str, float]:
        """Get full receptor state for debugging/analysis."""
        return {
            'bound': self.bound,
            'desensitized': self.desensitized,
            'internalized': self.internalized,
            'sensitivity': self.sensitivity,
            'noise_state': self._noise_state,
            'effective_kon': self._last_kon,
            'effective_koff': self._last_koff
        }


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
    Complete kinetic neuromodulation system (Enhanced)

    Integrates:
    - Neuromodulator release dynamics (with stochastic noise)
    - Receptor binding kinetics (Michaelis-Menten + channel noise)
    - Cross-modulator antagonism matrix (DA-5HT, NE-ACh, etc.)
    - Second messenger cascades
    - Three-factor learning rules
    - Time-asymmetric plasticity (retrospective credit assignment)
    - Epigenetic learning modifiers (permanent switches)
    - Cross-system error bargaining
    - Homeostatic stability mechanisms (BCM, weight bounds)
    - Metaplasticity
    - Chemical interactions (from original chemicals.py)

    This is SYSTEM 3 of the Three-System Brain:
    Motivation + Plasticity + Value Assignment

    It governs WHEN the cortex and reservoir change, not computation itself.
    """

    def __init__(
        self,
        enable_stochasticity: bool = True,
        enable_cross_modulator: bool = True,
        enable_epigenetics: bool = True,
        enable_temporal_credit: bool = True,
        enable_error_bargaining: bool = True
    ):
        """
        Initialize the enhanced neuromodulation system.

        Args:
            enable_stochasticity: Enable stochastic noise in kinetics
            enable_cross_modulator: Enable cross-modulator interactions
            enable_epigenetics: Enable epigenetic learning switches
            enable_temporal_credit: Enable time-asymmetric plasticity
            enable_error_bargaining: Enable cross-system error bargaining
        """
        # Feature flags
        self.enable_stochasticity = enable_stochasticity
        self.enable_cross_modulator = enable_cross_modulator
        self.enable_epigenetics = enable_epigenetics
        self.enable_temporal_credit = enable_temporal_credit
        self.enable_error_bargaining = enable_error_bargaining

        # Stochastic parameters (shared across receptors)
        self.stochastic_params = StochasticKineticParams(enabled=enable_stochasticity)

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

        # =================================================================
        # NEW ENHANCED COMPONENTS (System 3 enhancements)
        # =================================================================

        # Cross-modulator interaction matrix (DA-5HT antagonism, etc.)
        if enable_cross_modulator:
            self.cross_modulator_matrix = CrossModulatorMatrix()
        else:
            self.cross_modulator_matrix = None

        # Epigenetic learning modifiers (permanent switches)
        if enable_epigenetics:
            self.epigenetic_modifiers = EpigeneticLearningModifiers()
        else:
            self.epigenetic_modifiers = None

        # Time-asymmetric plasticity with retrospective credit
        if enable_temporal_credit:
            self.temporal_credit = TemporalCreditAssignment(
                history_window=10.0,  # 10 second history
                temporal_resolution=0.01,  # 10ms bins
                decay_rate=0.3
            )
        else:
            self.temporal_credit = None

        # Cross-system error bargaining
        if enable_error_bargaining:
            self.error_bargaining = ErrorBargainingSystem()
        else:
            self.error_bargaining = None

        # Homeostatic stability controller
        self.homeostatic = HomeostaticController()

        # Pass stochastic params to all receptors
        for receptor in self.receptors.values():
            receptor.stochastic_params = self.stochastic_params

        # Track for epigenetic triggers
        self._failure_count: int = 0
        self._success_count: int = 0
        self._high_stress_duration: float = 0.0
        self._high_arousal_duration: float = 0.0
    
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
    
    def update(self, dt: float, activations: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Update entire neuromodulation system with all enhanced components.

        Args:
            dt: Time step in seconds
            activations: Optional current neural activations for credit assignment

        Returns:
            Dict of current levels, states, and learning modifiers
        """
        state = {}

        # Update each neuromodulator release system
        for mod_type, release_sys in self.release_systems.items():
            level = release_sys.update(dt)
            self._current_levels[mod_type] = level
            state[f'{mod_type.value}_level'] = level
            state[f'{mod_type.value}_vesicles'] = release_sys.vesicle_pool

        # Update receptors based on neuromodulator levels (with stochastic noise)
        total_excitation = 0.0
        total_inhibition = 0.0

        for mod_type, receptor_types in self.modulator_receptors.items():
            ligand_conc = self._current_levels.get(mod_type, 0.5)
            for rec_type in receptor_types:
                receptor = self.receptors[rec_type]
                # Use stochastic update if enabled
                activation = receptor.update(
                    ligand_conc, dt,
                    stochastic=self.enable_stochasticity
                )
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

        # Apply basic chemical interactions (legacy)
        self._apply_chemical_interactions(dt)

        # =================================================================
        # ENHANCED: Apply cross-modulator interaction matrix
        # =================================================================
        if self.cross_modulator_matrix is not None:
            # Get all current levels including simple chemicals
            all_levels = {**self._current_levels}
            for chem, level in self.simple_chemicals.items():
                all_levels[chem] = level

            # Compute interaction effects
            effects = self.cross_modulator_matrix.compute_all_effects(all_levels, dt)

            # Apply effects to release systems
            for mod_type, effect in effects.items():
                if mod_type in self.release_systems:
                    self.release_systems[mod_type].tonic_level += effect
                    self.release_systems[mod_type].tonic_level = np.clip(
                        self.release_systems[mod_type].tonic_level, 0.0, 1.0
                    )
                elif mod_type in self.simple_chemicals:
                    self.simple_chemicals[mod_type] += effect
                    self.simple_chemicals[mod_type] = np.clip(
                        self.simple_chemicals[mod_type], 0.0, 1.0
                    )

            state['cross_modulator_active'] = True
        else:
            state['cross_modulator_active'] = False

        # =================================================================
        # ENHANCED: Update epigenetic switches
        # =================================================================
        if self.epigenetic_modifiers is not None:
            cortisol = self.simple_chemicals.get(ModulatorType.CORTISOL, 0.3)
            dopamine = self._current_levels.get(ModulatorType.DOPAMINE, 0.5)
            norepinephrine = self._current_levels.get(ModulatorType.NOREPINEPHRINE, 0.4)

            # Track chronic stress
            if cortisol > 0.6:
                self._high_stress_duration += dt
            else:
                self._high_stress_duration = max(0, self._high_stress_duration - dt * 0.1)

            # Track chronic arousal
            if norepinephrine > 0.7:
                self._high_arousal_duration += dt
            else:
                self._high_arousal_duration = max(0, self._high_arousal_duration - dt * 0.1)

            # Build epigenetic trigger signals
            epigenetic_signals = {
                'chronic_stress': cortisol,
                'learned_helplessness': 1.0 - dopamine if self._failure_count > 10 else 0.0,
                'success_consolidation': dopamine if self._success_count > 5 else 0.0,
                'novelty_seeking': dopamine,
                'emotional_dampening': norepinephrine if self._high_arousal_duration > 100 else 0.0,
            }

            # Update and get modifiers
            epigenetic_modifiers = self.epigenetic_modifiers.update_all(epigenetic_signals, dt)
            state['epigenetic_modifiers'] = epigenetic_modifiers
            state['active_switches'] = self.epigenetic_modifiers.get_active_switches()
        else:
            state['epigenetic_modifiers'] = {}
            state['active_switches'] = []

        # =================================================================
        # ENHANCED: Record activations for temporal credit assignment
        # =================================================================
        if self.temporal_credit is not None and activations is not None:
            # Compute current time from some state
            current_time = state.get('simulation_time', 0.0)
            self.temporal_credit.record_state(activations, current_time)
            state['temporal_credit_active'] = True
        else:
            state['temporal_credit_active'] = False

        # =================================================================
        # ENHANCED: Update homeostatic controller
        # =================================================================
        avg_activity = (total_excitation - total_inhibition) / max(1, len(self.receptors))
        self.homeostatic.update_threshold(avg_activity, dt * 1000)  # Convert to ms
        state['bcm_theta'] = self.homeostatic.theta
        state['plasticity_gate'] = self.homeostatic.compute_plasticity_gate(avg_activity)

        # Add simple chemical levels to state
        for chem_type, level in self.simple_chemicals.items():
            state[f'{chem_type.value}_level'] = level
            self._current_levels[chem_type] = level

        # Compute total excitation/inhibition balance
        state['excitation'] = total_excitation
        state['inhibition'] = total_inhibition
        state['ei_balance'] = total_excitation / max(0.01, total_inhibition)

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
        current_time: float,
        apply_epigenetics: bool = True
    ) -> float:
        """
        Compute weight change using enhanced three-factor rule.

        Now incorporates:
        - STDP eligibility traces
        - Neuromodulator gating
        - BCM homeostatic threshold
        - Epigenetic modifiers (if enabled)
        - Temporal credit assignment (if available)

        Args:
            pre_spike_time: Time of pre-synaptic spike
            post_spike_time: Time of post-synaptic spike
            current_time: Current simulation time
            apply_epigenetics: Whether to apply epigenetic modifiers

        Returns:
            Weight change (dw)
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

        # Compute base weight change
        dw = self.stdp.compute_weight_change(
            ltp_elig, ltd_elig, modulators, camp_level, metaplasticity
        )

        # =================================================================
        # ENHANCED: Apply BCM homeostatic gating
        # =================================================================
        # Get plasticity gate from BCM controller
        activity_estimate = abs(dw) * 10  # Rough activity proxy
        bcm_gate = self.homeostatic.compute_plasticity_gate(activity_estimate)

        # Modulate dw by BCM (positive gate amplifies, negative suppresses)
        if bcm_gate > 0:
            dw *= (1 + bcm_gate * 0.5)  # Boost if above threshold
        else:
            dw *= max(0.1, 1 + bcm_gate)  # Suppress if below

        # =================================================================
        # ENHANCED: Apply epigenetic modifiers
        # =================================================================
        if apply_epigenetics and self.epigenetic_modifiers is not None:
            active_modifiers = {}
            for switch in self.epigenetic_modifiers.switches.values():
                if switch.is_active:
                    active_modifiers[switch.target_parameter] = switch.modifier

            # Apply modifiers
            if 'global_plasticity' in active_modifiers:
                dw *= active_modifiers['global_plasticity']
            if 'ltp_amplitude' in active_modifiers and dw > 0:
                dw *= active_modifiers['ltp_amplitude']

        # =================================================================
        # ENHANCED: Apply homeostatic weight bounds
        # =================================================================
        # Clip to prevent runaway (will be applied to actual weights elsewhere)
        dw = np.clip(dw, -0.1, 0.1)  # Per-update limit

        return dw

    def assign_retrospective_credit(
        self,
        outcome: float,
        current_time: float
    ) -> Optional[np.ndarray]:
        """
        Assign credit to past states based on outcome.

        This implements time-asymmetric plasticity - when we know an outcome,
        we retroactively assign credit to past activations that contributed.

        Args:
            outcome: Reward/error signal (-1 to 1)
            current_time: Current simulation time

        Returns:
            Credit assignment vector, or None if temporal credit disabled
        """
        if self.temporal_credit is None:
            return None

        # Assign credit using temporal attribution kernel
        credit = self.temporal_credit.assign_credit(outcome, current_time)

        # Track success/failure for epigenetic triggers
        if outcome > 0.3:
            self._success_count += 1
            self._failure_count = max(0, self._failure_count - 1)
        elif outcome < -0.3:
            self._failure_count += 1
            self._success_count = max(0, self._success_count - 1)

        return credit

    def report_system_error(
        self,
        system: str,
        error: float,
        confidence: float = 0.5
    ):
        """
        Report prediction error from a brain system (for error bargaining).

        Args:
            system: 'cortex' or 'reservoir'
            error: Error magnitude (0 to 1)
            confidence: How confident is this system? (0 to 1)
        """
        if self.error_bargaining is not None:
            self.error_bargaining.report_error(system, error, confidence)

    def get_learning_decisions(self) -> Dict[str, float]:
        """
        Get cross-system learning decisions from error bargaining.

        Neuromodulators arbitrate between cortex and reservoir errors
        to determine how much each system should learn.

        Returns:
            Dict with cortex_learns, reservoir_learns, agreed_error, conflict
        """
        if self.error_bargaining is None:
            return {
                'cortex_learns': 0.5,
                'reservoir_learns': 0.5,
                'agreed_error': 0.0,
                'conflict': 0.0
            }

        # Get current modulator levels for arbitration
        da = self._current_levels.get(ModulatorType.DOPAMINE, 0.5)
        ne = self._current_levels.get(ModulatorType.NOREPINEPHRINE, 0.5)
        ht = self._current_levels.get(ModulatorType.SEROTONIN, 0.5)
        ach = self._current_levels.get(ModulatorType.ACETYLCHOLINE, 0.5)

        return self.error_bargaining.arbitrate(da, ne, ht, ach)

    def get_learning_modulation(self) -> Dict[str, float]:
        """
        Get current learning modulation factors (Enhanced).

        Now includes:
        - Base neuromodulator effects
        - Epigenetic modifiers
        - BCM threshold
        - Error bargaining results
        - Temporal credit availability

        Returns:
            Dict with comprehensive learning modulation factors
        """
        da = self._current_levels.get(ModulatorType.DOPAMINE, 0.5)
        ne = self._current_levels.get(ModulatorType.NOREPINEPHRINE, 0.5)
        ach = self._current_levels.get(ModulatorType.ACETYLCHOLINE, 0.5)
        ht = self._current_levels.get(ModulatorType.SEROTONIN, 0.5)

        camp = self.second_messengers['cAMP'].level
        pka = self.second_messengers['PKA'].level

        # Base modulation factors
        result = {
            'ltp_modulation': da * 1.5 + camp * 0.5 + pka,
            'ltd_modulation': (1.0 - da) * 0.5 + 0.5,
            'attention': ach * 1.5 + ne * 0.5,
            'consolidation': ne * 0.8 + ht * 0.4,
            'stability': ht * 0.6 + 0.4,
            'exploration': ne * 0.5 + (1.0 - ht) * 0.3,
            'plasticity_threshold': self.second_messengers['PKA'].metaplasticity_factor,
        }

        # Add BCM homeostatic info
        result['bcm_theta'] = self.homeostatic.theta
        result['should_learn'] = self.homeostatic.should_learn(da)

        # Add epigenetic modifiers
        if self.epigenetic_modifiers is not None:
            active = self.epigenetic_modifiers.get_active_switches()
            result['active_epigenetic_switches'] = len(active)
            result['epigenetic_switches'] = active

            # Apply global plasticity modifier
            for switch in self.epigenetic_modifiers.switches.values():
                if switch.is_active and switch.target_parameter == 'global_plasticity':
                    result['ltp_modulation'] *= switch.modifier
                    result['ltd_modulation'] *= switch.modifier
        else:
            result['active_epigenetic_switches'] = 0
            result['epigenetic_switches'] = []

        # Add error bargaining results
        if self.error_bargaining is not None:
            bargaining = self.get_learning_decisions()
            result['cortex_learns'] = bargaining['cortex_learns']
            result['reservoir_learns'] = bargaining['reservoir_learns']
            result['system_conflict'] = bargaining['conflict']
        else:
            result['cortex_learns'] = 0.5
            result['reservoir_learns'] = 0.5
            result['system_conflict'] = 0.0

        # Add temporal credit info
        if self.temporal_credit is not None:
            eligibility = self.temporal_credit.get_eligibility()
            result['temporal_credit_mean'] = float(np.mean(np.abs(eligibility))) if len(eligibility) > 1 else 0.0
            result['temporal_credit_active'] = True
        else:
            result['temporal_credit_mean'] = 0.0
            result['temporal_credit_active'] = False

        return result

    def report_outcome(self, success: bool, magnitude: float = 0.5):
        """
        Report outcome for epigenetic and learning tracking.

        Args:
            success: Whether the outcome was successful
            magnitude: How significant (0-1)
        """
        if success:
            self._success_count += 1
            # Trigger dopamine burst for success
            self.trigger_phasic_release(ModulatorType.DOPAMINE, magnitude * 0.3)
        else:
            self._failure_count += 1
            # Trigger cortisol for failure (stress)
            if ModulatorType.CORTISOL in self.simple_chemicals:
                self.simple_chemicals[ModulatorType.CORTISOL] = min(
                    1.0, self.simple_chemicals[ModulatorType.CORTISOL] + magnitude * 0.1
                )

    def clear_episode(self):
        """
        Clear episode-specific state (call at episode boundaries).

        Resets temporal credit history but preserves epigenetic state.
        """
        if self.temporal_credit is not None:
            self.temporal_credit.clear_history()

        # Reset short-term counters but not epigenetic switches
        self._failure_count = max(0, self._failure_count - 5)
        self._success_count = max(0, self._success_count - 5)
    
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
