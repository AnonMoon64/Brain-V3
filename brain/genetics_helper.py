
import numpy as np
from typing import Dict, Any, Optional
from .creature import Phenotype, CreatureBody
from .instincts import InstinctSystem
from .three_system_brain import BrainConfig

def blend_values(a: float, b: float, mutation_rate: float = 0.1, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Blend two float values with mutation."""
    base = (a + b) / 2
    val = base + np.random.normal(0, mutation_rate)
    return float(np.clip(val, min_val, max_val))

def create_child_phenotype(p1: Phenotype, p2: Phenotype) -> Phenotype:
    """Generate a child phenotype from two parents."""
    return Phenotype(
        size=blend_values(p1.size, p2.size, 0.1, 0.5, 3.0),
        hue=blend_values(p1.hue, p2.hue, 0.05, 0.0, 1.0),
        saturation=blend_values(p1.saturation, p2.saturation, 0.1, 0.0, 1.0),
        pattern_type=np.random.choice([p1.pattern_type, p2.pattern_type]),
        pattern_density=blend_values(p1.pattern_density, p2.pattern_density, 0.1),
        max_speed=blend_values(p1.max_speed / 5, p2.max_speed / 5, 0.1) * 5,
        jump_power=blend_values(p1.jump_power / 12, p2.jump_power / 12, 0.1) * 12,
        
        # Metabolic Evolution
        metabolic_rate=blend_values(p1.metabolic_rate, p2.metabolic_rate, 0.1, 0.5, 3.0),
        digestive_efficiency=blend_values(p1.digestive_efficiency, p2.digestive_efficiency, 0.1, 0.5, 2.0),
        heat_generation=blend_values(p1.heat_generation, p2.heat_generation, 0.1, 0.2, 5.0)
    )

def create_child_brain_config(c1: BrainConfig, c2: BrainConfig) -> BrainConfig:
    """
    Generate a child brain config from two parents.
    
    TIER 2: Evolving Brain Paradigms - Brain architecture parameters 
    are inherited with mutation, allowing different brain "paradigms"
    to emerge:
    
    - High spectral_radius = more chaotic/creative reservoir dynamics
    - Low spectral_radius = more stable/predictable 
    - High leak_rate = fast adaptation, short memory
    - Low leak_rate = slow adaptation, long memory
    - High lateral_inhibition = sharp, focused representations
    - Low lateral_inhibition = distributed, blurry representations
    - High neurogenesis_rate = more structural plasticity
    - High predictive_boost = more prediction-driven behavior
    
    Brain SIZE is randomly inherited from one parent (not blended) for
    clearer size inheritance patterns.
    """
    # Randomly pick one parent's brain size (not blended)
    size_parent = c1 if np.random.random() < 0.5 else c2
    
    return BrainConfig(
        # =====================================================================
        # Structural Parameters (inherited from one parent randomly)
        # =====================================================================
        input_dim=size_parent.input_dim,  # CRITICAL: Must be 128 to match SensoryEncoder
        num_columns=size_parent.num_columns,
        cells_per_column=size_parent.cells_per_column,
        reservoir_size=size_parent.reservoir_size,
        output_dim=size_parent.output_dim,
        
        # =====================================================================
        # Reservoir Dynamics (memory/chaos balance)
        # =====================================================================
        # High = chaotic edge-of-chaos dynamics (creative but unstable)
        # Low = stable attractor dynamics (predictable but limited)
        spectral_radius=blend_values(c1.spectral_radius, c2.spectral_radius, 0.05, 0.5, 1.2),
        
        # High = fast forgetting, quick adaptation
        # Low = long memory, slow learning
        leak_rate=blend_values(c1.leak_rate, c2.leak_rate, 0.05, 0.05, 0.95),
        
        # Connection density in reservoir
        reservoir_sparsity=blend_values(c1.reservoir_sparsity, c2.reservoir_sparsity, 0.02, 0.01, 0.5),
        
        # =====================================================================
        # Cortical Paradigm (representation style)
        # =====================================================================
        # High = sharp winner-take-all (categorical thinking)
        # Low = distributed representations (holistic thinking)
        lateral_inhibition_strength=blend_values(
            c1.lateral_inhibition_strength, c2.lateral_inhibition_strength, 
            0.05, 0.1, 0.8
        ),
        
        # How much prediction influences activation
        # High = strongly prediction-driven (anticipatory)
        # Low = reactive/stimulus-driven
        predictive_boost=blend_values(
            c1.predictive_boost, c2.predictive_boost,
            0.03, 0.05, 0.5
        ),
        
        # Target sparsity (how many neurons active at once)
        target_sparsity=blend_values(
            c1.target_sparsity, c2.target_sparsity,
            0.005, 0.01, 0.1
        ),
        
        # =====================================================================
        # Structural Plasticity (neurogenesis)
        # =====================================================================
        neurogenesis_rate=blend_values(c1.neurogenesis_rate, c2.neurogenesis_rate, 0.05, 0.0, 1.0),
        pruning_threshold=blend_values(c1.pruning_threshold, c2.pruning_threshold, 0.005, 0.001, 0.1),
        
        # =====================================================================
        # Neuromodulator Sensitivity (emotional regulation)
        # =====================================================================
        # How sensitive to reward signals
        dopamine_baseline=blend_values(c1.dopamine_baseline, c2.dopamine_baseline, 0.05, 0.2, 0.8),
        
        # How sensitive to social/calm signals
        serotonin_baseline=blend_values(c1.serotonin_baseline, c2.serotonin_baseline, 0.05, 0.2, 0.8),
        
        # How sensitive to arousal/attention
        norepinephrine_baseline=blend_values(c1.norepinephrine_baseline, c2.norepinephrine_baseline, 0.05, 0.1, 0.6),
        
        # How sensitive to stress
        cortisol_baseline=blend_values(c1.cortisol_baseline, c2.cortisol_baseline, 0.05, 0.1, 0.6),    )


def describe_brain_paradigm(config: BrainConfig) -> Dict[str, Any]:
    """
    Describe a brain's cognitive paradigm based on its configuration.
    
    TIER 2: Evolving Brain Paradigms - Different parameter combinations
    lead to different cognitive styles:
    
    Returns:
        Dict with paradigm description and traits
    """
    traits = []
    paradigm = "balanced"
    
    # Chaos vs Stability (spectral_radius)
    if config.spectral_radius > 1.0:
        traits.append("chaotic")
        paradigm = "creative"
    elif config.spectral_radius < 0.7:
        traits.append("stable")
        paradigm = "methodical"
    
    # Memory span (leak_rate)
    if config.leak_rate > 0.6:
        traits.append("reactive")  # Fast forgetting, quick to adapt
    elif config.leak_rate < 0.2:
        traits.append("contemplative")  # Long memory, slow adaptation
    
    # Representation style (lateral_inhibition)
    if config.lateral_inhibition_strength > 0.5:
        traits.append("focused")  # Sharp, categorical thinking
    elif config.lateral_inhibition_strength < 0.2:
        traits.append("holistic")  # Distributed, integrative thinking
    
    # Prediction vs Reaction (predictive_boost)
    if config.predictive_boost > 0.3:
        traits.append("anticipatory")
    elif config.predictive_boost < 0.1:
        traits.append("reactive")
    
    # Plasticity (neurogenesis_rate)
    if config.neurogenesis_rate > 0.7:
        traits.append("adaptive")
    elif config.neurogenesis_rate < 0.2:
        traits.append("crystallized")
    
    # Emotional style (neuromodulator baselines)
    if config.dopamine_baseline > 0.6:
        traits.append("reward-seeking")
    if config.cortisol_baseline > 0.4:
        traits.append("anxious")
    if config.serotonin_baseline > 0.6:
        traits.append("calm")
    
    # Determine dominant paradigm
    if "chaotic" in traits and "adaptive" in traits:
        paradigm = "explorer"
    elif "stable" in traits and "focused" in traits:
        paradigm = "specialist"
    elif "contemplative" in traits and "holistic" in traits:
        paradigm = "philosopher"
    elif "reactive" in traits and "reward-seeking" in traits:
        paradigm = "opportunist"
    elif "anticipatory" in traits and "calm" in traits:
        paradigm = "strategist"
    elif "anxious" in traits:
        paradigm = "vigilant"
    
    return {
        'paradigm': paradigm,
        'traits': traits,
        'spectral_radius': config.spectral_radius,
        'leak_rate': config.leak_rate,
        'lateral_inhibition': config.lateral_inhibition_strength,
        'predictive_boost': config.predictive_boost,
        'neurogenesis_rate': config.neurogenesis_rate,
    }