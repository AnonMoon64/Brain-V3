"""
DNA Engine - Digital Genetics for Synthetic Organisms

A complete genetic system for encoding brain structure, body morphology,
drives/instincts, and biochemistry. Supports heredity, mutation, crossover,
dominance, and developmental growth rules.

Inspired by the Creatures game series but built for the Three-System Brain.

LAYERS:
1. Genotype (DNA) - The genetic code stored as genes with alleles
2. Development - Rules that grow the brain/body from DNA
3. Phenotype - The expressed organism (brain + body + drives)

USAGE:
    from brain.dna import Genome, GenePool, breed, mutate
    
    # Create a random genome
    genome = Genome.random()
    
    # Build a brain from DNA
    brain = genome.develop_brain()
    
    # Breed two organisms
    child_genome = breed(parent1.genome, parent2.genome)
    
    # Mutate
    mutated = mutate(genome, rate=0.02)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from enum import Enum
import json
import hashlib
from copy import deepcopy


# =============================================================================
# GENE CATEGORIES
# =============================================================================

class GeneCategory(Enum):
    """Categories of genes in the genome."""
    BRAIN_STRUCTURE = "brain_structure"      # Cortex, reservoir, connectivity
    BRAIN_DYNAMICS = "brain_dynamics"        # Time constants, oscillations
    NEUROMODULATION = "neuromodulation"      # Receptor densities, sensitivities
    BODY_MORPHOLOGY = "body_morphology"      # Size, speed, appearance
    DRIVES = "drives"                        # Instincts, baseline motivations
    METABOLISM = "metabolism"                # Energy, fatigue, healing
    SENSORY = "sensory"                      # Vision, hearing, smell ranges
    LIFESPAN = "lifespan"                    # Aging, fertility, longevity
    LEARNING = "learning"                    # Plasticity rates, memory
    SOCIAL = "social"                        # Bonding, aggression, territoriality


class Dominance(Enum):
    """Allele dominance types."""
    DOMINANT = "dominant"           # Always expressed if present
    RECESSIVE = "recessive"         # Only expressed if homozygous
    CODOMINANT = "codominant"       # Both alleles blend
    INCOMPLETE = "incomplete"       # Partial expression


# =============================================================================
# GENE DEFINITION
# =============================================================================

@dataclass
class Gene:
    """
    A single gene with alleles, dominance, and mutation properties.
    
    Each gene has:
    - locus: Unique identifier/position in genome
    - name: Human-readable name
    - category: What aspect of the organism it affects
    - allele_a/b: Two alleles (diploid organism)
    - dominance: How alleles interact
    - mutation_rate: Probability of mutation per generation
    - min_val/max_val: Valid range for allele values
    - lethal_below/above: Lethal allele thresholds (organism dies)
    """
    locus: int
    name: str
    category: GeneCategory
    allele_a: float
    allele_b: float
    dominance: Dominance = Dominance.CODOMINANT
    mutation_rate: float = 0.01
    mutation_magnitude: float = 0.1
    min_val: float = 0.0
    max_val: float = 1.0
    lethal_below: Optional[float] = None
    lethal_above: Optional[float] = None
    description: str = ""
    
    def express(self) -> float:
        """
        Express the gene based on dominance rules.
        Returns the phenotypic value.
        """
        a, b = self.allele_a, self.allele_b
        
        if self.dominance == Dominance.DOMINANT:
            # Dominant allele wins (higher value considered dominant)
            return max(a, b)
        
        elif self.dominance == Dominance.RECESSIVE:
            # Recessive only shows if both alleles similar
            if abs(a - b) < 0.2:
                return (a + b) / 2
            else:
                return max(a, b)  # Dominant masks recessive
        
        elif self.dominance == Dominance.CODOMINANT:
            # Both alleles contribute equally
            return (a + b) / 2
        
        elif self.dominance == Dominance.INCOMPLETE:
            # Blend with slight bias toward higher
            return 0.6 * max(a, b) + 0.4 * min(a, b)
        
        return (a + b) / 2
    
    def express_scaled(self, target_min: float, target_max: float) -> float:
        """Express gene and scale to target range."""
        raw = self.express()
        return target_min + (target_max - target_min) * raw
    
    def is_lethal(self) -> bool:
        """Check if gene expression is in lethal range."""
        val = self.express()
        if self.lethal_below is not None and val < self.lethal_below:
            return True
        if self.lethal_above is not None and val > self.lethal_above:
            return True
        return False
    
    def mutate(self, rate_multiplier: float = 1.0) -> 'Gene':
        """Return a mutated copy of this gene."""
        new_gene = deepcopy(self)
        
        effective_rate = self.mutation_rate * rate_multiplier
        
        # Mutate allele A
        if np.random.random() < effective_rate:
            delta = np.random.normal(0, self.mutation_magnitude)
            new_gene.allele_a = np.clip(
                new_gene.allele_a + delta, 
                self.min_val, 
                self.max_val
            )
        
        # Mutate allele B
        if np.random.random() < effective_rate:
            delta = np.random.normal(0, self.mutation_magnitude)
            new_gene.allele_b = np.clip(
                new_gene.allele_b + delta,
                self.min_val,
                self.max_val
            )
        
        return new_gene
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'locus': self.locus,
            'name': self.name,
            'category': self.category.value,
            'allele_a': self.allele_a,
            'allele_b': self.allele_b,
            'dominance': self.dominance.value,
            'mutation_rate': self.mutation_rate,
            'mutation_magnitude': self.mutation_magnitude,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'lethal_below': self.lethal_below,
            'lethal_above': self.lethal_above,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Gene':
        """Deserialize from dictionary."""
        return cls(
            locus=d['locus'],
            name=d['name'],
            category=GeneCategory(d['category']),
            allele_a=d['allele_a'],
            allele_b=d['allele_b'],
            dominance=Dominance(d['dominance']),
            mutation_rate=d.get('mutation_rate', 0.01),
            mutation_magnitude=d.get('mutation_magnitude', 0.1),
            min_val=d.get('min_val', 0.0),
            max_val=d.get('max_val', 1.0),
            lethal_below=d.get('lethal_below'),
            lethal_above=d.get('lethal_above'),
            description=d.get('description', '')
        )


# =============================================================================
# GENE LIBRARY - All possible genes
# =============================================================================

class GeneLibrary:
    """
    Library of all gene definitions for the organism.
    This defines what genes exist and their default properties.
    """
    
    # =========================================================================
    # BRAIN STRUCTURE GENES (locus 1-99)
    # =========================================================================
    
    CORTICAL_COLUMNS = Gene(
        locus=1, name="cortical_columns", category=GeneCategory.BRAIN_STRUCTURE,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.008, mutation_magnitude=0.15,
        description="Number of cortical columns (50-500)"
    )
    
    CELLS_PER_COLUMN = Gene(
        locus=2, name="cells_per_column", category=GeneCategory.BRAIN_STRUCTURE,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.005, mutation_magnitude=0.1,
        description="Cells per minicolumn (8-64)"
    )
    
    CORTICAL_SPARSITY = Gene(
        locus=3, name="cortical_sparsity", category=GeneCategory.BRAIN_STRUCTURE,
        allele_a=0.3, allele_b=0.3, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.01, mutation_magnitude=0.08,
        min_val=0.01, max_val=0.2,
        lethal_below=0.005, lethal_above=0.3,
        description="Cortical activation sparsity (1-20%)"
    )
    
    RESERVOIR_SIZE = Gene(
        locus=4, name="reservoir_size", category=GeneCategory.BRAIN_STRUCTURE,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.008, mutation_magnitude=0.12,
        description="Reservoir neuron count (500-5000)"
    )
    
    RESERVOIR_SPARSITY = Gene(
        locus=5, name="reservoir_sparsity", category=GeneCategory.BRAIN_STRUCTURE,
        allele_a=0.3, allele_b=0.3, dominance=Dominance.CODOMINANT,
        mutation_rate=0.01, mutation_magnitude=0.1,
        description="Reservoir connection sparsity"
    )
    
    SPECTRAL_RADIUS = Gene(
        locus=6, name="spectral_radius", category=GeneCategory.BRAIN_STRUCTURE,
        allele_a=0.9, allele_b=0.9, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.005, mutation_magnitude=0.05,
        min_val=0.5, max_val=1.0,
        lethal_above=1.05,  # Chaos if > 1
        description="Reservoir spectral radius (edge of chaos)"
    )
    
    LATERAL_INHIBITION = Gene(
        locus=7, name="lateral_inhibition", category=GeneCategory.BRAIN_STRUCTURE,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.01, mutation_magnitude=0.1,
        description="Lateral inhibition strength"
    )
    
    FEEDBACK_STRENGTH = Gene(
        locus=8, name="feedback_strength", category=GeneCategory.BRAIN_STRUCTURE,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.01, mutation_magnitude=0.1,
        description="Top-down feedback connection strength"
    )
    
    CORTICAL_LAYERS = Gene(
        locus=9, name="cortical_layers", category=GeneCategory.BRAIN_STRUCTURE,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.DOMINANT,
        mutation_rate=0.005, mutation_magnitude=0.15,
        description="Number of cortical hierarchy layers (2-8)"
    )
    
    # =========================================================================
    # BRAIN DYNAMICS GENES (locus 100-199)
    # =========================================================================
    
    LEAK_RATE = Gene(
        locus=100, name="leak_rate", category=GeneCategory.BRAIN_DYNAMICS,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.01, mutation_magnitude=0.1,
        description="Reservoir leak rate (memory decay)"
    )
    
    GAMMA_POWER = Gene(
        locus=101, name="gamma_power", category=GeneCategory.BRAIN_DYNAMICS,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.01, mutation_magnitude=0.1,
        description="Gamma oscillation power (fast processing)"
    )
    
    THETA_POWER = Gene(
        locus=102, name="theta_power", category=GeneCategory.BRAIN_DYNAMICS,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.01, mutation_magnitude=0.1,
        description="Theta oscillation power (memory/navigation)"
    )
    
    REFRACTORY_PERIOD = Gene(
        locus=103, name="refractory_period", category=GeneCategory.BRAIN_DYNAMICS,
        allele_a=0.3, allele_b=0.3, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.01, mutation_magnitude=0.08,
        description="Neuron refractory period"
    )
    
    PREDICTION_WEIGHT = Gene(
        locus=104, name="prediction_weight", category=GeneCategory.BRAIN_DYNAMICS,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.01, mutation_magnitude=0.1,
        description="Weight of predictive coding"
    )
    
    # =========================================================================
    # NEUROMODULATION GENES (locus 200-299)
    # =========================================================================
    
    DOPAMINE_BASELINE = Gene(
        locus=200, name="dopamine_baseline", category=GeneCategory.NEUROMODULATION,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Baseline dopamine level"
    )
    
    DOPAMINE_SENSITIVITY = Gene(
        locus=201, name="dopamine_sensitivity", category=GeneCategory.NEUROMODULATION,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.012, mutation_magnitude=0.1,
        description="Dopamine receptor density"
    )
    
    SEROTONIN_BASELINE = Gene(
        locus=202, name="serotonin_baseline", category=GeneCategory.NEUROMODULATION,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Baseline serotonin level"
    )
    
    SEROTONIN_SENSITIVITY = Gene(
        locus=203, name="serotonin_sensitivity", category=GeneCategory.NEUROMODULATION,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.012, mutation_magnitude=0.1,
        description="Serotonin receptor density"
    )
    
    NOREPINEPHRINE_BASELINE = Gene(
        locus=204, name="norepinephrine_baseline", category=GeneCategory.NEUROMODULATION,
        allele_a=0.4, allele_b=0.4, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Baseline norepinephrine level"
    )
    
    ACETYLCHOLINE_BASELINE = Gene(
        locus=205, name="acetylcholine_baseline", category=GeneCategory.NEUROMODULATION,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Baseline acetylcholine level"
    )
    
    CORTISOL_SENSITIVITY = Gene(
        locus=206, name="cortisol_sensitivity", category=GeneCategory.NEUROMODULATION,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.01, mutation_magnitude=0.1,
        description="Stress hormone sensitivity"
    )
    
    OXYTOCIN_BASELINE = Gene(
        locus=207, name="oxytocin_baseline", category=GeneCategory.NEUROMODULATION,
        allele_a=0.4, allele_b=0.4, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Baseline oxytocin (bonding)"
    )
    
    GABA_GLUTAMATE_RATIO = Gene(
        locus=208, name="gaba_glutamate_ratio", category=GeneCategory.NEUROMODULATION,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.01, mutation_magnitude=0.08,
        lethal_below=0.1, lethal_above=0.9,  # Must be balanced
        description="Inhibition/excitation balance"
    )
    
    MODULATOR_DECAY_RATE = Gene(
        locus=209, name="modulator_decay_rate", category=GeneCategory.NEUROMODULATION,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.01, mutation_magnitude=0.1,
        description="How fast neuromodulators decay"
    )
    
    # =========================================================================
    # DRIVE GENES (locus 300-399)
    # =========================================================================
    
    HUNGER_BASELINE = Gene(
        locus=300, name="hunger_baseline", category=GeneCategory.DRIVES,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.02, mutation_magnitude=0.12,
        description="Base hunger drive intensity"
    )
    
    CURIOSITY_DRIVE = Gene(
        locus=301, name="curiosity_drive", category=GeneCategory.DRIVES,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.02, mutation_magnitude=0.15,
        description="Exploration/novelty seeking"
    )
    
    FEAR_SENSITIVITY = Gene(
        locus=302, name="fear_sensitivity", category=GeneCategory.DRIVES,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.DOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Fear response intensity"
    )
    
    AGGRESSION_BASELINE = Gene(
        locus=303, name="aggression_baseline", category=GeneCategory.DRIVES,
        allele_a=0.3, allele_b=0.3, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Base aggression level"
    )
    
    SOCIAL_DRIVE = Gene(
        locus=304, name="social_drive", category=GeneCategory.DRIVES,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.02, mutation_magnitude=0.12,
        description="Need for social interaction"
    )
    
    SLEEP_NEED = Gene(
        locus=305, name="sleep_need", category=GeneCategory.DRIVES,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Sleep/rest requirement"
    )
    
    PAIN_SENSITIVITY = Gene(
        locus=306, name="pain_sensitivity", category=GeneCategory.DRIVES,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.01, mutation_magnitude=0.1,
        description="Pain perception intensity"
    )
    
    REWARD_SENSITIVITY = Gene(
        locus=307, name="reward_sensitivity", category=GeneCategory.DRIVES,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.12,
        description="How strongly rewards are felt"
    )
    
    TERRITORIAL_DRIVE = Gene(
        locus=308, name="territorial_drive", category=GeneCategory.DRIVES,
        allele_a=0.4, allele_b=0.4, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Territorial behavior intensity"
    )
    
    # =========================================================================
    # BODY MORPHOLOGY GENES (locus 400-499)
    # =========================================================================
    
    BODY_SIZE = Gene(
        locus=400, name="body_size", category=GeneCategory.BODY_MORPHOLOGY,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.02, mutation_magnitude=0.15,
        description="Overall body size"
    )
    
    MOVEMENT_SPEED = Gene(
        locus=401, name="movement_speed", category=GeneCategory.BODY_MORPHOLOGY,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.02, mutation_magnitude=0.12,
        description="Base movement speed"
    )
    
    MUSCLE_STRENGTH = Gene(
        locus=402, name="muscle_strength", category=GeneCategory.BODY_MORPHOLOGY,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.02, mutation_magnitude=0.12,
        description="Muscle power"
    )
    
    LIMB_COUNT = Gene(
        locus=403, name="limb_count", category=GeneCategory.BODY_MORPHOLOGY,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.DOMINANT,
        mutation_rate=0.005, mutation_magnitude=0.2,
        description="Number of limbs (discrete)"
    )
    
    COLOR_HUE = Gene(
        locus=404, name="color_hue", category=GeneCategory.BODY_MORPHOLOGY,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.03, mutation_magnitude=0.15,
        description="Primary body color hue"
    )
    
    COLOR_SATURATION = Gene(
        locus=405, name="color_saturation", category=GeneCategory.BODY_MORPHOLOGY,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.03, mutation_magnitude=0.15,
        description="Color intensity"
    )
    
    PATTERN_TYPE = Gene(
        locus=406, name="pattern_type", category=GeneCategory.BODY_MORPHOLOGY,
        allele_a=0.3, allele_b=0.3, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.02, mutation_magnitude=0.2,
        description="Body pattern (stripes, spots, solid)"
    )
    
    # =========================================================================
    # METABOLISM GENES (locus 500-599)
    # =========================================================================
    
    METABOLIC_RATE = Gene(
        locus=500, name="metabolic_rate", category=GeneCategory.METABOLISM,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Energy consumption rate"
    )
    
    ENERGY_STORAGE = Gene(
        locus=501, name="energy_storage", category=GeneCategory.METABOLISM,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Max energy capacity"
    )
    
    HEALING_RATE = Gene(
        locus=502, name="healing_rate", category=GeneCategory.METABOLISM,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Recovery speed"
    )
    
    FATIGUE_RESISTANCE = Gene(
        locus=503, name="fatigue_resistance", category=GeneCategory.METABOLISM,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Resistance to fatigue"
    )
    
    # =========================================================================
    # SENSORY GENES (locus 600-699)
    # =========================================================================
    
    VISION_RANGE = Gene(
        locus=600, name="vision_range", category=GeneCategory.SENSORY,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.02, mutation_magnitude=0.12,
        description="Visual perception range"
    )
    
    HEARING_RANGE = Gene(
        locus=601, name="hearing_range", category=GeneCategory.SENSORY,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.02, mutation_magnitude=0.12,
        description="Auditory perception range"
    )
    
    SMELL_SENSITIVITY = Gene(
        locus=602, name="smell_sensitivity", category=GeneCategory.SENSORY,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.02, mutation_magnitude=0.12,
        description="Olfactory sensitivity"
    )
    
    PROPRIOCEPTION = Gene(
        locus=603, name="proprioception", category=GeneCategory.SENSORY,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Body position awareness"
    )
    
    # =========================================================================
    # LIFESPAN GENES (locus 700-799)
    # =========================================================================
    
    LONGEVITY = Gene(
        locus=700, name="longevity", category=GeneCategory.LIFESPAN,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.RECESSIVE,
        mutation_rate=0.01, mutation_magnitude=0.1,
        description="Natural lifespan"
    )
    
    FERTILITY = Gene(
        locus=701, name="fertility", category=GeneCategory.LIFESPAN,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.12,
        description="Reproductive fitness"
    )
    
    MATURATION_RATE = Gene(
        locus=702, name="maturation_rate", category=GeneCategory.LIFESPAN,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.01, mutation_magnitude=0.1,
        description="Development speed"
    )
    
    AGING_RATE = Gene(
        locus=703, name="aging_rate", category=GeneCategory.LIFESPAN,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.01, mutation_magnitude=0.1,
        description="Rate of biological aging"
    )
    
    # =========================================================================
    # LEARNING GENES (locus 800-899)
    # =========================================================================
    
    LEARNING_RATE = Gene(
        locus=800, name="learning_rate", category=GeneCategory.LEARNING,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.12,
        description="Speed of learning"
    )
    
    MEMORY_RETENTION = Gene(
        locus=801, name="memory_retention", category=GeneCategory.LEARNING,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Memory stability"
    )
    
    PLASTICITY = Gene(
        locus=802, name="plasticity", category=GeneCategory.LEARNING,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Neural plasticity level"
    )
    
    HABITUATION_RATE = Gene(
        locus=803, name="habituation_rate", category=GeneCategory.LEARNING,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="How fast stimuli are ignored"
    )
    
    # =========================================================================
    # SOCIAL GENES (locus 900-999)
    # =========================================================================
    
    BONDING_STRENGTH = Gene(
        locus=900, name="bonding_strength", category=GeneCategory.SOCIAL,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.02, mutation_magnitude=0.12,
        description="Attachment bond intensity"
    )
    
    EMPATHY = Gene(
        locus=901, name="empathy", category=GeneCategory.SOCIAL,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Emotional resonance with others"
    )
    
    DOMINANCE_DRIVE = Gene(
        locus=902, name="dominance_drive", category=GeneCategory.SOCIAL,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.INCOMPLETE,
        mutation_rate=0.015, mutation_magnitude=0.1,
        description="Social hierarchy ambition"
    )
    
    COOPERATION_TENDENCY = Gene(
        locus=903, name="cooperation_tendency", category=GeneCategory.SOCIAL,
        allele_a=0.5, allele_b=0.5, dominance=Dominance.CODOMINANT,
        mutation_rate=0.02, mutation_magnitude=0.12,
        description="Willingness to cooperate"
    )
    
    @classmethod
    def get_all_genes(cls) -> List[Gene]:
        """Return list of all gene definitions."""
        genes = []
        for name in dir(cls):
            attr = getattr(cls, name)
            if isinstance(attr, Gene):
                genes.append(deepcopy(attr))
        return genes
    
    @classmethod
    def get_gene_by_locus(cls, locus: int) -> Optional[Gene]:
        """Get gene definition by locus number."""
        for name in dir(cls):
            attr = getattr(cls, name)
            if isinstance(attr, Gene) and attr.locus == locus:
                return deepcopy(attr)
        return None
    
    @classmethod
    def get_gene_by_name(cls, name: str) -> Optional[Gene]:
        """Get gene definition by name."""
        attr = getattr(cls, name.upper(), None)
        if isinstance(attr, Gene):
            return deepcopy(attr)
        return None


# =============================================================================
# GENOME - Complete genetic code
# =============================================================================

class Genome:
    """
    Complete genome for an organism.
    Contains all genes and provides methods for expression, mutation, breeding.
    """
    
    def __init__(self, genes: Optional[Dict[int, Gene]] = None):
        """
        Initialize genome with genes indexed by locus.
        
        Args:
            genes: Dict mapping locus -> Gene. If None, creates default genome.
        """
        if genes is None:
            self.genes = {g.locus: g for g in GeneLibrary.get_all_genes()}
        else:
            self.genes = genes
        
        self._phenotype_cache = None
        self._id = self._compute_id()
    
    def _compute_id(self) -> str:
        """Compute unique genome ID from alleles."""
        allele_str = ""
        for locus in sorted(self.genes.keys()):
            gene = self.genes[locus]
            allele_str += f"{locus}:{gene.allele_a:.4f},{gene.allele_b:.4f}|"
        return hashlib.md5(allele_str.encode()).hexdigest()[:16]
    
    @property
    def id(self) -> str:
        """Unique genome identifier."""
        return self._id
    
    def get(self, name_or_locus: Union[str, int]) -> Optional[Gene]:
        """Get gene by name or locus."""
        if isinstance(name_or_locus, int):
            return self.genes.get(name_or_locus)
        else:
            for gene in self.genes.values():
                if gene.name == name_or_locus:
                    return gene
        return None
    
    def express(self, name_or_locus: Union[str, int]) -> float:
        """Express a gene and return its phenotypic value."""
        gene = self.get(name_or_locus)
        if gene:
            return gene.express()
        return 0.5  # Default
    
    def express_scaled(self, name_or_locus: Union[str, int], 
                       min_val: float, max_val: float) -> float:
        """Express gene scaled to range."""
        gene = self.get(name_or_locus)
        if gene:
            return gene.express_scaled(min_val, max_val)
        return (min_val + max_val) / 2
    
    def is_viable(self) -> bool:
        """Check if genome produces a viable organism (no lethal alleles)."""
        for gene in self.genes.values():
            if gene.is_lethal():
                return False
        return True
    
    def get_lethal_genes(self) -> List[Gene]:
        """Return list of genes with lethal expressions."""
        return [g for g in self.genes.values() if g.is_lethal()]
    
    def mutate(self, rate_multiplier: float = 1.0) -> 'Genome':
        """Return a mutated copy of this genome."""
        new_genes = {}
        for locus, gene in self.genes.items():
            new_genes[locus] = gene.mutate(rate_multiplier)
        return Genome(new_genes)
    
    def get_phenotype(self) -> Dict[str, float]:
        """
        Express all genes and return phenotype dictionary.
        Cached for performance.
        """
        if self._phenotype_cache is not None:
            return self._phenotype_cache
        
        phenotype = {}
        for gene in self.genes.values():
            phenotype[gene.name] = gene.express()
        
        self._phenotype_cache = phenotype
        return phenotype
    
    def invalidate_cache(self):
        """Invalidate phenotype cache (call after mutations)."""
        self._phenotype_cache = None
        self._id = self._compute_id()
    
    @classmethod
    def random(cls, randomness: float = 0.3) -> 'Genome':
        """
        Create a genome with randomized alleles.
        
        Args:
            randomness: How much to vary from defaults (0-1)
        """
        genes = {}
        for gene in GeneLibrary.get_all_genes():
            new_gene = deepcopy(gene)
            # Add random variation to alleles
            new_gene.allele_a = np.clip(
                gene.allele_a + np.random.normal(0, randomness * 0.3),
                gene.min_val, gene.max_val
            )
            new_gene.allele_b = np.clip(
                gene.allele_b + np.random.normal(0, randomness * 0.3),
                gene.min_val, gene.max_val
            )
            genes[gene.locus] = new_gene
        return cls(genes)
    
    def to_dict(self) -> dict:
        """Serialize genome to dictionary."""
        return {
            'id': self._id,
            'genes': {locus: gene.to_dict() for locus, gene in self.genes.items()}
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Genome':
        """Deserialize genome from dictionary."""
        genes = {int(locus): Gene.from_dict(gd) for locus, gd in d['genes'].items()}
        return cls(genes)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, s: str) -> 'Genome':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(s))
    
    def __repr__(self) -> str:
        return f"Genome(id={self._id}, genes={len(self.genes)})"


# =============================================================================
# BREEDING AND GENETIC OPERATIONS
# =============================================================================

def crossover(parent1: Genome, parent2: Genome, 
              crossover_points: int = 2) -> Tuple[Genome, Genome]:
    """
    Perform meiosis-like crossover between two genomes.
    
    Args:
        parent1: First parent genome
        parent2: Second parent genome
        crossover_points: Number of crossover points
        
    Returns:
        Tuple of two child genomes
    """
    loci = sorted(set(parent1.genes.keys()) | set(parent2.genes.keys()))
    
    # Generate crossover points
    if len(loci) > crossover_points + 1:
        points = sorted(np.random.choice(len(loci), crossover_points, replace=False))
    else:
        points = []
    
    child1_genes = {}
    child2_genes = {}
    
    use_parent1 = True
    point_idx = 0
    
    for i, locus in enumerate(loci):
        # Check for crossover
        if point_idx < len(points) and i >= points[point_idx]:
            use_parent1 = not use_parent1
            point_idx += 1
        
        p1_gene = parent1.genes.get(locus)
        p2_gene = parent2.genes.get(locus)
        
        if p1_gene is None:
            p1_gene = deepcopy(p2_gene)
        if p2_gene is None:
            p2_gene = deepcopy(p1_gene)
        
        if p1_gene is None:
            continue
        
        # Create child genes by mixing alleles
        if use_parent1:
            # Child 1 gets allele_a from parent1, allele_b from parent2
            c1_gene = deepcopy(p1_gene)
            c1_gene.allele_b = p2_gene.allele_a if np.random.random() < 0.5 else p2_gene.allele_b
            
            c2_gene = deepcopy(p2_gene)
            c2_gene.allele_b = p1_gene.allele_a if np.random.random() < 0.5 else p1_gene.allele_b
        else:
            c1_gene = deepcopy(p2_gene)
            c1_gene.allele_b = p1_gene.allele_a if np.random.random() < 0.5 else p1_gene.allele_b
            
            c2_gene = deepcopy(p1_gene)
            c2_gene.allele_b = p2_gene.allele_a if np.random.random() < 0.5 else p2_gene.allele_b
        
        child1_genes[locus] = c1_gene
        child2_genes[locus] = c2_gene
    
    return Genome(child1_genes), Genome(child2_genes)


def breed(parent1: Genome, parent2: Genome, 
          mutation_rate: float = 1.0,
          ensure_viable: bool = True,
          max_attempts: int = 10) -> Genome:
    """
    Breed two genomes to produce offspring.
    
    Args:
        parent1: First parent
        parent2: Second parent  
        mutation_rate: Mutation rate multiplier
        ensure_viable: Keep trying until viable offspring
        max_attempts: Max breeding attempts if ensuring viability
        
    Returns:
        Child genome
    """
    for attempt in range(max_attempts):
        # Crossover
        child1, child2 = crossover(parent1, parent2)
        
        # Pick one child randomly
        child = child1 if np.random.random() < 0.5 else child2
        
        # Mutate
        child = child.mutate(mutation_rate)
        child.invalidate_cache()
        
        if not ensure_viable or child.is_viable():
            return child
    
    # If all attempts failed, return last child anyway
    return child


def mutate(genome: Genome, rate: float = 1.0) -> Genome:
    """Mutate a genome and return new copy."""
    new_genome = genome.mutate(rate)
    new_genome.invalidate_cache()
    return new_genome


# =============================================================================
# DEVELOPMENT SYSTEM - Grow brain/body from DNA
# =============================================================================

class DevelopmentalSystem:
    """
    Grows a brain and body from a genome.
    
    This is the bridge between genotype and phenotype:
    - Reads genes
    - Applies developmental rules
    - Produces BrainConfig and body parameters
    """
    
    def __init__(self, genome: Genome):
        self.genome = genome
        self.phenotype = genome.get_phenotype()
    
    def develop_brain_config(self) -> 'BrainConfig':
        """
        Grow a BrainConfig from the genome.
        
        Returns a BrainConfig with all parameters set from DNA.
        """
        from .three_system_brain import BrainConfig
        
        p = self.phenotype
        
        # Brain structure
        num_columns = int(50 + 450 * p.get('cortical_columns', 0.5))
        cells_per_column = int(8 + 56 * p.get('cells_per_column', 0.5))
        reservoir_size = int(500 + 4500 * p.get('reservoir_size', 0.5))
        
        config = BrainConfig(
            # Structure
            num_columns=num_columns,
            cells_per_column=cells_per_column,
            reservoir_size=reservoir_size,
            
            # Sparsity
            target_sparsity=0.01 + 0.19 * p.get('cortical_sparsity', 0.3),
            
            # Reservoir
            spectral_radius=0.5 + 0.5 * p.get('spectral_radius', 0.9),
            reservoir_sparsity=0.05 + 0.25 * p.get('reservoir_sparsity', 0.3),
            leak_rate=0.1 + 0.5 * p.get('leak_rate', 0.5),
            
            # Lateral inhibition
            lateral_inhibition_strength=0.1 + 0.5 * p.get('lateral_inhibition', 0.5),
            
            # Neuromodulator baselines
            dopamine_baseline=0.2 + 0.6 * p.get('dopamine_baseline', 0.5),
            serotonin_baseline=0.2 + 0.6 * p.get('serotonin_baseline', 0.5),
            norepinephrine_baseline=0.1 + 0.5 * p.get('norepinephrine_baseline', 0.4),
            acetylcholine_baseline=0.2 + 0.6 * p.get('acetylcholine_baseline', 0.5),
            
            # Decay rates (from modulator_decay_rate)
            dopamine_decay=0.03 + 0.07 * (0.5 + p.get('modulator_decay_rate', 0.5)),
            serotonin_decay=0.02 + 0.04 * (0.5 + p.get('modulator_decay_rate', 0.5)),
            norepinephrine_decay=0.05 + 0.1 * (0.5 + p.get('modulator_decay_rate', 0.5)),
            acetylcholine_decay=0.04 + 0.08 * (0.5 + p.get('modulator_decay_rate', 0.5)),
            
            # System glue effects (from sensitivity genes)
            ach_sparsity_gain=0.3 + 0.6 * p.get('acetylcholine_baseline', 0.5),
            da_gain_effect=0.2 + 0.5 * p.get('dopamine_sensitivity', 0.5),
            serotonin_plasticity_effect=0.2 + 0.5 * p.get('serotonin_sensitivity', 0.5),
            
            # Learning (from learning genes)
            neurogenesis_rate=0.1 + 0.3 * p.get('plasticity', 0.5),
            
            # Dreaming (influenced by theta power)
            dream_enabled=True,
            dream_steps=int(3 + 7 * p.get('theta_power', 0.5)),
        )
        
        return config
    
    def develop_body_params(self) -> Dict[str, float]:
        """
        Develop body parameters from genome.
        
        Returns dict of body configuration values.
        """
        p = self.phenotype
        
        return {
            # Physical
            'size': 0.5 + p.get('body_size', 0.5),
            'speed': 0.3 + 0.7 * p.get('movement_speed', 0.5),
            'strength': 0.3 + 0.7 * p.get('muscle_strength', 0.5),
            'limb_count': int(2 + 6 * p.get('limb_count', 0.5)),
            
            # Appearance
            'color_hue': p.get('color_hue', 0.5),
            'color_saturation': 0.3 + 0.7 * p.get('color_saturation', 0.5),
            'pattern': p.get('pattern_type', 0.3),
            
            # Metabolism
            'metabolic_rate': 0.5 + 0.5 * p.get('metabolic_rate', 0.5),
            'max_energy': 50 + 100 * p.get('energy_storage', 0.5),
            'healing_rate': 0.5 + 0.5 * p.get('healing_rate', 0.5),
            'fatigue_resistance': 0.3 + 0.7 * p.get('fatigue_resistance', 0.5),
            
            # Sensory
            'vision_range': 5 + 15 * p.get('vision_range', 0.5),
            'hearing_range': 3 + 12 * p.get('hearing_range', 0.5),
            'smell_range': 2 + 8 * p.get('smell_sensitivity', 0.5),
            
            # Lifespan
            'max_age': int(100 + 400 * p.get('longevity', 0.5)),
            'fertility': 0.3 + 0.7 * p.get('fertility', 0.5),
            'maturation_age': int(10 + 20 * (1 - p.get('maturation_rate', 0.5))),
            'maturation_speed': p.get('maturation_rate', 0.5),  # For Homeostasis aging
        }
    
    def develop_drives(self) -> Dict[str, float]:
        """
        Develop drive/instinct parameters from genome.
        
        Returns dict of drive baseline values.
        """
        p = self.phenotype
        
        return {
            'hunger': 0.3 + 0.4 * p.get('hunger_baseline', 0.5),
            'curiosity': 0.2 + 0.6 * p.get('curiosity_drive', 0.5),
            'fear': 0.2 + 0.6 * p.get('fear_sensitivity', 0.5),
            'aggression': 0.1 + 0.5 * p.get('aggression_baseline', 0.3),
            'social': 0.3 + 0.5 * p.get('social_drive', 0.5),
            'sleep': 0.3 + 0.4 * p.get('sleep_need', 0.5),
            'pain': 0.3 + 0.5 * p.get('pain_sensitivity', 0.5),
            'reward': 0.3 + 0.5 * p.get('reward_sensitivity', 0.5),
            'territorial': 0.2 + 0.5 * p.get('territorial_drive', 0.4),
            'bonding': 0.3 + 0.5 * p.get('bonding_strength', 0.5),
            'empathy': 0.2 + 0.6 * p.get('empathy', 0.5),
            'dominance': 0.2 + 0.6 * p.get('dominance_drive', 0.5),
            'cooperation': 0.3 + 0.5 * p.get('cooperation_tendency', 0.5),
        }
    
    def develop_learning_params(self) -> Dict[str, float]:
        """Develop learning-related parameters."""
        p = self.phenotype
        
        return {
            'learning_rate': 0.001 + 0.01 * p.get('learning_rate', 0.5),
            'memory_decay': 0.001 + 0.01 * (1 - p.get('memory_retention', 0.5)),
            'plasticity': 0.5 + 0.5 * p.get('plasticity', 0.5),
            'habituation': 0.01 + 0.05 * p.get('habituation_rate', 0.5),
        }


# =============================================================================
# GENE POOL - Population genetics
# =============================================================================

class GenePool:
    """
    Manages a population of genomes.
    Tracks allele frequencies, genetic diversity, etc.
    """
    
    def __init__(self, population: Optional[List[Genome]] = None):
        self.population = population or []
        self.generation = 0
        self.history = []
    
    def add(self, genome: Genome):
        """Add genome to pool."""
        self.population.append(genome)
    
    def remove(self, genome: Genome):
        """Remove genome from pool."""
        self.population = [g for g in self.population if g.id != genome.id]
    
    def get_allele_frequencies(self, locus: int) -> Dict[str, float]:
        """Get allele frequency distribution for a locus."""
        if not self.population:
            return {}
        
        alleles = []
        for genome in self.population:
            gene = genome.genes.get(locus)
            if gene:
                alleles.extend([gene.allele_a, gene.allele_b])
        
        if not alleles:
            return {}
        
        return {
            'mean': np.mean(alleles),
            'std': np.std(alleles),
            'min': np.min(alleles),
            'max': np.max(alleles),
        }
    
    def get_genetic_diversity(self) -> float:
        """
        Calculate genetic diversity index (0-1).
        Higher = more diverse population.
        """
        if len(self.population) < 2:
            return 0.0
        
        diversities = []
        for locus in self.population[0].genes.keys():
            freqs = self.get_allele_frequencies(locus)
            if freqs:
                diversities.append(freqs['std'])
        
        return np.mean(diversities) if diversities else 0.0
    
    def get_fittest(self, fitness_fn: Callable[[Genome], float], 
                    n: int = 1) -> List[Genome]:
        """Get top N fittest genomes."""
        scored = [(g, fitness_fn(g)) for g in self.population]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [g for g, _ in scored[:n]]
    
    def evolve_generation(self, 
                          fitness_fn: Callable[[Genome], float],
                          survival_rate: float = 0.5,
                          mutation_rate: float = 1.0) -> 'GenePool':
        """
        Evolve population to next generation.
        
        Args:
            fitness_fn: Function that scores genomes
            survival_rate: Fraction that survive to breed
            mutation_rate: Mutation rate multiplier
            
        Returns:
            New GenePool with next generation
        """
        if len(self.population) < 2:
            return self
        
        # Score and sort
        scored = [(g, fitness_fn(g)) for g in self.population]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Select survivors
        n_survivors = max(2, int(len(scored) * survival_rate))
        survivors = [g for g, _ in scored[:n_survivors]]
        
        # Breed to replenish population
        new_population = list(survivors)
        target_size = len(self.population)
        
        while len(new_population) < target_size:
            parent1, parent2 = np.random.choice(survivors, 2, replace=False)
            child = breed(parent1, parent2, mutation_rate=mutation_rate)
            if child.is_viable():
                new_population.append(child)
        
        new_pool = GenePool(new_population)
        new_pool.generation = self.generation + 1
        
        # Record history
        self.history.append({
            'generation': self.generation,
            'diversity': self.get_genetic_diversity(),
            'mean_fitness': np.mean([s for _, s in scored]),
            'max_fitness': scored[0][1] if scored else 0,
        })
        
        return new_pool
    
    @classmethod
    def create_random(cls, size: int, randomness: float = 0.3) -> 'GenePool':
        """Create a pool of random genomes."""
        population = [Genome.random(randomness) for _ in range(size)]
        return cls(population)
    
    def __len__(self) -> int:
        return len(self.population)
    
    def __repr__(self) -> str:
        return f"GenePool(size={len(self)}, generation={self.generation})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_organism_from_dna(genome: Genome) -> Tuple[Any, Dict, Dict]:
    """
    Create a complete organism from DNA.
    
    Returns:
        Tuple of (brain, body_params, drives)
    """
    dev = DevelopmentalSystem(genome)
    
    config = dev.develop_brain_config()
    body = dev.develop_body_params()
    drives = dev.develop_drives()
    
    # Create brain from config
    from .three_system_brain import ThreeSystemBrain
    brain = ThreeSystemBrain(config)
    
    # Store genome reference
    brain.genome = genome
    brain.drives = drives
    brain.body_params = body
    
    return brain, body, drives


def breed_organisms(parent1_brain, parent2_brain,
                    mutation_rate: float = 1.0) -> Tuple[Any, Dict, Dict]:
    """
    Breed two organisms to create offspring.
    
    Args:
        parent1_brain: First parent (must have .genome attribute)
        parent2_brain: Second parent
        mutation_rate: Mutation rate multiplier
        
    Returns:
        Tuple of (child_brain, body_params, drives)
    """
    child_genome = breed(
        parent1_brain.genome,
        parent2_brain.genome,
        mutation_rate=mutation_rate
    )
    
    return create_organism_from_dna(child_genome)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'Gene',
    'GeneCategory',
    'Dominance',
    'GeneLibrary',
    'Genome',
    'GenePool',
    'DevelopmentalSystem',
    'crossover',
    'breed',
    'mutate',
    'create_organism_from_dna',
    'breed_organisms',
]
