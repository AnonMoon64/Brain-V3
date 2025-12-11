"""
RNA System - The Developmental Steering Wheel

DNA is the library. RNA is the librarian deciding what gets read, when, where, and how often.

This module implements a compressed developmental model inspired by real RNA:

1. **mRNA (Messenger RNA)** - Expression levels per gene per tissue
   - Controls HOW STRONGLY each gene is expressed
   - Varies by tissue type (brain, sensory, muscle, skin)
   - Creates natural phenotype variation even with identical DNA

2. **miRNA (MicroRNA)** - Gene silencing and activation
   - Turns genes OFF/DOWN/ON based on conditions
   - Triggers at developmental stages
   - Creates recessive traits, morph changes, dimorphism

3. **Regulatory RNA (lncRNA)** - Patterning and body layout
   - Controls macro structure: segments, symmetry, proportions
   - Acts as "master switches" for development
   - Enables diverse body plans from similar genomes

The RNA layer sits between DNA (genotype) and the organism (phenotype),
providing the developmental context that determines final form.

USAGE:
    from brain.rna import RNASystem, Transcriptome
    
    # Create RNA system for a genome
    rna = RNASystem(genome)
    
    # Develop with environmental context
    transcriptome = rna.transcribe(
        tissue='brain',
        stage=DevelopmentalStage.JUVENILE,
        stress=0.3
    )
    
    # Get final expression levels
    expression = transcriptome.get_expression('cortical_columns')
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from enum import Enum, auto
from copy import deepcopy
import json
import hashlib


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class TissueType(Enum):
    """Tissue types where genes are expressed differently."""
    BRAIN_CORTEX = "brain_cortex"
    BRAIN_RESERVOIR = "brain_reservoir"
    BRAIN_MODULATION = "brain_modulation"
    SENSORY_VISUAL = "sensory_visual"
    SENSORY_AUDITORY = "sensory_auditory"
    SENSORY_CHEMICAL = "sensory_chemical"
    MUSCULATURE = "musculature"
    SKELETON = "skeleton"
    SKIN_PIGMENT = "skin_pigment"
    METABOLISM = "metabolism"
    REPRODUCTIVE = "reproductive"
    NEURAL_CREST = "neural_crest"  # For pattern/color


class DevelopmentalStage(Enum):
    """Stages of organism development."""
    EMBRYONIC = 0       # Initial formation
    LARVAL = 1          # Early growth
    JUVENILE = 2        # Rapid development
    ADOLESCENT = 3      # Maturation
    ADULT = 4           # Fully developed
    SENESCENT = 5       # Aging


class RNAType(Enum):
    """Types of RNA molecules."""
    MRNA = "mRNA"           # Messenger - expression levels
    MIRNA = "miRNA"         # Micro - gene silencing
    SIRNA = "siRNA"         # Small interfering - targeted knockdown
    LNCRNA = "lncRNA"       # Long noncoding - regulation/scaffolding
    CIRCRNA = "circRNA"     # Circular - stability/sponging


# =============================================================================
# mRNA - MESSENGER RNA (Expression Levels)
# =============================================================================

@dataclass
class MessengerRNA:
    """
    mRNA controls how strongly each gene is expressed per tissue.
    
    This creates natural variation - same DNA can produce different
    phenotypes based on expression patterns.
    """
    gene_locus: int
    base_level: float = 0.5          # 0.0-1.0 baseline expression
    tissue_modifiers: Dict[TissueType, float] = field(default_factory=dict)
    stage_modifiers: Dict[DevelopmentalStage, float] = field(default_factory=dict)
    half_life: float = 1.0           # Stability (affects expression duration)
    noise_amplitude: float = 0.05    # Stochastic expression noise
    
    def __post_init__(self):
        # Default tissue modifiers if not specified
        if not self.tissue_modifiers:
            self.tissue_modifiers = {t: 1.0 for t in TissueType}
        if not self.stage_modifiers:
            self.stage_modifiers = {s: 1.0 for s in DevelopmentalStage}
    
    def get_expression(self, 
                       tissue: TissueType,
                       stage: DevelopmentalStage,
                       noise: bool = True) -> float:
        """
        Calculate expression level for given tissue and stage.
        
        Returns:
            Expression level 0.0-1.0
        """
        level = self.base_level
        
        # Apply tissue modifier
        level *= self.tissue_modifiers.get(tissue, 1.0)
        
        # Apply stage modifier
        level *= self.stage_modifiers.get(stage, 1.0)
        
        # Apply stochastic noise (biological realism)
        if noise:
            level += np.random.normal(0, self.noise_amplitude)
        
        return np.clip(level, 0.0, 1.0)
    
    def mutate(self, rate: float = 0.01) -> 'MessengerRNA':
        """Create mutated copy."""
        new_mrna = deepcopy(self)
        
        if np.random.random() < rate:
            new_mrna.base_level = np.clip(
                new_mrna.base_level + np.random.normal(0, 0.1),
                0.0, 1.0
            )
        
        # Mutate tissue modifiers
        for tissue in new_mrna.tissue_modifiers:
            if np.random.random() < rate * 0.5:
                new_mrna.tissue_modifiers[tissue] = np.clip(
                    new_mrna.tissue_modifiers[tissue] + np.random.normal(0, 0.1),
                    0.0, 2.0
                )
        
        # Mutate stage modifiers
        for stage in new_mrna.stage_modifiers:
            if np.random.random() < rate * 0.5:
                new_mrna.stage_modifiers[stage] = np.clip(
                    new_mrna.stage_modifiers[stage] + np.random.normal(0, 0.1),
                    0.0, 2.0
                )
        
        return new_mrna


# =============================================================================
# miRNA - MICRO RNA (Gene Silencing & Activation)
# =============================================================================

@dataclass
class MicroRNA:
    """
    miRNA controls gene silencing and activation.
    
    Can:
    - Turn genes OFF completely
    - Turn genes DOWN (partial silencing)
    - Turn genes ON under stress
    - Activate only at certain developmental stages
    
    This enables:
    - Recessive traits appearing
    - Sudden morph changes
    - Personality shifts
    - Sexual dimorphism
    - Random mutant variants
    """
    id: str
    target_loci: List[int]                      # Which genes it affects
    silencing_strength: float = 0.5             # 0.0 = no effect, 1.0 = complete silence
    activation_threshold: float = 0.5           # When does it activate
    trigger_condition: str = "always"           # Condition for activation
    stage_specific: Optional[DevelopmentalStage] = None
    stress_responsive: bool = False             # Activates under stress
    stress_threshold: float = 0.7               # Stress level to trigger
    
    def is_active(self,
                  stage: DevelopmentalStage,
                  stress: float = 0.0,
                  hormones: Dict[str, float] = None) -> bool:
        """Check if this miRNA is currently active."""
        hormones = hormones or {}
        
        # Stage-specific check
        if self.stage_specific is not None:
            if stage != self.stage_specific:
                return False
        
        # Stress-responsive check
        if self.stress_responsive:
            if stress < self.stress_threshold:
                return False
        
        # Condition parsing
        if self.trigger_condition == "always":
            return True
        elif self.trigger_condition == "never":
            return False
        elif self.trigger_condition.startswith("hormone:"):
            hormone_name = self.trigger_condition.split(":")[1]
            return hormones.get(hormone_name, 0) > self.activation_threshold
        elif self.trigger_condition == "random":
            return np.random.random() < 0.5
        
        return True
    
    def get_silencing(self, locus: int) -> float:
        """Get silencing factor for a specific gene locus."""
        if locus in self.target_loci:
            return 1.0 - self.silencing_strength
        return 1.0
    
    def mutate(self, rate: float = 0.01) -> 'MicroRNA':
        """Create mutated copy."""
        new_mirna = deepcopy(self)
        
        if np.random.random() < rate:
            new_mirna.silencing_strength = np.clip(
                new_mirna.silencing_strength + np.random.normal(0, 0.15),
                0.0, 1.0
            )
        
        if np.random.random() < rate * 0.3:
            new_mirna.activation_threshold = np.clip(
                new_mirna.activation_threshold + np.random.normal(0, 0.1),
                0.0, 1.0
            )
        
        # Rarely gain/lose targets
        if np.random.random() < rate * 0.1:
            if new_mirna.target_loci and np.random.random() < 0.5:
                new_mirna.target_loci.pop(np.random.randint(len(new_mirna.target_loci)))
            else:
                new_mirna.target_loci.append(np.random.randint(1, 100))
        
        return new_mirna


# =============================================================================
# lncRNA - LONG NONCODING RNA (Patterning & Structure)
# =============================================================================

@dataclass
class RegulatoryRNA:
    """
    Long noncoding RNA controls patterning and body layout.
    
    Acts as "macro switches" for development:
    - Head vs tail formation
    - Left-right symmetry
    - Number of segments
    - Pigment patterns
    - Brain region scaling
    
    This enables diverse body plans from similar genomes.
    """
    id: str
    
    # Segmentation and body plan
    segment_repeat: float = 0.5          # 0=few segments, 1=many
    symmetry_bias: float = 0.5           # 0=asymmetric, 1=bilateral
    anterior_posterior_gradient: float = 0.5  # Head-tail axis strength
    dorsal_ventral_gradient: float = 0.5      # Top-bottom axis
    
    # Pigmentation and appearance
    pigment_pattern: float = 0.5         # Pattern complexity
    stripe_frequency: float = 0.3        # Stripe/spot repetition
    color_gradient: float = 0.5          # Color variation along body
    
    # Brain region scaling
    brain_scaling: Dict[str, float] = field(default_factory=dict)
    
    # Appendage patterning
    limb_positioning: float = 0.5        # Where limbs form
    appendage_branching: float = 0.3     # Complexity of appendages
    
    def __post_init__(self):
        if not self.brain_scaling:
            self.brain_scaling = {
                'cortex': 1.0,
                'reservoir': 1.0,
                'modulation': 1.0,
                'sensory': 1.0,
                'motor': 1.0
            }
    
    def get_segment_count(self, base: int = 8) -> int:
        """Calculate number of body segments."""
        multiplier = 0.5 + 1.5 * self.segment_repeat
        return max(1, int(base * multiplier))
    
    def get_symmetry_type(self) -> str:
        """Determine body symmetry."""
        if self.symmetry_bias > 0.8:
            return "bilateral"
        elif self.symmetry_bias > 0.5:
            return "biradial"
        elif self.symmetry_bias > 0.2:
            return "radial"
        else:
            return "asymmetric"
    
    def get_pattern_type(self) -> str:
        """Determine pigmentation pattern."""
        if self.pigment_pattern > 0.8:
            return "complex_spots"
        elif self.pigment_pattern > 0.6:
            return "stripes"
        elif self.pigment_pattern > 0.4:
            return "gradient"
        elif self.pigment_pattern > 0.2:
            return "patches"
        else:
            return "solid"
    
    def scale_brain_region(self, region: str, base_size: int) -> int:
        """Apply scaling to brain region size."""
        scale = self.brain_scaling.get(region, 1.0)
        return max(1, int(base_size * scale))
    
    def mutate(self, rate: float = 0.01) -> 'RegulatoryRNA':
        """Create mutated copy."""
        new_rna = deepcopy(self)
        
        # Mutate continuous values
        for attr in ['segment_repeat', 'symmetry_bias', 'anterior_posterior_gradient',
                     'dorsal_ventral_gradient', 'pigment_pattern', 'stripe_frequency',
                     'color_gradient', 'limb_positioning', 'appendage_branching']:
            if np.random.random() < rate:
                current = getattr(new_rna, attr)
                setattr(new_rna, attr, np.clip(
                    current + np.random.normal(0, 0.1),
                    0.0, 1.0
                ))
        
        # Mutate brain scaling
        for region in new_rna.brain_scaling:
            if np.random.random() < rate:
                new_rna.brain_scaling[region] = np.clip(
                    new_rna.brain_scaling[region] + np.random.normal(0, 0.15),
                    0.3, 2.0
                )
        
        return new_rna


# =============================================================================
# TRANSCRIPTOME - Complete RNA State
# =============================================================================

@dataclass
class Transcriptome:
    """
    Complete RNA expression state at a point in development.
    
    This is the "snapshot" of all RNA activity that determines
    how genes are expressed in a particular context.
    """
    mrna_levels: Dict[int, float]           # Locus -> expression level
    active_mirnas: List[str]                # Currently active miRNAs
    silencing_factors: Dict[int, float]     # Locus -> silencing multiplier
    regulatory_state: Dict[str, float]      # Regulatory RNA parameters
    tissue: TissueType
    stage: DevelopmentalStage
    
    def get_expression(self, locus: int, base_value: float = 1.0) -> float:
        """
        Get final expression level for a gene.
        
        Combines mRNA level with miRNA silencing.
        """
        mrna = self.mrna_levels.get(locus, 0.5)
        silencing = self.silencing_factors.get(locus, 1.0)
        return base_value * mrna * silencing
    
    def get_effective_allele(self, locus: int, allele_value: float) -> float:
        """Apply expression modulation to an allele value."""
        expression = self.get_expression(locus)
        # Expression scales the allele's effect
        return allele_value * expression
    
    def get_brain_scaling(self, region: str) -> float:
        """Get scaling factor for a brain region."""
        return self.regulatory_state.get(f'brain_{region}', 1.0)


# =============================================================================
# RNA SYSTEM - Main Orchestrator
# =============================================================================

class RNASystem:
    """
    Complete RNA system for an organism.
    
    Manages:
    - mRNA expression profiles for all genes
    - miRNA silencing/activation rules
    - Regulatory RNA patterning controls
    
    Transcribes DNA into context-dependent expression patterns.
    """
    
    def __init__(self, genome=None):
        """
        Initialize RNA system, optionally from a genome.
        
        Args:
            genome: Optional Genome object to derive RNA from
        """
        self.mrna_pool: Dict[int, MessengerRNA] = {}
        self.mirna_pool: Dict[str, MicroRNA] = {}
        self.regulatory_rna: RegulatoryRNA = RegulatoryRNA(id="main")
        
        if genome is not None:
            self._derive_from_genome(genome)
    
    def _derive_from_genome(self, genome):
        """
        Derive RNA profiles from genome.
        
        In biology, DNA encodes both genes AND their regulatory elements.
        This simulates that relationship.
        """
        # Create mRNA for each gene in genome
        for locus, gene in genome.genes.items():
            # Base mRNA level influenced by gene properties
            base_level = 0.3 + 0.4 * ((gene.allele_a + gene.allele_b) / 2)
            
            # Tissue modifiers based on gene category
            tissue_mods = self._get_tissue_modifiers(gene.category)
            
            # Stage modifiers (most genes peak in juvenile-adult)
            stage_mods = {
                DevelopmentalStage.EMBRYONIC: 0.3,
                DevelopmentalStage.LARVAL: 0.5,
                DevelopmentalStage.JUVENILE: 0.9,
                DevelopmentalStage.ADOLESCENT: 1.0,
                DevelopmentalStage.ADULT: 0.95,
                DevelopmentalStage.SENESCENT: 0.7,
            }
            
            self.mrna_pool[locus] = MessengerRNA(
                gene_locus=locus,
                base_level=base_level,
                tissue_modifiers=tissue_mods,
                stage_modifiers=stage_mods,
                noise_amplitude=gene.mutation_rate * 2  # More mutable = noisier
            )
        
        # Create some default miRNAs
        self._create_default_mirnas(genome)
        
        # Create regulatory RNA from genome averages
        self._create_regulatory_rna(genome)
    
    def _get_tissue_modifiers(self, category) -> Dict[TissueType, float]:
        """Get tissue expression modifiers based on gene category."""
        from .dna import GeneCategory
        
        base = {t: 1.0 for t in TissueType}
        
        if category == GeneCategory.BRAIN_STRUCTURE:
            base[TissueType.BRAIN_CORTEX] = 1.5
            base[TissueType.BRAIN_RESERVOIR] = 1.5
            base[TissueType.MUSCULATURE] = 0.2
            base[TissueType.SKIN_PIGMENT] = 0.1
            
        elif category == GeneCategory.BRAIN_DYNAMICS:
            base[TissueType.BRAIN_CORTEX] = 1.3
            base[TissueType.BRAIN_RESERVOIR] = 1.4
            base[TissueType.BRAIN_MODULATION] = 1.2
            
        elif category == GeneCategory.NEUROMODULATION:
            base[TissueType.BRAIN_MODULATION] = 2.0
            base[TissueType.BRAIN_CORTEX] = 1.2
            base[TissueType.METABOLISM] = 0.8
            
        elif category == GeneCategory.BODY_MORPHOLOGY:
            base[TissueType.SKELETON] = 1.5
            base[TissueType.MUSCULATURE] = 1.3
            base[TissueType.SKIN_PIGMENT] = 1.4
            base[TissueType.BRAIN_CORTEX] = 0.1
            
        elif category == GeneCategory.DRIVES:
            base[TissueType.BRAIN_MODULATION] = 1.5
            base[TissueType.BRAIN_CORTEX] = 1.2
            base[TissueType.METABOLISM] = 1.1
            
        elif category == GeneCategory.METABOLISM:
            base[TissueType.METABOLISM] = 2.0
            base[TissueType.MUSCULATURE] = 1.3
            
        elif category == GeneCategory.SENSORY:
            base[TissueType.SENSORY_VISUAL] = 1.8
            base[TissueType.SENSORY_AUDITORY] = 1.8
            base[TissueType.SENSORY_CHEMICAL] = 1.8
            base[TissueType.BRAIN_CORTEX] = 1.2
            
        elif category == GeneCategory.LIFESPAN:
            base[TissueType.METABOLISM] = 1.5
            base[TissueType.REPRODUCTIVE] = 1.3
            
        elif category == GeneCategory.LEARNING:
            base[TissueType.BRAIN_CORTEX] = 1.6
            base[TissueType.BRAIN_RESERVOIR] = 1.4
            base[TissueType.BRAIN_MODULATION] = 1.3
            
        elif category == GeneCategory.SOCIAL:
            base[TissueType.BRAIN_MODULATION] = 1.4
            base[TissueType.BRAIN_CORTEX] = 1.2
        
        return base
    
    def _create_default_mirnas(self, genome):
        """Create default miRNA regulators."""
        # Stress-responsive miRNA that silences growth genes
        self.mirna_pool['mir_stress_1'] = MicroRNA(
            id='mir_stress_1',
            target_loci=[1, 2, 3],  # Cortical structure genes
            silencing_strength=0.4,
            stress_responsive=True,
            stress_threshold=0.6,
            trigger_condition="always"
        )
        
        # Development stage miRNA - silences juvenile genes in adults
        self.mirna_pool['mir_maturation'] = MicroRNA(
            id='mir_maturation',
            target_loci=[35, 36, 37],  # Growth-related
            silencing_strength=0.6,
            stage_specific=DevelopmentalStage.ADULT,
            trigger_condition="always"
        )
        
        # Embryonic patterning miRNA
        self.mirna_pool['mir_embryo'] = MicroRNA(
            id='mir_embryo',
            target_loci=[80, 81, 82, 83],  # Body structure
            silencing_strength=0.3,
            stage_specific=DevelopmentalStage.EMBRYONIC,
            trigger_condition="always"
        )
        
        # Random expression variation
        self.mirna_pool['mir_noise'] = MicroRNA(
            id='mir_noise',
            target_loci=list(range(1, 20)),
            silencing_strength=0.15,
            trigger_condition="random"
        )
    
    def _create_regulatory_rna(self, genome):
        """Create regulatory RNA from genome characteristics."""
        # Average allele values to determine regulatory patterns
        alleles = []
        for gene in genome.genes.values():
            alleles.extend([gene.allele_a, gene.allele_b])
        
        avg = np.mean(alleles) if alleles else 0.5
        std = np.std(alleles) if alleles else 0.1
        
        # Use genome statistics to set regulatory parameters
        self.regulatory_rna = RegulatoryRNA(
            id="main",
            segment_repeat=np.clip(avg + np.random.normal(0, 0.1), 0, 1),
            symmetry_bias=np.clip(0.8 + np.random.normal(0, 0.1), 0, 1),  # Most creatures bilateral
            anterior_posterior_gradient=np.clip(avg, 0, 1),
            dorsal_ventral_gradient=np.clip(avg, 0, 1),
            pigment_pattern=np.clip(std * 3, 0, 1),  # More genetic variation = more patterns
            stripe_frequency=np.random.random(),
            color_gradient=np.random.random(),
            brain_scaling={
                'cortex': 0.8 + 0.4 * avg,
                'reservoir': 0.8 + 0.4 * avg,
                'modulation': 1.0,
                'sensory': 0.9 + 0.2 * avg,
                'motor': 0.9 + 0.2 * avg
            },
            limb_positioning=0.5,
            appendage_branching=np.clip(std * 2, 0, 1)
        )
    
    def transcribe(self,
                   tissue: TissueType = TissueType.BRAIN_CORTEX,
                   stage: DevelopmentalStage = DevelopmentalStage.ADULT,
                   stress: float = 0.0,
                   hormones: Dict[str, float] = None) -> Transcriptome:
        """
        Transcribe DNA into expression levels for given context.
        
        This is the main function - it determines what genes are
        expressed and at what levels for a specific tissue/stage.
        
        Args:
            tissue: Target tissue type
            stage: Developmental stage
            stress: Current stress level (0-1)
            hormones: Hormone levels affecting expression
            
        Returns:
            Transcriptome with all expression information
        """
        hormones = hormones or {}
        
        # Calculate mRNA levels for each gene
        mrna_levels = {}
        for locus, mrna in self.mrna_pool.items():
            mrna_levels[locus] = mrna.get_expression(tissue, stage)
        
        # Determine active miRNAs
        active_mirnas = []
        silencing_factors = {locus: 1.0 for locus in mrna_levels}
        
        for mirna_id, mirna in self.mirna_pool.items():
            if mirna.is_active(stage, stress, hormones):
                active_mirnas.append(mirna_id)
                # Apply silencing to target genes
                for locus in mirna.target_loci:
                    if locus in silencing_factors:
                        silencing_factors[locus] *= mirna.get_silencing(locus)
        
        # Build regulatory state
        regulatory_state = {
            'brain_cortex': self.regulatory_rna.brain_scaling.get('cortex', 1.0),
            'brain_reservoir': self.regulatory_rna.brain_scaling.get('reservoir', 1.0),
            'brain_modulation': self.regulatory_rna.brain_scaling.get('modulation', 1.0),
            'brain_sensory': self.regulatory_rna.brain_scaling.get('sensory', 1.0),
            'brain_motor': self.regulatory_rna.brain_scaling.get('motor', 1.0),
            'segment_repeat': self.regulatory_rna.segment_repeat,
            'symmetry_bias': self.regulatory_rna.symmetry_bias,
            'pigment_pattern': self.regulatory_rna.pigment_pattern,
            'stripe_frequency': self.regulatory_rna.stripe_frequency,
        }
        
        return Transcriptome(
            mrna_levels=mrna_levels,
            active_mirnas=active_mirnas,
            silencing_factors=silencing_factors,
            regulatory_state=regulatory_state,
            tissue=tissue,
            stage=stage
        )
    
    def mutate(self, rate: float = 0.01) -> 'RNASystem':
        """
        Create mutated copy of RNA system.
        
        RNA mutations can have dramatic effects on phenotype
        even without DNA changes.
        """
        new_system = RNASystem()
        
        # Mutate mRNA pool
        for locus, mrna in self.mrna_pool.items():
            new_system.mrna_pool[locus] = mrna.mutate(rate)
        
        # Mutate miRNA pool
        for mirna_id, mirna in self.mirna_pool.items():
            new_system.mirna_pool[mirna_id] = mirna.mutate(rate)
        
        # Occasionally add new miRNA
        if np.random.random() < rate * 0.1:
            new_id = f"mir_new_{np.random.randint(1000)}"
            new_system.mirna_pool[new_id] = MicroRNA(
                id=new_id,
                target_loci=[np.random.randint(1, 100) for _ in range(np.random.randint(1, 5))],
                silencing_strength=np.random.random() * 0.5,
                trigger_condition=np.random.choice(["always", "random", "never"])
            )
        
        # Mutate regulatory RNA
        new_system.regulatory_rna = self.regulatory_rna.mutate(rate)
        
        return new_system
    
    def crossover(self, other: 'RNASystem') -> 'RNASystem':
        """
        Create offspring RNA system from two parents.
        
        RNA inheritance adds another layer of variation
        beyond DNA recombination.
        """
        child = RNASystem()
        
        # Inherit mRNA from random parent per gene
        all_loci = set(self.mrna_pool.keys()) | set(other.mrna_pool.keys())
        for locus in all_loci:
            if locus in self.mrna_pool and locus in other.mrna_pool:
                parent = self if np.random.random() < 0.5 else other
            elif locus in self.mrna_pool:
                parent = self
            else:
                parent = other
            child.mrna_pool[locus] = deepcopy(parent.mrna_pool[locus])
        
        # Inherit miRNAs (can get from either parent)
        all_mirnas = set(self.mirna_pool.keys()) | set(other.mirna_pool.keys())
        for mirna_id in all_mirnas:
            # Each miRNA has chance to be inherited
            if mirna_id in self.mirna_pool and mirna_id in other.mirna_pool:
                if np.random.random() < 0.5:
                    child.mirna_pool[mirna_id] = deepcopy(self.mirna_pool[mirna_id])
                else:
                    child.mirna_pool[mirna_id] = deepcopy(other.mirna_pool[mirna_id])
            elif mirna_id in self.mirna_pool:
                if np.random.random() < 0.7:  # 70% chance to inherit
                    child.mirna_pool[mirna_id] = deepcopy(self.mirna_pool[mirna_id])
            else:
                if np.random.random() < 0.7:
                    child.mirna_pool[mirna_id] = deepcopy(other.mirna_pool[mirna_id])
        
        # Blend regulatory RNA
        child.regulatory_rna = RegulatoryRNA(
            id="main",
            segment_repeat=(self.regulatory_rna.segment_repeat + other.regulatory_rna.segment_repeat) / 2 + np.random.normal(0, 0.05),
            symmetry_bias=(self.regulatory_rna.symmetry_bias + other.regulatory_rna.symmetry_bias) / 2,
            anterior_posterior_gradient=(self.regulatory_rna.anterior_posterior_gradient + other.regulatory_rna.anterior_posterior_gradient) / 2,
            dorsal_ventral_gradient=(self.regulatory_rna.dorsal_ventral_gradient + other.regulatory_rna.dorsal_ventral_gradient) / 2,
            pigment_pattern=(self.regulatory_rna.pigment_pattern + other.regulatory_rna.pigment_pattern) / 2 + np.random.normal(0, 0.05),
            stripe_frequency=self.regulatory_rna.stripe_frequency if np.random.random() < 0.5 else other.regulatory_rna.stripe_frequency,
            color_gradient=self.regulatory_rna.color_gradient if np.random.random() < 0.5 else other.regulatory_rna.color_gradient,
            brain_scaling={
                k: (self.regulatory_rna.brain_scaling.get(k, 1.0) + other.regulatory_rna.brain_scaling.get(k, 1.0)) / 2
                for k in set(self.regulatory_rna.brain_scaling.keys()) | set(other.regulatory_rna.brain_scaling.keys())
            },
            limb_positioning=(self.regulatory_rna.limb_positioning + other.regulatory_rna.limb_positioning) / 2,
            appendage_branching=(self.regulatory_rna.appendage_branching + other.regulatory_rna.appendage_branching) / 2
        )
        
        return child
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'mrna_pool': {
                str(k): {
                    'gene_locus': v.gene_locus,
                    'base_level': v.base_level,
                    'tissue_modifiers': {t.value: m for t, m in v.tissue_modifiers.items()},
                    'stage_modifiers': {s.value: m for s, m in v.stage_modifiers.items()},
                    'half_life': v.half_life,
                    'noise_amplitude': v.noise_amplitude
                }
                for k, v in self.mrna_pool.items()
            },
            'mirna_pool': {
                k: {
                    'id': v.id,
                    'target_loci': v.target_loci,
                    'silencing_strength': v.silencing_strength,
                    'activation_threshold': v.activation_threshold,
                    'trigger_condition': v.trigger_condition,
                    'stage_specific': v.stage_specific.value if v.stage_specific else None,
                    'stress_responsive': v.stress_responsive,
                    'stress_threshold': v.stress_threshold
                }
                for k, v in self.mirna_pool.items()
            },
            'regulatory_rna': {
                'id': self.regulatory_rna.id,
                'segment_repeat': self.regulatory_rna.segment_repeat,
                'symmetry_bias': self.regulatory_rna.symmetry_bias,
                'anterior_posterior_gradient': self.regulatory_rna.anterior_posterior_gradient,
                'dorsal_ventral_gradient': self.regulatory_rna.dorsal_ventral_gradient,
                'pigment_pattern': self.regulatory_rna.pigment_pattern,
                'stripe_frequency': self.regulatory_rna.stripe_frequency,
                'color_gradient': self.regulatory_rna.color_gradient,
                'brain_scaling': self.regulatory_rna.brain_scaling,
                'limb_positioning': self.regulatory_rna.limb_positioning,
                'appendage_branching': self.regulatory_rna.appendage_branching
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RNASystem':
        """Deserialize from dictionary."""
        system = cls()
        
        # Restore mRNA pool
        for locus_str, mrna_data in data.get('mrna_pool', {}).items():
            locus = int(locus_str)
            system.mrna_pool[locus] = MessengerRNA(
                gene_locus=mrna_data['gene_locus'],
                base_level=mrna_data['base_level'],
                tissue_modifiers={
                    TissueType(k): v for k, v in mrna_data.get('tissue_modifiers', {}).items()
                },
                stage_modifiers={
                    DevelopmentalStage(int(k)) if k.isdigit() else DevelopmentalStage[k]: v
                    for k, v in mrna_data.get('stage_modifiers', {}).items()
                },
                half_life=mrna_data.get('half_life', 1.0),
                noise_amplitude=mrna_data.get('noise_amplitude', 0.05)
            )
        
        # Restore miRNA pool
        for mirna_id, mirna_data in data.get('mirna_pool', {}).items():
            stage_val = mirna_data.get('stage_specific')
            system.mirna_pool[mirna_id] = MicroRNA(
                id=mirna_data['id'],
                target_loci=mirna_data['target_loci'],
                silencing_strength=mirna_data['silencing_strength'],
                activation_threshold=mirna_data.get('activation_threshold', 0.5),
                trigger_condition=mirna_data.get('trigger_condition', 'always'),
                stage_specific=DevelopmentalStage(stage_val) if stage_val is not None else None,
                stress_responsive=mirna_data.get('stress_responsive', False),
                stress_threshold=mirna_data.get('stress_threshold', 0.7)
            )
        
        # Restore regulatory RNA
        reg_data = data.get('regulatory_rna', {})
        system.regulatory_rna = RegulatoryRNA(
            id=reg_data.get('id', 'main'),
            segment_repeat=reg_data.get('segment_repeat', 0.5),
            symmetry_bias=reg_data.get('symmetry_bias', 0.5),
            anterior_posterior_gradient=reg_data.get('anterior_posterior_gradient', 0.5),
            dorsal_ventral_gradient=reg_data.get('dorsal_ventral_gradient', 0.5),
            pigment_pattern=reg_data.get('pigment_pattern', 0.5),
            stripe_frequency=reg_data.get('stripe_frequency', 0.3),
            color_gradient=reg_data.get('color_gradient', 0.5),
            brain_scaling=reg_data.get('brain_scaling', {}),
            limb_positioning=reg_data.get('limb_positioning', 0.5),
            appendage_branching=reg_data.get('appendage_branching', 0.3)
        )
        
        return system


# =============================================================================
# INTEGRATED DEVELOPMENT SYSTEM
# =============================================================================

class RNADevelopmentalSystem:
    """
    Complete developmental system using DNA + RNA.
    
    This replaces the DNA-only DevelopmentalSystem with a
    more realistic model that includes RNA-mediated expression.
    """
    
    def __init__(self, genome, rna_system: Optional[RNASystem] = None):
        """
        Initialize with genome and optional RNA system.
        
        If no RNA system provided, one is derived from the genome.
        """
        self.genome = genome
        self.rna = rna_system or RNASystem(genome)
    
    def develop_brain_config(self,
                            stage: DevelopmentalStage = DevelopmentalStage.ADULT,
                            stress: float = 0.0):
        """
        Develop brain configuration using RNA expression.
        
        Unlike DNA-only development, this accounts for:
        - Tissue-specific expression
        - Developmental stage
        - Stress-induced changes
        - miRNA silencing
        - Regulatory scaling
        """
        from .three_system_brain import BrainConfig
        
        # Get transcriptome for brain tissue
        cortex_tx = self.rna.transcribe(TissueType.BRAIN_CORTEX, stage, stress)
        reservoir_tx = self.rna.transcribe(TissueType.BRAIN_RESERVOIR, stage, stress)
        modulation_tx = self.rna.transcribe(TissueType.BRAIN_MODULATION, stage, stress)
        
        # Helper to get expressed gene value
        def expr(locus: int, default: float = 0.5, tissue_tx: Transcriptome = cortex_tx) -> float:
            gene = self.genome.genes.get(locus)
            if gene is None:
                return default
            base = gene.express()
            return tissue_tx.get_expression(locus, base)
        
        # Calculate brain structure with RNA expression
        base_columns = int(50 + 450 * expr(1))
        base_cells = int(8 + 56 * expr(2))
        base_reservoir = int(500 + 4500 * expr(3, tissue_tx=reservoir_tx))
        
        # Apply regulatory RNA scaling
        num_columns = self.rna.regulatory_rna.scale_brain_region('cortex', base_columns)
        cells_per_column = base_cells
        reservoir_size = self.rna.regulatory_rna.scale_brain_region('reservoir', base_reservoir)
        
        # Neuromodulator levels from modulation tissue
        dopamine_baseline = 0.2 + 0.6 * expr(40, tissue_tx=modulation_tx)
        serotonin_baseline = 0.2 + 0.6 * expr(41, tissue_tx=modulation_tx)
        norepinephrine_baseline = 0.1 + 0.5 * expr(42, tissue_tx=modulation_tx)
        acetylcholine_baseline = 0.2 + 0.6 * expr(43, tissue_tx=modulation_tx)
        
        # Learning and plasticity
        plasticity = expr(50)
        neurogenesis_rate = 0.1 + 0.3 * plasticity
        
        # Decay rates
        decay_factor = 0.5 + expr(45, tissue_tx=modulation_tx)
        
        config = BrainConfig(
            num_columns=num_columns,
            cells_per_column=cells_per_column,
            reservoir_size=reservoir_size,
            target_sparsity=0.01 + 0.19 * expr(4),
            spectral_radius=0.5 + 0.5 * expr(10, tissue_tx=reservoir_tx),
            reservoir_sparsity=0.05 + 0.25 * expr(11, tissue_tx=reservoir_tx),
            leak_rate=0.1 + 0.5 * expr(12, tissue_tx=reservoir_tx),
            lateral_inhibition_strength=0.1 + 0.5 * expr(5),
            dopamine_baseline=dopamine_baseline,
            serotonin_baseline=serotonin_baseline,
            norepinephrine_baseline=norepinephrine_baseline,
            acetylcholine_baseline=acetylcholine_baseline,
            dopamine_decay=0.03 + 0.07 * decay_factor,
            serotonin_decay=0.02 + 0.04 * decay_factor,
            norepinephrine_decay=0.05 + 0.1 * decay_factor,
            acetylcholine_decay=0.04 + 0.08 * decay_factor,
            ach_sparsity_gain=0.3 + 0.6 * expr(43, tissue_tx=modulation_tx),
            da_gain_effect=0.2 + 0.5 * expr(44, tissue_tx=modulation_tx),
            serotonin_plasticity_effect=0.2 + 0.5 * expr(46, tissue_tx=modulation_tx),
            neurogenesis_rate=neurogenesis_rate,
            dream_enabled=True,
            dream_steps=int(3 + 7 * expr(30)),
        )
        
        return config
    
    def develop_body_params(self,
                           stage: DevelopmentalStage = DevelopmentalStage.ADULT) -> Dict[str, float]:
        """Develop body parameters with RNA expression."""
        # Get transcriptomes for relevant tissues
        muscle_tx = self.rna.transcribe(TissueType.MUSCULATURE, stage)
        skin_tx = self.rna.transcribe(TissueType.SKIN_PIGMENT, stage)
        sensory_tx = self.rna.transcribe(TissueType.SENSORY_VISUAL, stage)
        meta_tx = self.rna.transcribe(TissueType.METABOLISM, stage)
        
        def expr(locus: int, default: float = 0.5, tx: Transcriptome = muscle_tx) -> float:
            gene = self.genome.genes.get(locus)
            if gene is None:
                return default
            return tx.get_expression(locus, gene.express())
        
        reg = self.rna.regulatory_rna
        
        return {
            # Physical attributes
            'size': 0.5 + expr(80, tx=muscle_tx),
            'speed': 0.3 + 0.7 * expr(81, tx=muscle_tx),
            'strength': 0.3 + 0.7 * expr(82, tx=muscle_tx),
            'limb_count': 2 + int(6 * reg.limb_positioning),
            
            # Appearance (from regulatory RNA)
            'color_hue': expr(83, tx=skin_tx),
            'color_saturation': 0.3 + 0.7 * expr(84, tx=skin_tx),
            'pattern': reg.pigment_pattern,
            'pattern_type': reg.get_pattern_type(),
            'stripe_frequency': reg.stripe_frequency,
            'symmetry': reg.get_symmetry_type(),
            'segments': reg.get_segment_count(),
            
            # Metabolism
            'metabolic_rate': 0.5 + expr(90, tx=meta_tx),
            'max_energy': 50 + 100 * expr(91, tx=meta_tx),
            'healing_rate': 0.3 + 0.7 * expr(92, tx=meta_tx),
            'fatigue_resistance': 0.3 + 0.7 * expr(93, tx=meta_tx),
            
            # Senses
            'vision_range': 5 + 15 * expr(100, tx=sensory_tx),
            'hearing_range': 3 + 10 * expr(101, tx=sensory_tx),
            'smell_range': 2 + 8 * expr(102, tx=sensory_tx),
            
            # Lifespan
            'max_age': int(100 + 400 * expr(110, tx=meta_tx)),
            'fertility': 0.3 + 0.7 * expr(111),
            'maturation_age': int(5 + 30 * expr(112)),
        }
    
    def develop_drives(self,
                      stage: DevelopmentalStage = DevelopmentalStage.ADULT) -> Dict[str, float]:
        """Develop instinctual drives with RNA expression."""
        mod_tx = self.rna.transcribe(TissueType.BRAIN_MODULATION, stage)
        
        def expr(locus: int, default: float = 0.5) -> float:
            gene = self.genome.genes.get(locus)
            if gene is None:
                return default
            return mod_tx.get_expression(locus, gene.express())
        
        return {
            'hunger': 0.3 + 0.4 * expr(60),
            'curiosity': 0.2 + 0.6 * expr(61),
            'fear': 0.2 + 0.6 * expr(62),
            'aggression': 0.1 + 0.5 * expr(63),
            'social': 0.2 + 0.6 * expr(64),
            'sleep': 0.3 + 0.4 * expr(65),
            'pain': 0.3 + 0.5 * expr(66),
            'reward': 0.3 + 0.5 * expr(67),
            'territorial': 0.1 + 0.5 * expr(68),
            'bonding': 0.2 + 0.6 * expr(69),
            'empathy': 0.2 + 0.6 * expr(70),
            'dominance': 0.2 + 0.6 * expr(71),
            'cooperation': 0.2 + 0.6 * expr(72),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_organism_with_rna(genome, 
                             rna_system: Optional[RNASystem] = None,
                             stage: DevelopmentalStage = DevelopmentalStage.ADULT,
                             stress: float = 0.0):
    """
    Create a complete organism from DNA + RNA.
    
    This is the RNA-aware version of create_organism_from_dna.
    
    Returns:
        Tuple of (brain, body_params, drives, rna_system)
    """
    from .three_system_brain import ThreeSystemBrain
    
    rna = rna_system or RNASystem(genome)
    dev = RNADevelopmentalSystem(genome, rna)
    
    config = dev.develop_brain_config(stage, stress)
    body = dev.develop_body_params(stage)
    drives = dev.develop_drives(stage)
    
    brain = ThreeSystemBrain(config)
    brain.genome = genome
    brain.rna_system = rna
    brain.drives = drives
    brain.body_params = body
    brain.developmental_stage = stage
    
    return brain, body, drives, rna


def breed_with_rna(parent1, parent2, mutation_rate: float = 1.0):
    """
    Breed two organisms with full RNA inheritance.
    
    Both DNA and RNA are recombined and mutated.
    
    Returns:
        Tuple of (child_brain, body_params, drives, rna_system)
    """
    from .dna import breed as dna_breed
    
    # Breed DNA
    child_genome = dna_breed(parent1.genome, parent2.genome, mutation_rate=mutation_rate)
    
    # Breed RNA (if both parents have RNA systems)
    if hasattr(parent1, 'rna_system') and hasattr(parent2, 'rna_system'):
        child_rna = parent1.rna_system.crossover(parent2.rna_system)
        child_rna = child_rna.mutate(0.01 * mutation_rate)
    else:
        child_rna = RNASystem(child_genome)
    
    return create_organism_with_rna(child_genome, child_rna)


# =============================================================================
# VIRAL INFECTION SYSTEM
# =============================================================================

class VirusType(Enum):
    """Types of viral infections with different effects."""
    RETROVIRUS = "retrovirus"          # Integrates into genome, permanent changes
    RNA_VIRUS = "rna_virus"            # Hijacks transcription temporarily
    PRION = "prion"                    # Affects protein folding/brain
    LYSOGENIC = "lysogenic"            # Dormant, activates under stress


@dataclass
class Virus:
    """
    A virus that can infect creatures and affect their RNA.
    
    Viruses can:
    - Alter gene expression levels
    - Insert new sequences
    - Spread between creatures
    - Mutate as they spread
    """
    id: str
    name: str
    virus_type: VirusType
    
    # Infection dynamics
    infectivity: float = 0.3          # Chance to spread on contact
    virulence: float = 0.5            # How harmful (0=benign, 1=deadly)
    incubation_time: float = 100.0    # Time before symptoms
    
    # Effects on RNA
    target_genes: List[str] = field(default_factory=list)  # Genes affected
    expression_modifier: float = 0.5   # Multiply expression by this
    
    # Special effects
    brain_effect: float = 0.0         # Affects cognition (-1 to 1)
    metabolism_effect: float = 0.0    # Affects metabolism (-1 to 1)
    behavior_effect: str = "none"     # aggression, lethargy, social
    
    # Mutation
    mutation_rate: float = 0.05       # How fast virus mutates when spreading
    
    # Payload (for retroviruses)
    inserted_sequence: Optional[Dict[str, float]] = None
    
    def mutate(self) -> 'Virus':
        """Create mutated copy of virus."""
        mutated = Virus(
            id=f"{self.id}_m{np.random.randint(1000)}",
            name=self.name,
            virus_type=self.virus_type,
            infectivity=np.clip(self.infectivity + np.random.normal(0, 0.05), 0.01, 0.99),
            virulence=np.clip(self.virulence + np.random.normal(0, 0.05), 0, 1),
            incubation_time=max(10, self.incubation_time + np.random.normal(0, 20)),
            target_genes=self.target_genes.copy(),
            expression_modifier=np.clip(self.expression_modifier + np.random.normal(0, 0.1), 0.1, 2.0),
            brain_effect=np.clip(self.brain_effect + np.random.normal(0, 0.1), -1, 1),
            metabolism_effect=np.clip(self.metabolism_effect + np.random.normal(0, 0.1), -1, 1),
            behavior_effect=self.behavior_effect,
            mutation_rate=self.mutation_rate,
            inserted_sequence=self.inserted_sequence
        )
        
        # Occasionally change target genes
        if np.random.random() < self.mutation_rate:
            possible_genes = ['dopamine_baseline', 'serotonin_baseline', 'cortical_columns',
                            'metabolic_rate', 'aggression', 'social', 'fear']
            if np.random.random() < 0.3 and possible_genes:
                mutated.target_genes.append(np.random.choice(possible_genes))
        
        return mutated


@dataclass 
class Infection:
    """An active infection in a creature."""
    virus: Virus
    infection_time: float = 0.0       # When infected
    current_time: float = 0.0         # Current simulation time
    viral_load: float = 0.1           # 0-1, how much virus is present
    immune_response: float = 0.0      # 0-1, immune system fighting
    is_symptomatic: bool = False
    is_cured: bool = False
    
    @property
    def time_infected(self) -> float:
        return self.current_time - self.infection_time
    
    def update(self, dt: float, immune_strength: float = 0.5):
        """Update infection state."""
        self.current_time += dt
        
        # Incubation check
        if self.time_infected > self.virus.incubation_time:
            self.is_symptomatic = True
        
        # Viral load dynamics
        if not self.is_cured:
            # Virus replicates
            growth = 0.1 * self.viral_load * (1 - self.viral_load) * dt
            # Immune system fights
            immune_fight = immune_strength * self.immune_response * self.viral_load * dt
            
            self.viral_load = np.clip(self.viral_load + growth - immune_fight, 0, 1)
            
            # Immune response builds over time
            self.immune_response = min(1.0, self.immune_response + 0.01 * dt)
            
            # Check if cured
            if self.viral_load < 0.01:
                self.is_cured = True
    
    def get_rna_effects(self) -> Dict[str, float]:
        """Get current effects on gene expression."""
        if self.is_cured or not self.is_symptomatic:
            return {}
        
        effects = {}
        severity = self.viral_load * self.virus.virulence
        
        for gene in self.virus.target_genes:
            # Modify expression based on viral load
            modifier = 1.0 + (self.virus.expression_modifier - 1.0) * severity
            effects[gene] = modifier
        
        return effects


class ImmuneSystem:
    """
    Simple immune system that fights infections.
    """
    
    def __init__(self, strength: float = 0.5):
        self.base_strength = strength
        self.current_strength = strength
        self.infections: List[Infection] = []
        self.immunity: Dict[str, float] = {}  # Virus ID -> immunity level
        self.fatigue = 0.0
        
    def infect(self, virus: Virus, current_time: float) -> bool:
        """
        Attempt to infect with a virus.
        
        Returns True if infection successful.
        """
        # Check existing immunity
        immunity = self.immunity.get(virus.id, 0.0)
        if np.random.random() < immunity:
            return False  # Immune, blocked
        
        # Check if already infected with this virus
        for inf in self.infections:
            if inf.virus.id == virus.id and not inf.is_cured:
                return False  # Already infected
        
        # Infection chance based on infectivity and current immune state
        infection_chance = virus.infectivity * (1 - self.current_strength * 0.5)
        if np.random.random() < infection_chance:
            self.infections.append(Infection(
                virus=virus,
                infection_time=current_time,
                current_time=current_time
            ))
            return True
        
        return False
    
    def update(self, dt: float):
        """Update all infections and immune state."""
        # Update each infection
        for infection in self.infections:
            infection.update(dt, self.current_strength)
            
            # Build immunity when cured
            if infection.is_cured:
                self.immunity[infection.virus.id] = min(1.0, 
                    self.immunity.get(infection.virus.id, 0) + 0.3)
        
        # Remove cured infections after immunity built
        self.infections = [i for i in self.infections 
                         if not i.is_cured or i.time_infected < 50]
        
        # Immune fatigue from fighting infections
        active_infections = sum(1 for i in self.infections if not i.is_cured)
        if active_infections > 0:
            self.fatigue = min(0.5, self.fatigue + 0.01 * active_infections * dt)
        else:
            self.fatigue = max(0, self.fatigue - 0.005 * dt)
        
        self.current_strength = self.base_strength * (1 - self.fatigue)
    
    def get_combined_rna_effects(self) -> Dict[str, float]:
        """Get combined RNA effects from all active infections."""
        combined = {}
        for infection in self.infections:
            effects = infection.get_rna_effects()
            for gene, modifier in effects.items():
                if gene in combined:
                    combined[gene] *= modifier  # Multiplicative
                else:
                    combined[gene] = modifier
        return combined
    
    def can_spread_to(self, other: 'ImmuneSystem', contact_type: str = 'proximity') -> List[Virus]:
        """
        Check which viruses can spread to another creature.
        
        Returns list of viruses that could spread.
        """
        spreadable = []
        contact_multiplier = {'proximity': 0.3, 'touch': 0.7, 'bite': 1.0}.get(contact_type, 0.5)
        
        for infection in self.infections:
            if infection.is_cured or not infection.is_symptomatic:
                continue
            if infection.viral_load < 0.2:
                continue
            
            spread_chance = infection.virus.infectivity * infection.viral_load * contact_multiplier
            if np.random.random() < spread_chance:
                # Mutate as it spreads
                if np.random.random() < infection.virus.mutation_rate:
                    spreadable.append(infection.virus.mutate())
                else:
                    spreadable.append(infection.virus)
        
        return spreadable
    
    def to_dict(self) -> Dict:
        """Serialize immune system state."""
        return {
            'base_strength': self.base_strength,
            'current_strength': self.current_strength,
            'fatigue': self.fatigue,
            'immunity': self.immunity,
            'active_infections': len([i for i in self.infections if not i.is_cured])
        }


# Common virus templates
COMMON_VIRUSES = {
    'cold': Virus(
        id='cold_v1', name='Common Cold', virus_type=VirusType.RNA_VIRUS,
        infectivity=0.4, virulence=0.1, incubation_time=50,
        target_genes=['metabolic_rate'], expression_modifier=0.8,
        metabolism_effect=-0.2
    ),
    'rage': Virus(
        id='rage_v1', name='Rage Virus', virus_type=VirusType.PRION,
        infectivity=0.2, virulence=0.7, incubation_time=200,
        target_genes=['serotonin_baseline', 'dopamine_baseline'],
        expression_modifier=0.4, brain_effect=-0.5, behavior_effect='aggression'
    ),
    'social': Virus(
        id='social_v1', name='Bonding Virus', virus_type=VirusType.RNA_VIRUS,
        infectivity=0.5, virulence=0.05, incubation_time=100,
        target_genes=['oxytocin_baseline', 'social'], expression_modifier=1.5,
        behavior_effect='social'
    ),
    'retro_memory': Virus(
        id='retro_mem_v1', name='Memory Retrovirus', virus_type=VirusType.RETROVIRUS,
        infectivity=0.1, virulence=0.3, incubation_time=500,
        target_genes=['plasticity', 'acetylcholine_baseline'],
        expression_modifier=1.3, brain_effect=0.2,
        inserted_sequence={'memory_boost': 0.15}
    ),
}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'TissueType',
    'DevelopmentalStage',
    'RNAType',
    'VirusType',
    
    # RNA classes
    'MessengerRNA',
    'MicroRNA', 
    'RegulatoryRNA',
    'Transcriptome',
    'RNASystem',
    
    # Virus classes
    'Virus',
    'Infection',
    'ImmuneSystem',
    'COMMON_VIRUSES',
    
    # Development
    'RNADevelopmentalSystem',
    
    # Functions
    'create_organism_with_rna',
    'breed_with_rna',
]
