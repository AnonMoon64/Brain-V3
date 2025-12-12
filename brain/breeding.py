"""
Breeding System - Sexual reproduction with DNA/RNA inheritance

This module provides:
- Fertility cycles and pregnancy
- Mating behavior and partner selection
- Offspring creation with genetic inheritance
- Epigenetic marks and RNA inheritance
- Mutation and variation

Breeding is the core mechanism that allows evolution to occur.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

# Import genetics
from .dna import Genome, breed as dna_breed, GeneLibrary
from .rna import RNASystem, DevelopmentalStage
from .creature import CreatureBody, Phenotype, Homeostasis


# =============================================================================
# FERTILITY AND REPRODUCTION STATE
# =============================================================================

class ReproductiveState(Enum):
    """Reproductive cycle state."""
    IMMATURE = 0        # Too young
    FERTILE = 1         # Can mate
    PREGNANT = 2        # Carrying offspring
    NURSING = 3         # Caring for young
    INFERTILE = 4       # Too old or damaged


@dataclass
class ReproductiveSystem:
    """
    Manages reproduction for a creature.
    
    Tracks fertility cycles, pregnancy, and offspring.
    """
    # Current state
    state: ReproductiveState = ReproductiveState.IMMATURE
    
    # Fertility
    base_fertility: float = 0.5         # From DNA
    current_fertility: float = 0.0       # Current level
    cycle_phase: float = 0.0            # 0-1 through cycle
    cycle_length: float = 100.0         # Time units per cycle
    
    # Pregnancy
    pregnancy_progress: float = 0.0     # 0-1
    gestation_period: float = 200.0     # Time to birth
    offspring_count: int = 1            # Number of offspring
    partner_genome: Optional[Genome] = field(default=None, repr=False)
    partner_rna: Optional[RNASystem] = field(default=None, repr=False)
    
    # Maturation
    maturation_age: float = 50.0        # Age to become fertile
    menopause_age: float = 500.0        # Age to become infertile
    
    # Cooldowns
    mating_cooldown: float = 0.0
    mating_cooldown_max: float = 50.0
    
    # Lifetime stats
    total_offspring: int = 0
    total_matings: int = 0
    
    def update(self, dt: float, age: float, energy: float, health: float):
        """
        Update reproductive state.
        
        Args:
            dt: Time delta
            age: Creature's current age
            energy: Current energy level
            health: Current health level
        """
        # Update cooldowns
        self.mating_cooldown = max(0, self.mating_cooldown - dt)
        
        # State transitions based on age
        if age < self.maturation_age:
            self.state = ReproductiveState.IMMATURE
            self.current_fertility = 0
            return
        
        if age > self.menopause_age:
            self.state = ReproductiveState.INFERTILE
            self.current_fertility = 0
            return
        
        # Pregnancy progress
        if self.state == ReproductiveState.PREGNANT:
            if energy > 0.2 and health > 0.3:
                self.pregnancy_progress += dt / self.gestation_period
                
                if self.pregnancy_progress >= 1.0:
                    # Ready to give birth
                    pass  # Handled externally
            return
        
        # Fertility cycle
        self.cycle_phase = (self.cycle_phase + dt / self.cycle_length) % 1.0
        
        # Fertility peaks in middle of cycle
        cycle_fertility = np.sin(self.cycle_phase * np.pi) ** 2
        
        # Modify by health and energy
        health_mod = 0.5 + 0.5 * health
        energy_mod = 0.5 + 0.5 * min(1, energy * 2)
        
        self.current_fertility = self.base_fertility * cycle_fertility * health_mod * energy_mod
        
        if self.current_fertility > 0.1 and self.state != ReproductiveState.NURSING:
            self.state = ReproductiveState.FERTILE
        else:
            self.state = ReproductiveState.IMMATURE
    
    def can_mate(self) -> bool:
        """Check if creature can mate right now."""
        return (
            self.state == ReproductiveState.FERTILE and
            self.current_fertility > 0.2 and
            self.mating_cooldown <= 0
        )
    
    def initiate_pregnancy(self, partner_genome: Genome, partner_rna: Optional[RNASystem] = None):
        """Start pregnancy with partner's genetic material."""
        self.state = ReproductiveState.PREGNANT
        self.pregnancy_progress = 0.0
        self.partner_genome = partner_genome
        self.partner_rna = partner_rna
        self.mating_cooldown = self.mating_cooldown_max
        self.total_matings += 1
        
        # Determine offspring count (usually 1, sometimes more)
        if np.random.random() < 0.1:
            self.offspring_count = 2  # Twins
        else:
            self.offspring_count = 1
    
    def give_birth(self, own_genome: Genome, own_rna: Optional[RNASystem] = None) -> List[Tuple[Genome, Optional[RNASystem]]]:
        """
        Give birth to offspring.
        
        Returns:
            List of (genome, rna_system) tuples for each offspring
        """
        if self.state != ReproductiveState.PREGNANT or self.pregnancy_progress < 1.0:
            return []
        
        if self.partner_genome is None:
            return []
        
        offspring = []
        for _ in range(self.offspring_count):
            # Breed DNA
            child_genome = dna_breed(own_genome, self.partner_genome)
            
            # Breed RNA if available
            child_rna = None
            if own_rna and self.partner_rna:
                child_rna = own_rna.crossover(self.partner_rna)
                child_rna = child_rna.mutate(0.02)  # RNA mutates faster
            elif own_rna:
                child_rna = own_rna.mutate(0.05)
            
            offspring.append((child_genome, child_rna))
        
        # Reset state
        self.state = ReproductiveState.NURSING
        self.pregnancy_progress = 0.0
        self.partner_genome = None
        self.partner_rna = None
        self.total_offspring += self.offspring_count
        
        return offspring
    
    def finish_nursing(self):
        """Return to normal fertile state."""
        self.state = ReproductiveState.FERTILE
    
    def get_mating_attractiveness(self) -> float:
        """
        Calculate how attractive this creature is as a mate.
        
        Based on fertility, health indicators, etc.
        """
        if self.state != ReproductiveState.FERTILE:
            return 0.0
        
        return self.current_fertility


# =============================================================================
# EPIGENETIC MARKS
# =============================================================================

@dataclass
class EpigeneticMark:
    """
    An epigenetic modification that affects gene expression.
    
    These can be acquired during life and passed to offspring.
    """
    gene_locus: int
    expression_modifier: float      # Multiplier on gene expression
    cause: str                      # What caused this mark
    strength: float = 1.0           # How strong the mark is
    heritability: float = 0.5       # Chance to pass to offspring
    decay_rate: float = 0.001       # How fast it fades
    
    def update(self, dt: float):
        """Decay mark over time."""
        self.strength -= self.decay_rate * dt
        return self.strength > 0
    
    def apply_to_expression(self, base_expression: float) -> float:
        """Modify gene expression level."""
        return base_expression * (1 + (self.expression_modifier - 1) * self.strength)


class EpigeneticSystem:
    """
    Manages epigenetic marks for a creature.
    
    Epigenetic marks are modifications to gene expression caused by
    environmental factors that can be inherited.
    """
    
    def __init__(self):
        self.marks: List[EpigeneticMark] = []
    
    def add_mark(self, locus: int, modifier: float, cause: str, heritability: float = 0.5):
        """Add a new epigenetic mark."""
        # Check if already marked at this locus
        for mark in self.marks:
            if mark.gene_locus == locus:
                # Strengthen existing mark
                mark.strength = min(1.0, mark.strength + 0.2)
                mark.expression_modifier = (mark.expression_modifier + modifier) / 2
                return
        
        self.marks.append(EpigeneticMark(
            gene_locus=locus,
            expression_modifier=modifier,
            cause=cause,
            strength=0.5,
            heritability=heritability
        ))
    
    def update(self, dt: float):
        """Update all marks, removing faded ones."""
        self.marks = [m for m in self.marks if m.update(dt)]
    
    def apply_stress_response(self, stress_level: float):
        """
        Apply stress-induced epigenetic changes.
        
        High stress causes marks that:
        - Reduce growth genes
        - Increase fear/alertness
        - Reduce exploration
        """
        if stress_level > 0.7:
            # Stress marks growth-related genes
            for locus in [1, 2, 3]:  # Cortical structure genes
                self.add_mark(locus, 0.8, "stress", heritability=0.3)
            
            # Enhance fear genes
            self.add_mark(62, 1.3, "stress", heritability=0.4)  # Fear drive
    
    def apply_abundance_response(self, nutrition_level: float):
        """
        Apply abundance-induced epigenetic changes.
        
        High nutrition causes marks that:
        - Increase growth genes
        - Increase exploration
        """
        if nutrition_level > 0.8:
            for locus in [1, 2, 3]:
                self.add_mark(locus, 1.2, "abundance", heritability=0.2)
            
            self.add_mark(61, 1.2, "abundance", heritability=0.2)  # Curiosity
    
    def get_heritable_marks(self) -> List[EpigeneticMark]:
        """Get marks that can be passed to offspring."""
        heritable = []
        for mark in self.marks:
            if np.random.random() < mark.heritability:
                # Create weakened copy for offspring
                child_mark = EpigeneticMark(
                    gene_locus=mark.gene_locus,
                    expression_modifier=mark.expression_modifier,
                    cause=f"inherited_{mark.cause}",
                    strength=mark.strength * 0.5,
                    heritability=mark.heritability * 0.7,  # Decreasing heritability
                    decay_rate=mark.decay_rate * 1.5  # Faster decay in offspring
                )
                heritable.append(child_mark)
        return heritable
    
    def inherit_marks(self, parent_marks: List[EpigeneticMark]):
        """Inherit marks from parent."""
        for mark in parent_marks:
            self.marks.append(mark)
    
    def get_expression_modifier(self, locus: int) -> float:
        """Get combined expression modifier for a gene locus."""
        modifier = 1.0
        for mark in self.marks:
            if mark.gene_locus == locus:
                modifier = mark.apply_to_expression(modifier)
        return modifier


# =============================================================================
# MATE SELECTION
# =============================================================================

def calculate_mate_compatibility(creature1: CreatureBody, creature2: CreatureBody) -> float:
    """
    Calculate mating compatibility between two creatures.
    
    Based on:
    - Similar species (hue)
    - Complementary traits
    - Both fertile
    """
    p1 = creature1.phenotype
    p2 = creature2.phenotype
    
    # Must be similar species (hue within range)
    hue_diff = abs(p1.hue - p2.hue)
    if hue_diff > 0.15:  # Different species
        return 0.0
    
    species_similarity = 1 - (hue_diff / 0.15)
    
    # Size compatibility (prefer similar but not identical)
    size_diff = abs(p1.size - p2.size)
    size_compat = 1 - min(1, size_diff)
    
    # Health indicators
    health1 = creature1.homeostasis.health
    health2 = creature2.homeostasis.health
    health_factor = (health1 + health2) / 2
    
    return species_similarity * 0.5 + size_compat * 0.2 + health_factor * 0.3


def select_mate(creature: CreatureBody, candidates: List[CreatureBody]) -> Optional[CreatureBody]:
    """
    Select best mate from available candidates.
    
    Returns:
        Best compatible mate, or None if no suitable candidates
    """
    best_mate = None
    best_score = 0.0
    
    for candidate in candidates:
        if candidate == creature:
            continue
        
        score = calculate_mate_compatibility(creature, candidate)
        if score > best_score and score > 0.3:  # Minimum compatibility threshold
            best_score = score
            best_mate = candidate
    
    return best_mate


# =============================================================================
# OFFSPRING CREATION
# =============================================================================

def create_offspring(parent1_genome: Genome, parent2_genome: Genome,
                     parent1_rna: Optional[RNASystem] = None,
                     parent2_rna: Optional[RNASystem] = None,
                     parent1_epigenetics: Optional[EpigeneticSystem] = None,
                     parent2_epigenetics: Optional[EpigeneticSystem] = None,
                     x: float = 0, y: float = 0) -> Tuple[CreatureBody, Genome, Optional[RNASystem], EpigeneticSystem]:
    """
    Create a complete offspring from two parents.
    
    Returns:
        Tuple of (body, genome, rna_system, epigenetics)
    """
    # Breed DNA
    child_genome = dna_breed(parent1_genome, parent2_genome)
    
    # Breed RNA
    child_rna = None
    if parent1_rna and parent2_rna:
        child_rna = parent1_rna.crossover(parent2_rna)
        child_rna = child_rna.mutate(0.015)
    elif parent1_rna:
        child_rna = parent1_rna.mutate(0.03)
    elif parent2_rna:
        child_rna = parent2_rna.mutate(0.03)
    
    # Create epigenetic system with inherited marks
    child_epigenetics = EpigeneticSystem()
    if parent1_epigenetics:
        child_epigenetics.inherit_marks(parent1_epigenetics.get_heritable_marks())
    if parent2_epigenetics:
        child_epigenetics.inherit_marks(parent2_epigenetics.get_heritable_marks())
    
    # Develop phenotype from genetics
    if child_rna:
        from .rna import RNADevelopmentalSystem
        dev = RNADevelopmentalSystem(child_genome, child_rna)
        body_params = dev.develop_body_params(DevelopmentalStage.EMBRYONIC)
    else:
        # Fallback to DNA-only development
        body_params = _develop_body_from_genome(child_genome)
    
    # Apply epigenetic modifiers
    for locus in body_params:
        modifier = child_epigenetics.get_expression_modifier(hash(locus) % 100)
        if isinstance(body_params[locus], (int, float)):
            body_params[locus] *= modifier
    
    # Create phenotype
    phenotype = Phenotype.from_body_params(body_params)
    
    # Create body
    body = CreatureBody(phenotype=phenotype, x=x, y=y)
    
    # Start as baby (reduced stats)
    body.homeostasis.age = 0.0
    body.homeostasis.energy = 0.5
    body.homeostasis.health = 1.0
    
    return body, child_genome, child_rna, child_epigenetics


def _develop_body_from_genome(genome: Genome) -> Dict[str, float]:
    """Fallback body development from genome only."""
    def gene_val(locus: int, default: float = 0.5) -> float:
        gene = genome.genes.get(locus)
        if gene:
            return gene.express()
        return default
    
    return {
        'size': 0.5 + gene_val(80),
        'speed': 0.3 + 0.7 * gene_val(81),
        'strength': 0.3 + 0.7 * gene_val(82),
        'limb_count': 4,
        'color_hue': gene_val(83),
        'color_saturation': 0.3 + 0.7 * gene_val(84),
        'pattern': gene_val(85),
        'pattern_type': 'solid',
        'metabolic_rate': 0.5 + gene_val(90),
        'max_energy': 50 + 100 * gene_val(91),
        'vision_range': 5 + 15 * gene_val(100),
        'hearing_range': 3 + 10 * gene_val(101),
        'smell_range': 2 + 8 * gene_val(102),
    }



# =============================================================================
# NEURAL STRUCTURE BREEDING (SYSTEM 1)
# =============================================================================

def breed_neural_structure(parent1_brain: Any, parent2_brain: Optional[Any] = None) -> Dict[str, Any]:
    """
    Combine structural snapshots from parent brains (Instinct Inheritance).
    
    Extracts structural primitives (compression engine patterns) from parents
    and combines them to form the innate structure of the offspring.
    
    Args:
        parent1_brain: The first parent's brain (ThreeSystemBrain)
        parent2_brain: The second parent's brain (optional)
        
    Returns:
        Structural snapshot to load into offspring brain
    """
    snapshot = {
        'version': '1.0',
        'compression_engine': {},
    }
    
    # helper to get primitives
    def get_prims(brain):
        if hasattr(brain, 'get_structural_snapshot'):
            try:
                s = brain.get_structural_snapshot()
                return s.get('compression_engine', {}).get('cortical_primitives', [])
            except Exception as e:
                print(f"Error getting snapshot: {e}")
                return []
        return []

    prims1 = get_prims(parent1_brain)
    prims2 = get_prims(parent2_brain) if parent2_brain else []
    
    # Combine primitives
    # Strategy: Take 50% from each parent + small mutation chance
    combined_prims = []
    
    # If single parent (asexual/fallback), take mostly from valid parent
    if not prims2:
        combined_prims = [p.copy() for p in prims1]
    else:
        # Sexual reproduction - mix and match
        # Limit total primitives to prevent bloating (e.g. max 100 inherited)
        max_inherited = 100
        
        # Shuffle both lists (using numpy if available, else generic logic)
        # Note: numpy is imported as np at top of file
        p1_idxs = np.random.permutation(len(prims1))
        p2_idxs = np.random.permutation(len(prims2))
        
        # Take from parent 1
        for i in range(min(len(prims1), max_inherited // 2)):
            combined_prims.append(prims1[p1_idxs[i]].copy())
            
        # Take from parent 2
        for i in range(min(len(prims2), max_inherited // 2)):
            combined_prims.append(prims2[p2_idxs[i]].copy())
            
    snapshot['compression_engine']['cortical_primitives'] = combined_prims
    
    return snapshot


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ReproductiveState',
    'ReproductiveSystem',
    'EpigeneticMark',
    'EpigeneticSystem',
    'calculate_mate_compatibility',
    'select_mate',
    'create_offspring',
    'breed_neural_structure',
]
