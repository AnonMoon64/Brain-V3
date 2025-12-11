"""
Microbiome Simulation

The gut-brain axis: bacteria in the digestive system affect:
- Mood (serotonin production)
- Metabolism (nutrient extraction)
- Immune function
- Behavior (cravings, social tendencies)

Microbiome composition changes based on:
- Diet
- Stress
- Social contact
- Environment
- Antibiotics/illness
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum, auto
from collections import defaultdict


class BacteriaType(Enum):
    """Major bacterial phyla in the gut."""
    FIRMICUTES = "firmicutes"           # Energy extraction, obesity link
    BACTEROIDETES = "bacteroidetes"     # Fiber digestion, lean phenotype
    ACTINOBACTERIA = "actinobacteria"   # Immune modulation
    PROTEOBACTERIA = "proteobacteria"   # Inflammation (too much = bad)
    LACTOBACILLUS = "lactobacillus"     # Mood, GABA production
    BIFIDOBACTERIUM = "bifidobacterium" # Serotonin precursors


@dataclass
class BacterialStrain:
    """A specific bacterial strain with effects on the host."""
    id: str
    bacteria_type: BacteriaType
    name: str
    
    # Population dynamics
    growth_rate: float = 0.1          # How fast it multiplies
    death_rate: float = 0.05          # Natural death rate
    competition_factor: float = 0.5    # How much it competes with others
    
    # Diet preferences (what it feeds on)
    preferred_nutrients: List[str] = field(default_factory=lambda: ['fiber'])
    
    # Effects on host
    serotonin_effect: float = 0.0     # -1 to 1
    dopamine_effect: float = 0.0
    gaba_effect: float = 0.0
    metabolism_effect: float = 0.0    # Positive = better nutrient extraction
    inflammation_effect: float = 0.0  # Positive = more inflammation
    immune_effect: float = 0.0        # Positive = stronger immune
    
    # Behavioral effects
    social_tendency: float = 0.0      # More social behavior
    anxiety_effect: float = 0.0       # Positive = more anxiety
    appetite_effect: float = 0.0      # Positive = more hunger


class Microbiome:
    """
    The microbial community living in a creature's gut.
    """
    
    def __init__(self, host_id: str = "unknown"):
        self.host_id = host_id
        
        # Bacterial populations (strain_id -> population 0-1)
        self.populations: Dict[str, float] = {}
        self.strains: Dict[str, BacterialStrain] = {}
        
        # Initialize with default flora
        self._init_default_flora()
        
        # Environmental factors
        self.ph_level: float = 6.5     # Gut pH
        self.nutrient_availability: Dict[str, float] = {
            'fiber': 0.5,
            'sugar': 0.3,
            'protein': 0.4,
            'fat': 0.3,
        }
        
        # Stress affects microbiome
        self.stress_level: float = 0.0
        
        # Antibiotic exposure (kills bacteria non-selectively)
        self.antibiotic_exposure: float = 0.0
        
        # Diversity metrics
        self._last_diversity = 0.0
        
    def _init_default_flora(self):
        """Initialize with baseline healthy bacteria."""
        default_strains = [
            BacterialStrain(
                id='firm_1', bacteria_type=BacteriaType.FIRMICUTES,
                name='Firmicutes Alpha', growth_rate=0.12, 
                preferred_nutrients=['fiber', 'sugar'],
                metabolism_effect=0.3, serotonin_effect=0.1
            ),
            BacterialStrain(
                id='bact_1', bacteria_type=BacteriaType.BACTEROIDETES,
                name='Bacteroidetes Beta', growth_rate=0.1,
                preferred_nutrients=['fiber'],
                metabolism_effect=0.2, inflammation_effect=-0.1
            ),
            BacterialStrain(
                id='lacto_1', bacteria_type=BacteriaType.LACTOBACILLUS,
                name='Lactobacillus Calmis', growth_rate=0.08,
                preferred_nutrients=['fiber', 'sugar'],
                serotonin_effect=0.3, gaba_effect=0.2, anxiety_effect=-0.2
            ),
            BacterialStrain(
                id='bifido_1', bacteria_type=BacteriaType.BIFIDOBACTERIUM,
                name='Bifidobacterium Happy', growth_rate=0.09,
                preferred_nutrients=['fiber'],
                serotonin_effect=0.2, dopamine_effect=0.1, immune_effect=0.2
            ),
            BacterialStrain(
                id='proteo_1', bacteria_type=BacteriaType.PROTEOBACTERIA,
                name='Proteobacteria Minor', growth_rate=0.15,
                preferred_nutrients=['protein', 'fat'],
                inflammation_effect=0.3, immune_effect=-0.1
            ),
        ]
        
        for strain in default_strains:
            self.strains[strain.id] = strain
            self.populations[strain.id] = 0.3 + np.random.random() * 0.4
    
    def update(self, dt: float, diet: Dict[str, float] = None, 
               stress: float = 0.0):
        """
        Update microbiome based on conditions.
        
        Args:
            dt: Time step
            diet: What was eaten (nutrient_type -> amount)
            stress: Current stress level
        """
        diet = diet or {}
        self.stress_level = stress
        
        # Update nutrient availability from diet
        for nutrient, amount in diet.items():
            if nutrient in self.nutrient_availability:
                self.nutrient_availability[nutrient] = np.clip(
                    self.nutrient_availability[nutrient] + amount * 0.1, 0, 1
                )
        
        # Decay nutrients
        for nutrient in self.nutrient_availability:
            self.nutrient_availability[nutrient] *= 0.95
        
        # Update each bacterial population
        total_pop = sum(self.populations.values()) + 0.001
        
        for strain_id, pop in list(self.populations.items()):
            strain = self.strains[strain_id]
            
            # Growth based on preferred nutrients
            nutrient_score = sum(
                self.nutrient_availability.get(n, 0) 
                for n in strain.preferred_nutrients
            ) / max(1, len(strain.preferred_nutrients))
            
            growth = strain.growth_rate * nutrient_score * pop * (1 - pop) * dt
            
            # Competition (limited carrying capacity)
            competition = strain.competition_factor * (total_pop - pop) * pop * 0.1 * dt
            
            # Death
            death = strain.death_rate * pop * dt
            
            # Stress kills beneficial bacteria
            stress_death = stress * 0.1 * pop * dt if strain.serotonin_effect > 0 else 0
            
            # Antibiotics kill everything
            antibiotic_death = self.antibiotic_exposure * 0.5 * pop * dt
            
            # Update population
            new_pop = pop + growth - competition - death - stress_death - antibiotic_death
            self.populations[strain_id] = np.clip(new_pop, 0.01, 1.0)
        
        # Decay antibiotic exposure
        self.antibiotic_exposure *= 0.9
        
        # Normalize populations
        total = sum(self.populations.values())
        if total > 3.0:
            for sid in self.populations:
                self.populations[sid] /= total / 2.0
    
    def get_neurochemical_effects(self) -> Dict[str, float]:
        """
        Get combined effects on host neurochemistry.
        
        Returns modifiers for neuromodulator baselines.
        """
        effects = {
            'serotonin': 0.0,
            'dopamine': 0.0,
            'gaba': 0.0,
        }
        
        for strain_id, pop in self.populations.items():
            strain = self.strains[strain_id]
            weight = pop  # More bacteria = more effect
            
            effects['serotonin'] += strain.serotonin_effect * weight
            effects['dopamine'] += strain.dopamine_effect * weight
            effects['gaba'] += strain.gaba_effect * weight
        
        # Normalize
        for key in effects:
            effects[key] = np.clip(effects[key], -0.5, 0.5)
        
        return effects
    
    def get_metabolism_modifier(self) -> float:
        """Get combined effect on metabolism/nutrient extraction."""
        total = 0.0
        for strain_id, pop in self.populations.items():
            strain = self.strains[strain_id]
            total += strain.metabolism_effect * pop
        return np.clip(total, -0.5, 0.5)
    
    def get_immune_modifier(self) -> float:
        """Get combined effect on immune system."""
        total = 0.0
        for strain_id, pop in self.populations.items():
            strain = self.strains[strain_id]
            total += strain.immune_effect * pop
        return np.clip(total, -0.5, 0.5)
    
    def get_behavioral_effects(self) -> Dict[str, float]:
        """Get effects on behavior."""
        effects = {
            'social_tendency': 0.0,
            'anxiety': 0.0,
            'appetite': 0.0,
        }
        
        for strain_id, pop in self.populations.items():
            strain = self.strains[strain_id]
            effects['social_tendency'] += strain.social_tendency * pop
            effects['anxiety'] += strain.anxiety_effect * pop
            effects['appetite'] += strain.appetite_effect * pop
        
        return effects
    
    def get_diversity(self) -> float:
        """Calculate microbiome diversity (Shannon index)."""
        total = sum(self.populations.values())
        if total < 0.01:
            return 0.0
        
        diversity = 0.0
        for pop in self.populations.values():
            if pop > 0.01:
                p = pop / total
                diversity -= p * np.log(p + 1e-10)
        
        self._last_diversity = diversity
        return diversity
    
    def apply_antibiotic(self, strength: float = 0.5):
        """Apply antibiotic treatment (kills bacteria indiscriminately)."""
        self.antibiotic_exposure = min(1.0, self.antibiotic_exposure + strength)
    
    def add_probiotic(self, strain: BacterialStrain, amount: float = 0.3):
        """Add probiotic bacteria."""
        if strain.id not in self.strains:
            self.strains[strain.id] = strain
            self.populations[strain.id] = amount
        else:
            self.populations[strain.id] = min(1.0, self.populations[strain.id] + amount)
    
    def transfer_from(self, other: 'Microbiome', amount: float = 0.1):
        """
        Transfer bacteria from another microbiome (social contact).
        
        This enables horizontal transmission of microbiome.
        """
        for strain_id, pop in other.populations.items():
            if pop < 0.1:
                continue
            
            transfer_amount = pop * amount * np.random.random()
            
            if strain_id in self.strains:
                self.populations[strain_id] = min(1.0, 
                    self.populations[strain_id] + transfer_amount)
            else:
                # New strain!
                self.strains[strain_id] = other.strains[strain_id]
                self.populations[strain_id] = transfer_amount
    
    def to_dict(self) -> Dict:
        """Serialize microbiome state."""
        return {
            'diversity': self.get_diversity(),
            'total_population': sum(self.populations.values()),
            'dominant_type': self._get_dominant_type(),
            'neurochemical_effects': self.get_neurochemical_effects(),
            'metabolism_modifier': self.get_metabolism_modifier(),
        }
    
    def _get_dominant_type(self) -> str:
        """Get the dominant bacterial type."""
        type_pops = defaultdict(float)
        for strain_id, pop in self.populations.items():
            strain = self.strains[strain_id]
            type_pops[strain.bacteria_type.value] += pop
        
        if not type_pops:
            return "none"
        return max(type_pops.items(), key=lambda x: x[1])[0]


# =============================================================================
# PROBIOTIC STRAINS (beneficial supplements)
# =============================================================================

PROBIOTIC_STRAINS = {
    'calm': BacterialStrain(
        id='probiotic_calm', bacteria_type=BacteriaType.LACTOBACILLUS,
        name='Lactobacillus Tranquilis', growth_rate=0.08,
        preferred_nutrients=['fiber'],
        serotonin_effect=0.4, gaba_effect=0.3, anxiety_effect=-0.3
    ),
    'energy': BacterialStrain(
        id='probiotic_energy', bacteria_type=BacteriaType.FIRMICUTES,
        name='Firmicutes Energeticus', growth_rate=0.1,
        preferred_nutrients=['sugar', 'protein'],
        dopamine_effect=0.3, metabolism_effect=0.4
    ),
    'social': BacterialStrain(
        id='probiotic_social', bacteria_type=BacteriaType.BIFIDOBACTERIUM,
        name='Bifidobacterium Socialis', growth_rate=0.09,
        preferred_nutrients=['fiber'],
        serotonin_effect=0.2, social_tendency=0.4
    ),
    'immune': BacterialStrain(
        id='probiotic_immune', bacteria_type=BacteriaType.ACTINOBACTERIA,
        name='Actinobacteria Defensus', growth_rate=0.07,
        preferred_nutrients=['fiber', 'protein'],
        immune_effect=0.5, inflammation_effect=-0.3
    ),
}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BacteriaType',
    'BacterialStrain',
    'Microbiome',
    'PROBIOTIC_STRAINS',
]
