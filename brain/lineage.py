"""
Lineage Tracking System (System 9)

Tracks the evolution of families/species over generations.
Monitors behavioral trends, genetic drift, and emergence of "species personalities".
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

@dataclass
class LineageStats:
    """Statistics for a single genetic lineage."""
    id: str
    name: str = "Unknown"
    
    # Population
    total_born: int = 0
    living_members: int = 0
    generation_count: int = 0
    longest_lived: float = 0.0
    
    # Genetic Profile (Average Phenotype)
    avg_color_hue: float = 0.0
    avg_size: float = 1.0
    avg_limbs: float = 4.0
    
    # Behavioral Profile (Average Personality)
    avg_bravery: float = 0.5
    avg_aggression: float = 0.0
    avg_social: float = 0.0
    avg_exploration: float = 0.0
    
    # Performance
    total_food_eaten: int = 0
    total_kills: int = 0
    total_blocks_dug: int = 0
    total_blocks_built: int = 0
    avg_energy_efficiency: float = 0.0
    
    # History
    founder_id: str = ""
    extinct: bool = False
    
    def update_averages(self, creature, weight: float = 0.1):
        """Update rolling averages with a new sample."""
        # Phenotype
        p = creature.phenotype
        self.avg_color_hue = self._lerp(self.avg_color_hue, p.hue, weight)
        self.avg_size = self._lerp(self.avg_size, p.size, weight)
        self.avg_limbs = self._lerp(self.avg_limbs, p.limb_count, weight)
        
        # Behavior (Genes)
        self.avg_bravery = self._lerp(self.avg_bravery, p.bravery, weight)
        
        # Inferred Behavior (from stats)
        # Assuming we can infer these from actions if passed, but for now using genes
        # or homeostasis states if available
        pass

    def _lerp(self, a, b, t):
        return a + (b - a) * t

class LineageSystem:
    """
    Global manager for tracking all lineages.
    """
    def __init__(self):
        self.lineages: Dict[str, LineageStats] = {}
        self.active_lineage_ids: Set[str] = set()
        
    def register_birth(self, creature, parent_ids: List[str] = None):
        """Register a new creature birth."""
        body = creature.body if hasattr(creature, 'body') else creature
        lid = body.homeostasis.genetic_lineage_id
        
        if not lid:
            # Create new lineage for spontaneous generation
            lid = f"L_{np.random.randint(1000, 9999)}_{int(body.id)%100}"
            body.homeostasis.genetic_lineage_id = lid
            
        if lid not in self.lineages:
            self.lineages[lid] = LineageStats(
                id=lid,
                founder_id=str(body.id)
            )
            
        stats = self.lineages[lid]
        stats.total_born += 1
        stats.living_members += 1
        self.active_lineage_ids.add(lid)
        
        # Update initial stats
        stats.update_averages(body, weight=1.0 if stats.total_born == 1 else 0.1)
        
    def register_death(self, creature):
        """Register a death."""
        body = creature.body if hasattr(creature, 'body') else creature
        lid = body.homeostasis.genetic_lineage_id
        if lid in self.lineages:
            stats = self.lineages[lid]
            stats.living_members = max(0, stats.living_members - 1)
            stats.longest_lived = max(stats.longest_lived, body.lifetime)
            
            # Accumulate lifetime stats
            if hasattr(body, 'blocks_dug'):
                stats.total_blocks_dug += body.blocks_dug
            if hasattr(body, 'blocks_built'):
                stats.total_blocks_built += body.blocks_built
            
            if stats.living_members == 0:
                stats.extinct = True
                if lid in self.active_lineage_ids:
                    self.active_lineage_ids.remove(lid)

    def update_stats(self, creatures: List):
        """Periodic update of all active lineages."""
        # Reset counters for recalculation? 
        # No, rolling averages are better for stability.
        
        # Instead, we just iterate active creatures and nudge the averages?
        # That's O(N).
        for c in creatures:
            lid = c.homeostasis.genetic_lineage_id
            if lid in self.lineages:
                # Update dynamic stats like behavior
                pass

    def get_dominant_lineages(self, limit: int = 5) -> List[LineageStats]:
        """Get most successful active lineages."""
        active = [self.lineages[lid] for lid in self.active_lineage_ids]
        return sorted(active, key=lambda x: x.living_members, reverse=True)[:limit]

