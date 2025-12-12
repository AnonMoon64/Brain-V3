"""
True Social Bond Scoring (UPGRADE 7)

Implements realistic social relationships:
- Trust (will they help me?)
- Affinity (do I like them?)
- Fear (are they dangerous?)
- Prestige (are they high status?)

These blend into neurochemical changes:
- High trust/affinity → oxytocin release
- High prestige → dopamine
- High fear → cortisol
- Betrayal → serotonin drop

This is how cooperation and social hierarchies actually evolve.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class SocialBond:
    """
    Relationship between two creatures.
    
    All values 0-1:
    - trust: Will they cooperate? Builds from positive interactions
    - affinity: Do I like them? Based on familiarity + positive experiences
    - fear: Are they dangerous? Based on observed aggression
    - prestige: High status? Based on success observations
    """
    target_id: str
    
    # Core bond dimensions
    trust: float = 0.5              # 0 = distrust, 1 = full trust
    affinity: float = 0.5           # 0 = dislike, 1 = love
    fear: float = 0.0               # 0 = safe, 1 = terrified
    prestige: float = 0.5           # 0 = low status, 1 = high status
    
    # Interaction history
    positive_interactions: int = 0
    negative_interactions: int = 0
    total_interactions: int = 0
    
    # Recent interactions (for recency bias)
    recent_outcomes: List[float] = field(default_factory=list)  # -1 to 1
    max_recent_memory: int = 10
    
    # Familiarity (builds over time)
    familiarity: float = 0.0        # 0 = stranger, 1 = old friend
    time_since_last_interaction: float = 0.0
    
    # Observational learning
    observed_successes: int = 0      # Times seen them succeed at tasks
    observed_failures: int = 0       # Times seen them fail
    
    def update_from_interaction(self, outcome: float, interaction_type: str = 'neutral'):
        """
        Update bond based on interaction outcome.
        
        Args:
            outcome: -1 (very bad) to 1 (very good)
            interaction_type: 'cooperative', 'competitive', 'aggressive', 'sharing'
        """
        self.total_interactions += 1
        
        # Store recent outcome
        self.recent_outcomes.append(outcome)
        if len(self.recent_outcomes) > self.max_recent_memory:
            self.recent_outcomes.pop(0)
        
        # Update interaction counts
        if outcome > 0.2:
            self.positive_interactions += 1
        elif outcome < -0.2:
            self.negative_interactions += 1
        
        # Update trust based on outcome and type
        if interaction_type == 'cooperative':
            # Cooperation builds trust fast
            if outcome > 0:
                self.trust = min(1.0, self.trust + 0.1 * outcome)
            else:
                # Betrayal destroys trust
                self.trust = max(0.0, self.trust - 0.3 * abs(outcome))
        
        elif interaction_type == 'sharing':
            # Sharing resources builds trust and affinity
            if outcome > 0:
                self.trust = min(1.0, self.trust + 0.08 * outcome)
                self.affinity = min(1.0, self.affinity + 0.1 * outcome)
        
        elif interaction_type == 'aggressive':
            # Aggression increases fear
            if outcome < 0:
                self.fear = min(1.0, self.fear + 0.15 * abs(outcome))
                self.trust = max(0.0, self.trust - 0.1)
                self.affinity = max(0.0, self.affinity - 0.1)
        
        # Update affinity based on recent trend
        recent_avg = np.mean(self.recent_outcomes) if self.recent_outcomes else 0
        self.affinity = self.affinity * 0.95 + recent_avg * 0.05
        self.affinity = max(0.0, min(1.0, self.affinity))
        
        # Build familiarity
        self.familiarity = min(1.0, self.familiarity + 0.02)
        self.time_since_last_interaction = 0.0
    
    def observe_success(self, success: bool):
        """Observe target succeed or fail at a task."""
        if success:
            self.observed_successes += 1
        else:
            self.observed_failures += 1
        
        # Update prestige based on success rate
        total_obs = self.observed_successes + self.observed_failures
        if total_obs > 0:
            success_rate = self.observed_successes / total_obs
            # Prestige = success rate with a floor
            self.prestige = 0.3 + success_rate * 0.7
    
    def decay_over_time(self, dt: float):
        """Decay bond over time (forgetting)."""
        self.time_since_last_interaction += dt
        
        # Familiarity decays slowly
        if self.time_since_last_interaction > 60:  # 1 minute
            self.familiarity = max(0.0, self.familiarity - 0.001 * dt)
        
        # Fear decays faster (forget old threats)
        self.fear = max(0.0, self.fear - 0.002 * dt)
        
        # Trust/affinity decay slowly without reinforcement
        if self.time_since_last_interaction > 30:
            self.trust = self.trust * 0.999
            self.affinity = self.affinity * 0.999
    
    def get_neurochemical_influence(self) -> Dict[str, float]:
        """
        Get neurochemical changes caused by this relationship.
        
        Returns deltas for neurochemicals (-1 to 1).
        """
        # Trust + Affinity → Oxytocin (bonding hormone)
        oxytocin_boost = (self.trust * 0.5 + self.affinity * 0.5) * self.familiarity
        
        # Prestige → Dopamine (reward from being near high-status)
        dopamine_boost = self.prestige * 0.3 if self.familiarity > 0.3 else 0.0
        
        # Fear → Cortisol (stress)
        cortisol_boost = self.fear * 0.5
        
        # Betrayal (low trust + negative recent) → Serotonin drop
        serotonin_change = 0.0
        if self.trust < 0.3 and len(self.recent_outcomes) > 0:
            recent_avg = np.mean(self.recent_outcomes[-3:])
            if recent_avg < -0.3:
                serotonin_change = -0.2  # Depression from betrayal
        
        return {
            'oxytocin': oxytocin_boost,
            'dopamine': dopamine_boost,
            'cortisol': cortisol_boost,
            'serotonin': serotonin_change
        }
    
    def should_cooperate(self) -> bool:
        """Should I cooperate with this individual?"""
        # Cooperate if trust + affinity - fear > threshold
        cooperation_score = self.trust * 0.5 + self.affinity * 0.3 - self.fear * 0.2
        return cooperation_score > 0.5
    
    def should_avoid(self) -> bool:
        """Should I avoid this individual?"""
        return self.fear > 0.6 or (self.trust < 0.2 and self.affinity < 0.3)
    
    def should_follow(self) -> bool:
        """Should I follow/imitate this individual?"""
        # Follow high-prestige, trusted individuals
        return self.prestige > 0.6 and self.trust > 0.4
    
    def get_bond_strength(self) -> float:
        """Overall bond strength (0-1)."""
        # Positive bonds: trust + affinity
        # Negative bonds: fear
        positive = (self.trust + self.affinity) / 2 * self.familiarity
        negative = self.fear
        return max(0.0, positive - negative)


class SocialBondSystem:
    """
    Manages all social relationships for a creature.
    
    Tracks bonds with every other creature encountered.
    Updates neurochemistry based on social context.
    """
    
    def __init__(self, creature_id: str):
        self.creature_id = creature_id
        
        # All known relationships
        self.bonds: Dict[str, SocialBond] = {}
        
        # Social preferences (personality)
        self.sociability: float = 0.5       # 0 = loner, 1 = social butterfly
        self.aggressiveness: float = 0.3    # 0 = passive, 1 = aggressive
        self.cooperativeness: float = 0.6   # 0 = selfish, 1 = altruistic
        
        # Current social state
        self.current_group: List[str] = []
        self.social_stress: float = 0.0     # Crowding stress
        
    def get_or_create_bond(self, target_id: str) -> SocialBond:
        """Get existing bond or create new one."""
        if target_id not in self.bonds:
            self.bonds[target_id] = SocialBond(target_id=target_id)
        return self.bonds[target_id]
    
    def interact_with(self, target_id: str, outcome: float, 
                     interaction_type: str = 'neutral'):
        """Record interaction with another creature."""
        bond = self.get_or_create_bond(target_id)
        bond.update_from_interaction(outcome, interaction_type)
    
    def observe_other(self, target_id: str, success: bool):
        """Observe another creature's success/failure."""
        bond = self.get_or_create_bond(target_id)
        bond.observe_success(success)
    
    def update_social_context(self, nearby_creatures: List[str], dt: float):
        """
        Update based on current social context.
        
        Args:
            nearby_creatures: List of creature IDs currently nearby
            dt: Time delta
        """
        self.current_group = nearby_creatures
        
        # Social stress from crowding
        n_nearby = len(nearby_creatures)
        optimal_group_size = 3 if self.sociability > 0.5 else 1
        
        if n_nearby > optimal_group_size:
            # Too crowded
            self.social_stress = min(1.0, (n_nearby - optimal_group_size) * 0.1)
        else:
            # Comfortable or lonely
            if self.sociability > 0.6 and n_nearby == 0:
                # Lonely
                self.social_stress = 0.2
            else:
                self.social_stress = max(0.0, self.social_stress - 0.05 * dt)
        
        # Decay all bonds
        for bond in self.bonds.values():
            bond.decay_over_time(dt)
    
    def get_total_neurochemical_influence(self) -> Dict[str, float]:
        """
        Get total neurochemical influence from all relationships.
        
        Stronger for nearby individuals.
        """
        total_influence = defaultdict(float)
        
        for creature_id in self.current_group:
            if creature_id in self.bonds:
                bond = self.bonds[creature_id]
                influence = bond.get_neurochemical_influence()
                
                # Weight by bond strength
                weight = bond.get_bond_strength()
                
                for chem, value in influence.items():
                    total_influence[chem] += value * weight
        
        # Add social stress → cortisol
        total_influence['cortisol'] += self.social_stress * 0.3
        
        # Normalize to reasonable ranges
        for chem in total_influence:
            total_influence[chem] = max(-0.5, min(0.5, total_influence[chem]))
        
        return dict(total_influence)
    
    def get_strongest_bonds(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N strongest bonds (friends)."""
        bonds_with_strength = [
            (bond.target_id, bond.get_bond_strength())
            for bond in self.bonds.values()
        ]
        bonds_with_strength.sort(key=lambda x: x[1], reverse=True)
        return bonds_with_strength[:n]
    
    def get_most_feared(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get top N most feared individuals."""
        feared = [
            (bond.target_id, bond.fear)
            for bond in self.bonds.values()
            if bond.fear > 0.3
        ]
        feared.sort(key=lambda x: x[1], reverse=True)
        return feared[:n]
    
    def get_role_models(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get top N highest prestige individuals to imitate."""
        prestigious = [
            (bond.target_id, bond.prestige)
            for bond in self.bonds.values()
            if bond.prestige > 0.6 and bond.trust > 0.4
        ]
        prestigious.sort(key=lambda x: x[1], reverse=True)
        return prestigious[:n]
    
    def should_cooperate_with(self, target_id: str) -> bool:
        """Should cooperate with this specific individual?"""
        if target_id not in self.bonds:
            # Unknown - use personality default
            return self.cooperativeness > 0.5
        return self.bonds[target_id].should_cooperate()
    
    def should_avoid_creature(self, target_id: str) -> bool:
        """Should avoid this creature?"""
        if target_id not in self.bonds:
            return False
        return self.bonds[target_id].should_avoid()
