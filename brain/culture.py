"""
Cultural Transmission and Social Learning

Creatures can learn behaviors by observing others:
- Imitation learning (copy successful actions)
- Social cues (follow gaze, alarm calls)
- Tradition formation (behaviors persist across generations)
- Conformity bias (copy the majority)

This creates emergent culture - behaviors that spread memetically
rather than genetically.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum, auto
from collections import defaultdict


class BehaviorType(Enum):
    """Types of learnable behaviors."""
    FORAGING = "foraging"           # Where/how to find food
    AVOIDANCE = "avoidance"         # What to avoid
    SOCIAL = "social"               # Social interaction patterns
    TOOL_USE = "tool_use"           # Using objects
    VOCALIZATION = "vocalization"   # Communication sounds
    MOVEMENT = "movement"           # Movement patterns
    NESTING = "nesting"             # Shelter building/finding


@dataclass
class LearnedBehavior:
    """A behavior that can be transmitted culturally."""
    id: str
    behavior_type: BehaviorType
    name: str
    
    # What the behavior does (encoded as action preferences)
    action_biases: Dict[str, float] = field(default_factory=dict)
    
    # Context where behavior applies
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Transmission properties
    complexity: float = 0.5         # How hard to learn (0-1)
    salience: float = 0.5           # How noticeable when performed
    utility: float = 0.5            # How beneficial
    
    # Tracking
    times_performed: int = 0
    times_observed: int = 0
    
    def mutate(self, rate: float = 0.1) -> 'LearnedBehavior':
        """Create slightly modified copy (cultural drift)."""
        new_biases = {}
        for action, bias in self.action_biases.items():
            if np.random.random() < rate:
                new_biases[action] = np.clip(bias + np.random.normal(0, 0.1), -1, 1)
            else:
                new_biases[action] = bias
        
        return LearnedBehavior(
            id=f"{self.id}_v{np.random.randint(1000)}",
            behavior_type=self.behavior_type,
            name=self.name,
            action_biases=new_biases,
            context=self.context.copy(),
            complexity=self.complexity,
            salience=self.salience,
            utility=self.utility
        )


class CulturalMemory:
    """
    A creature's cultural knowledge - behaviors learned from others.
    """
    
    def __init__(self, learning_rate: float = 0.3, conformity_bias: float = 0.5):
        self.learning_rate = learning_rate
        self.conformity_bias = conformity_bias
        
        # Known behaviors
        self.behaviors: Dict[str, LearnedBehavior] = {}
        
        # Observation buffer
        self.observed_behaviors: List[Tuple[str, str, float]] = []  # (behavior_id, performer_id, reward)
        self.observation_counts: Dict[str, int] = defaultdict(int)
        
        # Social connections (who do we pay attention to)
        self.social_attention: Dict[str, float] = {}  # creature_id -> attention weight
        
        # Teaching/demonstration state
        self.currently_demonstrating: Optional[str] = None
        
    def observe(self, performer_id: str, behavior: LearnedBehavior, 
                observed_reward: float, attention: float = 1.0):
        """
        Observe another creature performing a behavior.
        
        Args:
            performer_id: Who performed the behavior
            behavior: The behavior observed
            observed_reward: How well it worked (estimated from outcome)
            attention: How much attention we were paying
        """
        self.observed_behaviors.append((behavior.id, performer_id, observed_reward))
        self.observation_counts[behavior.id] += 1
        
        # Update social attention based on success
        current_attention = self.social_attention.get(performer_id, 0.5)
        if observed_reward > 0:
            self.social_attention[performer_id] = min(1.0, current_attention + 0.1)
        else:
            self.social_attention[performer_id] = max(0.1, current_attention - 0.05)
        
        # Maybe learn the behavior
        if behavior.id not in self.behaviors:
            learn_chance = self._compute_learn_chance(behavior, observed_reward, attention)
            if np.random.random() < learn_chance:
                # Learn with possible cultural drift
                if np.random.random() < 0.1:
                    self.behaviors[behavior.id] = behavior.mutate(0.05)
                else:
                    self.behaviors[behavior.id] = behavior
    
    def _compute_learn_chance(self, behavior: LearnedBehavior, 
                              reward: float, attention: float) -> float:
        """Compute probability of learning a behavior."""
        # Base learning rate
        chance = self.learning_rate
        
        # Easier to learn simple, salient behaviors
        chance *= (1 - behavior.complexity * 0.5)
        chance *= (0.5 + behavior.salience * 0.5)
        
        # More likely to learn successful behaviors
        chance *= (0.5 + reward * 0.5)
        
        # Attention matters
        chance *= attention
        
        # Conformity: more likely if many others do it
        obs_count = self.observation_counts[behavior.id]
        if obs_count > 3:
            chance *= (1 + self.conformity_bias * min(1, obs_count / 10))
        
        return np.clip(chance, 0, 0.9)
    
    def get_behavior_bias(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Get action biases from cultural knowledge for current context.
        
        Returns dict of action -> bias value to add to instinct outputs.
        """
        combined_bias = defaultdict(float)
        
        for behavior in self.behaviors.values():
            # Check if context matches
            if not self._context_matches(behavior.context, context):
                continue
            
            # Add behavior's action biases, weighted by utility
            for action, bias in behavior.action_biases.items():
                combined_bias[action] += bias * behavior.utility * 0.5
        
        return dict(combined_bias)
    
    def _context_matches(self, behavior_context: Dict, current_context: Dict) -> bool:
        """Check if behavior's context matches current situation."""
        if not behavior_context:
            return True  # No context requirements
        
        for key, required in behavior_context.items():
            if key not in current_context:
                continue
            actual = current_context[key]
            
            if isinstance(required, bool):
                if actual != required:
                    return False
            elif isinstance(required, (int, float)):
                if abs(actual - required) > 0.3:
                    return False
        
        return True
    
    def perform_behavior(self, behavior_id: str) -> Optional[Dict[str, float]]:
        """
        Perform a learned behavior, returning action biases.
        """
        if behavior_id not in self.behaviors:
            return None
        
        behavior = self.behaviors[behavior_id]
        behavior.times_performed += 1
        self.currently_demonstrating = behavior_id
        
        return behavior.action_biases
    
    def forget_unused(self, threshold: int = 10):
        """Forget behaviors that haven't been used recently."""
        to_remove = []
        for bid, behavior in self.behaviors.items():
            if behavior.times_performed < threshold and behavior.utility < 0.3:
                to_remove.append(bid)
        
        for bid in to_remove:
            del self.behaviors[bid]
    
    def get_traditions(self) -> List[str]:
        """Get list of well-established behaviors (traditions)."""
        return [b.name for b in self.behaviors.values() 
                if b.times_performed > 20 or b.times_observed > 50]
    
    def to_dict(self) -> Dict:
        """Serialize cultural memory."""
        return {
            'learning_rate': self.learning_rate,
            'num_behaviors': len(self.behaviors),
            'traditions': self.get_traditions(),
            'social_connections': len(self.social_attention)
        }


# =============================================================================
# COMMON LEARNABLE BEHAVIORS
# =============================================================================

COMMON_BEHAVIORS = {
    'avoid_water_edge': LearnedBehavior(
        id='avoid_water_edge', behavior_type=BehaviorType.AVOIDANCE,
        name='Avoid Water Edge',
        action_biases={'move_left': 0.3, 'move_right': -0.3},
        context={'near_water': True},
        complexity=0.2, salience=0.7, utility=0.8
    ),
    'food_location_memory': LearnedBehavior(
        id='food_loc_1', behavior_type=BehaviorType.FORAGING,
        name='Remember Food Spot',
        action_biases={'move_right': 0.5},
        context={'hunger': 0.5},
        complexity=0.3, salience=0.5, utility=0.7
    ),
    'alarm_response': LearnedBehavior(
        id='alarm_resp', behavior_type=BehaviorType.AVOIDANCE,
        name='Respond to Alarm',
        action_biases={'flee': 0.8, 'jump': 0.3},
        context={'heard_alarm': True},
        complexity=0.1, salience=0.9, utility=0.9
    ),
    'social_grooming': LearnedBehavior(
        id='social_groom', behavior_type=BehaviorType.SOCIAL,
        name='Social Grooming',
        action_biases={'approach': 0.5, 'rest': 0.3},
        context={'near_same_species': True, 'safety': 0.7},
        complexity=0.4, salience=0.6, utility=0.5
    ),
}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BehaviorType',
    'LearnedBehavior', 
    'CulturalMemory',
    'COMMON_BEHAVIORS',
]
