"""
Procedural Language Evolution System

Implements emergent communication through:
- Phoneme-based vocalization (basic sound units)
- Gesture primitives (visual signals)
- Reinforcement from successful communication
- Symbol grounding in sensory experience
- Collaborative meaning negotiation

Language emerges from interaction, not pre-defined dictionaries.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from enum import Enum


# =============================================================================
# PHONEME SYSTEM - Basic Sound Units
# =============================================================================

class PhonemeType(Enum):
    """Basic phoneme categories (simplified IPA)"""
    VOWEL = "vowel"           # a, e, i, o, u
    PLOSIVE = "plosive"       # p, t, k, b, d, g
    FRICATIVE = "fricative"   # f, s, h, v, z
    NASAL = "nasal"           # m, n
    LIQUID = "liquid"         # l, r
    GLIDE = "glide"           # w, y


@dataclass
class Phoneme:
    """A basic speech sound"""
    symbol: str                  # Written representation (e.g., 'ka', 'mi')
    phoneme_type: PhonemeType
    frequency: float = 0.0       # How often used
    distinctiveness: float = 0.5 # How easy to distinguish (0-1)
    
    # Articulatory features (for similarity)
    voicing: float = 0.5         # Voiced vs unvoiced
    place: float = 0.5           # Front, mid, back of mouth
    manner: float = 0.5          # How air flows


# Starter phoneme inventory (universal across human languages)
UNIVERSAL_PHONEMES = [
    # Vowels
    Phoneme('a', PhonemeType.VOWEL, distinctiveness=0.9, place=0.5),
    Phoneme('i', PhonemeType.VOWEL, distinctiveness=0.9, place=0.1),
    Phoneme('u', PhonemeType.VOWEL, distinctiveness=0.9, place=0.9),
    Phoneme('e', PhonemeType.VOWEL, distinctiveness=0.7, place=0.3),
    Phoneme('o', PhonemeType.VOWEL, distinctiveness=0.7, place=0.7),
    
    # Plosives
    Phoneme('pa', PhonemeType.PLOSIVE, distinctiveness=0.8, voicing=0.0),
    Phoneme('ta', PhonemeType.PLOSIVE, distinctiveness=0.8, voicing=0.0, place=0.3),
    Phoneme('ka', PhonemeType.PLOSIVE, distinctiveness=0.8, voicing=0.0, place=0.7),
    Phoneme('ba', PhonemeType.PLOSIVE, distinctiveness=0.7, voicing=1.0),
    Phoneme('da', PhonemeType.PLOSIVE, distinctiveness=0.7, voicing=1.0, place=0.3),
    
    # Nasals
    Phoneme('ma', PhonemeType.NASAL, distinctiveness=0.8, voicing=1.0),
    Phoneme('na', PhonemeType.NASAL, distinctiveness=0.8, voicing=1.0, place=0.3),
    
    # Fricatives
    Phoneme('sa', PhonemeType.FRICATIVE, distinctiveness=0.7, voicing=0.0),
    Phoneme('ha', PhonemeType.FRICATIVE, distinctiveness=0.6, voicing=0.0),
    
    # Liquids
    Phoneme('la', PhonemeType.LIQUID, distinctiveness=0.6, voicing=1.0),
    Phoneme('ra', PhonemeType.LIQUID, distinctiveness=0.6, voicing=1.0),
]


# =============================================================================
# GESTURE SYSTEM - Visual Communication
# =============================================================================

class GestureType(Enum):
    """Basic gesture categories"""
    POINTING = "pointing"       # Directional reference
    ICONIC = "iconic"           # Shape/action mimicry
    EMPHATIC = "emphatic"       # Emphasis/emotion
    REGULATORY = "regulatory"   # Turn-taking, attention


@dataclass
class Gesture:
    """A visual communication signal"""
    name: str
    gesture_type: GestureType
    intensity: float = 0.5       # Strength/size of gesture
    frequency: float = 0.0       # Usage count
    
    # Meaning associations (learned)
    meanings: Dict[str, float] = field(default_factory=dict)  # concept -> strength


PRIMITIVE_GESTURES = [
    Gesture('point_up', GestureType.POINTING, intensity=0.7),
    Gesture('point_down', GestureType.POINTING, intensity=0.7),
    Gesture('point_forward', GestureType.POINTING, intensity=0.7),
    Gesture('wave', GestureType.EMPHATIC, intensity=0.6),
    Gesture('nod', GestureType.REGULATORY, intensity=0.5),
    Gesture('shake_head', GestureType.REGULATORY, intensity=0.5),
    Gesture('raise_arms', GestureType.EMPHATIC, intensity=0.8),
    Gesture('crouch', GestureType.ICONIC, intensity=0.6),
]


# =============================================================================
# SYMBOL - Phoneme Sequence + Meaning
# =============================================================================

@dataclass
class Symbol:
    """
    A learned word/sign mapping sound to meaning.
    
    Symbols emerge through:
    1. Random phoneme combination
    2. Association with context
    3. Reinforcement from successful use
    """
    phonemes: List[str]          # Sequence of phoneme symbols
    gesture: Optional[str] = None # Optional accompanying gesture
    
    # Grounded meaning
    referent_type: Optional[str] = None  # 'food', 'water', 'danger', 'greeting', etc.
    context_features: np.ndarray = field(default_factory=lambda: np.zeros(10))
    
    # Learning statistics
    uses: int = 0                # Times produced
    successes: int = 0           # Times understood
    last_reward: float = 0.0     # Last reinforcement value
    
    # Social transmission
    learned_from: Optional[str] = None  # Creature ID
    taught_to: Set[str] = field(default_factory=set)  # Creature IDs
    
    def get_word(self) -> str:
        """Get phoneme sequence as pronounceable word"""
        return ''.join(self.phonemes)
    
    def success_rate(self) -> float:
        """Communication success rate"""
        if self.uses == 0:
            return 0.0
        return self.successes / self.uses
    
    def reinforcement_strength(self) -> float:
        """How strongly this symbol is reinforced"""
        # Combine success rate with recency
        return self.success_rate() * (0.5 + 0.5 * np.tanh(self.last_reward))


# =============================================================================
# COMMUNICATION EVENT - Track Interaction Outcomes
# =============================================================================

@dataclass
class CommunicationEvent:
    """Record of a communication attempt"""
    speaker_id: str
    listener_id: Optional[str]
    symbol_used: str
    gesture_used: Optional[str]
    context: Dict[str, any]      # What was happening
    success: bool                # Did listener respond appropriately?
    reward: float               # Reinforcement signal
    timestamp: float


# =============================================================================
# PROCEDURAL LANGUAGE EVOLUTION ENGINE
# =============================================================================

class ProceduralLanguageSystem:
    """
    Manages emergent language development for creatures.
    
    Each creature has:
    - Phoneme inventory (what sounds they can make)
    - Gesture repertoire (what actions they can perform)
    - Symbol lexicon (learned word-meaning mappings)
    - Communication history (for reinforcement learning)
    """
    
    def __init__(self, creature_id: str, innovation_rate: float = 0.1):
        self.creature_id = creature_id
        self.innovation_rate = innovation_rate  # Chance to invent new words
        
        # Available phonemes (start with subset, can expand)
        self.phoneme_inventory = [p.symbol for p in UNIVERSAL_PHONEMES[:8]]  # Start simple
        
        # Available gestures
        self.gesture_inventory = [g.name for g in PRIMITIVE_GESTURES]
        
        # Learned symbols (words)
        self.lexicon: Dict[str, Symbol] = {}
        
        # Communication history
        self.communication_history: List[CommunicationEvent] = []
        
        # Innovation: can create new phonemes/gestures
        self.can_innovate = True
        
    def generate_utterance(self, context: Dict[str, any]) -> Tuple[str, Optional[str]]:
        """
        Generate a word + gesture for the current context.
        
        Args:
            context: Dict with 'need' (hunger/thirst/fear), 'target' (food/water/danger), etc.
        
        Returns:
            (word, gesture) tuple
        """
        need = context.get('need', 'idle')
        target = context.get('target', None)
        
        # Check if we have an existing symbol for this context
        matching_symbols = self._find_matching_symbols(context)
        
        if matching_symbols and np.random.random() > self.innovation_rate:
            # Use existing symbol (weighted by success rate)
            weights = np.array([s.reinforcement_strength() + 0.1 for s in matching_symbols])
            weights /= weights.sum()
            symbol = np.random.choice(matching_symbols, p=weights)
            
            symbol.uses += 1
            return symbol.get_word(), symbol.gesture
        
        else:
            # Innovate: create new symbol
            new_symbol = self._create_new_symbol(context)
            symbol_word = new_symbol.get_word()
            self.lexicon[symbol_word] = new_symbol
            
            return symbol_word, new_symbol.gesture
    
    def _find_matching_symbols(self, context: Dict[str, any]) -> List[Symbol]:
        """Find symbols previously used in similar contexts"""
        need = context.get('need', None)
        target = context.get('target', None)
        
        matches = []
        for symbol in self.lexicon.values():
            # Match by referent type
            if symbol.referent_type == need or symbol.referent_type == target:
                matches.append(symbol)
        
        return matches
    
    def _create_new_symbol(self, context: Dict[str, any]) -> Symbol:
        """Invent a new word for this context"""
        # Generate random phoneme sequence (1-3 syllables)
        n_syllables = np.random.randint(1, 4)
        phonemes = [np.random.choice(self.phoneme_inventory) for _ in range(n_syllables)]
        
        # Possibly add gesture (30% chance)
        gesture = None
        if np.random.random() < 0.3:
            gesture = np.random.choice(self.gesture_inventory)
        
        # Ground in context
        referent = context.get('need', context.get('target', 'unknown'))
        
        return Symbol(
            phonemes=phonemes,
            gesture=gesture,
            referent_type=referent,
            uses=1
        )
    
    def receive_utterance(self, word: str, gesture: Optional[str], 
                         context: Dict[str, any]) -> Optional[str]:
        """
        Hear another creature's utterance and try to understand.
        
        Returns:
            Interpreted meaning or None if unknown
        """
        # Check if we know this word
        if word in self.lexicon:
            symbol = self.lexicon[word]
            return symbol.referent_type
        
        # Unknown word - can we learn from context?
        if context.get('observable_action'):
            # Social learning: infer meaning from what speaker does
            self._learn_from_observation(word, gesture, context)
            return context.get('observable_action')
        
        return None
    
    def _learn_from_observation(self, word: str, gesture: Optional[str], context: Dict[str, any]):
        """Learn new word by observing speaker's behavior"""
        # Create new symbol based on observed context
        # This is SOCIAL LEARNING - acquiring words from others
        new_symbol = Symbol(
            phonemes=list(word),  # Store as character list
            gesture=gesture,
            referent_type=context.get('observable_action'),
            learned_from=context.get('speaker_id')
        )
        
        self.lexicon[word] = new_symbol
    
    def reinforce_communication(self, word: str, success: bool, reward: float):
        """
        Update symbol based on communication outcome.
        
        This is the KEY to language evolution - successful communication
        strengthens word-meaning associations.
        """
        if word in self.lexicon:
            symbol = self.lexicon[word]
            symbol.last_reward = reward
            
            if success:
                symbol.successes += 1
            
            # Prune rarely used, unsuccessful words
            if symbol.uses > 10 and symbol.success_rate() < 0.2:
                # Forget ineffective words
                del self.lexicon[word]
    
    def teach_word(self, learner_id: str, word: str):
        """Explicitly teach a word to another creature"""
        if word in self.lexicon:
            self.lexicon[word].taught_to.add(learner_id)
    
    def get_lexicon_summary(self) -> Dict[str, any]:
        """Get statistics about this creature's language"""
        return {
            'vocabulary_size': len(self.lexicon),
            'total_communications': len(self.communication_history),
            'avg_success_rate': np.mean([s.success_rate() for s in self.lexicon.values()]) if self.lexicon else 0.0,
            'phoneme_count': len(self.phoneme_inventory),
            'gesture_count': len(self.gesture_inventory),
            'most_used_words': sorted(
                [(s.get_word(), s.uses) for s in self.lexicon.values()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


# =============================================================================
# CULTURAL LANGUAGE - Shared Across Population
# =============================================================================

class CulturalLanguage:
    """
    Tracks language conventions emerging across creature population.
    
    Words become "culturally established" when multiple creatures
    use them successfully for the same meaning.
    """
    
    def __init__(self):
        # Conventional symbols (word -> (referent, frequency, success_rate))
        self.conventions: Dict[str, Tuple[str, int, float]] = {}
        
        # Dialect groups (creatures using similar vocabulary)
        self.dialect_groups: List[Set[str]] = []
    
    def register_communication(self, speaker_id: str, word: str, 
                              meaning: str, success: bool):
        """Record a communication event for cultural tracking"""
        if word not in self.conventions:
            self.conventions[word] = (meaning, 1, 1.0 if success else 0.0)
        else:
            ref, count, success_rate = self.conventions[word]
            new_count = count + 1
            new_success = (success_rate * count + (1.0 if success else 0.0)) / new_count
            self.conventions[word] = (ref, new_count, new_success)
    
    def is_conventional(self, word: str, threshold: int = 5) -> bool:
        """Check if word is culturally established (used by multiple creatures)"""
        if word not in self.conventions:
            return False
        _, count, success_rate = self.conventions[word]
        return count >= threshold and success_rate > 0.6
    
    def get_conventional_words(self) -> Dict[str, str]:
        """Get dictionary of conventional word -> meaning mappings"""
        return {
            word: meaning
            for word, (meaning, count, success_rate) in self.conventions.items()
            if self.is_conventional(word)
        }
