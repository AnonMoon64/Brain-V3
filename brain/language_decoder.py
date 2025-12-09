"""
Neural Language Decoder

Implements coherent text output from neural activity:
- Semantic embedding space (learned)
- Attractor dynamics for word selection
- Beam search decoding
- Context-aware generation
- Vocabulary learning from training

Key insight: Language emerges from learned mappings between
neural activity patterns and semantic concepts, not from
simple firing rate → word lookup.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import math
import re


@dataclass
class SemanticConcept:
    """
    A learned semantic concept with associated neural pattern
    """
    concept_id: str
    words: List[str]  # Words expressing this concept
    pattern: np.ndarray  # Neural activity pattern
    
    # Statistics
    activation_count: int = 0
    last_activation: float = 0.0
    
    # Relationships
    related_concepts: List[str] = field(default_factory=list)
    opposite_concepts: List[str] = field(default_factory=list)
    
    # Emotional valence
    valence: float = 0.0  # -1 (negative) to 1 (positive)
    arousal: float = 0.5  # 0 (calm) to 1 (excited)


class SemanticSpace:
    """
    Learned semantic embedding space
    
    Maps neural patterns to semantic concepts bidirectionally.
    Uses competitive learning to form concept clusters.
    """
    
    def __init__(self, pattern_dim: int, n_concepts: int = 500):
        self.pattern_dim = pattern_dim
        self.n_concepts = n_concepts
        
        # Concept storage
        self.concepts: Dict[str, SemanticConcept] = {}
        
        # Pattern → concept mapping (learned prototypes)
        self.prototypes = np.random.randn(n_concepts, pattern_dim) * 0.1
        self.prototype_words: List[List[str]] = [[] for _ in range(n_concepts)]
        self.prototype_counts = np.zeros(n_concepts)
        
        # Word → pattern mapping
        self.word_embeddings: Dict[str, np.ndarray] = {}
        
        # Initialize with common concepts
        self._initialize_base_concepts()
    
    def _initialize_base_concepts(self) -> None:
        """Initialize base semantic concepts"""
        base_concepts = {
            'greeting': (['hello', 'hi', 'hey', 'greetings'], 0.3, 0.4),
            'farewell': (['goodbye', 'bye', 'see you', 'farewell'], 0.1, 0.3),
            'affirmation': (['yes', 'correct', 'right', 'indeed', 'true'], 0.4, 0.3),
            'negation': (['no', 'not', 'never', 'wrong', 'false'], -0.2, 0.4),
            'question': (['what', 'why', 'how', 'when', 'where', 'who'], 0.0, 0.5),
            'gratitude': (['thanks', 'thank you', 'grateful', 'appreciate'], 0.6, 0.4),
            'help': (['help', 'assist', 'support', 'aid'], 0.2, 0.5),
            'understanding': (['understand', 'see', 'know', 'realize', 'get it'], 0.3, 0.3),
            'confusion': (['confused', 'unclear', "don't understand", 'puzzled'], -0.2, 0.5),
            'happiness': (['happy', 'glad', 'pleased', 'delighted', 'joy'], 0.8, 0.6),
            'sadness': (['sad', 'unhappy', 'sorry', 'regret'], -0.6, 0.3),
            'thinking': (['think', 'consider', 'ponder', 'believe', 'suppose'], 0.0, 0.4),
            'feeling': (['feel', 'sense', 'experience', 'emotion'], 0.0, 0.5),
            'curiosity': (['curious', 'interested', 'wonder', 'intrigued'], 0.4, 0.6),
        }
        
        for i, (concept_id, (words, valence, arousal)) in enumerate(base_concepts.items()):
            if i >= self.n_concepts:
                break
            
            pattern = np.random.randn(self.pattern_dim) * 0.3
            self.prototypes[i] = pattern
            self.prototype_words[i] = words
            self.prototype_counts[i] = 10  # Pre-trained
            
            concept = SemanticConcept(
                concept_id=concept_id,
                words=words,
                pattern=pattern,
                valence=valence,
                arousal=arousal,
            )
            self.concepts[concept_id] = concept
            
            # Initialize word embeddings
            for word in words:
                self.word_embeddings[word] = pattern + np.random.randn(self.pattern_dim) * 0.05
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to neural pattern
        """
        words = self._tokenize(text)
        
        if not words:
            return np.zeros(self.pattern_dim)
        
        # Sum embeddings of known words, create new for unknown
        pattern = np.zeros(self.pattern_dim)
        weight_sum = 0.0
        
        for i, word in enumerate(words):
            weight = 1.0 / (i + 1)  # Recency weighting
            
            if word in self.word_embeddings:
                pattern += self.word_embeddings[word] * weight
            else:
                # Create embedding from character hash
                char_pattern = self._hash_word(word)
                self.word_embeddings[word] = char_pattern
                pattern += char_pattern * weight
            
            weight_sum += weight
        
        if weight_sum > 0:
            pattern /= weight_sum
        
        return pattern
    
    def decode_pattern(self, pattern: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Decode neural pattern to concepts with confidence
        Returns list of (concept_id, confidence)
        """
        # Find closest prototypes
        similarities = self._compute_similarities(pattern)
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold
                concept_id = self._get_concept_id(idx)
                results.append((concept_id, float(similarities[idx])))
        
        return results
    
    def _compute_similarities(self, pattern: np.ndarray) -> np.ndarray:
        """Compute cosine similarity to all prototypes"""
        pattern_norm = np.linalg.norm(pattern)
        if pattern_norm < 1e-8:
            return np.zeros(self.n_concepts)
        
        pattern_normalized = pattern / pattern_norm
        
        proto_norms = np.linalg.norm(self.prototypes, axis=1, keepdims=True)
        proto_norms = np.maximum(proto_norms, 1e-8)
        protos_normalized = self.prototypes / proto_norms
        
        return np.dot(protos_normalized, pattern_normalized)
    
    def _get_concept_id(self, idx: int) -> str:
        """Get concept ID from prototype index"""
        for concept in self.concepts.values():
            proto_idx = self._find_prototype_idx(concept.pattern)
            if proto_idx == idx:
                return concept.concept_id
        return f"concept_{idx}"
    
    def _find_prototype_idx(self, pattern: np.ndarray) -> int:
        """Find prototype index closest to pattern"""
        sims = self._compute_similarities(pattern)
        return int(np.argmax(sims))
    
    def learn(self, pattern: np.ndarray, words: List[str], reward: float = 1.0) -> None:
        """
        Learn association between pattern and words
        Uses competitive learning
        """
        # Find closest prototype
        similarities = self._compute_similarities(pattern)
        winner_idx = np.argmax(similarities)
        
        # Update winner (competitive learning)
        learning_rate = 0.1 * reward
        self.prototypes[winner_idx] += learning_rate * (pattern - self.prototypes[winner_idx])
        self.prototype_counts[winner_idx] += 1
        
        # Associate words with winner
        for word in words:
            if word not in self.prototype_words[winner_idx]:
                self.prototype_words[winner_idx].append(word)
            
            # Update word embedding
            if word in self.word_embeddings:
                self.word_embeddings[word] += learning_rate * 0.5 * (
                    pattern - self.word_embeddings[word]
                )
            else:
                self.word_embeddings[word] = pattern.copy()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def _hash_word(self, word: str) -> np.ndarray:
        """Create pattern from word using character hashing"""
        pattern = np.zeros(self.pattern_dim)
        for i, char in enumerate(word[:20]):
            idx = (ord(char) * (i + 1) * 31) % self.pattern_dim
            pattern[idx] += 0.3
            # Spread activation
            for offset in [-1, 1]:
                neighbor = (idx + offset) % self.pattern_dim
                pattern[neighbor] += 0.1
        
        norm = np.linalg.norm(pattern)
        if norm > 0:
            pattern /= norm
        return pattern


class AttractorDynamics:
    """
    Attractor dynamics for stable word selection
    
    Activity settles into stable attractor states
    representing word/concept selections.
    """
    
    def __init__(self, n_states: int):
        self.n_states = n_states
        
        # Attractor basin centers
        self.attractors = np.eye(n_states)  # One-hot attractors
        
        # State
        self.state = np.zeros(n_states)
        
        # Dynamics parameters
        self.tau = 10.0  # Time constant
        self.noise_level = 0.01
        self.inhibition = 0.5  # Lateral inhibition strength
    
    def step(self, input_drive: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Run one step of attractor dynamics
        Returns updated state
        """
        # Softmax-like competition
        exp_state = np.exp(self.state * 3)
        competition = exp_state / (np.sum(exp_state) + 1e-8)
        
        # Lateral inhibition
        inhibition = self.inhibition * np.sum(competition) - competition
        
        # Dynamics: move toward input, with competition
        dstate = (input_drive - self.state - inhibition) / self.tau
        
        # Add noise
        dstate += np.random.randn(self.n_states) * self.noise_level
        
        self.state += dstate * dt
        self.state = np.clip(self.state, 0, 1)
        
        return self.state
    
    def get_winner(self) -> Tuple[int, float]:
        """Get winning attractor and confidence"""
        idx = np.argmax(self.state)
        confidence = self.state[idx] / (np.sum(self.state) + 1e-8)
        return int(idx), float(confidence)
    
    def settle(self, input_drive: np.ndarray, max_steps: int = 50) -> Tuple[int, float]:
        """Run dynamics until settled"""
        self.state = input_drive.copy()
        
        for _ in range(max_steps):
            old_state = self.state.copy()
            self.step(input_drive, dt=1.0)
            
            # Check convergence
            if np.max(np.abs(self.state - old_state)) < 0.01:
                break
        
        return self.get_winner()
    
    def reset(self) -> None:
        """Reset state"""
        self.state = np.zeros(self.n_states)


class BeamSearchDecoder:
    """
    Beam search for generating coherent text sequences
    """
    
    def __init__(self, beam_width: int = 5, max_length: int = 20):
        self.beam_width = beam_width
        self.max_length = max_length
    
    def decode(
        self,
        initial_pattern: np.ndarray,
        semantic_space: SemanticSpace,
        score_fn,  # Callable[[List[str]], float]
    ) -> str:
        """
        Generate text using beam search
        """
        # Initialize beams: (words, cumulative_score)
        beams = [([], 0.0)]
        
        for step in range(self.max_length):
            candidates = []
            
            for words, score in beams:
                # Get possible next concepts
                if words:
                    # Encode current sequence
                    current_text = ' '.join(words)
                    current_pattern = semantic_space.encode_text(current_text)
                    # Combine with initial pattern
                    combined = 0.7 * initial_pattern + 0.3 * current_pattern
                else:
                    combined = initial_pattern
                
                # Decode to concepts
                concepts = semantic_space.decode_pattern(combined, top_k=10)
                
                for concept_id, confidence in concepts:
                    if concept_id in semantic_space.concepts:
                        concept = semantic_space.concepts[concept_id]
                        for word in concept.words[:3]:  # Top 3 words per concept
                            new_words = words + [word]
                            new_score = score + math.log(confidence + 1e-8)
                            new_score += score_fn(new_words)
                            candidates.append((new_words, new_score))
            
            if not candidates:
                break
            
            # Keep top beams
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:self.beam_width]
            
            # Check for end conditions
            if beams and len(beams[0][0]) >= 3:  # Minimum length
                top_words = beams[0][0]
                if self._is_complete(top_words):
                    break
        
        if beams:
            return ' '.join(beams[0][0])
        return ""
    
    def _is_complete(self, words: List[str]) -> bool:
        """Check if sequence is complete"""
        if not words:
            return False
        last = words[-1].lower()
        # Simple heuristic: end on certain words
        end_words = {'you', 'it', 'that', 'this', 'me', 'now', 'here'}
        return last in end_words or len(words) >= 8


class NeuralLanguageDecoder:
    """
    Complete neural language decoder
    
    Converts neural activity patterns to coherent text output
    using learned semantic mappings and attractor dynamics.
    """
    
    def __init__(
        self,
        vocabulary_size: int = 1000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        beam_width: int = 5,
        input_dim: Optional[int] = None  # For backward compatibility
    ):
        # Handle backward compatibility
        self.input_dim = input_dim if input_dim is not None else embedding_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocabulary_size = vocabulary_size
        self.beam_width = beam_width
        
        # Components
        self.semantic_space = SemanticSpace(self.input_dim, n_concepts=min(500, vocabulary_size // 2))
        self.attractor = AttractorDynamics(n_states=100)
        self.beam_search = BeamSearchDecoder(beam_width=beam_width, max_length=15)
        
        # Response templates (for structure)
        self.templates = {
            'greeting': ["Hello!", "Hi there.", "Greetings."],
            'acknowledgment': ["I understand.", "I see.", "Got it."],
            'uncertainty': ["I'm not sure.", "Let me think...", "Hmm..."],
            'positive': ["That's great!", "Wonderful.", "Excellent."],
            'negative': ["I see the concern.", "That's difficult.", "I understand."],
            'question_response': ["That's an interesting question.", "Let me consider that."],
            'thinking': ["Let me think about that...", "Processing...", "Considering..."],
        }
        
        # Context
        self.conversation_history: List[Tuple[str, str]] = []  # (input, output) pairs
        self.current_mood = "neutral"
        self.context_pattern = np.zeros(self.input_dim)
        
        # Word invention capabilities
        self.word_inventor: Optional['WordInventor'] = None  # Lazy init
        self.invented_words: Dict[str, Dict] = {}
        self.invention_threshold = 0.15  # Invent when confidence below this
        self.creativity_level = 0.3  # 0 = never invent, 1 = always invent
    
    def decode(
        self,
        activity_pattern: np.ndarray,
        mood: str = "neutral",
        temperature: float = 0.7
    ) -> Tuple[str, float]:
        """
        Decode neural activity pattern to text
        
        Args:
            activity_pattern: Neural activity from executive region
            mood: Current emotional state
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Tuple of (response_text, confidence)
        """
        self.current_mood = mood
        
        # Normalize pattern
        pattern = activity_pattern.copy()
        
        # Ensure pattern is the right size
        if len(pattern) < self.input_dim:
            padded = np.zeros(self.input_dim)
            padded[:len(pattern)] = pattern
            pattern = padded
        elif len(pattern) > self.input_dim:
            pattern = pattern[:self.input_dim]
        
        norm = np.linalg.norm(pattern)
        if norm > 1e-8:
            pattern = pattern / norm
        else:
            return self._get_default_response(mood), 0.1
        
        # Combine with context
        combined_pattern = 0.8 * pattern + 0.2 * self.context_pattern
        
        # Decode to concepts
        concepts = self.semantic_space.decode_pattern(combined_pattern, top_k=5)
        
        if not concepts:
            return self._get_default_response(mood), 0.1
        
        # Get primary concept
        primary_concept, confidence = concepts[0]
        
        if confidence < 0.2:
            return self._get_default_response(mood), confidence
        
        # Generate response based on concept and mood
        response = self._generate_response(
            primary_concept, 
            concepts, 
            mood, 
            temperature
        )
        
        # Update context
        self.context_pattern = 0.9 * self.context_pattern + 0.1 * pattern
        
        return response, confidence
    
    def _generate_response(
        self,
        primary_concept: str,
        concepts: List[Tuple[str, float]],
        mood: str,
        temperature: float
    ) -> str:
        """Generate response from concepts and mood"""
        
        # Check for template match
        template_key = self._match_template(primary_concept, mood)
        if template_key and np.random.random() < 0.3:
            templates = self.templates.get(template_key, [])
            if templates:
                return np.random.choice(templates)
        
        # Build response from concepts
        words = []
        
        for concept_id, confidence in concepts[:3]:
            if confidence < 0.15:
                continue
            
            if concept_id in self.semantic_space.concepts:
                concept = self.semantic_space.concepts[concept_id]
                
                # Sample word based on temperature
                if concept.words:
                    if temperature < 0.3:
                        word = concept.words[0]  # Most likely
                    else:
                        # Temperature-based sampling
                        probs = np.exp(np.arange(len(concept.words), 0, -1) / temperature)
                        probs /= np.sum(probs)
                        word = np.random.choice(concept.words, p=probs)
                    
                    if word not in words:  # Avoid repetition
                        words.append(word)
        
        if not words:
            return self._get_default_response(mood)
        
        # Construct sentence
        response = self._construct_sentence(words, mood)
        
        return response
    
    def _match_template(self, concept: str, mood: str) -> Optional[str]:
        """Match concept to template category"""
        greeting_concepts = {'greeting', 'hello', 'hi'}
        positive_concepts = {'happiness', 'gratitude', 'affirmation'}
        negative_concepts = {'sadness', 'negation', 'confusion'}
        question_concepts = {'question', 'curiosity'}
        
        if concept in greeting_concepts:
            return 'greeting'
        if concept in positive_concepts:
            return 'positive'
        if concept in negative_concepts:
            return 'negative'
        if concept in question_concepts:
            return 'question_response'
        if mood == 'confused' or concept == 'uncertainty':
            return 'uncertainty'
        
        return None
    
    def _construct_sentence(self, words: List[str], mood: str) -> str:
        """Construct grammatical sentence from words"""
        if not words:
            return ""
        
        # Simple sentence construction
        sentence = ' '.join(words)
        
        # Add mood-based prefix/suffix
        if mood in ['happy', 'excited']:
            if np.random.random() < 0.3:
                sentence = "I feel that " + sentence
        elif mood in ['sad', 'depressed']:
            if np.random.random() < 0.3:
                sentence = "It seems " + sentence
        elif mood == 'curious':
            if np.random.random() < 0.3:
                sentence = "I wonder about " + sentence
        elif mood == 'thinking':
            sentence = "I'm considering " + sentence
        
        # Capitalize and punctuate
        sentence = sentence.strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
        
        return sentence
    
    def _get_default_response(self, mood: str) -> str:
        """Get default response based on mood"""
        defaults = {
            'happy': "I'm feeling positive about this.",
            'sad': "I'm processing this...",
            'curious': "That's interesting to consider.",
            'confused': "I need to think about that more.",
            'calm': "I understand.",
            'excited': "This is engaging!",
            'anxious': "Let me consider this carefully.",
            'focused': "I'm processing the information.",
            'neutral': "I see.",
        }
        return defaults.get(mood, "Processing...")
    
    def train(self, input_text: str, output_text: str, reward: float = 1.0) -> None:
        """Train decoder on input-output pair"""
        # Encode both
        input_pattern = self.semantic_space.encode_text(input_text)
        output_words = self.semantic_space._tokenize(output_text)
        
        # Learn associations
        self.semantic_space.learn(input_pattern, output_words, reward)
        
        # Store in history
        self.conversation_history.append((input_text, output_text))
        if len(self.conversation_history) > 100:
            self.conversation_history.pop(0)
    
    def get_vocabulary_stats(self) -> Dict:
        """Get vocabulary statistics"""
        return {
            'n_words': len(self.semantic_space.word_embeddings),
            'n_concepts': len(self.semantic_space.concepts),
            'n_prototypes_used': int(np.sum(self.semantic_space.prototype_counts > 0)),
            'conversation_length': len(self.conversation_history),
            'invented_words': len(self.invented_words),
        }
    
    def invent_response_word(self, pattern: np.ndarray, context: str = "") -> str:
        """
        Invent a new word when existing vocabulary is insufficient.
        """
        if self.word_inventor is None:
            self.word_inventor = WordInventor()
        
        # Get emotional context from current mood
        valence = 0.0
        arousal = 0.5
        mood_valence = {
            'happy': 0.7, 'sad': -0.6, 'curious': 0.3,
            'confused': -0.2, 'excited': 0.6, 'anxious': -0.3,
            'calm': 0.1, 'focused': 0.2, 'neutral': 0.0
        }
        mood_arousal = {
            'happy': 0.6, 'sad': 0.3, 'curious': 0.6,
            'confused': 0.5, 'excited': 0.8, 'anxious': 0.7,
            'calm': 0.2, 'focused': 0.4, 'neutral': 0.4
        }
        valence = mood_valence.get(self.current_mood, 0.0)
        arousal = mood_arousal.get(self.current_mood, 0.5)
        
        word, metadata = self.word_inventor.invent_word(
            pattern, 
            semantic_hint=context,
            valence=valence,
            arousal=arousal
        )
        
        self.invented_words[word] = metadata
        return word
    
    def maybe_invent_word(self, pattern: np.ndarray, confidence: float, context: str = "") -> Optional[str]:
        """
        Maybe invent a word if confidence is low and creativity is enabled.
        """
        if confidence < self.invention_threshold and np.random.random() < self.creativity_level:
            return self.invent_response_word(pattern, context)
        return None


class WordInventor:
    """
    System for inventing novel words based on neural patterns.
    
    The brain can create new words when existing vocabulary
    is insufficient to express a concept. Words are formed
    by combining phonemes based on the neural activation pattern.
    """
    
    def __init__(self):
        # Phoneme inventory (simplified English-like)
        self.consonants = [
            'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm',
            'n', 'p', 'r', 's', 't', 'v', 'w', 'x', 'z',
            'ch', 'sh', 'th', 'ph', 'wh', 'qu', 'br', 'cr',
            'dr', 'fr', 'gr', 'pr', 'tr', 'bl', 'cl', 'fl',
            'gl', 'pl', 'sl', 'sc', 'sk', 'sm', 'sn', 'sp', 'st'
        ]
        self.vowels = [
            'a', 'e', 'i', 'o', 'u', 'ai', 'au', 'ea', 'ee',
            'ei', 'ie', 'oa', 'oo', 'ou', 'ue', 'ay', 'ey', 'oy'
        ]
        self.endings = [
            '', 's', 'ly', 'ness', 'tion', 'ing', 'ful', 'less',
            'ment', 'able', 'ible', 'ive', 'ous', 'al', 'ic', 'y'
        ]
        
        # Track invented words and their meanings
        self.invented_words: Dict[str, Dict] = {}
        
        # Word creation history for consistency
        self.pattern_to_word: Dict[int, str] = {}
    
    def _pattern_to_seed(self, pattern: np.ndarray) -> int:
        """Convert neural pattern to a reproducible seed."""
        # Use pattern characteristics to generate seed
        return int(abs(hash(pattern.tobytes())) % (2**31))
    
    def invent_word(
        self,
        pattern: np.ndarray,
        semantic_hint: str = "",
        valence: float = 0.0,
        arousal: float = 0.5
    ) -> Tuple[str, Dict]:
        """
        Invent a new word based on neural activation pattern.
        
        Args:
            pattern: Neural activity pattern
            semantic_hint: Optional semantic category hint
            valence: Emotional valence (-1 to 1)
            arousal: Arousal level (0 to 1)
            
        Returns:
            (invented_word, metadata)
        """
        seed = self._pattern_to_seed(pattern)
        
        # Check if we already invented a word for similar pattern
        if seed in self.pattern_to_word:
            word = self.pattern_to_word[seed]
            return word, self.invented_words.get(word, {})
        
        rng = np.random.RandomState(seed)
        
        # Determine word structure based on pattern statistics
        pattern_energy = np.mean(np.abs(pattern))
        pattern_variance = np.var(pattern)
        
        # More complex patterns -> longer words
        if pattern_energy > 0.5:
            n_syllables = rng.choice([2, 3, 3, 4])
        elif pattern_energy > 0.3:
            n_syllables = rng.choice([1, 2, 2, 3])
        else:
            n_syllables = rng.choice([1, 1, 2])
        
        # Build syllables
        word = ""
        for i in range(n_syllables):
            # Use different parts of pattern for each syllable
            start_idx = int(i * len(pattern) / n_syllables)
            end_idx = int((i + 1) * len(pattern) / n_syllables)
            syllable_pattern = pattern[start_idx:end_idx]
            
            # Map pattern values to phoneme indices
            if len(syllable_pattern) > 0:
                c_idx = int(abs(np.mean(syllable_pattern) * 1000)) % len(self.consonants)
                v_idx = int(abs(np.sum(syllable_pattern) * 100)) % len(self.vowels)
            else:
                c_idx = rng.randint(len(self.consonants))
                v_idx = rng.randint(len(self.vowels))
            
            # Consonant-Vowel or Vowel-Consonant based on position
            if i == 0 or rng.random() > 0.3:
                syllable = self.consonants[c_idx] + self.vowels[v_idx]
            else:
                syllable = self.vowels[v_idx] + self.consonants[c_idx]
            
            word += syllable
        
        # Maybe add ending based on valence/arousal
        if arousal > 0.7 and rng.random() > 0.5:
            word += rng.choice(['!', 'ish', 'y'])
        elif valence > 0.5 and rng.random() > 0.6:
            word += rng.choice(self.endings[:8])  # Positive endings
        elif valence < -0.3 and rng.random() > 0.6:
            word += rng.choice(['less', 'nt', 'x'])  # Harsher sounds
        
        # Store metadata
        metadata = {
            'seed': seed,
            'pattern_energy': float(pattern_energy),
            'pattern_variance': float(pattern_variance),
            'semantic_hint': semantic_hint,
            'valence': valence,
            'arousal': arousal,
            'syllables': n_syllables,
            'invented_at': len(self.invented_words),
        }
        
        self.invented_words[word] = metadata
        self.pattern_to_word[seed] = word
        
        return word, metadata
    
    def get_word_meaning(self, word: str) -> Optional[Dict]:
        """Get the metadata/meaning of an invented word."""
        return self.invented_words.get(word)
    
    def create_neologism_for_concept(
        self,
        concept_pattern: np.ndarray,
        existing_words: List[str],
        creativity: float = 0.5
    ) -> str:
        """
        Create a neologism when existing words are insufficient.
        
        Blends characteristics of related existing words with
        novel phoneme sequences from the pattern.
        """
        if not existing_words or creativity > 0.8:
            # Pure invention
            return self.invent_word(concept_pattern)[0]
        
        # Blend existing words with novel elements
        seed = self._pattern_to_seed(concept_pattern)
        rng = np.random.RandomState(seed)
        
        # Take parts of existing words
        parts = []
        for word in existing_words[:2]:
            if len(word) > 3:
                # Take beginning or end
                if rng.random() > 0.5:
                    parts.append(word[:len(word)//2])
                else:
                    parts.append(word[len(word)//2:])
            else:
                parts.append(word)
        
        # Combine with novel phoneme
        v_idx = int(abs(np.mean(concept_pattern) * 100)) % len(self.vowels)
        connector = self.vowels[v_idx]
        
        if len(parts) >= 2:
            neologism = parts[0] + connector + parts[1]
        elif len(parts) == 1:
            neologism = parts[0] + connector + self.invent_word(concept_pattern)[0][:3]
        else:
            neologism = self.invent_word(concept_pattern)[0]
        
        self.invented_words[neologism] = {
            'type': 'blend',
            'source_words': existing_words,
            'pattern_based': True,
        }
        
        return neologism
