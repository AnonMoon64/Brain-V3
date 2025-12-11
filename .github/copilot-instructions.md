# Brain-V3 AI Agent Instructions

## Project Philosophy

**Memory IS Architecture**: This is NOT a traditional chatbot. Memories are not stored as text but as patterns of neural activation. The network topology itself encodes experience. Responses emerge from dynamic neural structure, not templates or LLM completions.

**Chemical Modulation First**: All behavior is influenced by 10 neurochemicals (dopamine, serotonin, cortisol, oxytocin, norepinephrine, GABA, glutamate, acetylcholine, endorphin, adrenaline). Learning modes, neuron spawning, and pruning decisions are gated by chemical state, not fixed algorithms.

**Emergent > Hardcoded**: Avoid hardcoded responses, separate memory stores, or traditional training loops. Behavior emerges from structure + chemistry + dynamics.

## Three-System Architecture (Primary Architecture)

The codebase implements a concentrated intelligence architecture with three interacting subsystems in `three_system_brain.py`:

1. **System 1 - Sparse Cortical Engine** (`SparseCorticalEngine`)
   - HTM-like sparse distributed representations (SDR), k-winners-take-all (2% sparsity)
   - Cortical microcircuits with minicolumns
   - Predictive coding with feedforward/feedback paths
   - All "concepts," "patterns," "features" live here

2. **System 2 - Dynamic Recurrent Core** (`DynamicRecurrentCore`)
   - Reservoir computing (Echo State Networks)
   - Working memory, temporal sequences, multi-step reasoning
   - Multi-rate simulation
   - The "temporal glue" that traditional brain simulators lack

3. **System 3 - Neuromodulated Learning** (`NeuromodulatedLearningSystem`)
   - Kinetic receptor binding with stochastic dynamics
   - Three-factor learning (pre × post × neuromodulator)
   - Cross-modulator antagonism (DA-5HT, NE-ACh)
   - Epigenetic learning switches
   - Governs WHEN cortex/reservoir change, not computation itself

## Advanced Features (V2)

4. **Self-Compression Engine** (`SelfCompressionEngine`)
   - Cortex invents new columns to represent recurring patterns
   - Reservoir discovers dynamical modes through SVD
   - Semantic concepts emerge from co-activation patterns

5. **Internal Body / Motor Loop** (`InternalBody`)
   - 8 "muscles" with activation, fatigue, and recovery
   - Proprioceptive feedback (body → brain → body)
   - Internal sensations: energy, arousal, comfort
   - Homeostatic drives affect neurochemistry

6. **Global Event Queue** (`GlobalEventQueue`)
   - Single priority queue replaces 4 separate temporal layers
   - Events: gamma (fast), beta, theta, delta (slow), modulator
   - Efficient scheduling by next-fire-time

7. **LSH Hash Lattice** (`LSHHashLattice`)
   - Locality-sensitive hashing for fast nearest-neighbor
   - O(1) pattern lookup vs O(n) exhaustive search

**Interaction loop**: Sensory input → Sparse Cortex (patterns) → Reservoir (trajectories) → Cortex (predictions) → Neuromodulation (plasticity decisions) → Language Decoder (output).

## Key Entry Points

### Creating Brains
```python
from brain import create_brain, ThreeSystemBrain

# Create brain (all scales use ThreeSystemBrain)
brain = create_brain(scale="small")
# Scales: "micro" (800 neurons), "small" (3.2K), "medium" (6.4K), "large" (16K)

# Or directly
brain = ThreeSystemBrain(config=BrainConfig(...))
```

### Processing Input
```python
# Text processing (chatbot interface)
result = brain.process("Hello, how are you?")
print(result['response'])  # Generated text
print(result['mood'])      # e.g., "excited", "calm", "stressed"

# With explicit reward/arousal
result = brain.process(
    text="exciting news!",
    reward=0.8,        # Positive feedback → dopamine release
    arousal=0.7        # Attention level → norepinephrine
)

# Raw array processing (for advanced use)
result = brain.process_raw(input_array, dt=0.01)
```

### Inspection & Debugging
```python
# Dashboard data for UI/monitoring
data = brain.get_dashboard_data()
print(f"Mood: {data['mood']}")
print(f"Active neurons: {data['neurons']['active']}")
print(f"Chemicals: {data['chemicals']}")

# Detailed statistics
stats = brain.get_stats()
print(f"Global activity: {stats['global_activity']:.3f}")
print(f"Sparsity: {stats['sparsity']:.3f}")
```

## Critical Conventions

### 1. No External Dependencies (Core)
Core functionality is **pure Python + NumPy**. Optional dependencies:
- `dill` for full object serialization (recommended)
- `PyQt6` for GUI dashboard (optional)
- `openai` for training experiments (optional)

### 2. State Persistence Pattern
Brain states save to `brain_saves/` using dill for complete object graph serialization:

```python
# Save (auto-generates timestamped filename)
filepath = brain.save()  # → brain_saves/brain_YYYYMMDD_HHMMSS.brain

# Load
brain.load('brain_saves/brain_20251209_020928.brain')
```

### 3. Configuration is Code-Based
All parameters are in `three_system_brain.py`:
- Brain scales: `BrainConfig` dataclass
- Chemical baselines: In `BrainConfig` and `NeuromodulatedLearningSystem`
- Network topology: `SparseCorticalEngine`, `DynamicRecurrentCore`

### 4. Dashboard Commands (CLI)
When in `chatbot.py` interactive mode:
- `/status` - Full brain dashboard
- `/chemicals` - Neuromodulator levels
- `/stats` - Detailed statistics
- `/regions` - Brain region activity
- `/save` - Save current state
- `/introspect` - Brain describes its own state
- `/clear`, `/help`, `/quit`

## Running the Project

### CLI Chatbot
```powershell
python chatbot.py
```

### GUI Dashboard
```powershell
# Requires: pip install PyQt6
python run_dashboard.py
```

### Quick Tests
```powershell
# Create and inspect a brain
python -c "from brain import create_brain; b = create_brain('micro'); print(b.get_stats())"

# Test text processing
python -c "from brain import create_brain; b = create_brain('micro'); r = b.process('Hello!'); print(r['response'])"
```

## Development Patterns

### Adding Neurobiological Features
1. **Start with chemical modulation**: New behaviors should be influenced by neurochemical state
2. **Add to existing systems**: Extend SparseCorticalEngine, DynamicRecurrentCore, or NeuromodulatedLearningSystem
3. **Preserve emergent behavior**: Don't hardcode responses; let structure determine output
4. **Test at multiple scales**: Verify at "micro" (fast) before larger scales

### The Three Systems Have Clear Roles
- **System 1 (Cortex)**: Pattern recognition, sparse codes, predictions
- **System 2 (Reservoir)**: Temporal dynamics, working memory, sequences  
- **System 3 (Learning)**: When to learn, what to reinforce, neurogenesis

### Dynamic Neurogenesis
Managed by `NeuromodulatedLearningSystem`:
```python
# Spawning triggers (in should_create_neurons):
# 1. High novelty (low prediction accuracy)
# 2. High dopamine (reward signal)
# 3. Not fatigued

# Death triggers:
# 1. Prolonged inactivity
# 2. High cortisol (stress pruning)
```

## Common Pitfalls

### ❌ Don't: Integrate Traditional NLP/LLMs
This project intentionally avoids GPT/BERT/transformers for response generation. Language emerges from `language_decoder.py` attractor dynamics.

### ❌ Don't: Create Separate Memory Databases
Structure IS memory. Don't add JSON logs, SQLite, or key-value stores.

### ❌ Don't: Use Fixed Training Loops
Learning happens through neuromodulation and three-factor STDP, not backpropagation epochs.

### ✅ Do: Test Chemical Cocktails
Different moods create different behaviors:
```python
# Stressed brain (aggressive pruning)
brain.process(text, reward=-0.5, arousal=0.9)

# Calm brain (consolidation)
brain.process(text, reward=0.5, arousal=0.2)
```

### ✅ Do: Monitor Sparsity
Cortex should maintain ~2% active neurons. Check `stats['sparsity']` regularly.

## File Organization

The codebase is concentrated into 6 core files (plus interface files):

```
brain/
├── three_system_brain.py     # PRIMARY: All 3 systems + ThreeSystemBrain + SDRMemory
├── neuromodulation.py        # 10 neurochemicals, kinetic binding, three-factor learning
├── language_decoder.py       # Attractor dynamics for text output
├── signal_processing.py      # Input normalization pipeline
├── hierarchical_time.py      # Multi-rate simulation, oscillations
├── persistence.py            # Save/load with dill
└── __init__.py               # Package exports

chatbot.py                    # Interactive CLI
run_dashboard.py              # GUI launcher
gui/dashboard.py              # PyQt6 visualization
```

## Current Research Questions

When assisting with development, consider these open questions:

1. **Neuron Spawning**: Should it be novelty-driven? Chemical threshold? Probabilistic?
2. **Neuron Death**: Inactivity timeout? Conflict detection? Energy constraints?
3. **Sleep Cycles**: How to implement offline replay/consolidation?
4. **Multi-Agent**: How should multiple brains communicate?

## Testing Strategy

No formal test suite yet. Validate changes by:
1. Running `chatbot.py` and checking `/status` output
2. Verifying structure changes appear after conversations
3. Testing save/load round-trips
4. Monitoring chemical levels respond to reward/arousal
5. Checking sparsity stays ~2% for cortex

## Backward Compatibility

- `IntegratedBrain` is now an alias for `ThreeSystemBrain`
- `create_brain()` now returns `ThreeSystemBrain`
- Old `integrated_brain.py` has been removed - all functionality is in `three_system_brain.py`
