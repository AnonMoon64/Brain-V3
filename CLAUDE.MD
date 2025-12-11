# CLAUDE.MD - Brain-V3 Project Guide

## Project Overview

**Brain-V3** is a proof-of-concept chatbot implementing a neurochemically-modulated neural network where **memory IS architecture**. Unlike traditional chatbots with fixed architectures and text-based memory, this system physically changes its neural structure through simulated neurochemistry during each conversation.

**Key Philosophy:** Knowledge and memory are woven into the architecture itself. Neurons ARE memories.

## Core Concepts

### 1. Simulated Neurochemistry

Ten neurochemicals modulate brain behavior in real-time:

- **Dopamine** - Reward, motivation, reinforcement learning
- **Serotonin** - Mood stability, pattern consolidation
- **Cortisol** - Stress, aggressive pruning, threat response
- **Oxytocin** - Social bonding, connection strengthening
- **Norepinephrine** - Alertness, attention, novelty detection
- **GABA** - Inhibition, calm, noise reduction
- **Glutamate** - Excitation, new connections, learning speed
- **Acetylcholine** - Attention, memory formation
- **Endorphin** - Pain modulation, reward
- **Adrenaline** - Fight/flight, arousal

Chemical combinations create distinct "moods" that influence learning:

- High dopamine + low cortisol = Curious, exploratory
- High cortisol + norepinephrine = Alert, defensive
- High serotonin + GABA = Calm, consolidating
- High oxytocin + dopamine = Bonding, trusting

### 2. Dynamic Neural Architecture

The network evolves through:

- **Neuron spawning** - New neurons emerge for novel patterns
- **Neuron death** - Unused or conflicting neurons are pruned
- **Hebbian learning** - "Neurons that fire together wire together"
- **Chemical modulation** - Mood affects growth rate, pruning, and plasticity

### 3. Emergent Memory

Memories are NOT stored as text but as patterns of neural activation:

- Chemical state during encoding affects memory strength
- Emotional memories (high chemical activity) are stronger
- Retrieval is mood-congruent (current chemistry affects recall)
- Memories decay without recall, consolidate when reinforced

## Architecture

### Three-System Brain (Primary Architecture)

All functionality is consolidated into 6 core files:

```text
brain/
├── __init__.py              # Package exports
├── three_system_brain.py    # PRIMARY: ThreeSystemBrain + 3 systems + SDRMemory
│
# === SUPPORT MODULES (5 files) ===
├── neuromodulation.py       # 10 neurochemicals, kinetic binding, three-factor learning
├── language_decoder.py      # Semantic space, attractor dynamics for text output
├── signal_processing.py     # Input normalization, noise handling
├── hierarchical_time.py     # Multi-rate simulation, nested oscillations
└── persistence.py           # Save/load with dill

chatbot.py                   # Interactive CLI with dashboard
run_dashboard.py             # GUI dashboard
```

### The Three Systems

**ThreeSystemBrain** combines three interacting subsystems:

**System 1 - Sparse Cortical Engine** (`SparseCorticalEngine`)

- HTM-like sparse distributed representations (SDR)
- K-winners-take-all (~2% sparsity like biological cortex)
- Cortical microcircuits with minicolumns
- Predictive coding with feedforward/feedback paths
- All "concepts," "patterns," "features" live here

**System 2 - Dynamic Recurrent Core** (`DynamicRecurrentCore`)

- Echo State Network reservoir computing
- Working memory and temporal sequences
- Multi-step reasoning and pattern generation
- The "temporal glue" that traditional brain simulators lack

**System 3 - Neuromodulated Learning** (`NeuromodulatedLearningSystem`)

- Kinetic receptor binding with stochastic dynamics
- Three-factor learning (pre × post × neuromodulator)
- Cross-modulator antagonism (DA-5HT, NE-ACh)
- Epigenetic learning switches
- Governs WHEN cortex/reservoir change, not computation itself

### Advanced Features (V2)

**Self-Compression Engine** (`SelfCompressionEngine`)

- Cortex invents new columns to represent recurring patterns
- Reservoir discovers dynamical modes through SVD
- Semantic concepts emerge from co-activation patterns
- Automatic primitives creation for efficient representation

**Internal Body / Motor Loop** (`InternalBody`)

- 8 "muscles" with activation, fatigue, and recovery
- Proprioceptive feedback loop (body → brain → body)
- Internal sensations: energy, arousal, comfort
- Internal reward signal for closed perception-action loop
- Homeostatic drives affect neurochemistry

**Global Event Queue** (`GlobalEventQueue`)

- Single priority queue replaces 4 separate temporal layers
- Events: gamma (fast), beta, theta, delta (slow), modulator
- Efficient scheduling by next-fire-time
- Properly handles multi-rate simulation

**LSH Hash Lattice** (`LSHHashLattice`)

- Locality-sensitive hashing for fast nearest-neighbor
- Multiple hash tables with random projections
- O(1) pattern lookup vs O(n) exhaustive search
- Used for SDR similarity and memory retrieval

## Running the Project

### Requirements

```bash
# Pure Python - no external dependencies for basic functionality!
python chatbot.py
```

For advanced features:

```bash
pip install -r requirements.txt
```

### Main Entry Points

**Interactive CLI:**

```bash
python chatbot.py
```

**GUI Dashboard:**

```bash
python run_dashboard.py
```

### CLI Commands

- `/status` - Show brain dashboard (chemicals, neurons, memory)
- `/introspect` - Brain describes its own state
- `/chemicals` - Show neuromodulator levels
- `/stats` - Show brain statistics
- `/regions` - Show brain region activity
- `/save` - Save brain state
- `/clear` - Clear screen
- `/help` - Show help
- `/quit` - Exit

## Code Patterns & Conventions

### Creating a Brain Instance

```python
from brain import create_brain, ThreeSystemBrain

# Create brain with preset scale (now uses ThreeSystemBrain)
brain = create_brain(scale="small")
# Scales: "micro" (800 neurons), "small" (3.2K), "medium" (6.4K), "large" (16K)

# Process input
result = brain.process(
    "Hello, how are you?",
    reward=0.5,      # Positive feedback (-1 to 1)
    arousal=0.6      # Attention level (0 to 1)
)

# Get dashboard data
data = brain.get_dashboard_data()
print(f"Mood: {data['mood']}")
print(f"Active neurons: {data['neurons']['active']}")

# Get detailed stats
stats = brain.get_stats()
print(f"Global activity: {stats['global_activity']:.3f}")
print(f"Dopamine: {stats['neuromodulation']['dopamine']:.3f}")
```

### Key Classes

**ThreeSystemBrain** (brain/three_system_brain.py)

- Main unified brain system
- Methods: `process()`, `process_raw()`, `get_stats()`, `get_dashboard_data()`, `save()`, `load()`

**BrainConfig** (brain/three_system_brain.py)

- Configuration dataclass for brain parameters
- Attributes: `input_dim`, `num_columns`, `cells_per_column`, `reservoir_size`, etc.

**BrainState** (brain/three_system_brain.py)

- Current state tracking
- Attributes: `simulation_step`, `dopamine_level`, `mood`, etc.

## Development Guidelines

### When Adding Features

1. **Preserve the core philosophy**: Memory = Architecture
2. **Chemical modulation first**: New behaviors should be influenced by neurochemical state
3. **Biological plausibility**: Features should have neuroscience justification
4. **Emergent behavior**: Avoid hardcoded responses; let structure determine behavior
5. **Pure Python preferred**: Minimize dependencies when possible

### Testing Changes

```bash
# Quick test: Run chatbot and check /status output
python chatbot.py

# Test specific brain scale
python -c "from brain import create_brain; b = create_brain('micro'); print(b.get_stats())"

# Test text processing
python -c "from brain import create_brain; b = create_brain('micro'); r = b.process('Hello!'); print(r['response'])"
```

### Common Pitfall: Over-Engineering

This project intentionally avoids:

- Traditional NLP/LLM integration (responses are emergent from structure)
- Separate memory databases (structure IS memory)
- Fixed response templates (behavior emerges from chemicals + structure)
- Traditional training loops (learning happens through growth/pruning)

## Future Development Areas

### Active Research Questions

1. **Neuron Spawning Rules**: When does a new neuron spawn?
   - Novelty detection threshold?
   - Chemical state triggers?
   - Probabilistic vs deterministic?

2. **Neuron Death Criteria**: When do neurons die?
   - Inactivity timeout?
   - Conflict detection?
   - Energy/metabolism constraints?

3. **Chemical Learning Modes**:
   - Anger (high cortisol) = aggressive pruning
   - Calm (high serotonin) = pattern reinforcement
   - How do emotional states shape long-term structure?

### Planned Features

- ✅ Self-compression (cortex/reservoir discover efficient representations)
- ✅ Motor loop (internal body with muscles/sensations)
- ✅ Event-driven scheduling (single priority queue)
- ✅ LSH hash lattice (O(1) pattern lookup)
- Inter-neuron competition for survival
- Sleep/consolidation cycles (offline replay)
- Long-term personality emergence
- Visual network graph visualization
- Enhanced persistent state (brain_saves/ directory)
- Multi-agent brain communication
- Sparse containers (COO/CSR) for memory efficiency
- Binary reservoir with lookup tables

## File Locations

### Save States

Brain states are saved to: `brain_saves/` with timestamped filenames like `brain_YYYYMMDD_HHMMSS.brain`

### Configuration

No external config files - all configuration is in code:

- Brain scales: `brain/three_system_brain.py` - `BrainConfig` class
- Chemical parameters: `brain/neuromodulation.py`
- Network parameters: `brain/three_system_brain.py`

## Debugging Tips

### Visualizing Brain State

```python
data = brain.get_dashboard_data()
print(f"Total neurons: {data['neurons']['total']}")
print(f"Active neurons: {data['neurons']['active']}")
print(f"Mood: {data['mood']}")
print(f"Chemicals: {data['chemicals']}")
print(f"Personality: {data['personality']}")
```

### Monitoring Chemical Changes

```python
# Before input
before_chems = brain.get_dashboard_data()['chemicals']

# Process input
brain.process("exciting message!", reward=0.8)

# After input
after_chems = brain.get_dashboard_data()['chemicals']

# Compare
for chem in before_chems:
    delta = after_chems[chem] - before_chems[chem]
    print(f"{chem}: {delta:+.3f}")
```

## Backward Compatibility

- `IntegratedBrain` is now an alias for `ThreeSystemBrain`
- `create_brain()` returns `ThreeSystemBrain`
- Old `integrated_brain.py` has been removed - all functionality is in `three_system_brain.py`

## Resources

### Key Papers (Inspiration)

- Hebbian Learning: "Cells that fire together wire together"
- Neuromodulation: Dopamine's role in reinforcement learning
- Sparse Coding: Cortical representation efficiency
- Reservoir Computing: Echo State Networks, Liquid State Machines

### Related Projects

This is a from-scratch implementation. Core inspiration:

- Spiking Neural Networks (SNN)
- Neuromorphic computing
- Artificial Life / ALife
- Self-organizing systems

---

*"The chat interface is just what emerges from that growing structure."*
