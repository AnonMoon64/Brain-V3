# CLAUDE.MD - Brain-V2 Project Guide

## Project Overview

**Brain-V2** is a proof-of-concept chatbot implementing a neurochemically-modulated neural network where **memory IS architecture**. Unlike traditional chatbots with fixed architectures and text-based memory, this system physically changes its neural structure through simulated neurochemistry during each conversation.

**Key Philosophy:** Knowledge and memory are woven into the architecture itself. Neurons ARE memories.

## Core Concepts

### 1. Simulated Neurochemistry
Seven neurochemicals modulate brain behavior in real-time:
- **Dopamine** - Reward, motivation, reinforcement learning
- **Serotonin** - Mood stability, pattern consolidation
- **Cortisol** - Stress, aggressive pruning, threat response
- **Oxytocin** - Social bonding, connection strengthening
- **Norepinephrine** - Alertness, attention, novelty detection
- **GABA** - Inhibition, calm, noise reduction
- **Glutamate** - Excitation, new connections, learning speed

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

### Directory Structure
```
brain/
├── __init__.py              # Package exports
├── brain.py                 # ChemicalBrain - original chatbot brain
├── chemicals.py             # ChemicalSystem - 7 neurochemicals
├── neurons.py               # SpikingNeuralNetwork - dynamic growth/pruning

# === MOUSE-LEVEL COMPLEXITY EXTENSIONS ===
├── integrated_brain.py      # IntegratedBrain - unified advanced system
├── neuromodulation.py       # Kinetic receptor binding, three-factor learning
├── cortical_architecture.py # Hierarchical cortex with minicolumns
├── metabolism.py            # Per-neuron ATP, homeostatic plasticity
├── signal_processing.py     # Robust normalization, noise handling
├── language_decoder.py      # Semantic space, attractor dynamics
├── sparse_network.py        # K-winners-take-all, event-driven sim
├── gpu_acceleration.py      # Quantized arrays, vectorized ops
├── reservoir.py             # Echo State Networks, Liquid State Machines
└── hierarchical_time.py     # Multi-rate simulation, nested oscillations

chatbot.py                   # Interactive CLI with dashboard
run_dashboard.py             # GUI dashboard
```

### Mouse-Level Complexity Features

**IntegratedBrain** combines 10 advanced neurobiological features:

1. **Kinetic Neuromodulation** (`neuromodulation.py`)
   - Michaelis-Menten receptor binding kinetics (D1/D2, alpha/beta adrenergic, muscarinic, 5-HT)
   - Three-factor learning (pre-synaptic × post-synaptic × neuromodulator)
   - Second messenger cascades (cAMP, PKA, MAPK)
   - Metaplasticity (plasticity rules change based on history)

2. **Hierarchical Cortical Architecture** (`cortical_architecture.py`)
   - 6-layer cortical structure (L1-L6)
   - Minicolumns (~100 neurons per functional unit)
   - Topographic mapping with preserved spatial relationships
   - Lateral inhibition (Mexican-hat connectivity)

3. **Neuron-Specific Metabolism** (`metabolism.py`)
   - Per-neuron ATP and mitochondrial dynamics
   - Synaptic scaling (homeostatic regulation)
   - Intrinsic plasticity (threshold adaptation)
   - Astrocyte metabolic coupling

4. **Sparse Representations** (`sparse_network.py`)
   - K-winners-take-all (only 2% neurons active, like cortex)
   - Event-driven simulation (update only active neurons)
   - Sparse Distributed Memory (content-addressable storage)

5. **Reservoir Computing** (`reservoir.py`)
   - Echo State Networks (fixed random recurrent layer)
   - Liquid State Machines (spiking reservoir for temporal patterns)
   - Hybrid reservoirs (fast ESN + slow LSM dynamics)

6. **Hierarchical Time Scales** (`hierarchical_time.py`)
   - Multi-rate simulation (sensory 1000Hz, association 100Hz, memory 1Hz)
   - Nested oscillations (gamma 40Hz nested in theta 6Hz)
   - Cross-frequency coupling (phase-amplitude modulation)

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
from brain import create_brain, IntegratedBrain

# Create brain with preset scale
brain = create_brain(scale="small", use_gpu=False)
# Scales: "micro" (10K neurons), "small" (100K), "medium" (1M), "large" (10M)

# Process input
result = brain.process(
    "Hello, how are you?",
    reward=0.5,      # Positive feedback (0-1)
    arousal=0.6      # Attention level (0-1)
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

**IntegratedBrain** (brain/integrated_brain.py)
- Main unified brain system
- Methods: `process()`, `get_stats()`, `get_dashboard_data()`, `save()`, `load()`

**ChemicalSystem** (brain/chemicals.py)
- Manages 7 neurochemicals
- Methods: `update()`, `get_mood()`, `modulate()`

**SpikingNeuralNetwork** (brain/neurons.py)
- Dynamic growth/pruning
- Methods: `activate()`, `learn()`, `prune()`, `spawn_neurons()`

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

# Save/load test
python -c "
from brain import create_brain
b = create_brain('micro')
b.process('test input')
b.save('test.brain')
b2 = create_brain('micro')
b2.load('test.brain')
print('Success!')
"
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
- Inter-neuron competition for survival
- Sleep/consolidation cycles (offline replay)
- Long-term personality emergence
- Visual network graph visualization
- Enhanced persistent state (brain_saves/ directory)
- Multi-agent brain communication

## File Locations

### Save States
Brain states are saved to: `brain_saves/` or current directory as `brain_state_<timestamp>.brain`

### Configuration
No external config files - all configuration is in code:
- Brain scales: `brain/integrated_brain.py` - `BrainConfig` class
- Chemical parameters: `brain/chemicals.py`
- Network parameters: `brain/neurons.py`

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

### Tracing Network Growth
Watch for neuron/synapse changes in chatbot output:
```
[Structure changed: +3 neurons, +47 synapses]
[Concepts: hello, excited, meet]
```

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
