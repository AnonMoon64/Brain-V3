# Brain-V2: Chemical Brain Chatbot 🧠🧪

A proof-of-concept chatbot where **memory IS architecture**. This isn't a traditional chatbot with text logs - each conversation physically changes the neural structure through simulated neurochemistry.

## The Core Idea

> Knowledge and memory is woven into the architecture itself. Your neurons *are* your memories.

Traditional AI approach:
- Fixed architecture → train weights

**Our approach:**
- Minimal starting structure → grows based on interaction → chemistry influences which neurons form and which die → emergent learning

## How It Works

### 🧪 Simulated Neurochemicals

Seven chemicals influence brain behavior:

| Chemical | Effect |
|----------|--------|
| **Dopamine** | Reward, motivation, reinforcement learning |
| **Serotonin** | Mood stability, pattern consolidation |
| **Cortisol** | Stress, aggressive pruning, threat response |
| **Oxytocin** | Social bonding, connection strengthening |
| **Norepinephrine** | Alertness, attention, novelty detection |
| **GABA** | Inhibition, calm, noise reduction |
| **Glutamate** | Excitation, new connections, learning speed |

Different chemical cocktails create different "moods":
- High dopamine + low cortisol = **Curious, exploratory**
- High cortisol + high norepinephrine = **Alert, defensive**
- High serotonin + high GABA = **Calm, consolidating**
- High oxytocin + high dopamine = **Bonding, trusting**

### 🔮 Dynamic Neural Network

The neural network grows and prunes itself:

- **Neurons spawn** when encountering novel patterns
- **Neurons die** when unused or causing conflict
- **Synapses strengthen** with use (Hebbian: "neurons that fire together wire together")
- **Synapses weaken** without use
- **Chemical state influences** growth rate, pruning rate, and plasticity

### 💭 Chemical Memory

Memories aren't stored as text - they're patterns of neural activation:

- Chemical state at encoding affects memory strength
- Emotional memories (high chemical activity) are stronger
- Retrieval is mood-congruent (current chemicals affect what surfaces)
- Memories decay if not recalled, consolidate if reinforced

## Running the Chatbot

```bash
# No dependencies required - pure Python!
python chatbot.py
```

### Commands

| Command | Description |
|---------|-------------|
| `/status` | Show brain dashboard (chemicals, neurons, memory) |
| `/introspect` | Have the brain describe its own state |
| `/chemicals` | Show chemical levels |
| `/memory` | Show memory statistics |
| `/network` | Show neural network statistics |
| `/clear` | Clear the screen |
| `/help` | Show help |
| `/quit` | Exit |

### Example Session

```
You: Hello! I'm excited to meet you!
Brain: That's fascinating This is activating many patterns in me. I feel rewarded by this interaction.
  [Structure changed: +1 neurons, +15 synapses]
  [Concepts: hello, excited, meet]

You: /status
═════════════════════════════════════════════════════════
  🧠 CHEMICAL BRAIN STATUS
═════════════════════════════════════════════════════════

  Mood: EXCITED

  Neurochemicals:
    Dopamine     [███████████░░░░] 0.78
    Serotonin    [█████████░░░░░░] 0.62
    ...
```

## Architecture

```
brain/
├── __init__.py              # Package exports
├── brain.py                 # ChemicalBrain - original chatbot brain
├── chemicals.py             # ChemicalSystem - 7 neurochemicals
├── neurons.py               # SpikingNeuralNetwork - dynamic growth/pruning
│
│ # === MOUSE-LEVEL COMPLEXITY EXTENSIONS ===
│
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

## 🐭 Mouse-Level Complexity Extensions

The Brain-V2 now includes advanced systems targeting biological mouse-level complexity:

### Kinetic Neuromodulation (`neuromodulation.py`)
- **Receptor binding kinetics**: Michaelis-Menten dynamics for D1/D2 dopamine, alpha/beta adrenergic, muscarinic, 5-HT receptors
- **Three-factor learning**: Weight changes depend on pre-synaptic, post-synaptic, AND neuromodulator levels
- **Second messenger cascades**: cAMP, PKA, MAPK pathways
- **Metaplasticity**: The plasticity rules themselves change based on history

### Hierarchical Cortical Architecture (`cortical_architecture.py`)
- **6-layer cortical structure**: L1 (input), L2/3 (lateral), L4 (thalamic input), L5 (output), L6 (feedback)
- **Minicolumns**: ~100 neurons per functional unit
- **Topographic mapping**: Preserved spatial relationships
- **Lateral inhibition**: Mexican-hat connectivity pattern

### Neuron-Specific Metabolism (`metabolism.py`)
- **Per-neuron ATP**: Mitochondrial dynamics, oxidative stress
- **Synaptic scaling**: Homeostatic up/down regulation
- **Intrinsic plasticity**: Threshold adaptation to maintain target rates
- **Astrocyte support**: Metabolic coupling between glia and neurons

### Sparse Representations (`sparse_network.py`)
- **K-winners-take-all**: Only 2% of neurons active (like cortex)
- **Event-driven simulation**: Update only active neurons
- **Sparse Distributed Memory**: Content-addressable storage

### Reservoir Computing (`reservoir.py`)
- **Echo State Networks**: Fixed random recurrent layer, trainable readout
- **Liquid State Machines**: Spiking reservoir for temporal patterns
- **Hybrid reservoirs**: Fast (ESN) + slow (LSM) dynamics

### Hierarchical Time Scales (`hierarchical_time.py`)
- **Multi-rate simulation**: Sensory at 1000Hz, association at 100Hz, memory at 1Hz
- **Nested oscillations**: Gamma (40Hz) nested in theta (6Hz)
- **Cross-frequency coupling**: Phase-amplitude modulation

### Using the Advanced Brain

```python
from brain import IntegratedBrain, create_brain

# Create brain with preset scale
brain = create_brain(scale="small", use_gpu=False)

# Process input
response = brain.process_input(
    "Hello, how are you?",
    reward=0.5,      # Positive feedback
    arousal=0.6      # Moderate attention
)

# Get detailed stats
stats = brain.get_stats()
print(f"Active neurons: {stats['global_activity']:.3f}")
print(f"Dopamine level: {stats['neuromodulation']['dopamine']:.3f}")
```

## Key Questions We're Exploring

1. **When does a new neuron spawn?**
   - When the system encounters something it can't handle?
   - When a chemical threshold hits?
   - Random but biased?

2. **When do neurons die?**
   - Unused ones?
   - Ones that cause conflict?
   - Based on chemical state?

3. **How do chemicals affect learning?**
   - "Anger" (high cortisol) = more aggressive pruning
   - "Calm" (high serotonin) = reinforce existing patterns
   - Different chemical states = different learning modes

## What Makes This Unique

1. **Structure IS Memory**: No separate memory store. The network topology encodes experience.

2. **Chemical Modulation**: Learning isn't fixed - it changes based on emotional state.

3. **Emergent Behavior**: Responses emerge from the dynamic structure, not templates.

4. **Visible Evolution**: Watch the brain develop through the dashboard.

5. **Biologically Inspired**: Modernizing old-school neural concepts with chemical simulation.

## Future Directions

- More sophisticated neuron spawning rules
- Inter-neuron competition for survival
- Sleep/consolidation cycles
- Longer-term personality emergence
- Visual network graph
- Persistent brain state (save/load)

---

*"The chat interface is just what emerges from that growing structure."*
