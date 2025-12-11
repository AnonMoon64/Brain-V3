# Brain-V3: Chemical Brain with Emergent Life Simulation 🧠🧪🦎

A biologically-inspired neural architecture where **memory IS structure**. This isn't a traditional AI with separate memory stores — the network topology itself encodes experience. Includes a complete creature simulation with DNA/RNA genetics, instincts, breeding, and evolution.

## The Core Philosophy

> **Structure IS Memory. Chemistry IS Mood. Behavior IS Emergent.**

- No hardcoded responses or templates
- No separate memory database
- No traditional training loops
- Behavior emerges from neural dynamics + neurochemistry + structural plasticity

## 🧬 Two Modes of Operation

### 1. Chatbot Mode (CLI)
Interactive conversation with a brain that grows and changes structure based on interaction.

```powershell
python chatbot.py
```

### 2. Life Simulation Mode (GUI)
Watch creatures with DNA-derived brains navigate a 2D world, eat, breed, sleep, and evolve.

```powershell
python run_dashboard.py
```

---

## 🧪 Neurochemical System

Ten neurochemicals modulate all brain behavior:

| Chemical | Effect |
|----------|--------|
| **Dopamine** | Reward, motivation, reinforcement learning |
| **Serotonin** | Mood stability, pattern consolidation |
| **Cortisol** | Stress, aggressive pruning, threat response |
| **Oxytocin** | Social bonding, connection strengthening |
| **Norepinephrine** | Alertness, attention, novelty detection |
| **GABA** | Inhibition, calm, noise reduction |
| **Glutamate** | Excitation, new connections, learning speed |
| **Acetylcholine** | Attention, memory formation |
| **Endorphin** | Pain modulation, reward boost |
| **Adrenaline** | Arousal, fight/flight responses |

### Chemical Cocktails Create Moods

- **High dopamine + low cortisol** → Curious, exploratory
- **High cortisol + norepinephrine** → Alert, defensive  
- **High serotonin + GABA** → Calm, consolidating
- **High oxytocin + dopamine** → Bonding, trusting

---

## 🏗️ Three-System Brain Architecture

The brain is organized into three interacting subsystems:

### System 1: Sparse Cortical Engine
- HTM-like sparse distributed representations (SDR)
- K-winners-take-all (~2% sparsity like real cortex)
- Cortical minicolumns with lateral inhibition
- Predictive coding with feedforward/feedback paths

### System 2: Dynamic Recurrent Core  
- Reservoir computing (Echo State Networks)
- Working memory and temporal sequences
- Multi-rate simulation for different time scales
- The "temporal glue" for sequence processing

### System 3: Neuromodulated Learning
- Kinetic receptor binding with stochastic dynamics
- Three-factor learning: pre × post × neuromodulator
- Cross-modulator antagonism (DA-5HT, NE-ACh)
- Epigenetic switches gate learning modes

---

## 🦎 Creature Simulation System

### DNA/RNA Genetics
- **60+ genes** controlling brain size, body shape, metabolism, behavior
- **Dominance patterns**: Complete, incomplete, codominant
- **RNA expression** with tissue-specific transcription
- **Developmental stages**: Embryo → Larva → Juvenile → Adult → Elder
- **Epigenetic marks** that can be inherited

### Phenotype Expression
```python
from brain import Genome, EmbodiedBrain

# Create a random genome
genome = Genome.random()

# Develop a creature with brain from DNA
creature = EmbodiedBrain.from_genome(genome)

# Phenotype varies: limbs, size, metabolism, brain architecture
print(f"Limbs: {creature.body.phenotype.num_limbs}")
print(f"Brain columns: {creature.brain.config.num_columns}")
```

### Instinct System with Arbitration
Eight core instincts compete for behavioral control:

| Instinct | Drive | Behavior |
|----------|-------|----------|
| **Hunger** | Energy deficit | Seek and consume food |
| **Thirst** | Hydration need | Find water |
| **Fear** | Threat detection | Flee from hazards |
| **Exploration** | Novelty seeking | Move and discover |
| **Social** | Proximity to others | Approach creatures |
| **Mating** | Reproductive drive | Find compatible mates |
| **Rest** | Fatigue | Sleep and dream |
| **Surface** | Oxygen need | Surface when drowning |

**Instinct Arbitration**: When instincts conflict, inhibition rules resolve them:
- Fear inhibits exploration (safety first)
- Hunger suppresses mating (survive before reproduce)
- Drowning overrides everything (emergency surface)

### Motor Constraints from Phenotype
Actions are gated by physical capability:
- **0 limbs (snake)**: Cannot walk or jump, can only slither
- **Has fins**: Can swim efficiently
- **Has wings**: Can fly
- **Body size**: Affects jump height and speed

### World Simulation
- **Procedural terrain**: Ground, platforms, water, hazards, shelters
- **Day/night cycle**: Affects light, temperature, creature behavior
- **Weather system**: Clear, rain, storm conditions
- **Food spawning**: Plants and meat with spoilage
- **Edge detection**: Creatures sense cliffs and avoid fatal drops

### Breeding System
```python
from brain import create_offspring, calculate_mate_compatibility

# Check compatibility (genes, species distance)
score = calculate_mate_compatibility(parent1, parent2)

# Create offspring with genetic crossover + mutation
child_genome = create_offspring(parent1.genome, parent2.genome)
```

---

## 🦠 Biological Systems

### Viral Infections
Viruses can infect creatures and modify gene expression:
```python
from brain import Virus, Infection, ImmuneSystem, COMMON_VIRUSES

# Infect a creature
infection = Infection(COMMON_VIRUSES['neural_virus'])
creature.infections.append(infection)

# Immune system fights back
immune = ImmuneSystem()
immune.respond(creature, infection)
```

### Microbiome (Gut-Brain Axis)
Gut bacteria affect mood and metabolism:
```python
from brain import Microbiome, PROBIOTIC_STRAINS

microbiome = Microbiome()
microbiome.add_strain(PROBIOTIC_STRAINS['serotonin_producer'])

# Effects on creature
mood_modifier = microbiome.get_serotonin_production()
metabolism_boost = microbiome.get_metabolism_effect()
```

### Cultural Transmission
Creatures can learn behaviors by observing others:
```python
from brain import CulturalMemory, LearnedBehavior, BehaviorType

culture = CulturalMemory(learning_rate=0.3)

# Observe another creature's successful foraging
culture.observe(
    performer_id="creature_42",
    behavior=LearnedBehavior(
        id="efficient_foraging",
        behavior_type=BehaviorType.FORAGING,
        action_biases={'move_to_food': 0.8}
    ),
    observed_reward=0.9
)
```

### Cross-Simulation Migration
Export creatures to share between simulations:
```python
from brain import save_creature_to_file, load_creature_from_file

# Export a creature
save_creature_to_file(creature, "my_creature.brain")

# Import in another simulation
imported = load_creature_from_file("my_creature.brain")
```

---

## 📁 Project Structure

```
brain/
├── __init__.py               # Package exports (all public API)
├── three_system_brain.py     # PRIMARY: ThreeSystemBrain + 3 subsystems
├── neuromodulation.py        # 10 neurochemicals, kinetic binding
├── language_decoder.py       # Attractor dynamics → text output
├── signal_processing.py      # Input normalization pipeline
├── hierarchical_time.py      # Multi-rate oscillations
├── persistence.py            # Save/load with dill
│
│ # Genetics & Development
├── dna.py                    # Genome, genes, crossover, mutation
├── rna.py                    # RNA expression, viruses, immune system
│
│ # Creature Simulation
├── world.py                  # 2D environment, terrain, weather
├── creature.py               # Physical body, phenotype, homeostasis
├── instincts.py              # Instinct system with arbitration
├── breeding.py               # Reproduction, epigenetics
├── embodiment.py             # Brain↔body wiring, sensory encoding
│
│ # Extended Biology
├── culture.py                # Cultural transmission, social learning
├── microbiome.py             # Gut-brain axis, bacteria
├── migration.py              # Cross-simulation creature transfer
│
│ # Optional/Advanced
├── cortical_architecture.py  # Extended cortical layers
├── metabolism.py             # Per-neuron ATP, homeostasis
├── sparse_network.py         # Sparse containers
├── reservoir.py              # Echo State Networks
└── gpu_acceleration.py       # Vectorized operations

gui/
├── __init__.py
└── dashboard.py              # PyQt6 GUI with game simulation

chatbot.py                    # CLI chatbot interface
run_dashboard.py              # GUI launcher
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NumPy (core dependency)
- PyQt6 (optional, for GUI)
- dill (optional, for full state persistence)

### Installation
```powershell
cd Brain-V3
pip install -r requirements.txt
```

### CLI Chatbot
```powershell
python chatbot.py
```

### GUI Dashboard
```powershell
python run_dashboard.py
```

### Programmatic Usage
```python
from brain import create_brain, World, EmbodiedBrain, Genome

# Create a brain directly
brain = create_brain(scale='micro')  # micro/small/medium/large
result = brain.process('Hello!', reward=0.5, arousal=0.3)
print(result['response'])
print(result['mood'])

# Create a DNA-derived creature
genome = Genome.random()
creature = EmbodiedBrain.from_genome(genome)

# Create a world
world = World(width=800, height=600)
```

---

## 🎮 Dashboard Commands

| Command | Description |
|---------|-------------|
| `/status` | Full brain dashboard |
| `/chemicals` | Neuromodulator levels |
| `/stats` | Detailed statistics |
| `/regions` | Brain region activity |
| `/introspect` | Brain describes its own state |
| `/save` | Save brain state |
| `/clear` | Clear screen |
| `/help` | Show help |
| `/quit` | Exit |

---

## 🔬 Advanced Features

### Self-Compression Engine
The cortex invents new columns to represent recurring patterns. Reservoir discovers dynamical modes through SVD for efficient representation.

### Internal Body / Motor Loop
8 virtual "muscles" with activation, fatigue, and recovery. Proprioceptive feedback creates perception-action loops.

### Global Event Queue
Single priority queue replaces separate temporal layers. Events: gamma (fast), beta, theta, delta (slow), modulator.

### LSH Hash Lattice
Locality-sensitive hashing for O(1) pattern lookup vs O(n) exhaustive search.

### Signal Processing Pipeline
Robust input normalization with outlier handling, noise injection for regularization.

---

## 🔮 Future Work

### Cross-System Brain Consolidation
Major architectural refactor to further unify the three brain systems into a more tightly integrated whole. This would enable:
- Seamless information flow between cortex, reservoir, and learning systems
- Unified plasticity rules across all components
- Better emergent dynamics from system interactions

### Other Research Directions
- Sleep/consolidation cycles with offline replay
- Multi-agent brain communication
- Visual network graph visualization
- Persistent personality emergence over long timescales
- More sophisticated neuron spawning heuristics

---

## 🧪 Testing

```powershell
# Quick import test
python -c "from brain import create_brain; b = create_brain('micro'); print('OK')"

# Run test suite
python tests/run_tests.py

# Or with pytest
pip install pytest
pytest -q
```

---

## 📊 Brain Scales

| Scale | Cortical Columns | Reservoir Size | Use Case |
|-------|------------------|----------------|----------|
| `micro` | ~200 | ~800 | Quick testing |
| `small` | ~400 | ~1600 | Development |
| `medium` | ~800 | ~3200 | Normal use |
| `large` | ~2000 | ~8000 | Complex tasks |

---

## 🧠 Key Principles

1. **Structure IS Memory**: No separate memory store. Network topology encodes experience.

2. **Chemical Modulation First**: All behavior influenced by 10 neurochemicals.

3. **Emergent > Hardcoded**: No templates. Behavior emerges from dynamics.

4. **Three-Factor Learning**: pre × post × neuromodulator influences all plasticity.

5. **Biological Plausibility**: Inspired by real neuroscience, not just metaphors.

---

## 📜 License

MIT License - See LICENSE file for details.

---

*"The chat interface is just what emerges from that growing structure."*

*"Memory IS architecture. Chemistry IS mood. Behavior IS emergent."*
