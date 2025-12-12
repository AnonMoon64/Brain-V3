# Brain-V3: Chemical Brain with Emergent Life Simulation 🧠🧪🦎

A biologically-inspired neural architecture where **memory IS structure**. This isn't a traditional AI with separate memory stores — the network topology itself encodes experience. Includes a complete creature simulation with DNA/RNA genetics, instincts, breeding, evolution, tool use, social structures, and a visual GUI simulation.

## The Core Philosophy

> **Structure IS Memory. Chemistry IS Mood. Behavior IS Emergent.**

- No hardcoded responses or templates
- No separate memory database
- No traditional training loops
- Behavior emerges from neural dynamics + neurochemistry + structural plasticity
- **Dynamic Neurogenesis**: Neurons are created when novelty is high, pruned when inactive
- **Sleep Consolidation**: Learning happens during sleep, not instantly

---

## 🏆 Implemented Feature Tiers

### TIER 1: Core Infrastructure ✅
- **Structural Memory Evolution**: Heritable neural architecture, neurons created/pruned
- **Metabolic Evolution**: DNA affects metabolism, energy, digestive efficiency
- **Pain-Based Reinforcement**: Dopamine/cortisol modulate learning

### TIER 2: Cognitive Systems ✅
- **Predictive Minds**: Pain prediction, novelty detection, anticipatory behavior
- **Evolving Brain Paradigms**: Cognitive diversity via topology differences

### TIER 3: Environmental & Social ✅
- **Cultural Evolution**: Social learning, observation, behavioral transmission
- **Environmental Intelligence**: Resource depletion, day/night cycles, weather

### TIER 4: Higher Cognition ✅ NEW
- **Tool Use Emergence**: Creatures learn to use sticks, stones, shells as tools
- **Abstract Reasoning**: Pattern discovery, concept formation, analogical thinking
- **Social Structures**: Hierarchy, trust/affinity tracking, cooperation, resource sharing

### NSM: Neural Sleep & Memory ✅
- **Sleep State Transitions**: Fatigue → sleep → consolidation → wake
- **Chemical Tagging**: Dopamine/cortisol mark synapses for consolidation
- **Replay During Sleep**: Experiences replayed in compressed form
- **Pruning Phase**: Unused pathways removed during sleep

### NSM V2: Advanced Neural Systems ✅ NEW

- **Hippocampal Replay**: Four replay modes for memory consolidation during sleep
  - **Forward Replay**: Sequential re-experience for procedural learning
  - **Reverse Replay**: TD-like credit assignment (outcome → cause)
  - **Stochastic Replay**: Random sampling for generalization
  - **Compressed Replay**: Key moment extraction (high-reward events)
- **Multi-Threaded Simulation**: Decoupled update loops for 30-50% performance gain
  - Physics: 60Hz (collision, movement, gravity)
  - Brain: 10Hz (neural processing, learning)
  - Metabolism: 1Hz (hunger, thirst, energy)
- **Metaplasticity**: Learning rate modulation based on recent activity
- **Synaptic Scaling**: Homeostatic regulation prevents runaway excitation
- **Sparse Coding**: HTM-like 2% sparsity in cortical representations
- **Predictive Coding**: Feedforward/feedback loops for anticipation
- **Reservoir Computing**: Echo State Networks for temporal sequence memory
- **Three-Factor Learning**: pre × post × neuromodulator plasticity rule

---

## 🎮 Two Modes of Operation

### 1. Chatbot Mode (CLI)
Interactive conversation with a brain that grows and changes structure based on interaction.

```powershell
python chatbot.py
```

### 2. Life Simulation Mode (GUI Dashboard)
Watch creatures with DNA-derived brains navigate a 2D world, eat, drink, breed, sleep, use tools, and evolve.

```powershell
python run_dashboard.py
```

The GUI Dashboard includes:
- **Chat Tab**: Interactive conversation with the brain
- **Status Tab**: Real-time brain activity visualization
- **Training Tab**: Guided learning experiments
- **Game Tab**: 2D world simulation with creatures
- **Settings Tab**: Visual customization for sprites, backgrounds, world objects

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

### Dynamic Neurogenesis
- Neurons are created when novelty is high and dopamine is elevated
- Neurons are pruned when inactive or during high cortisol (stress)
- Brain structure evolves based on experience
- Configurable neurogenesis rate and pruning thresholds

---

## 🦎 Creature Simulation System

### DNA/RNA Genetics
- **60+ genes** controlling brain size, body shape, metabolism, behavior
- **Dominance patterns**: Complete, incomplete, codominant
- **RNA expression** with tissue-specific transcription
- **Developmental stages**: Embryo → Larva → Juvenile → Adult → Elder
- **Epigenetic marks** that can be inherited

### Homeostasis System
Creatures maintain internal balance:
- **Energy**: Depletes with activity, recovers from food and sleep
- **Hydration**: Depletes over time, restored by drinking water
- **Nutrition**: Food fills stomach, converts to energy over time
- **Health**: Damaged by hazards, heals when resting
- **Fatigue**: Builds with activity, triggers sleep when high

### Behavior States
Creatures transition through behavioral states:
- **IDLE** → Default state
- **EXPLORING** → Wandering the world
- **SEEKING_FOOD** → Moving toward food when hungry
- **SEEKING_WATER** → Moving toward water when thirsty
- **EATING** → Consuming food
- **DRINKING** → Consuming water
- **SLEEPING** → Energy recovery, slowed metabolism (creatures lie on back!)
- **FLEEING** → Running from hazards/pain
- **SEEKING_TOOL** → Moving toward a tool (TIER 4)
- **PICKING_UP** → Grasping an object (TIER 4)
- **CARRYING** → Holding a tool (TIER 4)
- **THROWING** → Hurling held object (TIER 4)
- **USING_TOOL** → Using tool for a purpose (TIER 4)

### Tool Use System (TIER 4)
Creatures can find, pick up, and use tools in the world:
- **Sticks**: Long reach for poking, digging, reaching food
- **Stones**: Heavy, can be thrown for damage
- **Leaves**: Lightweight, can carry water
- **Shells**: Scooping, protection
- **Bones**: Digging, weapon

Tools become extensions of body schema through use (proprioceptive incorporation). Success with a tool strengthens neural pathways connecting action→tool→outcome.

### Procedural Language Evolution (TIER 4)
Emergent communication through phonemes, gestures, and reinforcement:
- **Phoneme System**: Basic sound units (vowels: a/i/u/e/o, plosives: pa/ta/ka, nasals: ma/na, etc.)
- **Gesture Primitives**: Visual signals (pointing, nodding, waving, raising arms, crouching)
- **Symbol Creation**: New words emerge from random phoneme combinations + context
- **Grounding**: Words become associated with meanings through use (food, water, danger, sleep, etc.)
- **Reinforcement Learning**: Successful communication strengthens word-meaning associations
- **Social Transmission**: Creatures learn words by observing others' actions
- **Cultural Language**: Population-wide conventions emerge when words are used successfully by multiple creatures
- **Vocabulary Pruning**: Ineffective words (low success rate) are forgotten over time

Each creature develops unique vocabulary based on experience, then conventions emerge through interaction.

### Sleep System
- Creatures sleep when energy drops below 20%
- During sleep, metabolism slows to 30% (less hunger/thirst buildup)
- Creatures visually rotate to lie on their back
- Wake up when energy exceeds 60% and fatigue is low
- Uses the homeostasis `is_sleeping` state for proper tracking

### Motor & Movement
- Goal-based movement controller: STAY, WANDER, GO_TO_TARGET, FLEE_FROM
- Physics simulation with gravity, platforms, and collisions
- Creatures detect and navigate to food/water sources
- Edge detection prevents falling off world

### World Environment
- **Procedural terrain**: Ground, platforms, water pools
- **Day/night cycle**: Affects light and creature behavior
- **Weather system**: Clear, rain, storm conditions
- **Food spawning**: Plants and berries with different effects
  - Sweet berries: Provide dopamine boost
  - Bitter berries: Low nutritional value
  - Poison berries: Cause pain and cortisol spike
- **Water sources**: Placed above ground for drinking (non-blocking)

---

## 🎨 Visual Customization (Settings Tab)

### Background Images
- Multiple backgrounds indexed by world position
- Index 0 = center, negative = left worlds, positive = right worlds

### Body Part Sprites
- Customize sprites for each body part and age stage
- Full-body sprite support with DNA-based coloring
- Hue, saturation, brightness adjustments per sprite

### World Objects
- **Water**: Custom water tile images with animation
- **Hazard**: Fire, lava effects
- **Ground**: Terrain textures
- **Shelter**: Safe zones
- **Edibles**: Multiple food type images
  - Sweet Berry, Bitter Berry, Poison Berry
  - Plant, Meat
  - Supports animation frames
- **Tools** (TIER 4): Customizable tool sprites
  - Stick, Stone, Leaf, Shell, Bone
  - Tools spawn naturally in the world
  - Creatures can pick up, carry, throw, and use
- Tile mode vs stretch mode per object type
- Scale and animation speed controls

---

## 🦠 Extended Biology

### Breeding System
```python
from brain import create_offspring, calculate_mate_compatibility

score = calculate_mate_compatibility(parent1, parent2)
child_genome = create_offspring(parent1.genome, parent2.genome)
```

### Viral Infections
Viruses can infect creatures and modify gene expression.

### Microbiome (Gut-Brain Axis)
Gut bacteria affect mood and metabolism.

### Cultural Transmission
Creatures can learn behaviors by observing others.

### Cross-Simulation Migration
Export and import creatures between simulations.

---

## 📁 Project Structure

```
brain/
├── __init__.py               # Package exports
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
├── world.py                  # 2D environment, terrain, weather, food
├── creature.py               # Physical body, phenotype, homeostasis
├── movement_controller.py    # Goal-based movement (WANDER, GO_TO, etc.)
├── behavior_state.py         # Behavior state machine
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
├── brainstem.py              # Autonomic functions
├── metabolism.py             # Per-neuron ATP, homeostasis
├── quadtree.py               # Spatial indexing
├── visualization.py          # Network visualization helpers

gui/
├── __init__.py
├── dashboard.py              # Main PyQt6 dashboard window
├── game_tab.py               # 2D world simulation with creatures
├── settings_tab.py           # Visual customization UI
├── brain_inspector.py        # Detailed brain visualization
├── graph_widget.py           # Graph plotting widgets
├── sound_manager.py          # Sound effects management

chatbot.py                    # CLI chatbot interface
run_dashboard.py              # GUI launcher
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NumPy (core dependency)
- PyQt6 (for GUI)
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

## 🎮 Dashboard Commands (Chat Tab)

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

## 🎮 Game Tab Controls

- **New World**: Generate a new procedural world
- **Spawn Creature**: Add a creature with full brain
- **Play/Pause**: Control simulation
- **Speed**: 0.5x to 20x simulation speed
- **Tools**: Select, Drag, Feed, Hurt, Heal creatures
- **Click creature**: View stats (health, energy, hunger, thirst, state, brain info)

---

## 🔬 Advanced Features

### Self-Compression Engine
The cortex invents new columns to represent recurring patterns.

### Internal Body / Motor Loop
8 virtual "muscles" with activation, fatigue, and recovery.

### Global Event Queue
Single priority queue with gamma, beta, theta, delta, and modulator events.

### LSH Hash Lattice
O(1) pattern lookup via locality-sensitive hashing.

### Signal Processing Pipeline
Robust input normalization with outlier handling and noise injection.

---

## 📊 Brain Scales

| Scale | Cortical Columns | Reservoir Size | Total Neurons | Use Case |
|-------|------------------|----------------|---------------|----------|
| `micro` | ~200 | ~800 | ~800 | Quick testing |
| `small` | ~400 | ~1600 | ~3,200 | Development |
| `medium` | ~800 | ~3200 | ~6,400 | Normal use |
| `large` | ~2000 | ~8000 | ~16,000 | Complex tasks |

---

## 🧠 Key Principles

1. **Structure IS Memory**: No separate memory store. Network topology encodes experience.
2. **Chemical Modulation First**: All behavior influenced by 10 neurochemicals.
3. **Emergent > Hardcoded**: No templates. Behavior emerges from dynamics.
4. **Three-Factor Learning**: pre × post × neuromodulator influences all plasticity.
5. **Biological Plausibility**: Inspired by real neuroscience, not just metaphors.
6. **Dynamic Neurogenesis**: Neurons are born and die based on experience and chemistry.

---

## 📜 License

MIT License - See LICENSE file for details.

---

*"The chat interface is just what emerges from that growing structure."*

*"Memory IS architecture. Chemistry IS mood. Behavior IS emergent."*
