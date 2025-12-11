# Chemical Brain - True Emergent Intelligence
# Three-System Brain Architecture (V3)
#
# Memory IS structure. Responses emerge from dynamics.
# No templates. No separate memory store.
#
# CONCENTRATED ARCHITECTURE (6 Files):
# ├── three_system_brain.py   - Main brain + 3 systems + SDRMemory
# ├── neuromodulation.py      - 10 neurochemicals, kinetics, plasticity
# ├── language_decoder.py     - Attractor dynamics text output
# ├── signal_processing.py    - Input normalization pipeline
# ├── hierarchical_time.py    - Multi-rate oscillations
# └── persistence.py          - Save/load with dill
#
# THREE-SYSTEM ARCHITECTURE:
# System 1: Sparse Cortical Engine (representation, perception, prediction)
# System 2: Dynamic Recurrent Core (memory, imagination, sequences)
# System 3: Neuromodulated Learning (motivation, plasticity, value)

# =============================================================================
# PRIMARY EXPORTS: Three-System Brain
# =============================================================================

from .three_system_brain import (
    # Main brain class and factory
    ThreeSystemBrain,
    create_brain,  # Primary factory function
    create_three_system_brain,
    
    # Three subsystems
    SparseCorticalEngine,
    DynamicRecurrentCore,
    NeuromodulatedLearningSystem,
    
    # Configuration and state
    BrainConfig,
    BrainState,
    
    # SDR Memory (inlined)
    SDRMemory,
)

# Backwards compatibility alias
IntegratedBrain = ThreeSystemBrain

# =============================================================================
# SUPPORTING MODULES
# =============================================================================

# Neuromodulation: 10 neurochemicals with kinetic binding
from .neuromodulation import (
    KineticNeuromodulationSystem,
    ModulatorType,
)

# Signal processing: Input normalization
from .signal_processing import (
    RobustInputPipeline as SignalProcessor,
)

# Language decoder: Attractor dynamics output
from .language_decoder import (
    NeuralLanguageDecoder,
)

# Hierarchical time: Multi-rate oscillations
from .hierarchical_time import (
    HierarchicalTimeManager,
    TemporalScale,
)

# Persistence: Save/load
from .persistence import (
    BrainPersistence,
    save_brain,
    load_brain,
)

# Visualization: Matplotlib plotting (optional dependency)
try:
    from .visualization import (
        BrainVisualizer,
        quick_plot,
    )
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False
    BrainVisualizer = None
    quick_plot = None

# DNA / Genetics: Heredity, mutation, breeding
from .dna import (
    Gene,
    GeneCategory,
    Dominance,
    GeneLibrary,
    Genome,
    GenePool,
    DevelopmentalSystem,
    crossover,
    breed,
    mutate,
    create_organism_from_dna,
    breed_organisms,
)

# RNA / Development: Expression, silencing, patterning
from .rna import (
    TissueType,
    DevelopmentalStage,
    RNAType,
    MessengerRNA,
    MicroRNA,
    RegulatoryRNA,
    Transcriptome,
    RNASystem,
    RNADevelopmentalSystem,
    create_organism_with_rna,
    breed_with_rna,
    # Viral infections
    VirusType,
    Virus,
    Infection,
    ImmuneSystem,
    COMMON_VIRUSES,
)

# World simulation
from .world import (
    World,
    TileType,
    Weather,
    FoodSource,
    Hazard,
    Shelter,
)

# Creature body and physiology
from .creature import (
    Phenotype,
    Homeostasis,
    Action,
    MotorState,
    CreatureBody,
)

# Instincts and reward
from .instincts import (
    InstinctType,
    Instinct,
    InstinctSystem,
    RewardSystem,
)

# Brainstem - survival foundation (reflexes, drives, dopamine)
from .brainstem import (
    ReflexType,
    DriveType,
    Reflex,
    DriveNucleus,
    MotorPattern,
    DopamineSystem,
    Brainstem,
)

# Breeding and reproduction
from .breeding import (
    ReproductiveState,
    ReproductiveSystem,
    EpigeneticMark,
    EpigeneticSystem,
    calculate_mate_compatibility,
    select_mate,
    create_offspring,
)

# Embodiment - Brain↔Body wiring
from .embodiment import (
    SensoryEncoder,
    ActionDecoder,
    DriveNeuromodulatorBridge,
    EmbodiedBrain,
)

# Cultural transmission and social learning
from .culture import (
    BehaviorType,
    LearnedBehavior,
    CulturalMemory,
    COMMON_BEHAVIORS,
)

# Microbiome and gut-brain axis
from .microbiome import (
    BacteriaType,
    BacterialStrain,
    Microbiome,
    PROBIOTIC_STRAINS,
)

# Cross-simulation migration
from .migration import (
    MIGRATION_VERSION,
    MigrationMetadata,
    CreatureExporter,
    CreatureImporter,
    save_creature_to_file,
    load_creature_from_file,
    export_population,
    import_population,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Primary exports (Three-System Brain)
    'ThreeSystemBrain',
    'create_brain',
    'create_three_system_brain',
    'SparseCorticalEngine',
    'DynamicRecurrentCore', 
    'NeuromodulatedLearningSystem',
    'BrainConfig',
    'BrainState',
    'SDRMemory',
    'IntegratedBrain',  # Backwards compatibility
    
    # Neuromodulation
    'KineticNeuromodulationSystem',
    'ModulatorType',
    
    # Signal processing
    'SignalProcessor',
    
    # Language decoder
    'NeuralLanguageDecoder',
    
    # Hierarchical time
    'HierarchicalTimeManager',
    'TemporalScale',
    
    # Persistence
    'BrainPersistence',
    'save_brain',
    'load_brain',
    
    # Visualization (optional)
    'BrainVisualizer',
    'quick_plot',
    
    # DNA / Genetics
    'Gene',
    'GeneCategory',
    'Dominance',
    'GeneLibrary',
    'Genome',
    'GenePool',
    'DevelopmentalSystem',
    'crossover',
    'breed',
    'mutate',
    'create_organism_from_dna',
    'breed_organisms',
    
    # RNA / Development
    'TissueType',
    'DevelopmentalStage',
    'RNAType',
    'MessengerRNA',
    'MicroRNA',
    'RegulatoryRNA',
    'Transcriptome',
    'RNASystem',
    'RNADevelopmentalSystem',
    'create_organism_with_rna',
    'breed_with_rna',
    
    # World simulation
    'World',
    'TileType',
    'Weather',
    'FoodSource',
    'Hazard',
    'Shelter',
    
    # Creature body
    'Phenotype',
    'Homeostasis',
    'Action',
    'MotorState',
    'CreatureBody',
    
    # Instincts
    'InstinctType',
    'Instinct',
    'InstinctSystem',
    'RewardSystem',
    
    # Breeding
    'ReproductiveState',
    'ReproductiveSystem',
    'EpigeneticMark',
    'EpigeneticSystem',
    'calculate_mate_compatibility',
    'select_mate',
    'create_offspring',
    
    # Embodiment
    'SensoryEncoder',
    'ActionDecoder',
    'DriveNeuromodulatorBridge',
    'EmbodiedBrain',
    
    # Viral infections (from RNA)
    'VirusType',
    'Virus',
    'Infection',
    'ImmuneSystem',
    'COMMON_VIRUSES',
    
    # Cultural transmission
    'BehaviorType',
    'LearnedBehavior',
    'CulturalMemory',
    'COMMON_BEHAVIORS',
    
    # Microbiome
    'BacteriaType',
    'BacterialStrain',
    'Microbiome',
    'PROBIOTIC_STRAINS',
    
    # Cross-simulation migration
    'MIGRATION_VERSION',
    'MigrationMetadata',
    'CreatureExporter',
    'CreatureImporter',
    'save_creature_to_file',
    'load_creature_from_file',
    'export_population',
    'import_population',
]
