# Chemical Brain - True Emergent Intelligence
# Enhanced with Mouse-Level Complexity Systems
# Now with Three-System Brain Architecture
#
# Memory IS structure. Responses emerge from dynamics.
# No templates. No separate memory store.
#
# THREE-SYSTEM ARCHITECTURE:
# System 1: Sparse Cortical Engine (representation, perception, prediction)
# System 2: Dynamic Recurrent Core (memory, imagination, sequences)
# System 3: Neuromodulated Learning (motivation, plasticity, value)

# Advanced neuromodulation with kinetic receptor binding (10 neurotransmitters)
# Enhanced with: stochastic kinetics, cross-modulator antagonism, epigenetic switches,
# time-asymmetric plasticity, cross-system error bargaining, homeostatic stability
from .neuromodulation import (
    KineticNeuromodulationSystem as NeuromodulationSystem,
    NeuromodulatorType, ReceptorType,
    NeuromodulatorReceptor, ReceptorField, KineticNeuromodulator,
    ThreeFactorLearning, ModulatorType,
    # New enhanced components
    StochasticKineticParams,
    CrossModulatorInteraction, CrossModulatorMatrix,
    EpigeneticSwitch, EpigeneticLearningModifiers,
    TemporalCreditAssignment,
    ErrorBargainingSystem,
    HomeostaticController
)

# Hierarchical cortical architecture
from .cortical_architecture import (
    HierarchicalCortex as CorticalHierarchy, 
    Minicolumn, CorticalLayer,
    TopographicMap, CorticalArea, LateralInhibition
)

# Neuron-specific metabolism and homeostasis
from .metabolism import (
    MetabolicNetwork, NeuronMetabolism, MitochondrialDynamics as MitochondrialState,
    SynapticScaling, IntrinsicPlasticity
)

# Signal processing and normalization
from .signal_processing import (
    RobustInputPipeline as SignalProcessor, 
    InputNormalizer, NoiseGenerator as NoiseInjector,
    OutlierHandler as OutlierFilter, NormalizationType
)

# Neural language decoder
from .language_decoder import (
    NeuralLanguageDecoder, AttractorDynamics as AttractorNetwork, 
    SemanticSpace, BeamSearchDecoder
)

# Sparse representations and event-driven simulation
from .sparse_network import (
    SparseNetwork as SparseNeuralNetwork, SparseNeuron as SparseNeuronState, 
    KWinnersNetwork as KWinnersLayer,
    SDRMemory as SparseDistributedMemory, EventDrivenSimulator
)

# GPU acceleration and quantization
from .gpu_acceleration import (
    GPUNeuralNetwork as GPUAccelerator, QuantizedArray, 
    VectorizedNeuronLayer as VectorizedNeuronState,
    VectorizedSynapseMatrix as BatchedSynapticUpdate
)

# Reservoir computing (ESN/LSM)
from .reservoir import (
    HybridReservoir, EchoStateReservoir as EchoStateNetwork, 
    LiquidStateReservoir as LiquidStateMachine,
    ReservoirConfig
)

# Hierarchical time scales
from .hierarchical_time import (
    HierarchicalTimeManager, TemporalScale, NestedOscillator,
    TemporalIntegrator, OscillatorBand, HierarchicalTimeProcessor
)

# Integrated brain system
from .integrated_brain import (
    IntegratedBrain, BrainConfig, BrainState, BrainRegion as IntegratedBrainRegion,
    RegionalProcessor, create_brain
)

# Three-System Brain Architecture (NEW)
from .three_system_brain import (
    ThreeSystemBrain,
    SparseCorticalEngine,
    DynamicRecurrentCore,
    NeuromodulatedLearningSystem,
    create_three_system_brain
)

# Persistence and auto-save
from .persistence import (
    BrainPersistence, AutoSaveMixin,
    save_brain, load_brain, check_dill_available
)

__all__ = [
    # Neuromodulation (10 chemicals with kinetic receptor binding)
    'NeuromodulationSystem', 'NeuromodulatorType', 'ReceptorType',
    'NeuromodulatorReceptor', 'ReceptorField', 'KineticNeuromodulator',
    'ThreeFactorLearning', 'ModulatorType',
    # Enhanced neuromodulation components
    'StochasticKineticParams', 'CrossModulatorInteraction', 'CrossModulatorMatrix',
    'EpigeneticSwitch', 'EpigeneticLearningModifiers',
    'TemporalCreditAssignment', 'ErrorBargainingSystem', 'HomeostaticController',

    # Cortical architecture
    'CorticalHierarchy', 'Minicolumn', 'CorticalLayer',
    'TopographicMap', 'CorticalArea', 'LateralInhibition',

    # Metabolism
    'MetabolicNetwork', 'NeuronMetabolism', 'MitochondrialState',
    'SynapticScaling', 'IntrinsicPlasticity',

    # Signal processing
    'SignalProcessor', 'InputNormalizer',
    'NoiseInjector', 'OutlierFilter', 'NormalizationType',

    # Language decoder
    'NeuralLanguageDecoder', 'AttractorNetwork', 'SemanticSpace',
    'BeamSearchDecoder',

    # Sparse network
    'SparseNeuralNetwork', 'SparseNeuronState', 'KWinnersLayer',
    'SparseDistributedMemory', 'EventDrivenSimulator',

    # GPU acceleration
    'GPUAccelerator', 'QuantizedArray', 'VectorizedNeuronState',
    'BatchedSynapticUpdate',

    # Reservoir
    'HybridReservoir', 'EchoStateNetwork', 'LiquidStateMachine',
    'ReservoirConfig',

    # Hierarchical time
    'HierarchicalTimeManager', 'TemporalScale', 'NestedOscillator',
    'TemporalIntegrator', 'OscillatorBand', 'HierarchicalTimeProcessor',

    # Integrated brain
    'IntegratedBrain', 'BrainConfig', 'BrainState', 'IntegratedBrainRegion',
    'RegionalProcessor', 'create_brain',

    # Three-System Brain Architecture (NEW)
    'ThreeSystemBrain', 'SparseCorticalEngine', 'DynamicRecurrentCore',
    'NeuromodulatedLearningSystem', 'create_three_system_brain',

    # Persistence
    'BrainPersistence', 'AutoSaveMixin',
    'save_brain', 'load_brain', 'check_dill_available',
]
