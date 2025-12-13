"""
Embodiment - Wiring the Brain to the Body

This module bridges the ThreeSystemBrain with the CreatureBody:
- Sensory encoding: World → Brain input
- Motor decoding: Brain output → Actions  
- Drive→Neuromodulator mapping: Physiology affects chemistry
- Reward signals: Drive satisfaction → Learning

The embodiment loop:
1. Creature senses world → SensoryEncoder → Brain input
2. Brain processes → ActionDecoder → Motor commands
3. Actions affect world → Drive changes
4. Drive satisfaction/frustration → Neuromodulator changes → Learning

This is where cognition meets behavior.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

from .three_system_brain import ThreeSystemBrain, BrainConfig
from .creature import CreatureBody, Action, Homeostasis
from .neuromodulation import ModulatorType
from .signal_processing import RobustInputPipeline, NormalizationType
from .spatial_memory import SpatialMemorySystem


# =============================================================================
# SENSORY ENCODING
# =============================================================================

class SensoryEncoder:
    """
    Encodes creature sensory data into brain-compatible input.
    
    Maps:
    - Internal state (hunger, fatigue, pain) → interoception channels
    - External senses (vision, smell) → exteroception channels  
    - Proprioception (velocity, position) → motor feedback
    
    Uses RobustInputPipeline for normalization and noise injection.
    """
    
    def __init__(self, input_dim: int = 128, use_signal_processing: bool = True):
        # Ensure minimum dimension to hold all hardcoded sensory channels (up to index 128)
        self.input_dim = max(input_dim, 128)
        
        # Channel allocations
        self.intero_start = 0
        self.intero_size = 16       # Internal body state
        
        self.drive_start = 16
        self.drive_size = 16        # Drive signals
        
        self.proprio_start = 32
        self.proprio_size = 8       # Body position/movement
        
        self.vision_start = 40
        self.vision_size = 32       # Visual field
        
        self.smell_start = 72
        self.smell_size = 16        # Chemical gradients
        
        self.audio_start = 88
        self.audio_size = 16        # Sound direction
        
        self.social_start = 104
        self.social_size = 24       # Other creatures
        
        # Signal processing pipeline
        self.use_signal_processing = use_signal_processing
        if use_signal_processing:
            self.signal_pipeline = RobustInputPipeline(
                input_dim=input_dim,
                norm_type=NormalizationType.ADAPTIVE,
                noise_scale=0.03,  # Light noise for exploration
                handle_outliers=False # Sparse signals (e.g. seeing food) are NOT outliers
            )
        else:
            self.signal_pipeline = None
        
    def encode(self, sensory_data: Dict, creature: CreatureBody) -> np.ndarray:
        """
        Encode sensory data to brain input vector.
        
        Args:
            sensory_data: From creature.get_sensory_input()
            creature: The creature body
            
        Returns:
            Input array for brain.process_raw()
        """
        vec = np.zeros(self.input_dim)
        
        # Interoception: Internal body state
        internal = sensory_data.get('internal', {})
        vec[0] = internal.get('energy', 0.5)
        vec[1] = internal.get('nutrition', 0.5)
        vec[2] = internal.get('hydration', 0.5)
        vec[3] = 1.0 - internal.get('fatigue', 0.0)  # Vitality
        vec[4] = 1.0 - internal.get('pain', 0.0)     # Comfort
        vec[5] = self._temp_to_signal(internal.get('temperature', 0.5))
        vec[6] = internal.get('health', 1.0)
        vec[7] = internal.get('fertility', 0.0)
        
        # Overall wellness signal
        vec[8] = (vec[0] + vec[1] + vec[2] + vec[4]) / 4
        
        # Drives: Motivational signals
        drives = sensory_data.get('drives', {})
        vec[16] = drives.get('hunger', 0.0)
        vec[17] = drives.get('thirst', 0.0)
        vec[18] = drives.get('rest', 0.0)
        vec[19] = drives.get('warmth', 0.0)
        vec[20] = drives.get('cooling', 0.0)
        vec[21] = drives.get('safety', 0.0)
        vec[22] = drives.get('reproduction', 0.0)
        vec[23] = drives.get('exploration', 0.0)
        
        # Urgency signal: Maximum drive strength
        vec[24] = max(drives.values()) if drives else 0.0
        
        # Proprioception: Body awareness
        motor = sensory_data.get('motor', {})
        vec[32] = motor.get('vx', 0) / 5 + 0.5    # Normalize to 0-1
        vec[33] = motor.get('vy', 0) / 10 + 0.5
        vec[34] = 1.0 if motor.get('on_ground', True) else 0.0
        vec[35] = 1.0 if motor.get('in_water', False) else 0.0
        vec[36] = 1.0 if motor.get('in_shelter', False) else 0.0
        vec[37] = sensory_data.get('time_of_day', 0.5)
        
        # Vision: Objects in view
        foods = sensory_data.get('visible_food', [])
        for i, food in enumerate(foods[:4]):  # Max 4 food items
            base = 40 + i * 4
            if base + 4 <= 72:
                vec[base] = food['dx'] / 200 + 0.5    # Direction
                vec[base + 1] = food['dy'] / 200 + 0.5
                vec[base + 2] = 1.0 / (1 + food['dist'] / 100)  # Proximity
                vec[base + 3] = food['nutrition'] / 100
        
        hazards = sensory_data.get('visible_hazards', [])
        for i, hazard in enumerate(hazards[:2]):  # Max 2 hazards
            base = 56 + i * 4
            if base + 4 <= 72:
                vec[base] = hazard['dx'] / 200 + 0.5
                vec[base + 1] = hazard['dy'] / 200 + 0.5
                vec[base + 2] = 1.0 / (1 + hazard['dist'] / 100)
                vec[base + 3] = hazard['damage'] / 20
        
        # Smell: Chemical gradients (simplified to nearest food smell)
        if foods:
            nearest = min(foods, key=lambda f: f['dist'])
            angle = np.arctan2(nearest['dy'], nearest['dx'])
            vec[72] = np.cos(angle) * 0.5 + 0.5
            vec[73] = np.sin(angle) * 0.5 + 0.5
            vec[74] = 1.0 / (1 + nearest['dist'] / 100)
        
        # Social: Other creatures
        creatures = sensory_data.get('visible_creatures', [])
        for i, other in enumerate(creatures[:4]):  # Max 4 creatures
            base = 104 + i * 6
            if base + 6 <= 128:
                vec[base] = other['dx'] / 200 + 0.5
                vec[base + 1] = other['dy'] / 200 + 0.5
                vec[base + 2] = 1.0 / (1 + other['dist'] / 100)
                vec[base + 3] = other['size']
                vec[base + 4] = 1.0 if other['is_threat'] else 0.0
                vec[base + 5] = 1.0 if other['is_same_species'] else 0.0
        
        # Apply signal processing pipeline (normalization, noise, outlier handling)
        if self.signal_pipeline is not None:
            vec = self.signal_pipeline.process(vec, add_noise=True)
        
        return vec
    
    def set_training(self, training: bool):
        """Set training mode (affects noise injection)."""
        if self.signal_pipeline is not None:
            self.signal_pipeline.set_training(training)
    
    def get_processing_stats(self) -> Dict:
        """Get signal processing statistics."""
        if self.signal_pipeline is not None:
            return self.signal_pipeline.get_stats()
        return {}
    
    def _temp_to_signal(self, temp: float) -> float:
        """Convert temperature to comfort signal (1=optimal, 0=extreme)."""
        return 1.0 - 2 * abs(temp - 0.5)


# =============================================================================
# ACTION DECODING
# =============================================================================

class ActionDecoder:
    """
    Decodes brain output into motor commands.
    
    Brain output is a high-dimensional vector. We decode it into:
    - Movement direction/speed
    - Jump trigger
    - Interaction actions (eat, drink, mate)
    """
    
    def __init__(self, output_dim: int = 64):
        self.output_dim = output_dim
        
        # Output channel allocations (learned through training)
        # For now, we use fixed channels
        self.move_x_channel = 0      # Movement direction
        self.move_y_channel = 1      # Up/down tendency
        self.jump_channel = 2        # Jump trigger
        self.eat_channel = 3         # Eating action
        self.drink_channel = 4       # Drinking action
        self.rest_channel = 5        # Rest action
        self.flee_channel = 6        # Flee response
        self.approach_channel = 7    # Approach response
        self.mate_channel = 8        # Mating action
        self.call_channel = 9        # Vocalization
        self.dig_channel = 10        # Digging
        self.build_channel = 11      # Building
        self.plant_channel = 12      # Planting (Cultivation)
        
    def decode(self, brain_output: np.ndarray, 
               creature: CreatureBody) -> Dict[str, float]:
        """
        Decode brain output to motor commands.
        
        Args:
            brain_output: Output from brain.process_raw()
            creature: The creature body for context
            
        Returns:
            Dict of motor commands for creature._process_brain_output()
        """
        # Ensure we have enough output
        if len(brain_output) < 13:
            brain_output = np.pad(brain_output, (0, 13 - len(brain_output)))
        
        # Decode with sigmoid for 0-1 range
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x * 3, -10, 10)))
        
        commands = {}
        
        # Movement: Use first two channels as direction vector
        move_x = brain_output[self.move_x_channel]
        move_y = brain_output[self.move_y_channel]
        
        # Convert to left/right commands
        if move_x > 0.1:
            commands['move_right'] = min(1.0, move_x * 2)
        elif move_x < -0.1:
            commands['move_left'] = min(1.0, -move_x * 2)
        
        # Jump when up channel is high and on ground
        jump_signal = sigmoid(brain_output[self.jump_channel])
        if jump_signal > 0.6:
            commands['jump'] = jump_signal
        
        # Interaction actions
        commands['eat'] = sigmoid(brain_output[self.eat_channel])
        commands['drink'] = sigmoid(brain_output[self.drink_channel])
        commands['rest'] = sigmoid(brain_output[self.rest_channel])
        commands['mate'] = sigmoid(brain_output[self.mate_channel])
        commands['dig'] = sigmoid(brain_output[self.dig_channel])
        commands['build'] = sigmoid(brain_output[self.build_channel])
        commands['plant'] = sigmoid(brain_output[self.plant_channel])
        
        # Combine flee/approach into movement bias
        flee_signal = sigmoid(brain_output[self.flee_channel])
        approach_signal = sigmoid(brain_output[self.approach_channel])
        
        # Flee overrides normal movement when high
        if flee_signal > 0.7:
            # Reverse direction
            if 'move_right' in commands:
                commands['move_left'] = commands.pop('move_right') * flee_signal
            elif 'move_left' in commands:
                commands['move_right'] = commands.pop('move_left') * flee_signal
        
        return commands


# =============================================================================
# DRIVE → NEUROMODULATOR MAPPING
# =============================================================================

class DriveNeuromodulatorBridge:
    """
    Maps drive states to neuromodulator changes.
    
    This implements the reward model where:
    - Drive satisfaction → Dopamine (reward)
    - Drive frustration → Cortisol (stress)
    - Safety drives → Norepinephrine (alertness)
    - Social drives → Oxytocin
    - Exploration → Acetylcholine
    """
    
    def __init__(self):
        # Track previous drive states for satisfaction detection
        self.prev_drives: Dict[str, float] = {}
        self.prev_internal: Dict[str, float] = {}
        
    def compute_neuromodulator_changes(self, 
                                        current_drives: Dict[str, float],
                                        current_internal: Dict[str, float]) -> Dict[str, float]:
        """
        Compute neuromodulator deltas based on drive changes.
        
        Returns dict of modulator_name → delta value
        
        NOTE: These deltas are applied every frame (~30-60Hz).
        They must be very small to avoid runaway accumulation.
        The homeostatic decay in KineticNeuromodulationSystem is weak
        (0.02 * dt per frame), so we must keep additions even smaller.
        """
        changes = {
            'dopamine': 0.0,
            'serotonin': 0.0,
            'cortisol': 0.0,
            'norepinephrine': 0.0,
            'oxytocin': 0.0,
            'acetylcholine': 0.0,
            'gaba': 0.0,
            'glutamate': 0.0,
            'endorphin': 0.0,
            'adrenaline': 0.0,
        }
        
        # Compute drive satisfaction/frustration
        # NOTE: Scale these down since they're applied every frame
        for drive, level in current_drives.items():
            prev_level = self.prev_drives.get(drive, level)
            delta = prev_level - level  # Positive = drive reduced = satisfaction
            
            if delta > 0.01:
                # Drive satisfaction → Dopamine + Endorphin (scaled down)
                changes['dopamine'] += delta * 0.05
                changes['endorphin'] += delta * 0.02
                changes['serotonin'] += delta * 0.02
                
            elif delta < -0.01:
                # Drive frustration → VERY small cortisol bump
                changes['cortisol'] += abs(delta) * 0.005  # Reduced from 0.05
        
        # Specific drive mappings
        # NOTE: These are per-frame deltas, MUST be very small to prevent runaway
        
        # High hunger/thirst → Cortisol (STARVATION STRESS)
        # Only critical levels trigger stress
        hunger = current_drives.get('hunger', 0)
        if hunger > 0.9:
            # Critical hunger - stress
            changes['cortisol'] += 0.002  # Reduced from 0.02
            changes['norepinephrine'] += 0.005
            changes['adrenaline'] += 0.002
        
        thirst = current_drives.get('thirst', 0)
        if thirst > 0.9:
            # Critical thirst - stress
            changes['cortisol'] += 0.003  # Reduced from 0.03
            changes['norepinephrine'] += 0.005
            changes['adrenaline'] += 0.002
        
        # Safety drive (pain/fear) → Norepinephrine + Adrenaline
        safety = current_drives.get('safety', 0)
        if safety > 0.5:
            changes['norepinephrine'] += safety * 0.005
            changes['adrenaline'] += safety * 0.008
            # Danger response - very small cortisol
            changes['cortisol'] += safety * 0.003  # Reduced from 0.03
        
        # Reproduction drive → Oxytocin + Dopamine
        repro = current_drives.get('reproduction', 0)
        if repro > 0.5:
            changes['oxytocin'] += repro * 0.005
            changes['dopamine'] += repro * 0.003
        
        # Exploration → Acetylcholine + Dopamine (curiosity)
        explore = current_drives.get('exploration', 0)
        if explore > 0.3:
            changes['acetylcholine'] += explore * 0.005
            changes['dopamine'] += explore * 0.003
        
        # Rest drive → GABA (need to slow down)
        rest = current_drives.get('rest', 0)
        if rest > 0.5:
            changes['gaba'] += rest * 0.005
        
        # Internal state changes
        
        # Energy gain → Serotonin (satisfaction)
        energy_now = current_internal.get('energy', 0.5)
        energy_prev = self.prev_internal.get('energy', energy_now)
        if energy_now > energy_prev + 0.01:
            changes['serotonin'] += (energy_now - energy_prev) * 0.1
            changes['dopamine'] += (energy_now - energy_prev) * 0.08
        
        # Pain → Endorphin (natural response)
        pain = current_internal.get('pain', 0)
        if pain > 0.3:
            changes['endorphin'] += pain * 0.005
            changes['adrenaline'] += pain * 0.003
            
            # Severe pain causes stress spike (but very small)
            if pain > 0.6:
                changes['cortisol'] += (pain - 0.6) * 0.01  # Reduced from 0.1
        
        # Temperature discomfort → Cortisol (only extreme)
        temp = current_internal.get('temperature', 0.5)
        temp_discomfort = abs(temp - 0.5) * 2
        if temp_discomfort > 0.5:
            changes['cortisol'] += temp_discomfort * 0.001  # Reduced from 0.01
        
        # Low health → Cortisol (only critical)
        health = current_internal.get('health', 1.0)
        if health < 0.3:
            changes['cortisol'] += (0.3 - health) * 0.005  # Reduced from 0.05
            changes['dopamine'] -= (0.3 - health) * 0.002
        
        # Update previous state
        self.prev_drives = current_drives.copy()
        self.prev_internal = current_internal.copy()
        
        return changes


# =============================================================================
# EMBODIED BRAIN - Complete integration
# =============================================================================

class EmbodiedBrain:
    """
    A brain embedded in a body, fully connected to sensorimotor loop.
    
    This is the complete creature: body + brain + wiring.
    """
    
    def __init__(self, 
                 brain: Optional[ThreeSystemBrain] = None,
                 body: Optional[CreatureBody] = None,
                 brain_scale: str = 'micro',
                 brain_config: Optional[BrainConfig] = None):
        """
        Create an embodied brain.
        
        Args:
            brain: Pre-existing brain (or create new one)
            body: Pre-existing body (or create new one)
            brain_scale: Size of brain to create if not provided
            brain_config: BrainConfig from DNA (overrides brain_scale)
        """
        from . import create_brain
        
        if brain is not None:
            self.brain = brain
        elif brain_config is not None:
            # Create brain from DNA-derived config
            self.brain = ThreeSystemBrain(config=brain_config)
        else:
            # Use default scale
            self.brain = create_brain(brain_scale)
            
        self.body = body or CreatureBody()
        
        # Wiring components
        self.sensory_encoder = SensoryEncoder(input_dim=128)
        self.action_decoder = ActionDecoder(output_dim=64)
        self.neuro_bridge = DriveNeuromodulatorBridge()
        
        # Initialize Spatial Memory (Structure-based navigation)
        # Scale resolution by brain size: Micro brains get coarse grids, Big brains get fine grids
        # World assumed ~3000x3000 max for now
        grid_res = 500 if brain_scale == 'micro' else 200
        self.spatial_memory = SpatialMemorySystem(
            world_width=3000, 
            world_height=3000, 
            grid_size=grid_res
        )
        
        # Statistics
        self.total_reward = 0.0
        self.step_count = 0
        self.actions_taken = []
    
    @classmethod
    def from_genome(cls, genome: 'Genome', x: float = 400.0, y: float = 200.0) -> 'EmbodiedBrain':
        """
        Create an embodied brain from DNA.
        
        The genome specifies both brain architecture and body phenotype.
        
        Args:
            genome: Genetic code (from brain.dna)
            x, y: Starting position in world
            
        Returns:
            EmbodiedBrain with DNA-configured brain and body
        """
        from .dna import DevelopmentalSystem
        from .creature import Phenotype
        
        # Develop organism from DNA
        dev = DevelopmentalSystem(genome)
        
        # Get brain config from DNA
        brain_config = dev.develop_brain_config()
        
        # Get body params from DNA
        body_params = dev.develop_body_params()
        
        # Create phenotype from body params
        phenotype = Phenotype.from_body_params(body_params)
        
        # Create body with phenotype
        body = CreatureBody(phenotype=phenotype, x=x, y=y)
        
        # Create brain from config
        brain = ThreeSystemBrain(config=brain_config)
        
        return cls(brain=brain, body=body)
        
    def step(self, world, other_creatures: List[CreatureBody] = None, 
             dt: float = 0.1) -> Dict[str, Any]:
        """
        Execute one step of the embodied loop.
        
        1. Sense environment
        2. Encode for brain
        3. Brain processes
        4. Decode to actions
        5. Update body
        6. Compute reward signal
        7. Update neuromodulators
        
        Returns:
            Dict with step information
        """
        self.step_count += 1
        
        # 1. Gather sensory input
        sensory_data = self.body.get_sensory_input(world, other_creatures)
        
        # 2. Encode for brain
        brain_input = self.sensory_encoder.encode(sensory_data, self.body)
        
        # 3. Process through brain
        # Get drive levels for reward/arousal
        drives = sensory_data.get('drives', {})
        internal = sensory_data.get('internal', {})
        
        # Compute neuromodulator effects from physiology
        neuro_changes = self.neuro_bridge.compute_neuromodulator_changes(
            drives, internal
        )
        
        # Apply neuromodulator changes
        self._apply_neuromodulator_changes(neuro_changes)
        
        # === SYSTEM 6: SPATIAL MEMORY UPDATE ===
        # Update Place Cells and Hebbian Synapses based on experience
        experiences = {}
        
        # Detect energy gain (Eating)
        curr_energy = internal.get('energy', 0.5)
        prev_energy = self.prev_internal.get('energy', curr_energy)
        if curr_energy > prev_energy + 0.001:
            experiences['energy_gain'] = curr_energy - prev_energy
            
        # Detect pain (Hazard)
        experiences['pain'] = internal.get('pain', 0)
        
        # Update synapses
        self.spatial_memory.update(
            self.body.motor.x, self.body.motor.y, experiences, dt
        )
        
        # Query for navigational target
        nav_target = self.spatial_memory.get_navigation_gradient(drives)
        
        # If we have a spatial goal, encode it into brain input AND bias motor system
        nav_bias_x, nav_bias_y = 0.0, 0.0
        if nav_target:
            tx, ty = nav_target
            dx = tx - self.body.motor.x
            dy = ty - self.body.motor.y
            dist = np.sqrt(dx*dx + dy*dy) + 1e-6
            nav_bias_x = dx / dist
            nav_bias_y = dy / dist
            
            # Encode into specific brain input channels (e.g., 60-61)
            # This allows the reservoir to "know" where it wants to go
            if self.sensory_encoder.input_dim > 61:
                brain_input[60] = nav_bias_x
                brain_input[61] = nav_bias_y
        
        # Compute reward and arousal for brain
        reward = self._compute_reward(sensory_data)
        arousal = self._compute_arousal(sensory_data)
        
        # Process through brain
        brain_result = self.brain.process_raw(
            brain_input,
            dt=dt,
            learning_enabled=True
        )
        
        # 4. Decode brain output to motor commands
        brain_output = brain_result.get('integrated_output', 
                                        brain_result.get('reservoir_output',
                                        np.zeros(64)))
        
        motor_commands = self.action_decoder.decode(brain_output, self.body)
        
        # Blend Spatial Memory drive (Hippocampal override)
        # If the creature remembers food is nearby and is hungry, it feels a "pull"
        if nav_target:
            # Strength of pull depends on hunger/fear intensity (already in nav_target logic)
            # We add a subtle bias to the motor cortex
            motor_commands['move_x'] = np.clip(motor_commands['move_x'] + nav_bias_x * 0.3, -1, 1)
            motor_commands['move_y'] = np.clip(motor_commands['move_y'] + nav_bias_y * 0.3, -1, 1)
        
        
        # 5. Update body with motor commands
        self.body.update(dt, world, motor_commands)
        
        # 6. Track reward
        self.total_reward += reward
        
        # Store for brain inspector visualization
        self.last_step_data = {
            'sensory_input': brain_input,
            'brain_output': brain_output,
            'motor_commands': motor_commands,
            'drives': drives,
            'neuro_changes': neuro_changes,
            'reward': reward,
            'arousal': arousal,
            'alive': self.body.is_alive(),
        }
        
        return self.last_step_data
    
    def _compute_reward(self, sensory_data: Dict) -> float:
        """
        Compute reward signal from current state.
        
        Positive reward for:
        - Drive reduction (satisfaction)
        - Energy gain
        - Health maintenance
        
        Negative reward for:
        - Pain
        - Health loss
        - Extreme drives
        """
        internal = sensory_data.get('internal', {})
        drives = sensory_data.get('drives', {})
        
        reward = 0.0
        
        # Survival bonus
        if internal.get('health', 0) > 0.5:
            reward += 0.01
        
        # Drive satisfaction (low drives = good)
        max_drive = max(drives.values()) if drives else 0
        if max_drive < 0.3:
            reward += 0.05  # All needs met
        elif max_drive > 0.7:
            reward -= 0.05  # Urgent unmet need
        
        # Pain penalty
        reward -= internal.get('pain', 0) * 0.2
        
        # Energy level
        energy = internal.get('energy', 0.5)
        if energy > 0.6:
            reward += 0.02
        elif energy < 0.2:
            reward -= 0.05
        
        return np.clip(reward, -1, 1)
    
    def _compute_arousal(self, sensory_data: Dict) -> float:
        """
        Compute arousal level from sensory state.
        
        High arousal from:
        - Danger nearby
        - Urgent drives
        - Novel stimuli
        """
        drives = sensory_data.get('drives', {})
        
        # Base arousal from drive urgency
        max_drive = max(drives.values()) if drives else 0
        arousal = max_drive * 0.5
        
        # Danger detection
        safety = drives.get('safety', 0)
        arousal += safety * 0.3
        
        # Nearby threats
        creatures = sensory_data.get('visible_creatures', [])
        threats = [c for c in creatures if c.get('is_threat', False)]
        if threats:
            nearest_threat = min(t['dist'] for t in threats)
            threat_arousal = 1.0 / (1 + nearest_threat / 50)
            arousal += threat_arousal * 0.3
        
        # Exploration excitement
        arousal += drives.get('exploration', 0) * 0.2
        
        return np.clip(arousal, 0, 1)
    
    def _apply_neuromodulator_changes(self, changes: Dict[str, float]):
        """Apply computed neuromodulator changes to brain."""
        # Map our names to brain's modulator types
        name_to_type = {
            'dopamine': ModulatorType.DOPAMINE,
            'serotonin': ModulatorType.SEROTONIN,
            'norepinephrine': ModulatorType.NOREPINEPHRINE,
            'acetylcholine': ModulatorType.ACETYLCHOLINE,
            'gaba': ModulatorType.GABA,
            'glutamate': ModulatorType.GLUTAMATE,
            'cortisol': ModulatorType.CORTISOL,
            'oxytocin': ModulatorType.OXYTOCIN,
            'endorphin': ModulatorType.ENDORPHIN,
            'adrenaline': ModulatorType.ADRENALINE,
        }
        
        # Access the kinetic neuromodulation system through learning.neuromod
        neuromod = self.brain.learning.neuromod
        
        for name, delta in changes.items():
            if name in name_to_type and abs(delta) > 0.001:
                mod_type = name_to_type[name]
                # Get current level and apply bounded change
                current = neuromod.get_level(mod_type)
                new_level = np.clip(current + delta, 0, 1)
                neuromod.set_level(mod_type, new_level)
    
    def get_state(self) -> Dict[str, Any]:
        """Get combined brain/body state for inspection."""
        brain_stats = self.brain.get_stats()
        
        return {
            'brain': brain_stats,
            'body': {
                'position': (self.body.motor.x, self.body.motor.y),
                'velocity': (self.body.motor.vx, self.body.motor.vy),
                'health': self.body.homeostasis.health,
                'energy': self.body.homeostasis.energy,
                'drives': self.body.homeostasis.get_drive_levels(),
                'alive': self.body.is_alive(),
            },
            'stats': {
                'total_reward': self.total_reward,
                'step_count': self.step_count,
                'food_eaten': self.body.food_eaten,
                'distance': self.body.distance_traveled,
            }
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SensoryEncoder',
    'ActionDecoder',
    'DriveNeuromodulatorBridge',
    'EmbodiedBrain',
]
