"""
Cross-Simulation Migration

Export and import creatures between different simulation instances:
- Serialize complete creature state (brain, body, DNA, RNA, culture, microbiome)
- Version compatibility checking
- Population mixing for genetic diversity
- Cloud/file-based creature sharing
"""

import numpy as np
import json
import base64
import hashlib
import gzip
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from pathlib import Path


# =============================================================================
# MIGRATION FORMAT
# =============================================================================

MIGRATION_VERSION = "1.0.0"
MAGIC_HEADER = b"BRAIN_CREATURE_V1"


@dataclass
class MigrationMetadata:
    """Metadata about a migrated creature."""
    version: str
    creature_id: str
    creature_name: str
    export_time: str
    source_simulation: str
    generation: int
    age: float
    
    # Lineage tracking
    parent_ids: List[str]
    origin_simulation: str
    migrations_count: int
    
    # Checksums
    dna_hash: str
    brain_hash: str


class CreatureExporter:
    """
    Exports creatures to portable format.
    """
    
    def __init__(self, simulation_id: str = "unknown"):
        self.simulation_id = simulation_id
    
    def export_creature(self, creature: Any, include_brain_state: bool = True) -> bytes:
        """
        Export a creature to bytes.
        
        Args:
            creature: The creature object to export
            include_brain_state: Whether to include full brain weights
            
        Returns:
            Compressed bytes representing the creature
        """
        data = {
            'version': MIGRATION_VERSION,
            'export_time': datetime.now().isoformat(),
            'source_simulation': self.simulation_id,
        }
        
        # Extract creature components
        if hasattr(creature, 'body'):
            data['body'] = self._serialize_body(creature.body)
        
        if hasattr(creature, 'genome'):
            data['genome'] = self._serialize_genome(creature.genome)
        elif hasattr(creature, 'dna'):
            data['genome'] = self._serialize_genome(creature.dna)
        
        if hasattr(creature, 'rna_system'):
            data['rna'] = self._serialize_rna(creature.rna_system)
        
        if hasattr(creature, 'brain') or hasattr(creature, 'embodied_brain'):
            brain = getattr(creature, 'brain', None) or getattr(creature, 'embodied_brain', None)
            data['brain'] = self._serialize_brain(brain, include_brain_state)
        
        if hasattr(creature, 'instincts'):
            data['instincts'] = self._serialize_instincts(creature.instincts)
        
        if hasattr(creature, 'culture') or hasattr(creature, 'cultural_memory'):
            culture = getattr(creature, 'culture', None) or getattr(creature, 'cultural_memory', None)
            data['culture'] = self._serialize_culture(culture)
        
        if hasattr(creature, 'microbiome'):
            data['microbiome'] = self._serialize_microbiome(creature.microbiome)
        
        if hasattr(creature, 'immune_system'):
            data['immune'] = self._serialize_immune(creature.immune_system)
        
        # Creature metadata
        data['metadata'] = {
            'id': getattr(creature, 'id', str(np.random.randint(1e9))),
            'name': getattr(creature, 'name', 'Unknown'),
            'generation': getattr(creature, 'generation', 0),
            'age': getattr(creature.body, 'lifetime', 0) if hasattr(creature, 'body') else 0,
            'parent_ids': getattr(creature, 'parent_ids', []),
        }
        
        # Compute checksums
        data['checksums'] = {
            'dna': self._compute_hash(data.get('genome', {})),
            'brain': self._compute_hash(data.get('brain', {})),
        }
        
        # Serialize and compress
        json_str = json.dumps(data, default=self._json_serializer)
        compressed = gzip.compress(json_str.encode('utf-8'))
        
        return MAGIC_HEADER + compressed
    
    def _serialize_body(self, body) -> Dict:
        """Serialize body state."""
        result = {}
        
        if hasattr(body, 'phenotype'):
            p = body.phenotype
            result['phenotype'] = {
                'size': p.size,
                'width': p.width,
                'height': p.height,
                'hue': p.hue,
                'saturation': p.saturation,
                'brightness': getattr(p, 'brightness', 0.8),
                'limb_count': p.limb_count,
                'has_tail': p.has_tail,
                'has_fins': getattr(p, 'has_fins', False),
                'has_wings': getattr(p, 'has_wings', False),
                'max_speed': p.max_speed,
                'jump_power': p.jump_power,
            }
        
        if hasattr(body, 'homeostasis'):
            h = body.homeostasis
            result['homeostasis'] = {
                'energy': h.energy,
                'nutrition': h.nutrition,
                'hydration': h.hydration,
                'health': h.health,
            }
        
        result['stats'] = {
            'lifetime': getattr(body, 'lifetime', 0),
            'food_eaten': getattr(body, 'food_eaten', 0),
            'distance_traveled': getattr(body, 'distance_traveled', 0),
            'offspring_count': getattr(body, 'offspring_count', 0),
        }
        
        return result
    
    def _serialize_genome(self, genome) -> Dict:
        """Serialize genome."""
        if genome is None:
            return {}
        
        result = {
            'id': getattr(genome, 'id', ''),
            'genes': {}
        }
        
        if hasattr(genome, 'genes'):
            for locus, gene in genome.genes.items():
                result['genes'][str(locus)] = {
                    'name': gene.name,
                    'allele_a': gene.allele_a,
                    'allele_b': gene.allele_b,
                    'dominance': getattr(gene, 'dominance', 0.5),
                }
        
        return result
    
    def _serialize_rna(self, rna) -> Dict:
        """Serialize RNA system."""
        if rna is None:
            return {}
        
        result = {}
        
        if hasattr(rna, 'mrna_levels'):
            result['mrna'] = {k: list(v) if isinstance(v, np.ndarray) else v 
                            for k, v in rna.mrna_levels.items()}
        
        if hasattr(rna, 'active_mirna'):
            result['mirna'] = list(rna.active_mirna)
        
        return result
    
    def _serialize_brain(self, brain, include_state: bool) -> Dict:
        """Serialize brain configuration and optionally state."""
        if brain is None:
            return {}
        
        result = {}
        
        # Get the actual brain if it's an EmbodiedBrain
        if hasattr(brain, 'brain'):
            brain = brain.brain
        
        # Config
        if hasattr(brain, 'config'):
            config = brain.config
            result['config'] = {
                'num_columns': config.num_columns,
                'cells_per_column': config.cells_per_column,
                'reservoir_size': config.reservoir_size,
                'target_sparsity': config.target_sparsity,
                'spectral_radius': getattr(config, 'spectral_radius', 0.9),
            }
        
        # Statistics
        if hasattr(brain, 'get_stats'):
            try:
                stats = brain.get_stats()
                result['stats'] = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                  for k, v in stats.items() if not isinstance(v, np.ndarray)}
            except:
                pass
        
        # Optionally include weights (large!)
        if include_state and hasattr(brain, 'cortex'):
            # Only save essential learned weights
            cortex = brain.cortex
            if hasattr(cortex, 'ff_weights') and cortex.ff_weights is not None:
                result['weights'] = {
                    'ff_weights_shape': list(cortex.ff_weights.shape),
                    'ff_weights': self._compress_array(cortex.ff_weights),
                }
        
        return result
    
    def _serialize_instincts(self, instincts) -> Dict:
        """Serialize instinct system."""
        if instincts is None:
            return {}
        
        result = {}
        if hasattr(instincts, 'to_dict'):
            result = instincts.to_dict()
        elif hasattr(instincts, 'instincts'):
            for itype, inst in instincts.instincts.items():
                result[itype.value] = {
                    'base_strength': inst.base_strength,
                    'learned_weight': getattr(inst, 'learned_weight', 1.0),
                }
        return result
    
    def _serialize_culture(self, culture) -> Dict:
        """Serialize cultural memory."""
        if culture is None:
            return {}
        
        if hasattr(culture, 'to_dict'):
            return culture.to_dict()
        return {}
    
    def _serialize_microbiome(self, microbiome) -> Dict:
        """Serialize microbiome."""
        if microbiome is None:
            return {}
        
        if hasattr(microbiome, 'to_dict'):
            return microbiome.to_dict()
        return {}
    
    def _serialize_immune(self, immune) -> Dict:
        """Serialize immune system."""
        if immune is None:
            return {}
        
        if hasattr(immune, 'to_dict'):
            return immune.to_dict()
        return {}
    
    def _compress_array(self, arr: np.ndarray) -> str:
        """Compress numpy array to base64 string."""
        # Quantize to reduce size
        arr_f16 = arr.astype(np.float16)
        compressed = gzip.compress(arr_f16.tobytes())
        return base64.b64encode(compressed).decode('ascii')
    
    def _compute_hash(self, data: Dict) -> str:
        """Compute hash of data for integrity checking."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)


class CreatureImporter:
    """
    Imports creatures from portable format.
    """
    
    def __init__(self, simulation_id: str = "unknown"):
        self.simulation_id = simulation_id
    
    def import_creature(self, data: bytes) -> Dict:
        """
        Import a creature from bytes.
        
        Args:
            data: Compressed bytes from export
            
        Returns:
            Dict with creature components ready for reconstruction
        """
        # Verify header
        if not data.startswith(MAGIC_HEADER):
            raise ValueError("Invalid creature file format")
        
        # Decompress
        compressed = data[len(MAGIC_HEADER):]
        json_str = gzip.decompress(compressed).decode('utf-8')
        creature_data = json.loads(json_str)
        
        # Version check
        version = creature_data.get('version', '0.0.0')
        if not self._check_version_compatible(version):
            raise ValueError(f"Incompatible creature version: {version}")
        
        # Verify checksums
        if not self._verify_checksums(creature_data):
            print("Warning: Creature checksum mismatch - data may be corrupted")
        
        # Process and return
        result = {
            'metadata': creature_data.get('metadata', {}),
            'source_simulation': creature_data.get('source_simulation', 'unknown'),
            'export_time': creature_data.get('export_time', ''),
        }
        
        # Extract components
        if 'body' in creature_data:
            result['body'] = creature_data['body']
        
        if 'genome' in creature_data:
            result['genome'] = self._reconstruct_genome(creature_data['genome'])
        
        if 'brain' in creature_data:
            result['brain_config'] = self._reconstruct_brain_config(creature_data['brain'])
            if 'weights' in creature_data['brain']:
                result['brain_weights'] = creature_data['brain']['weights']
        
        if 'instincts' in creature_data:
            result['instincts'] = creature_data['instincts']
        
        if 'culture' in creature_data:
            result['culture'] = creature_data['culture']
        
        if 'microbiome' in creature_data:
            result['microbiome'] = creature_data['microbiome']
        
        return result
    
    def _check_version_compatible(self, version: str) -> bool:
        """Check if version is compatible."""
        try:
            major, minor, patch = map(int, version.split('.'))
            our_major, our_minor, _ = map(int, MIGRATION_VERSION.split('.'))
            return major == our_major  # Same major version
        except:
            return False
    
    def _verify_checksums(self, data: Dict) -> bool:
        """Verify data integrity."""
        if 'checksums' not in data:
            return True
        
        # Recompute and compare
        exporter = CreatureExporter()
        if 'genome' in data:
            expected_dna = data['checksums'].get('dna', '')
            actual_dna = exporter._compute_hash(data['genome'])
            if expected_dna and expected_dna != actual_dna:
                return False
        
        return True
    
    def _reconstruct_genome(self, genome_data: Dict) -> Dict:
        """Reconstruct genome from serialized data."""
        return genome_data  # Return as-is, actual Genome object created by caller
    
    def _reconstruct_brain_config(self, brain_data: Dict) -> Dict:
        """Extract brain config parameters."""
        return brain_data.get('config', {})


# =============================================================================
# FILE I/O HELPERS
# =============================================================================

def save_creature_to_file(creature: Any, filepath: Union[str, Path], 
                          simulation_id: str = "local") -> Path:
    """
    Save a creature to a file.
    
    Args:
        creature: The creature to save
        filepath: Where to save
        simulation_id: ID of source simulation
        
    Returns:
        Path to saved file
    """
    exporter = CreatureExporter(simulation_id)
    data = exporter.export_creature(creature)
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        f.write(data)
    
    return filepath


def load_creature_from_file(filepath: Union[str, Path], 
                           simulation_id: str = "local") -> Dict:
    """
    Load a creature from a file.
    
    Args:
        filepath: Path to creature file
        simulation_id: ID of importing simulation
        
    Returns:
        Dict with creature data for reconstruction
    """
    filepath = Path(filepath)
    
    with open(filepath, 'rb') as f:
        data = f.read()
    
    importer = CreatureImporter(simulation_id)
    return importer.import_creature(data)


def export_population(creatures: List[Any], dirpath: Union[str, Path],
                      simulation_id: str = "local") -> List[Path]:
    """
    Export multiple creatures to a directory.
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for i, creature in enumerate(creatures):
        name = getattr(creature, 'name', f'creature_{i}')
        safe_name = "".join(c for c in name if c.isalnum() or c in '_-')
        filepath = dirpath / f"{safe_name}_{i}.creature"
        paths.append(save_creature_to_file(creature, filepath, simulation_id))
    
    return paths


def import_population(dirpath: Union[str, Path], 
                      simulation_id: str = "local") -> List[Dict]:
    """
    Import all creatures from a directory.
    """
    dirpath = Path(dirpath)
    creatures = []
    
    for filepath in dirpath.glob("*.creature"):
        try:
            creature_data = load_creature_from_file(filepath, simulation_id)
            creatures.append(creature_data)
        except Exception as e:
            print(f"Failed to import {filepath}: {e}")
    
    return creatures


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MIGRATION_VERSION',
    'MigrationMetadata',
    'CreatureExporter',
    'CreatureImporter',
    'save_creature_to_file',
    'load_creature_from_file',
    'export_population',
    'import_population',
]
