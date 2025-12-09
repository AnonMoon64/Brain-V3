"""
Brain Persistence Module

Full Python object graph serialization using dill.
Auto-save functionality with configurable intervals.
"""

import os
import time
import threading
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import pickle
import json

# Try to use dill for full object graph serialization
try:
    import dill
    HAS_DILL = True
except ImportError:
    dill = pickle  # Fallback to pickle
    HAS_DILL = False

import numpy as np


class BrainPersistence:
    """
    Handles saving and loading of brain state with full Python object graph.
    
    Features:
    - Uses dill for complete object serialization (lambdas, closures, etc.)
    - Auto-save at configurable intervals
    - Incremental saves (only if state changed)
    - Backup management with rotation
    - State versioning
    """
    
    VERSION = "2.0"
    
    def __init__(
        self,
        save_directory: str = "./brain_saves",
        auto_save_interval: float = 300.0,  # 5 minutes
        max_backups: int = 5,
        enable_auto_save: bool = True
    ):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        
        self.auto_save_interval = auto_save_interval
        self.max_backups = max_backups
        self.enable_auto_save = enable_auto_save
        
        # State tracking
        self._last_save_hash: Optional[str] = None
        self._last_save_time: float = 0
        self._save_count: int = 0
        
        # Auto-save threading
        self._auto_save_thread: Optional[threading.Thread] = None
        self._stop_auto_save = threading.Event()
        self._brain_ref: Optional[Any] = None
        
        # Callbacks
        self._on_save_callback: Optional[Callable] = None
        self._on_load_callback: Optional[Callable] = None
        
    def _compute_state_hash(self, brain) -> str:
        """Compute hash of brain state for change detection."""
        try:
            # Use a subset of state for fast hashing
            state_repr = {
                'interaction_count': getattr(brain, 'interaction_count', 0),
                'training_count': getattr(brain, 'training_count', 0),
                'total_surprise': getattr(brain, 'total_surprise', 0),
            }
            return hashlib.md5(json.dumps(state_repr, default=str).encode()).hexdigest()
        except Exception:
            return str(time.time())  # Fallback to timestamp
    
    def save(
        self,
        brain,
        filepath: Optional[str] = None,
        include_full_state: bool = True,
        create_backup: bool = True
    ) -> str:
        """
        Save brain state to file using dill for full object graph.
        
        Args:
            brain: The brain object to save
            filepath: Optional specific filepath, else uses default
            include_full_state: Include complete object graph
            create_backup: Whether to create a backup of existing save
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_directory / f"brain_{timestamp}.brain"
        else:
            filepath = Path(filepath)
        
        # Create backup if file exists
        if create_backup and filepath.exists():
            self._rotate_backups(filepath)
        
        # Prepare save data
        save_data = {
            'version': self.VERSION,
            'saved_at': datetime.now().isoformat(),
            'has_dill': HAS_DILL,
            'python_version': self._get_python_version(),
            'save_count': self._save_count,
        }
        
        if include_full_state:
            # Full object graph serialization
            save_data['brain'] = brain
            save_data['serialization'] = 'full'
        else:
            # Partial state serialization (more compatible)
            save_data['state'] = self._extract_serializable_state(brain)
            save_data['serialization'] = 'partial'
        
        # Save using dill (or pickle fallback)
        try:
            with open(filepath, 'wb') as f:
                dill.dump(save_data, f, protocol=dill.HIGHEST_PROTOCOL)
        except Exception as e:
            # Fallback to regular pickle if dill fails
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Update tracking
        self._last_save_hash = self._compute_state_hash(brain)
        self._last_save_time = time.time()
        self._save_count += 1
        
        # Save metadata separately for inspection
        meta_path = filepath.with_suffix('.meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'version': save_data['version'],
                'saved_at': save_data['saved_at'],
                'has_dill': save_data['has_dill'],
                'serialization': save_data['serialization'],
                'save_count': save_data['save_count'],
                'file_size_bytes': os.path.getsize(filepath),
            }, f, indent=2)
        
        if self._on_save_callback:
            try:
                self._on_save_callback(str(filepath))
            except Exception:
                pass
        
        return str(filepath)
    
    def load(
        self,
        filepath: str,
        brain_class=None
    ) -> Any:
        """
        Load brain state from file.
        
        Args:
            filepath: Path to the save file
            brain_class: Optional class to instantiate if needed
            
        Returns:
            Loaded brain object
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Save file not found: {filepath}")
        
        # Load using dill (or pickle fallback)
        try:
            with open(filepath, 'rb') as f:
                save_data = dill.load(f)
        except Exception:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
        
        version = save_data.get('version', '1.0')
        serialization = save_data.get('serialization', 'full')
        
        if serialization == 'full':
            brain = save_data.get('brain')
        else:
            # Reconstruct from partial state
            if brain_class is None:
                raise ValueError("brain_class required for partial state loading")
            brain = brain_class()
            self._restore_state(brain, save_data.get('state', {}))
        
        # Update tracking
        self._last_save_hash = self._compute_state_hash(brain)
        self._last_save_time = time.time()
        
        if self._on_load_callback:
            try:
                self._on_load_callback(brain)
            except Exception:
                pass
        
        return brain
    
    def _extract_serializable_state(self, brain) -> Dict:
        """Extract serializable state from brain."""
        state = {}
        
        # Common attributes
        for attr in ['interaction_count', 'training_count', 'total_surprise',
                     'mood_history', 'created_at']:
            if hasattr(brain, attr):
                value = getattr(brain, attr)
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, np.ndarray):
                    value = value.tolist()
                state[attr] = value
        
        # Word embeddings
        if hasattr(brain, 'word_embeddings'):
            state['word_embeddings'] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in brain.word_embeddings.items()
            }
        
        # Output patterns
        if hasattr(brain, 'output_patterns'):
            state['output_patterns'] = [
                (p.tolist() if isinstance(p, np.ndarray) else p, t)
                for p, t in brain.output_patterns
            ]
        
        # Personality
        if hasattr(brain, 'personality'):
            pers = brain.personality
            state['personality'] = {
                'baseline_shifts': dict(pers.baseline_shifts) if hasattr(pers, 'baseline_shifts') else {},
                'traits': dict(pers.traits) if hasattr(pers, 'traits') else {},
                'chemical_history': list(pers.chemical_history) if hasattr(pers, 'chemical_history') else [],
            }
        
        return state
    
    def _restore_state(self, brain, state: Dict):
        """Restore state to brain."""
        for attr in ['interaction_count', 'training_count', 'total_surprise',
                     'mood_history']:
            if attr in state:
                setattr(brain, attr, state[attr])
        
        if 'created_at' in state:
            brain.created_at = datetime.fromisoformat(state['created_at'])
        
        if 'word_embeddings' in state:
            brain.word_embeddings = {
                k: np.array(v) for k, v in state['word_embeddings'].items()
            }
        
        if 'output_patterns' in state:
            brain.output_patterns = [
                (np.array(p), t) for p, t in state['output_patterns']
            ]
        
        if 'personality' in state and hasattr(brain, 'personality'):
            pers_data = state['personality']
            brain.personality.baseline_shifts = dict(pers_data.get('baseline_shifts', {}))
            brain.personality.traits = dict(pers_data.get('traits', {}))
            brain.personality.chemical_history = list(pers_data.get('chemical_history', []))
    
    def _rotate_backups(self, filepath: Path):
        """Rotate backup files."""
        # Move existing backups
        for i in range(self.max_backups - 1, 0, -1):
            old_backup = filepath.with_suffix(f'.backup{i}')
            new_backup = filepath.with_suffix(f'.backup{i+1}')
            if old_backup.exists():
                if i == self.max_backups - 1:
                    old_backup.unlink()  # Delete oldest
                else:
                    old_backup.rename(new_backup)
        
        # Create new backup
        if filepath.exists():
            backup_path = filepath.with_suffix('.backup1')
            filepath.rename(backup_path)
    
    def _get_python_version(self) -> str:
        """Get Python version string."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # ========== Auto-Save Functionality ==========
    
    def start_auto_save(self, brain, callback: Optional[Callable] = None):
        """
        Start auto-save thread.
        
        Args:
            brain: Brain object to save
            callback: Optional callback after each save
        """
        if not self.enable_auto_save:
            return
        
        self._brain_ref = brain
        self._on_save_callback = callback
        self._stop_auto_save.clear()
        
        self._auto_save_thread = threading.Thread(
            target=self._auto_save_loop,
            daemon=True
        )
        self._auto_save_thread.start()
    
    def stop_auto_save(self):
        """Stop auto-save thread."""
        self._stop_auto_save.set()
        if self._auto_save_thread:
            self._auto_save_thread.join(timeout=5.0)
            self._auto_save_thread = None
    
    def _auto_save_loop(self):
        """Background auto-save loop."""
        while not self._stop_auto_save.is_set():
            # Wait for interval or stop signal
            self._stop_auto_save.wait(timeout=self.auto_save_interval)
            
            if self._stop_auto_save.is_set():
                break
            
            if self._brain_ref is None:
                continue
            
            # Check if state changed
            current_hash = self._compute_state_hash(self._brain_ref)
            if current_hash != self._last_save_hash:
                try:
                    # Save with auto-save filename
                    filepath = self.save_directory / "brain_autosave.brain"
                    self.save(
                        self._brain_ref,
                        filepath=str(filepath),
                        include_full_state=True,
                        create_backup=True
                    )
                except Exception as e:
                    # Log but don't crash
                    print(f"Auto-save failed: {e}")
    
    def list_saves(self) -> list:
        """List all saved brain files."""
        saves = []
        for path in self.save_directory.glob("*.brain"):
            if path.suffix == '.brain' and 'backup' not in path.stem:
                meta_path = path.with_suffix('.meta.json')
                meta = {}
                if meta_path.exists():
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                saves.append({
                    'path': str(path),
                    'name': path.stem,
                    'meta': meta
                })
        return sorted(saves, key=lambda x: x.get('meta', {}).get('saved_at', ''), reverse=True)
    
    def get_latest_save(self) -> Optional[str]:
        """Get path to latest save file."""
        saves = self.list_saves()
        if saves:
            return saves[0]['path']
        return None


class AutoSaveMixin:
    """
    Mixin class to add auto-save functionality to brain classes.
    
    Usage:
        class MyBrain(AutoSaveMixin, BaseBrain):
            pass
        
        brain = MyBrain()
        brain.enable_auto_save(interval=300)  # Every 5 minutes
    """
    
    _persistence: Optional[BrainPersistence] = None
    
    def enable_auto_save(
        self,
        save_directory: str = "./brain_saves",
        interval: float = 300.0,
        max_backups: int = 5
    ):
        """Enable auto-save functionality."""
        self._persistence = BrainPersistence(
            save_directory=save_directory,
            auto_save_interval=interval,
            max_backups=max_backups,
            enable_auto_save=True
        )
        self._persistence.start_auto_save(self)
    
    def disable_auto_save(self):
        """Disable auto-save functionality."""
        if self._persistence:
            self._persistence.stop_auto_save()
    
    def save_brain(self, filepath: Optional[str] = None) -> str:
        """Save brain state."""
        if self._persistence is None:
            self._persistence = BrainPersistence(enable_auto_save=False)
        return self._persistence.save(self, filepath)
    
    def load_brain(self, filepath: str):
        """Load brain state."""
        if self._persistence is None:
            self._persistence = BrainPersistence(enable_auto_save=False)
        loaded = self._persistence.load(filepath)
        # Copy attributes from loaded to self
        for attr in dir(loaded):
            if not attr.startswith('_'):
                try:
                    setattr(self, attr, getattr(loaded, attr))
                except Exception:
                    pass


# Convenience functions for direct use

def save_brain(brain, filepath: str = None, directory: str = "./brain_saves") -> str:
    """
    Save brain with full object graph using dill.
    
    Args:
        brain: Brain object to save
        filepath: Optional specific path
        directory: Save directory if filepath not specified
        
    Returns:
        Path to saved file
    """
    persistence = BrainPersistence(save_directory=directory, enable_auto_save=False)
    return persistence.save(brain, filepath)


def load_brain(filepath: str, brain_class=None):
    """
    Load brain from file.
    
    Args:
        filepath: Path to save file
        brain_class: Optional class for partial state restoration
        
    Returns:
        Loaded brain object
    """
    persistence = BrainPersistence(enable_auto_save=False)
    return persistence.load(filepath, brain_class)


# Check dill availability on import
def check_dill_available() -> bool:
    """Check if dill is available for full object serialization."""
    return HAS_DILL


if not HAS_DILL:
    import warnings
    warnings.warn(
        "dill not installed. Using pickle as fallback. "
        "For full object graph serialization, install dill: pip install dill"
    )
