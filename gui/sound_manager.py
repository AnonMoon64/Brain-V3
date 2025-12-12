from PyQt6.QtMultimedia import QSoundEffect, QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl, QObject
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

class SoundManager(QObject):
    """Manages sound effects for the simulation."""
    
import json

@dataclass
class SoundConfig:
    """Configuration for sound mappings."""
    sound_mappings: Dict[str, str] = field(default_factory=dict)
    muted: bool = False
    master_volume: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            'sound_mappings': self.sound_mappings,
            'muted': self.muted,
            'master_volume': self.master_volume
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SoundConfig':
        config = cls()
        config.sound_mappings = data.get('sound_mappings', {})
        config.muted = data.get('muted', False)
        config.master_volume = data.get('master_volume', 0.5)
        return config

class SoundManager(QObject):
    """Manages sound effects with customizable mappings."""
    
    DEFAULT_SOUNDS = {
        'eat': 'eat.wav',
        'drink': 'drink.wav',
        'sleep': 'sleep.wav',
        'hurt': 'hurt.wav',
        'die': 'die.wav',
        'breed': 'breed.wav'
    }
    
    AUTOSAVE_PATH = Path.home() / ".brain_v3_sound_config.json"
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sounds = {} # For QSoundEffect (WAV)
        self.players = {} # For QMediaPlayer (MP3/OGG)
        self.audio_outputs = {} # Keep references to outputs
        self.config = SoundConfig()
        self.base_path = Path(__file__).parent.parent / "sounds"
        
        # Ensure directory exists
        if not self.base_path.exists():
            try:
                self.base_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
                
        # Load config or defaults
        self._load_config()
        
        # Initialize default mappings if empty
        if not self.config.sound_mappings:
            self.config.sound_mappings = self.DEFAULT_SOUNDS.copy()
            
        self.load_sounds()
        
    def _load_config(self):
        """Load configuration from file."""
        if self.AUTOSAVE_PATH.exists():
            try:
                with open(self.AUTOSAVE_PATH, 'r') as f:
                    self.config = SoundConfig.from_dict(json.load(f))
            except Exception as e:
                print(f"Failed to load sound config: {e}")
        
    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.AUTOSAVE_PATH, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Failed to save sound config: {e}")
            
    def load_sounds(self):
        """Load sounds based on current config."""
        # Clear existing resources
        self.sounds.clear()
        for p in self.players.values():
            p.setSource(QUrl())
            p.deleteLater()
        self.players.clear()
        self.audio_outputs.clear()
        
        mappings_changed = False
        
        for name, filename in list(self.config.sound_mappings.items()):
            if not filename:
                continue
                
            path = Path(filename)
            
            # Locate file
            target_path = path
            if not target_path.exists():
                target_path = self.base_path / filename
                
            if not target_path.exists() and target_path.suffix != '.wav':
                 new_name = target_path.with_suffix('.wav').name
                 target_path = self.base_path / new_name
                 
            if not target_path.exists() and name in self.DEFAULT_SOUNDS:
                 default_file = self.DEFAULT_SOUNDS[name]
                 default_path = self.base_path / default_file
                 if default_path.exists():
                     target_path = default_path
                     self.config.sound_mappings[name] = str(default_file)
                     mappings_changed = True

            if target_path.exists():
                # DECIDE PLAYER TYPE
                is_wav = target_path.suffix.lower() == '.wav'
                
                if is_wav:
                    # Use QSoundEffect for WAV (Low latency, polyphonic)
                    effect = QSoundEffect()
                    effect.setSource(QUrl.fromLocalFile(str(target_path)))
                    effect.setVolume(self.config.master_volume)
                    self.sounds[name] = effect
                else:
                    # Use QMediaPlayer for MP3/OGG (Better codec support)
                    player = QMediaPlayer()
                    audio_out = QAudioOutput()
                    player.setAudioOutput(audio_out)
                    audio_out.setVolume(self.config.master_volume)
                    player.setSource(QUrl.fromLocalFile(str(target_path)))
                    
                    self.players[name] = player
                    self.audio_outputs[name] = audio_out 
            else:
                 print(f"Sound file not found: {path} (and default failed)")
                 
        if mappings_changed:
            self.save_config()

    def play(self, name: str, volume: float = 1.0):
        """Play a sound effect."""
        if self.config.muted:
            return
            
        final_vol = volume * self.config.master_volume
        
        # Try QSoundEffect (WAV)
        if name in self.sounds:
            effect = self.sounds[name]
            effect.setVolume(final_vol)
            if not effect.isPlaying():
                effect.play()
                print(f"Playing WAV: {name} ({final_vol:.2f})")
                
        # Try QMediaPlayer (MP3/OGG)
        elif name in self.players:
            player = self.players[name]
            out = self.audio_outputs[name]
            out.setVolume(final_vol)
            
            if player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                player.setPosition(0)
                player.play()
                print(f"Playing Media: {name} ({final_vol:.2f})")
        else:
            print(f"Sound not loaded: {name}")
    
    def set_muted(self, muted: bool):
        self.config.muted = muted
        self.save_config()
        
    def set_master_volume(self, volume: float):
        self.config.master_volume = max(0.0, min(1.0, volume))
        # Update loaded sounds
        for effect in self.sounds.values():
            effect.setVolume(self.config.master_volume)
        for out in self.audio_outputs.values():
            out.setVolume(self.config.master_volume)
        self.save_config()
        
    def set_sound_mapping(self, name: str, filepath: str):
        """Update mapping for a sound action."""
        self.config.sound_mappings[name] = filepath
        self.save_config()
        self.load_sounds() # Reload to apply changes
