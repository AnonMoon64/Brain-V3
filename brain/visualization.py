"""
Brain Visualization Tools

Simple matplotlib-based visualization for debugging and monitoring:
- Neuromodulator levels over time
- Cortex activity heatmaps
- Reservoir state trajectories
- Learning curves

Usage:
    from brain.visualization import BrainVisualizer
    
    viz = BrainVisualizer(brain)
    viz.start_recording()
    
    # ... interact with brain ...
    
    viz.plot_modulators()
    viz.plot_activity()
    viz.save_all("session_plots/")
"""

import numpy as np
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from collections import deque
from dataclasses import dataclass, field
import time

# Conditional import for matplotlib (optional dependency)
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

if TYPE_CHECKING:
    from .three_system_brain import ThreeSystemBrain


@dataclass
class RecordedStep:
    """Single timestep of recorded data."""
    timestamp: float
    step: int
    
    # Neuromodulators
    dopamine: float = 0.5
    serotonin: float = 0.5
    norepinephrine: float = 0.3
    acetylcholine: float = 0.5
    cortisol: float = 0.3
    gaba: float = 0.4
    glutamate: float = 0.5
    oxytocin: float = 0.3
    
    # Activity
    cortex_sparsity: float = 0.02
    reservoir_norm: float = 0.0
    confidence: float = 0.0
    novelty: float = 0.0
    
    # Mood
    mood: str = "neutral"


class BrainVisualizer:
    """
    Real-time and post-hoc visualization for Three-System Brain.
    
    Records brain state over time and provides matplotlib plots.
    """
    
    def __init__(self, brain: 'ThreeSystemBrain', max_history: int = 1000):
        """
        Initialize visualizer.
        
        Args:
            brain: ThreeSystemBrain instance to monitor
            max_history: Maximum timesteps to keep in memory
        """
        self.brain = brain
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)
        self.recording = False
        self._start_time = time.time()
        
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not installed. Install with: pip install matplotlib")
    
    def start_recording(self):
        """Start recording brain state."""
        self.recording = True
        self._start_time = time.time()
        self.history.clear()
    
    def stop_recording(self):
        """Stop recording brain state."""
        self.recording = False
    
    def record_step(self):
        """Record current brain state (call after each brain.process())."""
        if not self.recording:
            return
        
        data = self.brain.get_dashboard_data()
        chemicals = data.get('chemicals', {})
        neurons = data.get('neurons', {})
        
        step = RecordedStep(
            timestamp=time.time() - self._start_time,
            step=self.brain.step_count,
            dopamine=chemicals.get('dopamine', 0.5),
            serotonin=chemicals.get('serotonin', 0.5),
            norepinephrine=chemicals.get('norepinephrine', 0.3),
            acetylcholine=chemicals.get('acetylcholine', 0.5),
            cortisol=chemicals.get('cortisol', 0.3),
            gaba=chemicals.get('gaba', 0.4),
            glutamate=chemicals.get('glutamate', 0.5),
            oxytocin=chemicals.get('oxytocin', 0.3),
            cortex_sparsity=neurons.get('sparsity', 0.02) if neurons else 0.02,
            reservoir_norm=float(np.linalg.norm(self.brain.reservoir.state)) if hasattr(self.brain, 'reservoir') else 0,
            confidence=data.get('confidence', 0.0),
            novelty=data.get('novelty', 0.0),
            mood=data.get('mood', 'neutral')
        )
        
        self.history.append(step)
    
    def get_arrays(self) -> Dict[str, np.ndarray]:
        """Convert history to numpy arrays for plotting."""
        if not self.history:
            return {}
        
        steps = list(self.history)
        return {
            'time': np.array([s.timestamp for s in steps]),
            'step': np.array([s.step for s in steps]),
            'dopamine': np.array([s.dopamine for s in steps]),
            'serotonin': np.array([s.serotonin for s in steps]),
            'norepinephrine': np.array([s.norepinephrine for s in steps]),
            'acetylcholine': np.array([s.acetylcholine for s in steps]),
            'cortisol': np.array([s.cortisol for s in steps]),
            'gaba': np.array([s.gaba for s in steps]),
            'glutamate': np.array([s.glutamate for s in steps]),
            'oxytocin': np.array([s.oxytocin for s in steps]),
            'cortex_sparsity': np.array([s.cortex_sparsity for s in steps]),
            'reservoir_norm': np.array([s.reservoir_norm for s in steps]),
            'confidence': np.array([s.confidence for s in steps]),
            'novelty': np.array([s.novelty for s in steps]),
        }
    
    def plot_modulators(self, figsize: tuple = (12, 8), save_path: Optional[str] = None):
        """
        Plot neuromodulator levels over time.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return None
        
        data = self.get_arrays()
        if not data:
            print("No data recorded yet. Call record_step() after brain.process()")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Neuromodulator Dynamics', fontsize=14, fontweight='bold')
        
        time = data['time']
        
        # Plot 1: Core 4 modulators
        ax1 = axes[0, 0]
        ax1.plot(time, data['dopamine'], label='Dopamine', color='#FF5722', linewidth=2)
        ax1.plot(time, data['serotonin'], label='Serotonin', color='#2196F3', linewidth=2)
        ax1.plot(time, data['norepinephrine'], label='Norepinephrine', color='#FF9800', linewidth=1.5)
        ax1.plot(time, data['acetylcholine'], label='Acetylcholine', color='#9C27B0', linewidth=1.5)
        ax1.set_ylabel('Level')
        ax1.set_xlabel('Time (s)')
        ax1.set_title('Core Neuromodulators')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Stress/Inhibition
        ax2 = axes[0, 1]
        ax2.plot(time, data['cortisol'], label='Cortisol', color='#795548', linewidth=2)
        ax2.plot(time, data['gaba'], label='GABA', color='#4CAF50', linewidth=2)
        ax2.plot(time, data['glutamate'], label='Glutamate', color='#F44336', linewidth=1.5)
        ax2.set_ylabel('Level')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('Stress & Balance')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Activity metrics
        ax3 = axes[1, 0]
        ax3.plot(time, data['cortex_sparsity'], label='Cortex Sparsity', color='#3F51B5', linewidth=2)
        ax3.plot(time, data['confidence'], label='Confidence', color='#009688', linewidth=2)
        ax3.plot(time, data['novelty'], label='Novelty', color='#E91E63', linewidth=1.5)
        ax3.set_ylabel('Value')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Activity Metrics')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Reservoir dynamics
        ax4 = axes[1, 1]
        ax4.plot(time, data['reservoir_norm'], label='Reservoir Norm', color='#607D8B', linewidth=2)
        ax4.plot(time, data['oxytocin'], label='Oxytocin', color='#E91E63', linewidth=1.5, linestyle='--')
        ax4.set_ylabel('Value')
        ax4.set_xlabel('Time (s)')
        ax4.set_title('Reservoir & Social')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_modulator_heatmap(self, figsize: tuple = (14, 6), save_path: Optional[str] = None):
        """
        Plot all modulators as a heatmap over time.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save figure
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return None
        
        data = self.get_arrays()
        if not data:
            print("No data recorded")
            return None
        
        # Build matrix
        modulator_names = ['dopamine', 'serotonin', 'norepinephrine', 'acetylcholine',
                          'cortisol', 'gaba', 'glutamate', 'oxytocin']
        matrix = np.array([data[name] for name in modulator_names])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.set_yticks(range(len(modulator_names)))
        ax.set_yticklabels([n.capitalize() for n in modulator_names])
        ax.set_xlabel('Time Step')
        ax.set_title('Neuromodulator Levels Over Time')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Level')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def save_all(self, output_dir: str = "brain_plots"):
        """
        Save all available plots to a directory.
        
        Args:
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        self.plot_modulators(save_path=f"{output_dir}/modulators_{timestamp}.png")
        self.plot_modulator_heatmap(save_path=f"{output_dir}/heatmap_{timestamp}.png")
        
        print(f"Saved all plots to {output_dir}/")
    
    def show(self):
        """Show all current plots (interactive mode)."""
        if HAS_MATPLOTLIB:
            plt.show()


def quick_plot(brain: 'ThreeSystemBrain', steps: int = 50, input_text: str = "Hello"):
    """
    Quick utility to run brain for N steps and plot the results.
    
    Args:
        brain: ThreeSystemBrain instance
        steps: Number of processing steps
        input_text: Text to process each step
    """
    viz = BrainVisualizer(brain)
    viz.start_recording()
    
    for i in range(steps):
        brain.process(input_text)
        viz.record_step()
    
    viz.stop_recording()
    viz.plot_modulators()
    viz.show()
    
    return viz
