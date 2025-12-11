import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QWidget, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont

class BrainInspectorWindow(QDialog):
    """
    Visual debugger for the Three-System Brain.
    Shows real-time activations of sensory, cortex, reservoir, and motor systems.
    """
    
    def __init__(self, creature, parent=None):
        super().__init__(parent)
        self.creature = creature
        self.setWindowTitle(f"Brain Inspector: {creature.name}")
        self.resize(1000, 600)
        
        layout = QVBoxLayout(self)
        
        # Top Stats
        stats_layout = QHBoxLayout()
        self.lbl_energy = QLabel("Energy: 0%")
        self.lbl_health = QLabel("Health: 0%")
        self.lbl_mood = QLabel("Mood: Neutral")
        stats_layout.addWidget(self.lbl_energy)
        stats_layout.addWidget(self.lbl_health)
        stats_layout.addWidget(self.lbl_mood)
        layout.addLayout(stats_layout)
        
        # Main Visualization Areas
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 1. Sensory Inputs
        self.sensory_panel = SensoryPanel()
        splitter.addWidget(self.create_group("Sensory Input (128)", self.sensory_panel))
        
        # 2. Central Brain (Cortex + Reservoir)
        self.brain_panel = CentralBrainPanel()
        splitter.addWidget(self.create_group("System 1 & 2: Cortex & Reservoir", self.brain_panel))
        
        # 3. Outputs & Chemicals
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.motor_panel = MotorPanel()
        self.chemical_panel = ChemicalPanel()
        
        right_layout.addWidget(self.create_group("Motor Output (64)", self.motor_panel))
        right_layout.addWidget(self.create_group("System 3: Neuromodulators", self.chemical_panel))
        splitter.addWidget(right_widget)
        
        splitter.setSizes([200, 500, 300])
        layout.addWidget(splitter)
        
        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_view)
        self.timer.start(100) # 10fps update
        
    def create_group(self, title, widget):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        layout.setContentsMargins(0,0,0,0)
        return group
        
    def update_view(self):
        if not self.creature or not self.creature.body.is_alive():
            self.lbl_mood.setText("Creature Dead")
            return
            
        c = self.creature
        brain = c.embodied_brain.brain if c.embodied_brain else None
        
        # Update labels
        self.lbl_energy.setText(f"Energy: {c.body.homeostasis.energy:.1%}")
        self.lbl_health.setText(f"Health: {c.body.homeostasis.health:.1%}")
        
        if not brain:
            self.lbl_mood.setText("No Brain (Instinct Only)")
            return
            
        # Get Data
        # We need to access the last step's data from the embodied brain
        # Assuming the brain object holds state
        
        # 1. Sensory
        # Accessing private attribute is risky, but for debug it's okay?
        # Actually EmbodiedBrain.step returns dict with 'sensory_input'
        # But we don't have access to the return value of step() here easily unless we store it.
        # Let's hope EmbodiedBrain stores it.
        # It creates self.actions_taken = [], maybe we extend it to store last_state
        # For now, let's grab it from where it might be.
        # The SensoryEncoder does not store state.
        # We might need to modify EmbodiedBrain to store `last_sensory_input`
        
        # WORKAROUND: We can re-encode current state or check if stored.
        # Brain class usually has inputs?
        # `brain.reservoir.W_input` is weights.
        
        # Let's update `EmbodiedBrain` to store `last_state_dict` in next step.
        # For now, check if we can get it from creature
        if hasattr(c.embodied_brain, 'last_step_data'):
            data = c.embodied_brain.last_step_data
            if data:
                self.sensory_panel.update_data(data.get('sensory_input', np.zeros(128)))
                self.motor_panel.update_data(data.get('brain_output', np.zeros(64)))
                
        # 2. Central Brain
        stats = brain.get_stats() # Just config
        # We need state
        if hasattr(brain, 'reservoir'):
             res_state = brain.reservoir.state
             self.brain_panel.update_reservoir(res_state)
             
        if hasattr(brain, 'cortex'):
             cortex_act = brain.cortex.activation
             self.brain_panel.update_cortex(cortex_act)
             
        # 3. Chemicals
        if hasattr(brain, 'learning'):
            # Use get_all_levels() method instead of .levels attribute
            if hasattr(brain.learning.neuromod, 'get_all_levels'):
                chem = brain.learning.neuromod.get_all_levels()
            elif hasattr(brain.learning, 'get_neuromodulator_levels'):
                chem = brain.learning.get_neuromodulator_levels()
            else:
                chem = {}
            self.chemical_panel.update_data(chem)
            
            mood_data = brain.get_dashboard_data()
            self.lbl_mood.setText(f"Mood: {mood_data.get('mood', 'Unknown')}")

class SensoryPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.data = np.zeros(128)
        self.setMinimumWidth(100)
        
    def update_data(self, data):
        self.data = data
        self.update()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        
        if len(self.data) == 0: return
        
        h = self.height()
        w = self.width()
        bar_h = h / len(self.data)
        
        for i, val in enumerate(self.data):
            y = i * bar_h
            bw = val * w
            
            # Color coding by region (approximate based on encoder)
            # 0-16 Intero, 16-32 Drive ...
            if i < 16: c = QColor(100, 100, 255) # Intero
            elif i < 32: c = QColor(255, 100, 100) # Drives
            elif i < 40: c = QColor(100, 255, 100) # Proprio
            elif i < 72: c = QColor(255, 255, 0) # Vision
            else: c = QColor(200, 200, 200) # Other
            
            p.fillRect(QRectF(0, y, bw, bar_h-1), c)

class MotorPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.data = np.zeros(64)
        self.setMinimumHeight(100)
        
    def update_data(self, data):
        self.data = data
        self.update()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        
        if len(self.data) == 0: return
        
        w = self.width()
        h = self.height()
        bar_w = w / len(self.data)
        
        for i, val in enumerate(self.data):
            x = i * bar_w
            bh = val * h
            
            p.fillRect(QRectF(x, h - bh, bar_w-1, bh), QColor(255, 150, 50))

class ChemicalPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.levels = {}
        self.setMinimumHeight(100)
        
    def update_data(self, levels):
        self.levels = levels
        self.update()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        
        if not self.levels: return
        
        w = self.width()
        h = self.height()
        count = len(self.levels)
        bar_w = w / count
        
        keys = sorted(self.levels.keys())
        p.setFont(QFont("Arial", 8))
        
        for i, k in enumerate(keys):
            try:
                # k is Enum modulatortype
                name = k.name[:3]
                val = self.levels[k]
            except:
                name = str(k)[:3]
                val = 0
            
            x = i * bar_w
            bh = val * h
            
            # Color
            c = QColor(100, 200, 255)
            if "DOPAMINE" in str(k).upper(): c = QColor(255, 200, 0)
            if "CORTISOL" in str(k).upper(): c = QColor(255, 50, 50)
            if "SEROTONIN" in str(k).upper(): c = QColor(50, 50, 255)
            
            p.fillRect(QRectF(x, h - bh, bar_w-2, bh), c)
            
            p.setPen(QColor(255,255,255))
            p.drawText(QRectF(x, h-15, bar_w, 15), Qt.AlignmentFlag.AlignCenter, name)

class CentralBrainPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.res_state = np.zeros(1)
        self.cortex_act = np.zeros(1)
        self.setMinimumHeight(300)
        
    def update_reservoir(self, state):
        self.res_state = state
        self.update()
        
    def update_cortex(self, act):
        self.cortex_act = act
        self.update()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(10, 15, 20))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Draw Reservoir as scatter
        # We don't have XY coords for neurons, so we just project them randomly or linearly
        # To make it look "alive", we can map index to simple spiral or grid
        
        count = len(self.res_state)
        if count > 0:
            # Use pseudo-random positions seeded by index (stable)
            cx, cy = w/2, h/2
            radius = min(w, h) * 0.4
            
            # Draw only active ones to save perf
            active_indices = np.where(np.abs(self.res_state) > 0.05)[0]
            
            p.setPen(Qt.PenStyle.NoPen)
            
            for idx in active_indices:
                val = self.res_state[idx]
                
                # Simple spiral projection
                angle = idx * 0.1
                r = (idx / count) * radius
                x = cx + np.cos(angle) * r
                y = cy + np.sin(angle) * r
                
                # Color based on val
                alpha = int(min(255, abs(val) * 200 + 55))
                if val > 0:
                    c = QColor(100, 200, 255, alpha)
                else:
                    c = QColor(255, 100, 100, alpha)
                
                p.setBrush(QBrush(c))
                size = 3 + abs(val) * 5
                p.drawEllipse(QPointF(x, y), size, size)
                
        # Draw Cortex as overlay grid (top left)
        # TODO if needed. For now the reservoir is the "cool" part.
