import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QWidget, QFrame, QSplitter, QGridLayout, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient

class BrainInspectorWindow(QDialog):
    """
    Visual debugger for the Three-System Brain.
    Shows real-time activations of sensory, cortex, reservoir, and motor systems.
    Enhanced to show: neural activity, synaptic strengths, neuromodulators, and motor influence.
    """
    
    def __init__(self, creature, parent=None):
        super().__init__(parent)
        self.creature = creature
        self.setWindowTitle(f"🧠 Brain Inspector: {creature.name}")
        self.resize(1200, 700)
        self.setStyleSheet("background-color: #1a1a2e; color: #eee;")
        
        layout = QVBoxLayout(self)
        
        # Top Stats Row
        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        stats_layout.setContentsMargins(5, 5, 5, 5)
        
        self.lbl_energy = QLabel("⚡ Energy: 0%")
        self.lbl_health = QLabel("❤️ Health: 0%")
        self.lbl_mood = QLabel("😐 Mood: Neutral")
        self.lbl_state = QLabel("🎯 State: Idle")
        self.lbl_reward = QLabel("🎁 Reward: 0.00")
        
        for lbl in [self.lbl_energy, self.lbl_health, self.lbl_mood, self.lbl_state, self.lbl_reward]:
            lbl.setStyleSheet("font-size: 12px; padding: 5px; background: #2a2a4e; border-radius: 5px;")
            stats_layout.addWidget(lbl)
        
        layout.addWidget(stats_widget)
        
        # Main Visualization Areas
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 1. Left Panel: Sensory + Drives
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.sensory_panel = SensoryPanel()
        left_layout.addWidget(self.create_group("📡 Sensory Input (128 channels)", self.sensory_panel))
        
        self.drive_panel = DrivePanel()
        left_layout.addWidget(self.create_group("🔥 Active Drives", self.drive_panel))
        
        splitter.addWidget(left_widget)
        
        # 2. Central Panel: Brain Activity (Reservoir + Cortex)
        self.brain_panel = CentralBrainPanel()
        splitter.addWidget(self.create_group("🧠 Neural Activity (Reservoir + Cortex)", self.brain_panel))
        
        # 3. Right Panel: Outputs + Chemicals + Motor
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.motor_panel = MotorPanel()
        right_layout.addWidget(self.create_group("🎮 Motor Output (64 channels)", self.motor_panel))
        
        self.chemical_panel = ChemicalPanel()
        right_layout.addWidget(self.create_group("💉 Neuromodulators", self.chemical_panel))
        
        self.motor_influence_panel = MotorInfluencePanel()
        right_layout.addWidget(self.create_group("⚙️ Motor Command Influence", self.motor_influence_panel))
        
        splitter.addWidget(right_widget)
        
        splitter.setSizes([250, 500, 350])
        layout.addWidget(splitter)
        
        # Timer for real-time updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_view)
        self.timer.start(100)  # 10fps update
        
    def create_group(self, title, widget):
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        layout.setContentsMargins(5, 15, 5, 5)
        return group
        
    def update_view(self):
        if not self.creature or not self.creature.body.is_alive():
            self.lbl_mood.setText("💀 Creature Dead")
            return
            
        c = self.creature
        brain = c.embodied_brain.brain if c.embodied_brain else None
        
        # Update top labels
        h = c.body.homeostasis
        self.lbl_energy.setText(f"⚡ Energy: {h.energy:.0%}")
        self.lbl_health.setText(f"❤️ Health: {h.health:.0%}")
        
        # State
        state_name = c.behavior_state.state.name if c.behavior_state else "Unknown"
        self.lbl_state.setText(f"🎯 State: {state_name}")
        
        if not brain:
            self.lbl_mood.setText("😐 Instinct Only")
            return
        
        # Get last step data for visualization
        data = None
        if hasattr(c.embodied_brain, 'last_step_data') and c.embodied_brain.last_step_data:
            data = c.embodied_brain.last_step_data
            
            # Update panels with real data
            self.sensory_panel.update_data(data.get('sensory_input', np.zeros(128)))
            self.motor_panel.update_data(data.get('brain_output', np.zeros(64)))
            self.drive_panel.update_data(data.get('drives', {}))
            self.motor_influence_panel.update_data(data.get('motor_commands', {}), data.get('neuro_changes', {}))
            
            # Reward
            reward = data.get('reward', 0)
            self.lbl_reward.setText(f"🎁 Reward: {reward:+.3f}")
        
        # Brain visualization (reservoir state)
        if hasattr(brain, 'reservoir') and brain.reservoir:
            self.brain_panel.update_reservoir(brain.reservoir.state)
            
        if hasattr(brain, 'cortex') and brain.cortex:
            self.brain_panel.update_cortex(brain.cortex.activation)
        
        # Chemicals
        if hasattr(brain, 'learning') and hasattr(brain.learning, 'neuromod'):
            if hasattr(brain.learning.neuromod, 'get_all_levels'):
                chem = brain.learning.neuromod.get_all_levels()
            else:
                chem = {}
            self.chemical_panel.update_data(chem)
            
            # Mood from dashboard
            mood_data = brain.get_dashboard_data()
            self.lbl_mood.setText(f"😊 Mood: {mood_data.get('mood', 'Unknown')}")


class SensoryPanel(QWidget):
    """Vertical bar chart of sensory input channels."""
    
    def __init__(self):
        super().__init__()
        self.data = np.zeros(128)
        self.setMinimumWidth(150)
        self.setMinimumHeight(200)
        
    def update_data(self, data):
        self.data = np.array(data) if data is not None else np.zeros(128)
        self.update()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(15, 15, 25))
        
        if len(self.data) == 0:
            return
        
        h = self.height()
        w = self.width()
        bar_h = max(1, h / len(self.data))
        
        # Region labels
        regions = [
            (0, 16, "Intero", QColor(100, 100, 255)),
            (16, 32, "Drives", QColor(255, 100, 100)),
            (32, 40, "Proprio", QColor(100, 255, 100)),
            (40, 72, "Vision", QColor(255, 255, 0)),
            (72, 88, "Smell", QColor(255, 150, 0)),
            (88, 104, "Audio", QColor(150, 100, 255)),
            (104, 128, "Social", QColor(255, 100, 255)),
        ]
        
        for start, end, name, color in regions:
            for i in range(start, min(end, len(self.data))):
                y = i * bar_h
                val = abs(self.data[i])
                bw = min(val * (w - 30), w - 30)
                
                # Intensity affects alpha
                alpha = int(100 + val * 155)
                c = QColor(color.red(), color.green(), color.blue(), alpha)
                
                p.fillRect(QRectF(25, y, bw, max(1, bar_h - 1)), c)
        
        # Draw region labels
        p.setPen(QColor(200, 200, 200))
        p.setFont(QFont("Arial", 7))
        for start, end, name, color in regions:
            y = start * bar_h + (end - start) * bar_h / 2
            p.drawText(0, int(y), name[:3])


class DrivePanel(QWidget):
    """Horizontal bars showing active drives."""
    
    def __init__(self):
        super().__init__()
        self.drives = {}
        self.setMinimumHeight(120)
        
    def update_data(self, drives):
        self.drives = drives or {}
        self.update()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(15, 15, 25))
        
        if not self.drives:
            p.setPen(QColor(100, 100, 100))
            p.drawText(10, 20, "No drive data")
            return
        
        w = self.width()
        h = self.height()
        bar_h = h / max(1, len(self.drives))
        
        colors = {
            'hunger': QColor(255, 150, 50),
            'thirst': QColor(50, 150, 255),
            'rest': QColor(100, 255, 100),
            'safety': QColor(255, 50, 50),
            'reproduction': QColor(255, 100, 200),
            'exploration': QColor(200, 200, 50),
        }
        
        p.setFont(QFont("Arial", 9))
        
        for i, (name, val) in enumerate(sorted(self.drives.items(), key=lambda x: -x[1])):
            y = i * bar_h
            bw = val * (w - 80)
            
            color = colors.get(name, QColor(150, 150, 150))
            
            # Gradient bar
            gradient = QLinearGradient(60, y, 60 + bw, y)
            gradient.setColorAt(0, color.darker(150))
            gradient.setColorAt(1, color)
            
            p.fillRect(QRectF(60, y + 2, bw, bar_h - 4), gradient)
            
            # Label
            p.setPen(QColor(255, 255, 255))
            p.drawText(5, int(y + bar_h / 2 + 4), name[:6].title())
            
            # Value
            p.drawText(int(65 + bw), int(y + bar_h / 2 + 4), f"{val:.0%}")


class MotorPanel(QWidget):
    """Vertical bars showing motor output channels."""
    
    def __init__(self):
        super().__init__()
        self.data = np.zeros(64)
        self.setMinimumHeight(100)
        
    def update_data(self, data):
        self.data = np.array(data) if data is not None else np.zeros(64)
        self.update()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(15, 15, 25))
        
        if len(self.data) == 0:
            return
        
        w = self.width()
        h = self.height()
        bar_w = w / len(self.data)
        
        for i, val in enumerate(self.data):
            x = i * bar_w
            bh = abs(val) * (h - 10)
            
            # Color based on value
            if val > 0:
                c = QColor(100, 255, 150, int(100 + abs(val) * 155))
            else:
                c = QColor(255, 100, 100, int(100 + abs(val) * 155))
            
            p.fillRect(QRectF(x, h - bh - 5, max(1, bar_w - 1), bh), c)
        
        # Label important channels
        p.setPen(QColor(200, 200, 200))
        p.setFont(QFont("Arial", 7))
        labels = [(0, "MX"), (2, "Jump"), (3, "Eat"), (4, "Drink")]
        for idx, label in labels:
            if idx < len(self.data):
                x = idx * bar_w
                p.drawText(int(x), h - 2, label)


class ChemicalPanel(QWidget):
    """Bar chart of neuromodulator levels."""
    
    def __init__(self):
        super().__init__()
        self.levels = {}
        self.setMinimumHeight(120)
        
    def update_data(self, levels):
        self.levels = levels or {}
        self.update()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(15, 15, 25))
        
        if not self.levels:
            p.setPen(QColor(100, 100, 100))
            p.drawText(10, 20, "No chemical data")
            return
        
        w = self.width()
        h = self.height()
        count = len(self.levels)
        bar_w = (w - 10) / count
        
        # Sort and process
        items = []
        for k, v in self.levels.items():
            try:
                name = k.name if hasattr(k, 'name') else str(k)
                val = float(v)
            except:
                name = str(k)[:8]
                val = 0
            items.append((name, val))
        
        colors = {
            'DOPAMINE': QColor(255, 200, 50),
            'CORTISOL': QColor(255, 50, 50),
            'SEROTONIN': QColor(50, 100, 255),
            'NOREPINEPHRINE': QColor(255, 150, 50),
            'OXYTOCIN': QColor(255, 100, 200),
            'GABA': QColor(100, 200, 100),
            'ACETYLCHOLINE': QColor(200, 200, 100),
            'ENDORPHIN': QColor(200, 100, 255),
            'ADRENALINE': QColor(255, 50, 100),
            'GLUTAMATE': QColor(100, 255, 200),
        }
        
        p.setFont(QFont("Arial", 7))
        
        for i, (name, val) in enumerate(items):
            x = 5 + i * bar_w
            bh = val * (h - 25)
            
            color = colors.get(name.upper(), QColor(150, 150, 150))
            
            # Gradient bar
            gradient = QLinearGradient(x, h - bh, x, h)
            gradient.setColorAt(0, color)
            gradient.setColorAt(1, color.darker(200))
            
            p.fillRect(QRectF(x, h - bh - 15, bar_w - 2, bh), gradient)
            
            # Label
            p.setPen(QColor(230, 230, 230))
            p.drawText(QRectF(x, h - 12, bar_w, 12), Qt.AlignmentFlag.AlignCenter, name[:3])


class MotorInfluencePanel(QWidget):
    """Shows how neuromodulators are influencing current motor commands."""
    
    def __init__(self):
        super().__init__()
        self.commands = {}
        self.neuro_changes = {}
        self.setMinimumHeight(100)
        
    def update_data(self, commands, neuro_changes):
        self.commands = commands or {}
        self.neuro_changes = neuro_changes or {}
        self.update()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(15, 15, 25))
        
        h = self.height()
        w = self.width()
        
        p.setFont(QFont("Arial", 10))
        y = 15
        
        # Show active motor commands
        p.setPen(QColor(100, 255, 150))
        p.drawText(5, y, "Active Commands:")
        y += 15
        
        for name, val in self.commands.items():
            if val > 0.1:
                bar_w = val * (w - 100)
                p.fillRect(QRectF(80, y - 10, bar_w, 12), QColor(100, 200, 150, 150))
                p.setPen(QColor(255, 255, 255))
                p.drawText(5, y, f"{name}: {val:.0%}")
                y += 15
                if y > h - 30:
                    break
        
        # Show chemical influence summary
        y = max(y + 10, h - 40)
        p.setPen(QColor(255, 200, 100))
        
        # Summarize chemical effects
        dopamine = self.neuro_changes.get('dopamine', 0)
        cortisol = self.neuro_changes.get('cortisol', 0)
        
        if dopamine > 0.05:
            p.drawText(5, y, f"📈 Dopamine boost: +{dopamine:.2f} (motivation)")
            y += 12
        if cortisol > 0.1:
            p.setPen(QColor(255, 100, 100))
            p.drawText(5, y, f"⚠️ Stress response: +{cortisol:.2f} (cortisol)")


class CentralBrainPanel(QWidget):
    """Visualization of reservoir and cortex neural activity."""
    
    def __init__(self):
        super().__init__()
        self.res_state = np.zeros(1)
        self.cortex_act = np.zeros(1)
        self.setMinimumHeight(300)
        
    def update_reservoir(self, state):
        self.res_state = np.array(state) if state is not None else np.zeros(1)
        self.update()
        
    def update_cortex(self, act):
        self.cortex_act = np.array(act) if act is not None else np.zeros(1)
        self.update()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Dark gradient background
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(10, 15, 25))
        gradient.setColorAt(1, QColor(5, 10, 20))
        p.fillRect(self.rect(), gradient)
        
        w = self.width()
        h = self.height()
        cx, cy = w / 2, h / 2
        radius = min(w, h) * 0.4
        
        # Draw reservoir neurons as spiral
        count = len(self.res_state)
        if count > 0:
            # Find active neurons for efficient drawing
            active_mask = np.abs(self.res_state) > 0.02
            active_indices = np.where(active_mask)[0]
            
            p.setPen(Qt.PenStyle.NoPen)
            
            # Draw all neurons as faint dots first
            p.setBrush(QBrush(QColor(30, 40, 60)))
            for idx in range(0, count, max(1, count // 200)):  # Sample for performance
                angle = idx * 0.15
                r = (idx / count) * radius
                x = cx + np.cos(angle) * r
                y = cy + np.sin(angle) * r
                p.drawEllipse(QPointF(x, y), 2, 2)
            
            # Draw active neurons brightly
            for idx in active_indices:
                val = self.res_state[idx]
                
                # Spiral projection
                angle = idx * 0.15
                r = (idx / count) * radius
                x = cx + np.cos(angle) * r
                y = cy + np.sin(angle) * r
                
                # Color based on activation value
                intensity = min(255, int(abs(val) * 300 + 50))
                if val > 0:
                    c = QColor(50, 150, 255, intensity)  # Blue for positive
                else:
                    c = QColor(255, 100, 100, intensity)  # Red for negative
                
                p.setBrush(QBrush(c))
                size = 2 + abs(val) * 8
                p.drawEllipse(QPointF(x, y), size, size)
        
        # Draw wave pattern indicator
        p.setPen(QPen(QColor(100, 150, 200, 100), 1))
        p.drawEllipse(QPointF(cx, cy), radius * 0.3, radius * 0.3)
        p.drawEllipse(QPointF(cx, cy), radius * 0.6, radius * 0.6)
        p.drawEllipse(QPointF(cx, cy), radius * 0.9, radius * 0.9)
        
        # Stats overlay
        p.setPen(QColor(200, 200, 200))
        p.setFont(QFont("Arial", 9))
        active_count = np.sum(np.abs(self.res_state) > 0.05) if len(self.res_state) > 1 else 0
        total = len(self.res_state) if len(self.res_state) > 1 else 0
        sparsity = active_count / max(1, total)
        
        p.drawText(10, 20, f"Active neurons: {active_count}/{total} ({sparsity:.1%} firing)")
        
        if len(self.res_state) > 1:
            mean_act = np.mean(np.abs(self.res_state))
            max_act = np.max(np.abs(self.res_state))
            p.drawText(10, 35, f"Mean: {mean_act:.3f} | Max: {max_act:.3f}")
