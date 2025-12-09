"""
Chemical Brain Dashboard - PyQt6 GUI with three tabs

Tab 1: Chat - Interact with the brain
Tab 2: Status - Real-time visualization of brain state
Tab 3: Training - Use OpenAI to teach the brain
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QTextEdit, QLineEdit, QPushButton, QLabel,
    QProgressBar, QGroupBox, QGridLayout, QScrollArea,
    QSpinBox, QComboBox, QFileDialog, QMessageBox,
    QFrame, QSplitter, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor, QPalette, QTextCursor

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from brain import IntegratedBrain, create_brain


class TrainingWorker(QThread):
    """Background worker for OpenAI training"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, brain: IntegratedBrain, api_key: str, topic: str, num_examples: int):
        super().__init__()
        self.brain = brain
        self.api_key = api_key
        self.topic = topic
        self.num_examples = num_examples

    def run(self):
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)

            self.progress.emit(f"Generating {self.num_examples} training examples about '{self.topic}'...")

            # Generate training data from OpenAI
            prompt = f"""Generate {self.num_examples} diverse conversation examples for training a chatbot about: {self.topic}

Each example should be a user message and an ideal assistant response.
Format as JSON array:
[
  {{"user": "user message", "assistant": "ideal response"}},
  ...
]

Make the examples varied, covering different aspects of the topic.
Keep responses concise but informative."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
            )

            import json
            content = response.choices[0].message.content
            # Extract JSON from response
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                examples = json.loads(content[start:end])
            else:
                self.error.emit("Failed to parse training data from OpenAI")
                return

            self.progress.emit(f"Training brain with {len(examples)} examples...")

            results = []
            for i, example in enumerate(examples):
                user_msg = example.get('user', '')
                assistant_msg = example.get('assistant', '')
                if user_msg and assistant_msg:
                    result = self.brain.train(user_msg, assistant_msg)
                    results.append({
                        'input': user_msg,
                        'expected': assistant_msg,
                        'mood': result['mood']
                    })
                    self.progress.emit(f"[{i+1}/{len(examples)}] Training: \"{user_msg[:40]}...\"")
                    self.progress.emit(f"    → Response: \"{assistant_msg[:40]}...\"")

            self.finished.emit({
                'topic': self.topic,
                'examples_trained': len(results),
                'results': results
            })

        except ImportError:
            self.error.emit("OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            self.error.emit(str(e))


class ChemicalBar(QWidget):
    """Custom widget for displaying a chemical level"""

    def __init__(self, name: str, color: str = "#4CAF50"):
        super().__init__()
        self.name = name
        self.color = color

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        self.label = QLabel(name.capitalize()[:12])
        self.label.setFixedWidth(90)
        self.label.setStyleSheet("font-size: 11px;")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(True)
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2a2a2a;
                height: 16px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """)

        self.effective_label = QLabel("1.00x")
        self.effective_label.setFixedWidth(45)
        self.effective_label.setStyleSheet("font-size: 10px; color: #888;")

        layout.addWidget(self.label)
        layout.addWidget(self.bar)
        layout.addWidget(self.effective_label)

    def set_value(self, value: float, effective_multiplier: float = 1.0):
        self.bar.setValue(int(value * 100))
        self.bar.setFormat(f"{value:.2f}")
        self.effective_label.setText(f"{effective_multiplier:.2f}x")


class ChatTab(QWidget):
    """Tab for chatting with the brain"""

    def __init__(self, brain: IntegratedBrain):
        super().__init__()
        self.brain = brain
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
            }
        """)

        # Input area
        input_layout = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
        """)
        self.input_field.returnPressed.connect(self.send_message)

        self.send_btn = QPushButton("Send")
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.send_btn.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)

        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")

        layout.addWidget(self.chat_history)
        layout.addLayout(input_layout)
        layout.addWidget(self.status_label)

        # Welcome message
        self.append_system("Chemical Brain initialized. Start chatting!")

    def send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return

        self.input_field.clear()
        self.append_message("You", text, "#64B5F6")

        # Process through brain
        result = self.brain.process(text)
        response = result['response']
        mood = result['mood']

        self.append_message(f"Brain [{mood}]", response, "#81C784")

        # Update status
        stats = result['growth_stats']
        changes = []
        if stats['neurons_born'] > 0:
            changes.append(f"+{stats['neurons_born']} neurons")
        if stats['neurons_died'] > 0:
            changes.append(f"-{stats['neurons_died']} neurons")
        if stats['synapses_formed'] > 0:
            changes.append(f"+{stats['synapses_formed']} synapses")

        if changes:
            self.status_label.setText(f"Structure changed: {', '.join(changes)}")
        else:
            self.status_label.setText(f"Mood: {mood} | Concepts: {', '.join(result['concepts_detected'][:3])}")

    def append_message(self, sender: str, text: str, color: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        html = f'<p><span style="color: #888;">[{timestamp}]</span> <b style="color: {color};">{sender}:</b> {text}</p>'
        self.chat_history.append(html)
        self.chat_history.moveCursor(QTextCursor.MoveOperation.End)

    def append_system(self, text: str):
        html = f'<p style="color: #FFB74D; font-style: italic;">{text}</p>'
        self.chat_history.append(html)


class StatusTab(QWidget):
    """Tab for viewing brain status"""

    def __init__(self, brain: IntegratedBrain):
        super().__init__()
        self.brain = brain
        self.chemical_bars = {}
        self.setup_ui()

        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(500)  # Update every 500ms

    def setup_ui(self):
        layout = QHBoxLayout(self)

        # Left column - Chemicals
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Chemicals group
        chem_group = QGroupBox("Neurochemicals")
        chem_group.setStyleSheet("""
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
        chem_layout = QVBoxLayout(chem_group)

        # Color mapping for chemicals
        colors = {
            'dopamine': '#FF5722',
            'serotonin': '#2196F3',
            'norepinephrine': '#FF9800',
            'acetylcholine': '#9C27B0',
            'gaba': '#4CAF50',
            'glutamate': '#F44336',
            'cortisol': '#795548',
            'oxytocin': '#E91E63',
            'endorphin': '#00BCD4',
            'adrenaline': '#FFEB3B',
        }

        # All 10 chemicals
        chemical_names = [
            'dopamine', 'serotonin', 'norepinephrine', 'acetylcholine',
            'gaba', 'glutamate', 'cortisol', 'oxytocin', 'endorphin', 'adrenaline'
        ]

        for chem in chemical_names:
            bar = ChemicalBar(chem, colors.get(chem, '#4CAF50'))
            self.chemical_bars[chem] = bar
            chem_layout.addWidget(bar)

        left_layout.addWidget(chem_group)

        # Mood display
        mood_group = QGroupBox("Current State")
        mood_layout = QVBoxLayout(mood_group)

        self.mood_label = QLabel("Mood: neutral")
        self.mood_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;")
        mood_layout.addWidget(self.mood_label)

        self.interactions_label = QLabel("Interactions: 0")
        mood_layout.addWidget(self.interactions_label)

        self.training_label = QLabel("Training samples: 0")
        mood_layout.addWidget(self.training_label)

        left_layout.addWidget(mood_group)
        left_layout.addStretch()

        # Right column - Network and Memory
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Network stats
        net_group = QGroupBox("Neural Network")
        net_layout = QGridLayout(net_group)

        self.neurons_label = QLabel("Hidden Neurons: 0")
        self.synapses_label = QLabel("Synapses: 0")
        self.density_label = QLabel("Density: 0.000")
        self.born_label = QLabel("Born: 0")
        self.died_label = QLabel("Died: 0")

        net_layout.addWidget(self.neurons_label, 0, 0)
        net_layout.addWidget(self.synapses_label, 0, 1)
        net_layout.addWidget(self.density_label, 1, 0)
        net_layout.addWidget(self.born_label, 1, 1)
        net_layout.addWidget(self.died_label, 2, 0)

        right_layout.addWidget(net_group)

        # Memory stats
        mem_group = QGroupBox("Memory")
        mem_layout = QGridLayout(mem_group)

        self.memories_label = QLabel("Stored: 0")
        self.encoded_label = QLabel("Encoded: 0")
        self.forgotten_label = QLabel("Forgotten: 0")
        self.strength_label = QLabel("Avg Strength: 0.00")

        mem_layout.addWidget(self.memories_label, 0, 0)
        mem_layout.addWidget(self.encoded_label, 0, 1)
        mem_layout.addWidget(self.forgotten_label, 1, 0)
        mem_layout.addWidget(self.strength_label, 1, 1)

        right_layout.addWidget(mem_group)

        # Learning mode
        learn_group = QGroupBox("Learning Mode")
        learn_layout = QGridLayout(learn_group)

        self.learning_bars = {}
        learning_params = ['growth_rate', 'pruning_rate', 'plasticity', 'reinforcement_rate', 'attention', 'stability']

        for i, param in enumerate(learning_params):
            label = QLabel(param.replace('_', ' ').title()[:12])
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #555;
                    border-radius: 3px;
                    background-color: #2a2a2a;
                    height: 14px;
                }
                QProgressBar::chunk {
                    background-color: #2196F3;
                }
            """)
            self.learning_bars[param] = bar
            learn_layout.addWidget(label, i // 2, (i % 2) * 2)
            learn_layout.addWidget(bar, i // 2, (i % 2) * 2 + 1)

        right_layout.addWidget(learn_group)
        right_layout.addStretch()

        # Add to main layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 400])

        layout.addWidget(splitter)

    def update_display(self):
        data = self.brain.get_dashboard_data()

        # Update chemicals
        chemicals = data['chemicals']
        receptors = data.get('receptors', {})
        for name, bar in self.chemical_bars.items():
            value = chemicals.get(name, 0)
            receptor = receptors.get(name, 1.0)
            bar.set_value(value, receptor)

        # Update mood
        mood = data['mood']
        mood_colors = {
            'fight_or_flight': '#F44336',
            'stressed': '#FF5722',
            'anxious': '#FF9800',
            'excited': '#FFEB3B',
            'happy': '#8BC34A',
            'connected': '#E91E63',
            'focused': '#9C27B0',
            'euphoric': '#00BCD4',
            'calm': '#4CAF50',
            'relaxed': '#2196F3',
            'depressed': '#607D8B',
            'curious': '#FF9800',
            'drowsy': '#9E9E9E',
            'neutral': '#BDBDBD',
        }
        color = mood_colors.get(mood, '#BDBDBD')
        self.mood_label.setText(f"Mood: {mood.upper()}")
        self.mood_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")

        # Update stats
        self.interactions_label.setText(f"Interactions: {data['interactions']}")
        self.training_label.setText(f"Training samples: {data['training_count']}")

        neurons = data.get('neurons', {})
        self.neurons_label.setText(f"Neurons: {neurons.get('hidden', neurons.get('total', 0))}")
        self.born_label.setText(f"Born: {neurons.get('born', 0)}")
        self.died_label.setText(f"Died: {neurons.get('died', 0)}")

        synapses = data.get('synapses', {})
        self.synapses_label.setText(f"Synapses: {synapses.get('total', 0)}")
        self.density_label.setText(f"Density: {synapses.get('density', 0):.4f}")

        memory = data.get('memory', {})
        self.memories_label.setText(f"Stored: {memory.get('total', 0)}")
        self.encoded_label.setText(f"Encoded: {memory.get('encoded', 0)}")
        self.forgotten_label.setText(f"Forgotten: {memory.get('forgotten', 0)}")
        self.strength_label.setText(f"Avg Strength: {memory.get('avg_strength', 0):.2f}")

        # Update learning bars
        learning = data['learning']
        for param, bar in self.learning_bars.items():
            value = learning.get(param, 0)
            bar.setValue(int(min(1.0, value) * 100))


class TrainingTab(QWidget):
    """Tab for training with OpenAI"""

    def __init__(self, brain: IntegratedBrain, config: Dict[str, Any] = None):
        super().__init__()
        self.brain = brain
        self.config = config if config is not None else {}
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # API Key section
        api_group = QGroupBox("OpenAI Configuration")
        api_layout = QHBoxLayout(api_group)

        api_layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("sk-...")

        # Try to get from config first, then environment
        saved_key = self.config.get('openai_api_key', '')
        env_key = os.environ.get('OPENAI_API_KEY', '')
        if saved_key:
            self.api_key_input.setText(saved_key)
        elif env_key:
            self.api_key_input.setText(env_key)
        
        # Save API key when changed
        self.api_key_input.textChanged.connect(self._save_api_key)

        api_layout.addWidget(self.api_key_input)
        layout.addWidget(api_group)

        # Training config
        config_group = QGroupBox("Training Configuration")
        config_layout = QGridLayout(config_group)

        config_layout.addWidget(QLabel("Topic:"), 0, 0)
        self.topic_input = QLineEdit()
        self.topic_input.setPlaceholderText("e.g., Python programming, cooking, science...")
        config_layout.addWidget(self.topic_input, 0, 1, 1, 2)

        config_layout.addWidget(QLabel("Examples:"), 1, 0)
        self.num_examples = QSpinBox()
        self.num_examples.setRange(1, 20)
        self.num_examples.setValue(5)
        config_layout.addWidget(self.num_examples, 1, 1)

        self.train_btn = QPushButton("Generate & Train")
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)
        self.train_btn.clicked.connect(self.start_training)
        config_layout.addWidget(self.train_btn, 1, 2)

        layout.addWidget(config_group)

        # Manual training
        manual_group = QGroupBox("Manual Training")
        manual_layout = QGridLayout(manual_group)

        manual_layout.addWidget(QLabel("Input:"), 0, 0)
        self.manual_input = QLineEdit()
        self.manual_input.setPlaceholderText("User message")
        manual_layout.addWidget(self.manual_input, 0, 1)

        manual_layout.addWidget(QLabel("Expected:"), 1, 0)
        self.manual_response = QLineEdit()
        self.manual_response.setPlaceholderText("Ideal response")
        manual_layout.addWidget(self.manual_response, 1, 1)

        self.manual_train_btn = QPushButton("Train")
        self.manual_train_btn.clicked.connect(self.manual_train)
        manual_layout.addWidget(self.manual_train_btn, 0, 2, 2, 1)

        layout.addWidget(manual_group)

        # Progress/Log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #444;
                font-family: monospace;
            }
        """)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

        # Save/Load buttons
        io_layout = QHBoxLayout()

        self.save_btn = QPushButton("Save Brain")
        self.save_btn.clicked.connect(self.save_brain)
        io_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load Brain")
        self.load_btn.clicked.connect(self.load_brain)
        io_layout.addWidget(self.load_btn)

        layout.addLayout(io_layout)

    def start_training(self):
        api_key = self.api_key_input.text().strip()
        topic = self.topic_input.text().strip()

        if not api_key:
            self.log("ERROR: Please enter your OpenAI API key")
            return

        if not topic:
            self.log("ERROR: Please enter a topic to train on")
            return

        self.train_btn.setEnabled(False)
        self.log(f"Starting training on topic: {topic}")

        self.worker = TrainingWorker(
            self.brain,
            api_key,
            topic,
            self.num_examples.value()
        )
        self.worker.progress.connect(self.log)
        self.worker.finished.connect(self.training_finished)
        self.worker.error.connect(self.training_error)
        self.worker.start()

    def training_finished(self, result):
        self.train_btn.setEnabled(True)
        self.log(f"✅ Training complete! Trained {result['examples_trained']} examples on '{result['topic']}'")
        self.log("")
        self.log("=" * 50)
        self.log("TRAINED CONVERSATIONS:")
        self.log("=" * 50)
        for i, r in enumerate(result['results'], 1):
            self.log(f"\n[Example {i}]")
            self.log(f"👤 User: {r['input']}")
            self.log(f"🤖 Brain learned: {r['expected']}")
            self.log(f"   Mood after: {r['mood']}")
        self.log("")
        self.log("=" * 50)

    def training_error(self, error):
        self.train_btn.setEnabled(True)
        self.log(f"ERROR: {error}")

    def _save_api_key(self, key: str):
        """Save API key to config when changed."""
        self.config['openai_api_key'] = key

    def manual_train(self):
        input_text = self.manual_input.text().strip()
        response = self.manual_response.text().strip()

        if not input_text or not response:
            self.log("ERROR: Please enter both input and expected response")
            return

        result = self.brain.train(input_text, response)
        self.log(f"Trained: '{input_text[:30]}...' -> '{response[:30]}...' (mood: {result['mood']})")

        self.manual_input.clear()
        self.manual_response.clear()

    def save_brain(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Brain", "", "Brain Files (*.brain);;All Files (*)"
        )
        if filepath:
            if not filepath.endswith('.brain'):
                filepath += '.brain'
            try:
                self.brain.save(filepath)
                self.log(f"Brain saved to {filepath}")
            except Exception as e:
                self.log(f"ERROR saving: {e}")

    def load_brain(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Brain", "", "Brain Files (*.brain);;All Files (*)"
        )
        if filepath:
            try:
                self.brain.load(filepath)
                self.log(f"Brain loaded from {filepath}")
            except Exception as e:
                self.log(f"ERROR loading: {e}")

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")


class ChemicalBrainDashboard(QMainWindow):
    """Main dashboard window - Now using IntegratedBrain with all mouse-level features"""
    
    CONFIG_FILE = Path.home() / ".chemical_brain_config.json"
    DEFAULT_BRAIN_FILE = Path.home() / ".chemical_brain_autosave.brain"

    def __init__(self):
        super().__init__()
        # Load config (API key, etc.)
        self.config = self._load_config()
        
        # Try to load existing brain, otherwise create new
        self.brain = self._load_or_create_brain()
        self.setup_ui()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load config from file."""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_config(self):
        """Save config to file."""
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Failed to save config: {e}")
    
    def _load_or_create_brain(self) -> IntegratedBrain:
        """Try to load existing brain from autosave, otherwise create new."""
        if self.DEFAULT_BRAIN_FILE.exists():
            try:
                from brain.persistence import BrainPersistence
                persistence = BrainPersistence(enable_auto_save=False)
                brain = persistence.load(str(self.DEFAULT_BRAIN_FILE))
                print(f"Loaded brain from {self.DEFAULT_BRAIN_FILE}")
                return brain
            except Exception as e:
                print(f"Could not load existing brain: {e}")
                print("Creating new brain...")
        
        return create_brain('small', use_gpu=False)
    
    def closeEvent(self, event):
        """Save brain and config on close."""
        try:
            # Save brain state
            from brain.persistence import BrainPersistence
            persistence = BrainPersistence(enable_auto_save=False)
            persistence.save(self.brain, str(self.DEFAULT_BRAIN_FILE))
            print(f"Brain auto-saved to {self.DEFAULT_BRAIN_FILE}")
            
            # Save config
            self._save_config()
        except Exception as e:
            print(f"Error during auto-save: {e}")
        
        event.accept()

    def setup_ui(self):
        self.setWindowTitle("Chemical Brain Dashboard")
        self.setMinimumSize(900, 700)

        # Dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
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
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #333;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QLineEdit, QSpinBox, QComboBox {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #888;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QTabBar::tab:hover {
                color: #fff;
            }
        """)

        # Central widget with tabs
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Header
        header = QLabel("Chemical Brain Dashboard")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50; padding: 10px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(ChatTab(self.brain), "Chat")
        tabs.addTab(StatusTab(self.brain), "Status")
        self.training_tab = TrainingTab(self.brain, self.config)
        tabs.addTab(self.training_tab, "Training")

        layout.addWidget(tabs)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Base, QColor(42, 42, 42))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Button, QColor(51, 51, 51))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(76, 175, 80))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = ChemicalBrainDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
