from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath, QPalette
from PyQt6.QtCore import Qt, QRect

class StatsGraphWidget(QWidget):
    """
    Real-time graph widget for simulation statistics.
    Tracks Population, Food, and Average Energy.
    """
    
    def __init__(self, parent=None, max_points=200):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.max_points = max_points
        
        # Data buffers
        self.pop_history = []
        self.food_history = []
        self.energy_history = []
        
        # Colors
        self.pop_color = QColor(100, 200, 255)  # Blue
        self.food_color = QColor(100, 255, 100) # Green
        self.energy_color = QColor(255, 200, 100) # Orange
        
        self.setBackgroundRole(QPalette.ColorRole.NoRole)
    
    def add_data_point(self, pop: int, food: int, avg_energy: float):
        """Add a new frame of data."""
        self.pop_history.append(pop)
        self.food_history.append(food)
        self.energy_history.append(avg_energy * 10) # Scale energy for visibility (0-1 -> 0-10)
        
        # Trim
        if len(self.pop_history) > self.max_points:
            self.pop_history.pop(0)
            self.food_history.pop(0)
            self.energy_history.pop(0)
            
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(40, 44, 52))
        
        if not self.pop_history:
            return
            
        # Layout
        margin = 10
        w = self.width() - margin * 2
        h = self.height() - margin * 2
        
        x0 = margin
        y0 = self.height() - margin
        
        # Determine strict range
        # Max value across all datasets (min 10)
        max_val = max(10, max(self.pop_history + self.food_history + self.energy_history))
        
        # Draw Grids
        painter.setPen(QPen(QColor(60, 60, 60), 1, Qt.PenStyle.DashLine))
        painter.drawLine(x0, y0, x0 + w, y0) # Bottom
        painter.drawLine(x0, margin, x0 + w, margin) # Top
        
        # Scale factors
        x_step = w / (self.max_points - 1)
        y_scale = h / max_val
        
        # Draw Paths
        self._draw_line(painter, self.pop_history, self.pop_color, x0, y0, x_step, y_scale)
        self._draw_line(painter, self.food_history, self.food_color, x0, y0, x_step, y_scale)
        self._draw_line(painter, self.energy_history, self.energy_color, x0, y0, x_step, y_scale)
        
        # Draw Legend
        self._draw_legend(painter)

    def _draw_line(self, painter, data, color, x0, y0, x_step, y_scale):
        if len(data) < 2:
            return
            
        path = QPainterPath()
        path.moveTo(x0, y0 - data[0] * y_scale)
        
        for i, val in enumerate(data[1:], 1):
            path.lineTo(x0 + i * x_step, y0 - val * y_scale)
            
        painter.setPen(QPen(color, 2))
        painter.drawPath(path)
        
    def _draw_legend(self, painter):
        painter.setFont(QFont("Segoe UI", 8))
        
        # Pop
        painter.setPen(self.pop_color)
        painter.drawText(10, 15, "Population")
        
        # Food
        painter.setPen(self.food_color)
        painter.drawText(80, 15, "Food")
        
        # Energy
        painter.setPen(self.energy_color)
        painter.drawText(130, 15, "Energy (x10)")
