"""
Hazard Pathfinding Awareness (UPGRADE 5)

Implements intelligent navigation that avoids dangers:
- Heatmap-based navigation
- Hazard cost fields
- Gradient descent toward safe/rewarding areas
- Creatures stop jumping into fire unless DNA says they're deranged
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class HazardCostField:
    """
    2D cost field for navigation.
    
    Higher values = more dangerous/costly.
    Creatures navigate via gradient descent toward low-cost areas.
    """
    width: int
    height: int
    tile_size: int
    
    # Cost grid (higher = avoid)
    cost_grid: np.ndarray = None
    
    # Reward grid (higher = approach)
    reward_grid: np.ndarray = None
    
    # Combined potential field
    potential_field: np.ndarray = None
    
    # Decay rates
    cost_decay: float = 0.95
    reward_decay: float = 0.98
    
    def __post_init__(self):
        grid_w = self.width // self.tile_size
        grid_h = self.height // self.tile_size
        
        if self.cost_grid is None:
            self.cost_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        if self.reward_grid is None:
            self.reward_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        if self.potential_field is None:
            self.potential_field = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    def add_hazard(self, x: float, y: float, intensity: float = 1.0, radius: float = 100.0):
        """Add hazard influence to cost field."""
        grid_x = int(x / self.tile_size)
        grid_y = int(y / self.tile_size)
        grid_radius = int(radius / self.tile_size)
        
        grid_h, grid_w = self.cost_grid.shape
        
        # Add Gaussian hazard influence
        for dy in range(-grid_radius, grid_radius + 1):
            for dx in range(-grid_radius, grid_radius + 1):
                gx = grid_x + dx
                gy = grid_y + dy
                
                if 0 <= gx < grid_w and 0 <= gy < grid_h:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= grid_radius:
                        # Exponential decay
                        cost = intensity * np.exp(-dist**2 / (2 * (grid_radius / 2)**2))
                        self.cost_grid[gy, gx] = max(self.cost_grid[gy, gx], cost)
    
    def add_reward(self, x: float, y: float, value: float = 0.5, radius: float = 80.0):
        """Add reward (food/water) to reward field."""
        grid_x = int(x / self.tile_size)
        grid_y = int(y / self.tile_size)
        grid_radius = int(radius / self.tile_size)
        
        grid_h, grid_w = self.reward_grid.shape
        
        for dy in range(-grid_radius, grid_radius + 1):
            for dx in range(-grid_radius, grid_radius + 1):
                gx = grid_x + dx
                gy = grid_y + dy
                
                if 0 <= gx < grid_w and 0 <= gy < grid_h:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= grid_radius:
                        reward = value * np.exp(-dist**2 / (2 * (grid_radius / 3)**2))
                        self.reward_grid[gy, gx] = max(self.reward_grid[gy, gx], reward)
    
    def update_potential_field(self):
        """Combine cost and reward into navigation potential."""
        # Potential = Reward - Cost
        # Navigate toward high potential (high reward, low cost)
        self.potential_field = self.reward_grid - self.cost_grid * 2.0  # Cost weighted higher
    
    def get_gradient_at(self, x: float, y: float) -> Tuple[float, float]:
        """
        Get gradient direction at position.
        
        Returns (dx, dy) normalized direction toward better areas.
        Positive values = move in that direction to improve situation.
        """
        grid_x = int(x / self.tile_size)
        grid_y = int(y / self.tile_size)
        
        grid_h, grid_w = self.potential_field.shape
        
        # Clamp to valid range
        grid_x = max(1, min(grid_w - 2, grid_x))
        grid_y = max(1, min(grid_h - 2, grid_y))
        
        # Compute gradient via finite differences
        dx = self.potential_field[grid_y, grid_x + 1] - self.potential_field[grid_y, grid_x - 1]
        dy = self.potential_field[grid_y + 1, grid_x] - self.potential_field[grid_y - 1, grid_x]
        
        # Normalize
        mag = np.sqrt(dx**2 + dy**2)
        if mag > 0.001:
            dx /= mag
            dy /= mag
        
        return (dx, dy)
    
    def get_danger_level_at(self, x: float, y: float) -> float:
        """Get danger level at position (0 = safe, 1 = deadly)."""
        grid_x = int(x / self.tile_size)
        grid_y = int(y / self.tile_size)
        
        grid_h, grid_w = self.cost_grid.shape
        
        if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
            return min(1.0, self.cost_grid[grid_y, grid_x])
        return 0.0
    
    def decay(self, dt: float):
        """Decay cost and reward over time (memories fade)."""
        self.cost_grid *= self.cost_decay
        self.reward_grid *= self.reward_decay
        
        # Zero out very small values
        self.cost_grid[self.cost_grid < 0.01] = 0
        self.reward_grid[self.reward_grid < 0.01] = 0


class IntelligentNavigator:
    """
    Intelligent navigation system using hazard awareness.
    
    Creatures with this navigator:
    - Build mental maps of dangerous areas
    - Route around hazards intelligently
    - Balance risk vs reward based on desperation
    """
    
    def __init__(self, creature_id: str, world_width: int, world_height: int, tile_size: int = 32):
        self.creature_id = creature_id
        self.hazard_field = HazardCostField(world_width, world_height, tile_size)
        
        # Risk tolerance (modified by hunger/desperation)
        self.base_risk_tolerance = 0.3  # 0 = coward, 1 = reckless
        self.current_risk_tolerance = 0.3
        
        # Learning: remember where we've been hurt
        self.pain_memory: List[Tuple[float, float, float]] = []  # (x, y, pain_level)
        self.max_pain_memories = 50
    
    def observe_hazard(self, x: float, y: float, intensity: float = 1.0):
        """Creature observes a hazard."""
        self.hazard_field.add_hazard(x, y, intensity, radius=120)
    
    def observe_reward(self, x: float, y: float, value: float = 0.5):
        """Creature observes food/water."""
        self.hazard_field.add_reward(x, y, value, radius=80)
    
    def remember_pain(self, x: float, y: float, pain: float):
        """Remember location where pain was experienced."""
        self.pain_memory.append((x, y, pain))
        if len(self.pain_memory) > self.max_pain_memories:
            self.pain_memory.pop(0)
        
        # Add to hazard field
        self.hazard_field.add_hazard(x, y, intensity=pain, radius=80)
    
    def update_risk_tolerance(self, hunger: float, thirst: float, bravery: float = 0.5):
        """
        Adjust risk tolerance based on desperation.
        
        Starving creatures take more risks.
        """
        desperation = max(hunger, thirst)
        
        if desperation > 0.8:
            # DESPERATE: Will walk through fire for food
            self.current_risk_tolerance = 0.9
        elif desperation > 0.6:
            # HUNGRY: Higher risk tolerance
            self.current_risk_tolerance = 0.6 + bravery * 0.3
        else:
            # SAFE: Use base + bravery
            self.current_risk_tolerance = self.base_risk_tolerance + bravery * 0.4
    
    def get_safe_direction(self, x: float, y: float, 
                          target_x: float, target_y: float) -> Tuple[float, float]:
        """
        Get direction to move that balances reaching target with avoiding danger.
        
        Returns (dx, dy) normalized direction.
        """
        # Update potential field
        self.hazard_field.update_potential_field()
        
        # Get gradient (direction of improvement)
        grad_x, grad_y = self.hazard_field.get_gradient_at(x, y)
        
        # Get direct direction to target
        target_dx = target_x - x
        target_dy = target_y - y
        target_dist = np.sqrt(target_dx**2 + target_dy**2)
        
        if target_dist > 0.1:
            target_dx /= target_dist
            target_dy /= target_dist
        else:
            return (0.0, 0.0)
        
        # Get danger at current location
        danger = self.hazard_field.get_danger_level_at(x, y)
        
        # Blend gradient and target based on danger and risk tolerance
        if danger > self.current_risk_tolerance:
            # High danger - prioritize safety (gradient)
            blend = 0.8  # 80% gradient, 20% target
        elif danger > self.current_risk_tolerance * 0.5:
            # Moderate danger - balance
            blend = 0.5
        else:
            # Low danger - prioritize target
            blend = 0.2
        
        # Blend directions
        dx = grad_x * blend + target_dx * (1 - blend)
        dy = grad_y * blend + target_dy * (1 - blend)
        
        # Normalize
        mag = np.sqrt(dx**2 + dy**2)
        if mag > 0.001:
            dx /= mag
            dy /= mag
        
        return (dx, dy)
    
    def should_avoid_area(self, x: float, y: float) -> bool:
        """Check if creature should avoid this area."""
        danger = self.hazard_field.get_danger_level_at(x, y)
        return danger > self.current_risk_tolerance
    
    def update(self, dt: float):
        """Update navigator (decay memories)."""
        self.hazard_field.decay(dt)
