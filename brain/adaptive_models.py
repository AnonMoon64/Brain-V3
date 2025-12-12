"""
Adaptive Internal Models (UPGRADE 6)

Creatures learn predictive models of their world:
- Expected hunger increase per minute
- Expected travel cost
- Expected danger from actions

Animals that mispredict die. Animals that get it right thrive.
This is the foundation of planning and foresight.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class HungerModel:
    """Learnable model of hunger dynamics."""
    # Predicted hunger increase per second at rest
    base_rate: float = 0.01
    
    # Predicted hunger increase per distance traveled
    movement_cost: float = 0.0001
    
    # Predicted time until critical hunger (seconds)
    time_to_starvation: float = 100.0
    
    # Learning rate
    learning_rate: float = 0.01
    
    # Prediction error accumulator
    recent_errors: List[float] = field(default_factory=list)
    max_error_memory: int = 20
    
    def predict_hunger_at_time(self, current_hunger: float, dt: float, 
                              distance_traveled: float = 0.0) -> float:
        """Predict future hunger level."""
        predicted = current_hunger
        predicted += self.base_rate * dt
        predicted += self.movement_cost * distance_traveled
        return min(1.0, predicted)
    
    def update_model(self, actual_hunger: float, predicted_hunger: float):
        """Update model based on prediction error."""
        error = actual_hunger - predicted_hunger
        
        # Store error
        self.recent_errors.append(abs(error))
        if len(self.recent_errors) > self.max_error_memory:
            self.recent_errors.pop(0)
        
        # Adjust base rate based on error
        if error > 0.01:
            # Underestimated hunger increase
            self.base_rate += self.learning_rate * error
        elif error < -0.01:
            # Overestimated hunger increase
            self.base_rate -= self.learning_rate * abs(error)
        
        # Clamp
        self.base_rate = max(0.001, min(0.05, self.base_rate))
    
    def get_prediction_accuracy(self) -> float:
        """Get model accuracy (0 = terrible, 1 = perfect)."""
        if not self.recent_errors:
            return 0.5
        avg_error = np.mean(self.recent_errors)
        return max(0.0, 1.0 - avg_error * 5)  # Scale to 0-1


@dataclass
class TravelCostModel:
    """Learnable model of movement energy costs."""
    # Energy cost per pixel traveled
    cost_per_distance: float = 0.0002
    
    # Energy cost per jump
    cost_per_jump: float = 0.02
    
    # Speed-dependent multiplier (running costs more)
    speed_multiplier: float = 1.5
    
    # Learning rate
    learning_rate: float = 0.01
    
    # Prediction history
    recent_errors: List[float] = field(default_factory=list)
    max_error_memory: int = 20
    
    def predict_energy_cost(self, distance: float, n_jumps: int = 0, 
                           speed_factor: float = 1.0) -> float:
        """Predict energy cost of movement."""
        cost = distance * self.cost_per_distance * (speed_factor * self.speed_multiplier)
        cost += n_jumps * self.cost_per_jump
        return cost
    
    def update_model(self, actual_cost: float, predicted_cost: float):
        """Update model based on observed energy changes."""
        error = actual_cost - predicted_cost
        
        self.recent_errors.append(abs(error))
        if len(self.recent_errors) > self.max_error_memory:
            self.recent_errors.pop(0)
        
        # Adjust cost per distance
        if error > 0.005:
            self.cost_per_distance += self.learning_rate * error * 0.01
        elif error < -0.005:
            self.cost_per_distance -= self.learning_rate * abs(error) * 0.01
        
        # Clamp
        self.cost_per_distance = max(0.0001, min(0.001, self.cost_per_distance))
    
    def get_prediction_accuracy(self) -> float:
        """Get model accuracy."""
        if not self.recent_errors:
            return 0.5
        avg_error = np.mean(self.recent_errors)
        return max(0.0, 1.0 - avg_error * 10)


@dataclass
class DangerModel:
    """Learnable model of environmental dangers."""
    # Probability of taking damage in next N seconds
    danger_probability: float = 0.1
    
    # Expected damage magnitude if hit
    expected_damage: float = 0.1
    
    # Learning from experience
    times_damaged: int = 0
    times_safe: int = 0
    
    # Recent danger experiences
    recent_dangers: List[Tuple[float, float]] = field(default_factory=list)  # (danger_level, actual_damage)
    max_danger_memory: int = 30
    
    def predict_danger(self, visible_hazards: int = 0, 
                      hazard_distance: float = 1000.0) -> float:
        """
        Predict danger level (0-1) in current situation.
        
        Returns expected damage in next few seconds.
        """
        if visible_hazards == 0:
            return self.danger_probability * self.expected_damage * 0.1
        
        # Danger increases with proximity
        proximity_factor = max(0.0, 1.0 - hazard_distance / 200)
        
        predicted_danger = (
            self.danger_probability * 
            self.expected_damage * 
            (1 + proximity_factor * 3)
        )
        
        return min(1.0, predicted_danger)
    
    def observe_danger_outcome(self, predicted_danger: float, actual_damage: float):
        """Update model after experiencing (or not) danger."""
        self.recent_dangers.append((predicted_danger, actual_damage))
        if len(self.recent_dangers) > self.max_danger_memory:
            self.recent_dangers.pop(0)
        
        if actual_damage > 0.01:
            self.times_damaged += 1
            # Increase expected damage if we underestimated
            if actual_damage > self.expected_damage:
                self.expected_damage = (
                    self.expected_damage * 0.9 + 
                    actual_damage * 0.1
                )
        else:
            self.times_safe += 1
        
        # Update probability
        total = self.times_damaged + self.times_safe
        if total > 0:
            self.danger_probability = self.times_damaged / total
        
        # Clamp
        self.expected_damage = max(0.01, min(0.5, self.expected_damage))
        self.danger_probability = max(0.01, min(0.9, self.danger_probability))
    
    def get_prediction_accuracy(self) -> float:
        """How well does this model predict danger?"""
        if len(self.recent_dangers) < 5:
            return 0.5
        
        errors = []
        for predicted, actual in self.recent_dangers:
            error = abs(predicted - actual)
            errors.append(error)
        
        avg_error = np.mean(errors)
        return max(0.0, 1.0 - avg_error * 2)


class AdaptiveInternalModels:
    """
    Complete internal model system for a creature.
    
    Tracks:
    - Hunger dynamics
    - Travel costs
    - Danger predictions
    
    Creatures use these models for:
    - Planning (should I go for that food?)
    - Risk assessment (is it worth crossing the hazard?)
    - Resource management (do I have enough energy?)
    """
    
    def __init__(self, creature_id: str):
        self.creature_id = creature_id
        
        # Sub-models
        self.hunger_model = HungerModel()
        self.travel_model = TravelCostModel()
        self.danger_model = DangerModel()
        
        # Tracking for updates
        self.last_hunger = 0.5
        self.last_energy = 1.0
        self.last_position = (0.0, 0.0)
        self.last_update_time = 0.0
        
        # Prediction performance
        self.prediction_accuracy = 0.5
    
    def predict_action_outcome(self, 
                              current_hunger: float,
                              current_energy: float,
                              action: str,
                              action_params: Dict) -> Dict[str, float]:
        """
        Predict outcome of taking an action.
        
        Returns dict with:
        - 'hunger_after': Expected hunger level
        - 'energy_after': Expected energy level  
        - 'danger': Expected damage
        - 'success_prob': Probability of achieving goal
        """
        dt = action_params.get('time', 1.0)
        distance = action_params.get('distance', 0.0)
        
        # Predict hunger
        predicted_hunger = self.hunger_model.predict_hunger_at_time(
            current_hunger, dt, distance
        )
        
        # Predict energy cost
        energy_cost = self.travel_model.predict_energy_cost(
            distance,
            n_jumps=action_params.get('jumps', 0),
            speed_factor=action_params.get('speed', 1.0)
        )
        predicted_energy = current_energy - energy_cost
        
        # Predict danger
        danger = self.danger_model.predict_danger(
            visible_hazards=action_params.get('hazards', 0),
            hazard_distance=action_params.get('hazard_dist', 1000)
        )
        
        return {
            'hunger_after': predicted_hunger,
            'energy_after': predicted_energy,
            'danger': danger,
            'success_prob': max(0.1, 1.0 - danger - (predicted_hunger > 0.9) * 0.5)
        }
    
    def update_from_observation(self,
                               current_hunger: float,
                               current_energy: float,
                               current_position: Tuple[float, float],
                               damage_taken: float,
                               current_time: float):
        """Update models based on actual observations."""
        dt = current_time - self.last_update_time
        if dt < 0.1:
            return  # Too soon
        
        # Update hunger model
        predicted_hunger = self.hunger_model.predict_hunger_at_time(
            self.last_hunger, dt
        )
        self.hunger_model.update_model(current_hunger, predicted_hunger)
        
        # Update travel model
        distance = np.sqrt(
            (current_position[0] - self.last_position[0])**2 +
            (current_position[1] - self.last_position[1])**2
        )
        predicted_cost = self.travel_model.predict_energy_cost(distance)
        actual_cost = self.last_energy - current_energy
        self.travel_model.update_model(actual_cost, predicted_cost)
        
        # Update danger model
        predicted_danger = self.danger_model.predict_danger()
        self.danger_model.observe_danger_outcome(predicted_danger, damage_taken)
        
        # Update tracking
        self.last_hunger = current_hunger
        self.last_energy = current_energy
        self.last_position = current_position
        self.last_update_time = current_time
        
        # Calculate overall accuracy
        self.prediction_accuracy = (
            self.hunger_model.get_prediction_accuracy() * 0.4 +
            self.travel_model.get_prediction_accuracy() * 0.3 +
            self.danger_model.get_prediction_accuracy() * 0.3
        )
    
    def should_attempt_action(self, outcome: Dict[str, float], 
                             desperation: float = 0.5) -> bool:
        """
        Decide if action is worth attempting based on predicted outcome.
        
        Creatures with better models make better decisions.
        """
        # Check if we'll survive
        if outcome['energy_after'] < 0.1:
            # Will run out of energy
            if desperation < 0.8:
                return False
        
        if outcome['hunger_after'] > 0.95:
            # Will starve
            return True  # Desperate
        
        # Check danger vs reward
        if outcome['danger'] > 0.5 and desperation < 0.7:
            return False
        
        # Check success probability
        if outcome['success_prob'] < 0.3 and desperation < 0.5:
            return False
        
        return True
    
    def get_model_quality(self) -> float:
        """Overall model prediction accuracy (0-1)."""
        return self.prediction_accuracy
