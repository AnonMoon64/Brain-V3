"""
Reservoir Computing Layer

Implements:
- Echo State Networks (ESN)
- Liquid State Machines (LSM)
- Fixed random recurrent reservoir
- Trainable readout only
- Spectral radius control for edge of chaos

Key insight: A large random recurrent network with fixed weights
can provide rich dynamics. Only the output layer needs training.
This is 1000x less compute than full backprop through time.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import math


class ReservoirType(Enum):
    """Types of reservoir"""
    ESN = "echo_state"      # Rate-based
    LSM = "liquid_state"    # Spiking
    HYBRID = "hybrid"        # Combination


@dataclass
class ReservoirConfig:
    """Configuration for reservoir computing"""
    n_input: int = 64
    n_reservoir: int = 1000
    n_output: int = 32
    
    # Connectivity
    input_density: float = 0.1
    reservoir_density: float = 0.1
    
    # Spectral radius (controls dynamics)
    spectral_radius: float = 0.95  # < 1 for echo state property
    
    # Input/output scaling
    input_scaling: float = 0.1
    output_scaling: float = 1.0
    
    # Leak rate (for leaky integrator neurons)
    leak_rate: float = 0.3
    
    # Noise
    noise_level: float = 0.001
    
    # Reservoir type
    reservoir_type: ReservoirType = ReservoirType.ESN


class EchoStateReservoir:
    """
    Echo State Network reservoir
    
    A large random recurrent network with:
    - Fixed input weights
    - Fixed recurrent weights (tuned for spectral radius)
    - Only output weights are trained
    
    The reservoir acts as a nonlinear expansion that creates
    rich dynamics from input sequences.
    """
    
    def __init__(self, config: ReservoirConfig = None):
        self.config = config or ReservoirConfig()
        
        self.n_input = self.config.n_input
        self.n_reservoir = self.config.n_reservoir
        self.n_output = self.config.n_output
        
        # Initialize weights
        self._init_weights()
        
        # Reservoir state
        self.state = np.zeros(self.n_reservoir)
        
        # State history (for training)
        self.state_history: List[np.ndarray] = []
        self.max_history = 10000
        
        # Output weights (trainable)
        self.W_out = np.zeros((self.n_output, self.n_reservoir + self.n_input))
        
        # Training data collection
        self.training_states: List[np.ndarray] = []
        self.training_targets: List[np.ndarray] = []
    
    def _init_weights(self) -> None:
        """Initialize fixed weights with proper spectral radius"""
        np.random.seed(42)  # Reproducible reservoir
        
        # Input weights (sparse, random)
        self.W_in = np.zeros((self.n_reservoir, self.n_input))
        n_input_conn = int(self.n_reservoir * self.n_input * self.config.input_density)
        
        for _ in range(n_input_conn):
            i = np.random.randint(self.n_reservoir)
            j = np.random.randint(self.n_input)
            self.W_in[i, j] = np.random.randn() * self.config.input_scaling
        
        # Recurrent weights (sparse, scaled for spectral radius)
        self.W_res = np.zeros((self.n_reservoir, self.n_reservoir))
        n_res_conn = int(self.n_reservoir * self.n_reservoir * self.config.reservoir_density)
        
        for _ in range(n_res_conn):
            i = np.random.randint(self.n_reservoir)
            j = np.random.randint(self.n_reservoir)
            self.W_res[i, j] = np.random.randn()
        
        # Scale to target spectral radius
        eigenvalues = np.linalg.eigvals(self.W_res)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            self.W_res *= self.config.spectral_radius / current_radius
        
        # Add small feedback weights (optional)
        self.W_fb = np.random.randn(self.n_reservoir, self.n_output) * 0.01
    
    def step(self, input_vec: np.ndarray, target: np.ndarray = None) -> np.ndarray:
        """
        Run one step of reservoir dynamics
        
        x(t+1) = (1-α)x(t) + α·tanh(W_in·u(t) + W_res·x(t) + W_fb·y(t))
        """
        # Ensure input is right size
        if len(input_vec) < self.n_input:
            padded = np.zeros(self.n_input)
            padded[:len(input_vec)] = input_vec
            input_vec = padded
        else:
            input_vec = input_vec[:self.n_input]
        
        # Compute new state
        pre_activation = (
            np.dot(self.W_in, input_vec) +
            np.dot(self.W_res, self.state)
        )
        
        # Add noise for regularization
        pre_activation += np.random.randn(self.n_reservoir) * self.config.noise_level
        
        # Leaky integration with tanh nonlinearity
        new_state = (
            (1 - self.config.leak_rate) * self.state +
            self.config.leak_rate * np.tanh(pre_activation)
        )
        
        self.state = new_state
        
        # Store history
        self.state_history.append(self.state.copy())
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        # Compute output
        extended_state = np.concatenate([self.state, input_vec])
        output = np.dot(self.W_out, extended_state) * self.config.output_scaling
        
        # Collect training data if target provided
        if target is not None:
            self.training_states.append(extended_state.copy())
            self.training_targets.append(target.copy())
        
        return output
    
    def train_readout(self, ridge_alpha: float = 1e-6) -> float:
        """
        Train output weights using ridge regression
        
        This is the only learning in reservoir computing.
        Uses collected state-target pairs.
        
        Returns: training error
        """
        if len(self.training_states) < 10:
            return float('inf')
        
        # Stack training data
        X = np.array(self.training_states)  # (n_samples, n_reservoir + n_input)
        Y = np.array(self.training_targets)  # (n_samples, n_output)
        
        # Ridge regression: W = Y^T X (X^T X + αI)^{-1}
        XtX = np.dot(X.T, X)
        XtX += ridge_alpha * np.eye(XtX.shape[0])
        XtY = np.dot(X.T, Y)
        
        try:
            self.W_out = np.linalg.solve(XtX, XtY).T
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            self.W_out = np.dot(np.linalg.pinv(X), Y).T
        
        # Compute training error
        predictions = np.dot(X, self.W_out.T)
        mse = np.mean((predictions - Y) ** 2)
        
        # Clear training data
        self.training_states = []
        self.training_targets = []
        
        return mse
    
    def reset_state(self) -> None:
        """Reset reservoir state"""
        self.state = np.zeros(self.n_reservoir)
        self.state_history = []
    
    def get_memory_capacity(self, test_length: int = 1000) -> float:
        """
        Estimate memory capacity of reservoir
        
        Memory capacity measures how well the reservoir can 
        reconstruct delayed input signals.
        """
        # Generate random test signal
        test_signal = np.random.randn(test_length, self.n_input)
        
        # Run through reservoir
        self.reset_state()
        states = []
        for t in range(test_length):
            self.step(test_signal[t])
            states.append(self.state.copy())
        
        states = np.array(states)
        
        # Compute memory capacity for different delays
        mc_total = 0.0
        max_delay = min(100, test_length // 2)
        
        for delay in range(1, max_delay):
            # Target is delayed input (just first component)
            target = test_signal[:-delay, 0]
            reservoir_states = states[delay:, :]
            
            # Linear regression
            try:
                weights = np.linalg.lstsq(reservoir_states, target, rcond=None)[0]
                prediction = np.dot(reservoir_states, weights)
                
                # Correlation coefficient squared
                corr = np.corrcoef(prediction, target)[0, 1] ** 2
                if not np.isnan(corr):
                    mc_total += corr
            except:
                pass
        
        return mc_total
    
    def get_stats(self) -> Dict:
        """Get reservoir statistics"""
        return {
            'n_reservoir': self.n_reservoir,
            'spectral_radius': self.config.spectral_radius,
            'leak_rate': self.config.leak_rate,
            'state_mean': float(np.mean(self.state)),
            'state_std': float(np.std(self.state)),
            'state_sparsity': float(np.mean(np.abs(self.state) < 0.1)),
            'n_training_samples': len(self.training_states),
        }


class LiquidStateReservoir:
    """
    Liquid State Machine (spiking reservoir)
    
    Uses spiking neurons instead of rate-based.
    Better temporal processing but more complex.
    """
    
    def __init__(self, config: ReservoirConfig = None):
        self.config = config or ReservoirConfig()
        self.config.reservoir_type = ReservoirType.LSM
        
        self.n_input = self.config.n_input
        self.n_reservoir = self.config.n_reservoir
        self.n_output = self.config.n_output
        
        # Neuron state
        self.membrane = np.zeros(self.n_reservoir)
        self.threshold = np.ones(self.n_reservoir) * 1.0
        self.tau = 20.0  # ms
        self.refractory = np.zeros(self.n_reservoir)
        self.refractory_period = 2.0  # ms
        
        # Spike history
        self.spike_trains: Dict[int, List[float]] = {i: [] for i in range(self.n_reservoir)}
        self.current_time = 0.0
        
        # Weights (same as ESN)
        self._init_weights()
        
        # Output (trained on spike counts/rates)
        self.W_out = np.zeros((self.n_output, self.n_reservoir))
        
        # Training
        self.training_responses: List[np.ndarray] = []
        self.training_targets: List[np.ndarray] = []
    
    def _init_weights(self) -> None:
        """Initialize weights with biological constraints"""
        # Input weights
        self.W_in = np.zeros((self.n_reservoir, self.n_input))
        n_conn = int(self.n_reservoir * self.n_input * self.config.input_density)
        
        for _ in range(n_conn):
            i = np.random.randint(self.n_reservoir)
            j = np.random.randint(self.n_input)
            # Dale's law: ~80% excitatory, ~20% inhibitory
            sign = 1 if np.random.random() < 0.8 else -1
            self.W_in[i, j] = sign * np.abs(np.random.randn()) * self.config.input_scaling
        
        # Recurrent weights with Dale's law
        # First 80% are excitatory, last 20% are inhibitory
        n_exc = int(self.n_reservoir * 0.8)
        
        self.W_res = np.zeros((self.n_reservoir, self.n_reservoir))
        n_conn = int(self.n_reservoir * self.n_reservoir * self.config.reservoir_density)
        
        for _ in range(n_conn):
            i = np.random.randint(self.n_reservoir)
            j = np.random.randint(self.n_reservoir)
            if i != j:
                # Sign determined by source neuron type
                sign = 1 if j < n_exc else -1
                self.W_res[i, j] = sign * np.abs(np.random.randn()) * 0.1
        
        # Scale recurrent weights
        self.W_res *= self.config.spectral_radius
    
    def step(self, input_vec: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Run one timestep of LSM"""
        self.current_time += dt
        
        # Pad input
        if len(input_vec) < self.n_input:
            padded = np.zeros(self.n_input)
            padded[:len(input_vec)] = input_vec
            input_vec = padded
        else:
            input_vec = input_vec[:self.n_input]
        
        # Update refractory
        self.refractory = np.maximum(0, self.refractory - dt)
        
        # Compute input current
        input_current = np.dot(self.W_in, input_vec)
        
        # Compute recurrent current from recent spikes
        recurrent_current = np.zeros(self.n_reservoir)
        for i in range(self.n_reservoir):
            recent_spikes = [t for t in self.spike_trains[i] 
                           if self.current_time - t < 20]
            if recent_spikes:
                # Exponential decay kernel
                for spike_time in recent_spikes:
                    delay = self.current_time - spike_time
                    kernel = np.exp(-delay / 5.0)  # 5ms decay
                    recurrent_current += self.W_res[:, i] * kernel
        
        # Total current
        total_current = input_current + recurrent_current
        total_current += np.random.randn(self.n_reservoir) * self.config.noise_level
        
        # LIF dynamics
        active = self.refractory <= 0
        dv = (-self.membrane + total_current) / self.tau * dt
        self.membrane = np.where(active, self.membrane + dv, self.membrane)
        
        # Check for spikes
        spikes = (self.membrane >= self.threshold) & active
        
        # Record spikes
        for i in np.where(spikes)[0]:
            self.spike_trains[i].append(self.current_time)
            # Trim old spikes
            self.spike_trains[i] = [t for t in self.spike_trains[i] 
                                   if self.current_time - t < 1000]
        
        # Reset spiking neurons
        self.membrane = np.where(spikes, 0.0, self.membrane)
        self.refractory = np.where(spikes, self.refractory_period, self.refractory)
        
        # Compute output (based on spike counts in window)
        spike_counts = np.array([
            len([t for t in self.spike_trains[i] if self.current_time - t < 50])
            for i in range(self.n_reservoir)
        ])
        
        # Normalize to firing rate
        response = spike_counts / 50.0 * 1000.0  # Convert to Hz
        
        # Linear readout
        output = np.dot(self.W_out, response) * self.config.output_scaling
        
        return output
    
    def collect_response(self, window: float = 100.0) -> np.ndarray:
        """Collect liquid response over window"""
        response = np.array([
            len([t for t in self.spike_trains[i] 
                 if self.current_time - t < window])
            for i in range(self.n_reservoir)
        ])
        return response / (window / 1000.0)  # Hz
    
    def train_readout(self, target: np.ndarray) -> None:
        """Collect training sample"""
        response = self.collect_response()
        self.training_responses.append(response)
        self.training_targets.append(target)
    
    def fit_readout(self, ridge_alpha: float = 1e-4) -> float:
        """Fit readout weights"""
        if len(self.training_responses) < 10:
            return float('inf')
        
        X = np.array(self.training_responses)
        Y = np.array(self.training_targets)
        
        # Ridge regression
        XtX = np.dot(X.T, X) + ridge_alpha * np.eye(X.shape[1])
        XtY = np.dot(X.T, Y)
        
        self.W_out = np.linalg.solve(XtX, XtY).T
        
        # Error
        pred = np.dot(X, self.W_out.T)
        mse = np.mean((pred - Y) ** 2)
        
        self.training_responses = []
        self.training_targets = []
        
        return mse
    
    def reset(self) -> None:
        """Reset liquid state"""
        self.membrane = np.zeros(self.n_reservoir)
        self.refractory = np.zeros(self.n_reservoir)
        self.spike_trains = {i: [] for i in range(self.n_reservoir)}
        self.current_time = 0.0


class HybridReservoir:
    """
    Hybrid reservoir combining ESN and LSM
    
    Uses rate-based dynamics for fast computation
    with spiking output for biological compatibility.
    """
    
    def __init__(
        self,
        input_dim: int = 100,
        reservoir_size: int = 1000,
        output_dim: int = 100,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        config: ReservoirConfig = None
    ):
        # Support both config object and direct parameters
        if config is not None:
            self.config = config
        else:
            self.config = ReservoirConfig(
                n_input=input_dim,
                n_reservoir=reservoir_size,
                n_output=output_dim,
                spectral_radius=spectral_radius,
                reservoir_density=sparsity  # Map sparsity to reservoir_density
            )
        
        # Two reservoirs: fast (ESN) and slow (LSM-like)
        self.fast_reservoir = EchoStateReservoir(ReservoirConfig(
            n_input=self.config.n_input,
            n_reservoir=self.config.n_reservoir // 2,
            n_output=self.config.n_reservoir // 4,
            spectral_radius=0.9,
            leak_rate=0.5,  # Fast
        ))
        
        self.slow_reservoir = EchoStateReservoir(ReservoirConfig(
            n_input=self.config.n_input + self.config.n_reservoir // 4,
            n_reservoir=self.config.n_reservoir // 2,
            n_output=self.config.n_output,
            spectral_radius=0.99,
            leak_rate=0.1,  # Slow
        ))
        
        # Combined output
        self.n_output = self.config.n_output
    
    def step(self, input_vec: np.ndarray, target: np.ndarray = None) -> np.ndarray:
        """Process through both reservoirs"""
        # Fast reservoir processes raw input
        fast_output = self.fast_reservoir.step(input_vec)
        
        # Slow reservoir gets input + fast output
        combined_input = np.concatenate([input_vec, fast_output])
        slow_output = self.slow_reservoir.step(combined_input, target)
        
        return slow_output
    
    def process(self, input_vec: np.ndarray) -> np.ndarray:
        """Process input through reservoir (alias for step)."""
        return self.step(input_vec)
    
    def train(self, ridge_alpha: float = 1e-6) -> Tuple[float, float]:
        """Train both readouts"""
        # Train slow reservoir (has targets)
        slow_error = self.slow_reservoir.train_readout(ridge_alpha)
        
        # Fast reservoir is self-supervised (predict next input)
        fast_error = self.fast_reservoir.train_readout(ridge_alpha)
        
        return fast_error, slow_error
    
    def reset(self) -> None:
        """Reset both reservoirs"""
        self.fast_reservoir.reset_state()
        self.slow_reservoir.reset_state()
    
    def get_state(self) -> Dict:
        """Get reservoir state for serialization"""
        return {
            'fast_state': self.fast_reservoir.state.tolist() if hasattr(self.fast_reservoir, 'state') else [],
            'slow_state': self.slow_reservoir.state.tolist() if hasattr(self.slow_reservoir, 'state') else [],
            'stats': self.get_stats()
        }
    
    def get_stats(self) -> Dict:
        """Get combined statistics"""
        return {
            'fast': self.fast_reservoir.get_stats(),
            'slow': self.slow_reservoir.get_stats(),
            'total_reservoir_size': (
                self.fast_reservoir.n_reservoir + 
                self.slow_reservoir.n_reservoir
            ),
        }


def create_reservoir(
    reservoir_type: ReservoirType,
    config: ReservoirConfig = None
):
    """Factory function to create appropriate reservoir"""
    config = config or ReservoirConfig()
    config.reservoir_type = reservoir_type
    
    if reservoir_type == ReservoirType.ESN:
        return EchoStateReservoir(config)
    elif reservoir_type == ReservoirType.LSM:
        return LiquidStateReservoir(config)
    elif reservoir_type == ReservoirType.HYBRID:
        return HybridReservoir(config)
    else:
        return EchoStateReservoir(config)
