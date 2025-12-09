"""
Robust Signal Processing and Noise Handling

Implements:
- Input normalization (layer normalization, batch normalization concepts)
- Noise injection for regularization and exploration
- Long-tailed data handling (adaptive binning, rank normalization)
- Signal-to-noise ratio optimization
- Adaptive gain control
- Outlier detection and handling

Key insight: Biological neural systems have evolved sophisticated
mechanisms for handling noisy, non-stationary inputs with heavy tails.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import math
from collections import deque


class NormalizationType(Enum):
    """Types of normalization"""
    NONE = "none"
    MINMAX = "minmax"
    ZSCORE = "zscore"
    LAYER = "layer"
    RANK = "rank"  # For long-tailed data
    ADAPTIVE = "adaptive"


@dataclass
class RunningStatistics:
    """
    Efficient online computation of mean and variance
    Using Welford's algorithm for numerical stability
    """
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences
    min_val: float = float('inf')
    max_val: float = float('-inf')
    
    # Exponential moving average (for non-stationary data)
    ema_mean: float = 0.0
    ema_var: float = 1.0
    ema_alpha: float = 0.01
    
    def update(self, value: float) -> None:
        """Update statistics with new value"""
        self.count += 1
        
        # Welford's online algorithm
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        
        # Track range
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        # EMA update (more weight on recent values)
        self.ema_mean = (1 - self.ema_alpha) * self.ema_mean + self.ema_alpha * value
        diff_sq = (value - self.ema_mean) ** 2
        self.ema_var = (1 - self.ema_alpha) * self.ema_var + self.ema_alpha * diff_sq
    
    def update_batch(self, values: np.ndarray) -> None:
        """Update with batch of values"""
        for v in values.flatten():
            self.update(v)
    
    @property
    def variance(self) -> float:
        if self.count < 2:
            return 1.0
        return self.m2 / (self.count - 1)
    
    @property
    def std(self) -> float:
        return math.sqrt(max(1e-8, self.variance))
    
    @property
    def ema_std(self) -> float:
        return math.sqrt(max(1e-8, self.ema_var))


class InputNormalizer:
    """
    Adaptive input normalization
    
    Handles:
    - Online mean/variance estimation
    - Multiple normalization strategies
    - Non-stationary input distributions
    - Long-tailed distributions
    """
    
    def __init__(
        self,
        dim: int,
        norm_type: NormalizationType = NormalizationType.ADAPTIVE,
        eps: float = 1e-8
    ):
        self.dim = dim
        self.norm_type = norm_type
        self.eps = eps
        
        # Per-dimension statistics
        self.stats = [RunningStatistics() for _ in range(dim)]
        
        # Global statistics
        self.global_stats = RunningStatistics()
        
        # Adaptive parameters
        self.adaptive_scale = np.ones(dim)
        self.adaptive_shift = np.zeros(dim)
        
        # History for rank normalization
        self.value_history: List[np.ndarray] = []
        self.max_history = 1000
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input according to strategy"""
        x = np.atleast_1d(x).astype(float)
        
        # Update statistics
        self._update_stats(x)
        
        if self.norm_type == NormalizationType.NONE:
            return x
        
        if self.norm_type == NormalizationType.MINMAX:
            return self._minmax_normalize(x)
        
        if self.norm_type == NormalizationType.ZSCORE:
            return self._zscore_normalize(x)
        
        if self.norm_type == NormalizationType.LAYER:
            return self._layer_normalize(x)
        
        if self.norm_type == NormalizationType.RANK:
            return self._rank_normalize(x)
        
        if self.norm_type == NormalizationType.ADAPTIVE:
            return self._adaptive_normalize(x)
        
        return x
    
    def _update_stats(self, x: np.ndarray) -> None:
        """Update running statistics"""
        for i, val in enumerate(x[:self.dim]):
            self.stats[i].update(val)
        self.global_stats.update_batch(x)
        
        # Store for rank normalization
        self.value_history.append(x.copy())
        if len(self.value_history) > self.max_history:
            self.value_history.pop(0)
    
    def _minmax_normalize(self, x: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]"""
        result = np.zeros_like(x)
        for i in range(min(len(x), self.dim)):
            range_val = self.stats[i].max_val - self.stats[i].min_val
            if range_val > self.eps:
                result[i] = (x[i] - self.stats[i].min_val) / range_val
            else:
                result[i] = 0.5
        return result
    
    def _zscore_normalize(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalization using EMA statistics"""
        result = np.zeros_like(x)
        for i in range(min(len(x), self.dim)):
            std = self.stats[i].ema_std
            if std > self.eps:
                result[i] = (x[i] - self.stats[i].ema_mean) / std
            else:
                result[i] = 0.0
        return result
    
    def _layer_normalize(self, x: np.ndarray) -> np.ndarray:
        """Layer normalization (normalize across features)"""
        mean = np.mean(x)
        std = np.std(x) + self.eps
        return (x - mean) / std
    
    def _rank_normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Rank normalization for long-tailed distributions
        Maps values to their percentile rank
        """
        if len(self.value_history) < 10:
            return self._zscore_normalize(x)
        
        # Build empirical CDF from history
        all_values = np.concatenate(self.value_history, axis=0)
        
        result = np.zeros_like(x)
        for i in range(len(x)):
            # Compute percentile rank
            rank = np.sum(all_values <= x[i]) / len(all_values)
            # Transform to roughly Gaussian
            result[i] = np.clip(rank * 2 - 1, -1, 1)
        
        return result
    
    def _adaptive_normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Adaptive normalization that handles:
        - Non-stationarity (uses EMA)
        - Outliers (soft clipping)
        - Long tails (combines z-score and rank)
        """
        # First, z-score normalize
        z_normalized = self._zscore_normalize(x)
        
        # Soft clip outliers (beyond 3 sigma)
        clipped = np.tanh(z_normalized / 3.0) * 3.0
        
        # Apply learned scale/shift
        result = clipped * self.adaptive_scale[:len(x)] + self.adaptive_shift[:len(x)]
        
        return result
    
    def adapt(self, target_mean: float = 0.0, target_std: float = 1.0) -> None:
        """Adapt normalization parameters toward target distribution"""
        for i in range(self.dim):
            current_mean = self.stats[i].ema_mean
            current_std = self.stats[i].ema_std
            
            if current_std > self.eps:
                self.adaptive_scale[i] = target_std / current_std
            self.adaptive_shift[i] = target_mean - current_mean * self.adaptive_scale[i]


class NoiseGenerator:
    """
    Biologically-inspired noise generation
    
    Types of noise:
    - White noise (thermal)
    - Pink noise (1/f, common in biology)
    - Poisson noise (spike-like)
    - Ornstein-Uhlenbeck (correlated noise)
    """
    
    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)
        
        # OU process state
        self.ou_state: Dict[str, float] = {}
        self.ou_theta = 0.1  # Mean reversion rate
        self.ou_sigma = 0.3  # Volatility
    
    def white_noise(self, shape: Tuple, scale: float = 1.0) -> np.ndarray:
        """Gaussian white noise"""
        return self.rng.randn(*shape) * scale
    
    def pink_noise(self, n_samples: int, scale: float = 1.0) -> np.ndarray:
        """
        1/f (pink) noise - ubiquitous in biological systems
        Using Voss-McCartney algorithm
        """
        n_rows = 16  # Number of random sources
        
        # Initialize random sources
        array = self.rng.randn(n_rows, n_samples)
        
        # Create pink noise by summing octaves
        pink = np.zeros(n_samples)
        for i in range(n_rows):
            # Each row updates at different rate (power of 2)
            step = 2 ** i
            held = np.repeat(array[i, ::step], step)[:n_samples]
            pink += held
        
        # Normalize
        pink = (pink - np.mean(pink)) / (np.std(pink) + 1e-8)
        return pink * scale
    
    def poisson_noise(self, rate: float, n_samples: int) -> np.ndarray:
        """Poisson process (spike train)"""
        return self.rng.poisson(rate, n_samples).astype(float)
    
    def ornstein_uhlenbeck(
        self, 
        key: str, 
        n_samples: int = 1,
        mu: float = 0.0,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Ornstein-Uhlenbeck process - correlated noise
        Good for modeling slowly-varying neuromodulatory fluctuations
        """
        if key not in self.ou_state:
            self.ou_state[key] = mu
        
        samples = []
        x = self.ou_state[key]
        
        for _ in range(n_samples):
            dx = self.ou_theta * (mu - x) * dt + self.ou_sigma * math.sqrt(dt) * self.rng.randn()
            x += dx
            samples.append(x)
        
        self.ou_state[key] = x
        return np.array(samples)
    
    def neural_noise(
        self,
        shape: Tuple,
        activity_level: float = 0.5,
        scale: float = 0.1
    ) -> np.ndarray:
        """
        Activity-dependent neural noise
        Higher activity = more noise (matches biology)
        """
        base_noise = self.white_noise(shape, scale)
        activity_modulation = 0.5 + activity_level  # 0.5 to 1.5x
        return base_noise * activity_modulation


class OutlierHandler:
    """
    Robust outlier detection and handling
    
    Methods:
    - IQR-based detection
    - MAD (Median Absolute Deviation)
    - Adaptive thresholds
    """
    
    def __init__(self, method: str = 'mad', threshold: float = 3.0):
        self.method = method
        self.threshold = threshold
        
        # History for adaptive threshold
        self.value_history = deque(maxlen=1000)
        self.outlier_rate = 0.0
    
    def detect(self, x: np.ndarray) -> np.ndarray:
        """
        Detect outliers
        Returns boolean mask (True = outlier)
        """
        x = np.atleast_1d(x)
        
        if self.method == 'iqr':
            return self._iqr_detect(x)
        elif self.method == 'mad':
            return self._mad_detect(x)
        else:
            return self._zscore_detect(x)
    
    def _iqr_detect(self, x: np.ndarray) -> np.ndarray:
        """IQR-based outlier detection"""
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        
        lower = q1 - self.threshold * iqr
        upper = q3 + self.threshold * iqr
        
        return (x < lower) | (x > upper)
    
    def _mad_detect(self, x: np.ndarray) -> np.ndarray:
        """MAD-based outlier detection (robust to outliers)"""
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        
        if mad < 1e-8:
            return np.zeros(len(x), dtype=bool)
        
        # Modified z-score
        modified_z = 0.6745 * (x - median) / mad
        
        return np.abs(modified_z) > self.threshold
    
    def _zscore_detect(self, x: np.ndarray) -> np.ndarray:
        """Simple z-score detection"""
        z = (x - np.mean(x)) / (np.std(x) + 1e-8)
        return np.abs(z) > self.threshold
    
    def handle(
        self, 
        x: np.ndarray, 
        strategy: str = 'clip'
    ) -> np.ndarray:
        """
        Handle outliers according to strategy
        
        Strategies:
        - clip: Clip to threshold
        - nan: Replace with NaN
        - median: Replace with median
        - winsorize: Replace with nearest non-outlier
        """
        x = x.copy()
        outliers = self.detect(x)
        
        # Track outlier rate
        self.outlier_rate = 0.9 * self.outlier_rate + 0.1 * np.mean(outliers)
        
        if not np.any(outliers):
            return x
        
        if strategy == 'nan':
            x[outliers] = np.nan
        
        elif strategy == 'median':
            x[outliers] = np.median(x[~outliers]) if np.any(~outliers) else 0
        
        elif strategy == 'clip':
            median = np.median(x)
            mad = np.median(np.abs(x - median)) + 1e-8
            lower = median - self.threshold * 1.4826 * mad
            upper = median + self.threshold * 1.4826 * mad
            x = np.clip(x, lower, upper)
        
        elif strategy == 'winsorize':
            sorted_x = np.sort(x[~outliers])
            if len(sorted_x) > 0:
                x[outliers & (x < np.median(x))] = sorted_x[0]
                x[outliers & (x >= np.median(x))] = sorted_x[-1]
        
        return x


class AdaptiveGainControl:
    """
    Adaptive gain control (inspired by retinal adaptation)
    
    Maintains sensitivity across wide input range
    Implements:
    - Weber-Fechner law (logarithmic response)
    - Contrast gain control
    - Temporal adaptation
    """
    
    def __init__(self, tau_fast: float = 10.0, tau_slow: float = 1000.0):
        self.tau_fast = tau_fast   # Fast adaptation (ms)
        self.tau_slow = tau_slow   # Slow adaptation (ms)
        
        # Adaptation states
        self.fast_state = 0.0
        self.slow_state = 0.0
        
        # Gain
        self.gain = 1.0
        self.min_gain = 0.01
        self.max_gain = 100.0
        
        # Operating point
        self.operating_point = 0.5
    
    def adapt(self, input_level: float, dt: float) -> None:
        """Adapt to input level"""
        # Fast adaptation tracks recent input
        alpha_fast = dt / (self.tau_fast + dt)
        self.fast_state = (1 - alpha_fast) * self.fast_state + alpha_fast * input_level
        
        # Slow adaptation tracks longer-term average
        alpha_slow = dt / (self.tau_slow + dt)
        self.slow_state = (1 - alpha_slow) * self.slow_state + alpha_slow * input_level
        
        # Compute gain to maintain operating point
        if self.fast_state > 0.01:
            target_gain = self.operating_point / self.fast_state
            self.gain = np.clip(target_gain, self.min_gain, self.max_gain)
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply adaptive gain"""
        return x * self.gain
    
    def weber_fechner(self, x: np.ndarray, c: float = 1.0) -> np.ndarray:
        """
        Weber-Fechner logarithmic response
        Good for compressing wide dynamic range
        """
        return np.log(1 + c * np.abs(x)) * np.sign(x)
    
    def contrast_normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Divisive normalization (contrast gain control)
        Response = x / (sigma + pool(x))
        """
        sigma = 0.1
        pool = np.sqrt(np.mean(x ** 2))
        return x / (sigma + pool)


class RobustInputPipeline:
    """
    Complete robust input processing pipeline
    
    Combines:
    - Normalization
    - Noise injection
    - Outlier handling
    - Adaptive gain control
    """
    
    def __init__(
        self,
        input_dim: int,
        norm_type: NormalizationType = NormalizationType.ADAPTIVE,
        noise_scale: float = 0.05,
        handle_outliers: bool = True
    ):
        self.input_dim = input_dim
        
        # Components
        self.normalizer = InputNormalizer(input_dim, norm_type)
        self.noise_gen = NoiseGenerator()
        self.outlier_handler = OutlierHandler(method='mad', threshold=3.5)
        self.gain_control = AdaptiveGainControl()
        
        # Settings
        self.noise_scale = noise_scale
        self.handle_outliers = handle_outliers
        self.training_mode = True
    
    def process(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Process input through pipeline
        
        Steps:
        1. Handle outliers (if enabled)
        2. Normalize
        3. Apply adaptive gain
        4. Add noise (if training)
        """
        x = np.atleast_1d(x).astype(float)
        
        # 1. Outlier handling
        if self.handle_outliers:
            x = self.outlier_handler.handle(x, strategy='clip')
        
        # 2. Normalization
        x = self.normalizer.normalize(x)
        
        # 3. Adaptive gain control
        input_level = np.mean(np.abs(x))
        self.gain_control.adapt(input_level, dt=1.0)
        x = self.gain_control.apply(x)
        
        # 4. Noise injection (only in training mode)
        if add_noise and self.training_mode and self.noise_scale > 0:
            noise = self.noise_gen.neural_noise(
                x.shape, 
                activity_level=input_level,
                scale=self.noise_scale
            )
            x = x + noise
        
        return x
    
    def set_training(self, mode: bool) -> None:
        """Set training mode (affects noise injection)"""
        self.training_mode = mode
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            'outlier_rate': self.outlier_handler.outlier_rate,
            'current_gain': self.gain_control.gain,
            'global_mean': self.normalizer.global_stats.ema_mean,
            'global_std': self.normalizer.global_stats.ema_std,
        }
