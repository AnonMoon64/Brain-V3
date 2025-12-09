"""
GPU-Ready Operations and Quantization

Implements:
- Quantized neural states (int8, binary)
- GPU-compatible array operations (numpy with cupy fallback)
- Vectorized operations for massive parallelism
- Memory-efficient representations
- SIMD-friendly data layouts

Key insight: Modern GPUs can simulate millions of neurons
if operations are properly vectorized and memory-efficient.
Quantization enables 32x memory reduction with minimal accuracy loss.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import warnings

# Try to import GPU libraries (optional)
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

# Backend selection
class ComputeBackend(Enum):
    NUMPY = "numpy"
    CUPY = "cupy"


def get_array_module(use_gpu: bool = False):
    """Get the appropriate array module (numpy or cupy)"""
    if use_gpu and GPU_AVAILABLE:
        return cp
    return np


@dataclass
class QuantizationConfig:
    """Configuration for neural state quantization"""
    bits: int = 8  # 8-bit (int8) or 1-bit (binary)
    symmetric: bool = True  # Symmetric around zero
    per_channel: bool = False  # Per-channel vs per-tensor scaling
    
    # Dynamic range
    min_val: float = -1.0
    max_val: float = 1.0
    
    @property
    def scale(self) -> float:
        """Quantization scale factor"""
        if self.bits == 1:
            return 1.0
        n_levels = 2 ** self.bits
        return (self.max_val - self.min_val) / (n_levels - 1)
    
    @property
    def zero_point(self) -> int:
        """Zero point for asymmetric quantization"""
        if self.symmetric:
            return 0
        n_levels = 2 ** self.bits
        return int(-self.min_val / self.scale)


class QuantizedArray:
    """
    Quantized array with efficient storage
    
    Stores values as int8 or binary, with scale/offset for dequantization.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        config: QuantizationConfig = None,
        use_gpu: bool = False
    ):
        self.config = config or QuantizationConfig()
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = get_array_module(self.use_gpu)
        
        # Quantize the data
        self._quantized, self._scale, self._zero_point = self._quantize(data)
        self.shape = data.shape
    
    def _quantize(self, data: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """Quantize float data to int"""
        xp = self.xp
        
        if self.config.bits == 1:
            # Binary quantization
            quantized = (data > 0).astype(xp.uint8)
            return quantized, 1.0, 0
        
        # Compute scale for this data
        data_min = float(xp.min(data))
        data_max = float(xp.max(data))
        
        # Handle zero range
        if data_max - data_min < 1e-8:
            data_max = data_min + 1.0
        
        if self.config.symmetric:
            abs_max = max(abs(data_min), abs(data_max))
            n_levels = 2 ** (self.config.bits - 1) - 1
            scale = abs_max / n_levels if abs_max > 0 else 1.0
            zero_point = 0
        else:
            n_levels = 2 ** self.config.bits - 1
            scale = (data_max - data_min) / n_levels
            zero_point = int(-data_min / scale)
        
        # Quantize
        if scale > 0:
            quantized = xp.round((data - data_min) / scale).astype(xp.int8)
        else:
            quantized = xp.zeros_like(data, dtype=xp.int8)
        
        return quantized, scale, zero_point
    
    def dequantize(self) -> np.ndarray:
        """Convert back to float"""
        xp = self.xp
        
        if self.config.bits == 1:
            return (self._quantized.astype(xp.float32) * 2 - 1)
        
        return (self._quantized.astype(xp.float32) - self._zero_point) * self._scale
    
    def to_numpy(self) -> np.ndarray:
        """Ensure result is numpy array (not cupy)"""
        result = self.dequantize()
        if self.use_gpu:
            return cp.asnumpy(result)
        return result
    
    @property
    def nbytes(self) -> int:
        """Memory usage in bytes"""
        return self._quantized.nbytes
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs float32"""
        float_bytes = np.prod(self.shape) * 4
        return float_bytes / self.nbytes


class BinaryNeuralNetwork:
    """
    Binary Neural Network layer
    
    Uses XNOR and popcount for efficient computation.
    Achieves 32x speedup on compatible hardware.
    """
    
    def __init__(
        self,
        n_input: int,
        n_output: int,
        use_gpu: bool = False
    ):
        self.n_input = n_input
        self.n_output = n_output
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = get_array_module(self.use_gpu)
        
        # Binary weights stored as packed bits
        # Each uint64 holds 64 binary weights
        self.n_packed = (n_input + 63) // 64
        self.weights_packed = self.xp.random.randint(
            0, 2**63,
            size=(n_output, self.n_packed),
            dtype=self.xp.uint64
        )
        
        # Scale for output
        self.scale = 1.0 / n_input
        
        # Batch normalization parameters (learnable)
        self.gamma = self.xp.ones(n_output, dtype=self.xp.float32)
        self.beta = self.xp.zeros(n_output, dtype=self.xp.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with binary operations
        
        For explanation, falls back to explicit binary ops.
        Real hardware would use XNOR+popcount.
        """
        xp = self.xp
        
        # Binarize input (sign function)
        x_binary = (x > 0).astype(xp.float32)
        
        # Unpack weights for matmul (in practice, use bit operations)
        weights_float = self._unpack_weights()
        
        # Binary matmul: XNOR then sum
        # XNOR(a,b) = 1 if a==b else 0
        # This is equivalent to: 2 * (a @ b.T) - n_input
        xnor_sum = xp.dot(x_binary, weights_float.T)
        
        # Scale to [-1, 1]
        output = (2 * xnor_sum - self.n_input) * self.scale
        
        # Batch norm
        output = self.gamma * output + self.beta
        
        return output
    
    def _unpack_weights(self) -> np.ndarray:
        """Unpack binary weights to float (for fallback computation)"""
        xp = self.xp
        
        weights = xp.zeros((self.n_output, self.n_input), dtype=xp.float32)
        
        for i in range(self.n_output):
            for j in range(self.n_packed):
                packed = int(self.weights_packed[i, j])
                for k in range(64):
                    idx = j * 64 + k
                    if idx < self.n_input:
                        weights[i, idx] = float((packed >> k) & 1)
        
        return weights
    
    def update_weights(self, gradients: np.ndarray, learning_rate: float) -> None:
        """
        Update weights using straight-through estimator
        
        Binary weights can't use gradients directly.
        We accumulate gradients and flip bits probabilistically.
        """
        xp = self.xp
        
        # Probability of flipping based on gradient magnitude
        flip_prob = xp.abs(gradients) * learning_rate
        flip_prob = xp.clip(flip_prob, 0, 0.5)
        
        # Random flips
        flips = xp.random.random(gradients.shape) < flip_prob
        
        # Apply flips to packed weights
        for i in range(self.n_output):
            for j in range(self.n_input):
                if flips[i, j]:
                    pack_idx = j // 64
                    bit_idx = j % 64
                    self.weights_packed[i, pack_idx] ^= (1 << bit_idx)


class VectorizedNeuronLayer:
    """
    Fully vectorized neuron layer for GPU acceleration
    
    All operations are batched for maximum parallelism.
    """
    
    def __init__(
        self,
        n_neurons: int,
        use_gpu: bool = False,
        dtype: str = 'float32'
    ):
        self.n_neurons = n_neurons
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = get_array_module(self.use_gpu)
        
        # Set dtype
        self.dtype = getattr(self.xp, dtype)
        
        # Neuron state (vectorized)
        self.membrane_potential = self.xp.zeros(n_neurons, dtype=self.dtype)
        self.threshold = self.xp.ones(n_neurons, dtype=self.dtype) * (-55.0)
        self.v_rest = self.xp.ones(n_neurons, dtype=self.dtype) * (-70.0)
        self.v_reset = self.xp.ones(n_neurons, dtype=self.dtype) * (-75.0)
        self.tau = self.xp.ones(n_neurons, dtype=self.dtype) * 20.0
        
        # Refractory state
        self.refractory_remaining = self.xp.zeros(n_neurons, dtype=self.dtype)
        self.refractory_period = 2.0
        
        # Spike output (binary)
        self.spikes = self.xp.zeros(n_neurons, dtype=self.xp.uint8)
        
        # Statistics
        self.spike_counts = self.xp.zeros(n_neurons, dtype=self.xp.int32)
    
    def step(
        self,
        input_current: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Vectorized neuron update
        
        All neurons are updated in parallel.
        """
        xp = self.xp
        
        # Convert input if needed
        if self.use_gpu and not isinstance(input_current, cp.ndarray):
            input_current = cp.asarray(input_current)
        
        # Pad or truncate input
        if len(input_current) < self.n_neurons:
            padded = xp.zeros(self.n_neurons, dtype=self.dtype)
            padded[:len(input_current)] = input_current
            input_current = padded
        else:
            input_current = input_current[:self.n_neurons]
        
        # Update refractory
        self.refractory_remaining = xp.maximum(0, self.refractory_remaining - dt)
        
        # Mask for non-refractory neurons
        active_mask = self.refractory_remaining <= 0
        
        # Leaky integration (vectorized)
        dv = (self.v_rest - self.membrane_potential + input_current) / self.tau * dt
        self.membrane_potential = xp.where(
            active_mask,
            self.membrane_potential + dv,
            self.v_reset
        )
        
        # Check threshold (vectorized)
        spike_mask = (self.membrane_potential >= self.threshold) & active_mask
        self.spikes = spike_mask.astype(xp.uint8)
        
        # Reset spiking neurons
        self.membrane_potential = xp.where(
            spike_mask,
            self.v_reset,
            self.membrane_potential
        )
        
        # Set refractory
        self.refractory_remaining = xp.where(
            spike_mask,
            xp.full(self.n_neurons, self.refractory_period, dtype=self.dtype),
            self.refractory_remaining
        )
        
        # Update counts
        self.spike_counts += self.spikes.astype(xp.int32)
        
        return self.spikes
    
    def get_spikes_numpy(self) -> np.ndarray:
        """Get spikes as numpy array"""
        if self.use_gpu:
            return cp.asnumpy(self.spikes)
        return self.spikes
    
    def get_firing_rates(self, total_time: float) -> np.ndarray:
        """Get firing rates in Hz"""
        rates = self.spike_counts.astype(self.dtype) / (total_time / 1000.0)
        if self.use_gpu:
            return cp.asnumpy(rates)
        return rates


class VectorizedSynapseMatrix:
    """
    Vectorized synapse operations using sparse or dense matrices
    """
    
    def __init__(
        self,
        n_pre: int,
        n_post: int,
        density: float = 0.1,
        use_gpu: bool = False,
        quantize: bool = False
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = get_array_module(self.use_gpu)
        self.quantize = quantize
        
        # Initialize weights
        if quantize:
            # Quantized weights (int8)
            weights_float = self.xp.random.randn(n_post, n_pre).astype(self.xp.float32) * 0.1
            self.weights = QuantizedArray(
                weights_float if not self.use_gpu else cp.asnumpy(weights_float),
                QuantizationConfig(bits=8),
                use_gpu=use_gpu
            )
            self._weights_float = None  # Will dequantize when needed
        else:
            # Full precision weights
            self.weights = self.xp.random.randn(n_post, n_pre).astype(self.xp.float32) * 0.1
        
        # Mask for connectivity (sparse)
        self.mask = self.xp.random.random((n_post, n_pre)) < density
        if not quantize:
            self.weights *= self.mask
        
        # Delays (simplified: single delay)
        self.delay = 1.0
        
        # Eligibility traces (for learning)
        self.eligibility = self.xp.zeros((n_post, n_pre), dtype=self.xp.float32)
        self.eligibility_decay = 0.95
    
    def propagate(self, pre_spikes: np.ndarray) -> np.ndarray:
        """
        Propagate spikes through synapses
        
        post_current = W @ pre_spikes
        """
        xp = self.xp
        
        # Convert input if needed
        if self.use_gpu and not isinstance(pre_spikes, cp.ndarray):
            pre_spikes = cp.asarray(pre_spikes)
        
        # Get weights (dequantize if needed)
        if self.quantize:
            weights = self.weights.dequantize()
            if not self.use_gpu:
                weights = weights
            else:
                weights = cp.asarray(weights)
            weights = weights * self.mask
        else:
            weights = self.weights
        
        # Matrix multiply
        post_current = xp.dot(weights, pre_spikes.astype(xp.float32))
        
        return post_current
    
    def update_eligibility(self, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> None:
        """Update eligibility traces for STDP"""
        xp = self.xp
        
        if self.use_gpu:
            if not isinstance(pre_spikes, cp.ndarray):
                pre_spikes = cp.asarray(pre_spikes)
            if not isinstance(post_spikes, cp.ndarray):
                post_spikes = cp.asarray(post_spikes)
        
        # Decay
        self.eligibility *= self.eligibility_decay
        
        # Hebbian update: outer product of post and pre
        self.eligibility += xp.outer(
            post_spikes.astype(xp.float32),
            pre_spikes.astype(xp.float32)
        )
    
    def apply_reward(self, reward: float, learning_rate: float = 0.01) -> None:
        """Apply reward to eligibility traces"""
        xp = self.xp
        
        weight_update = self.eligibility * reward * learning_rate
        
        if self.quantize:
            # Dequantize, update, requantize
            weights_float = self.weights.dequantize()
            if self.use_gpu and not isinstance(weights_float, cp.ndarray):
                weights_float = cp.asarray(weights_float)
            weights_float += weight_update
            weights_float = xp.clip(weights_float, -1, 1)
            self.weights = QuantizedArray(
                weights_float if not self.use_gpu else cp.asnumpy(weights_float),
                self.weights.config,
                use_gpu=self.use_gpu
            )
        else:
            self.weights += weight_update * self.mask
            self.weights = xp.clip(self.weights, -1, 1)


class GPUNeuralNetwork:
    """
    Complete GPU-accelerated neural network
    
    Combines:
    - Vectorized neuron layers
    - Efficient synapse propagation
    - Optional quantization
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        use_gpu: bool = False,
        quantize: bool = False,
        sparsity: float = 0.1
    ):
        self.layer_sizes = layer_sizes
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.quantize = quantize
        
        # Create layers
        self.layers: List[VectorizedNeuronLayer] = []
        self.synapses: List[VectorizedSynapseMatrix] = []
        
        for i, size in enumerate(layer_sizes):
            self.layers.append(VectorizedNeuronLayer(
                size,
                use_gpu=self.use_gpu,
                dtype='float32'
            ))
            
            if i > 0:
                self.synapses.append(VectorizedSynapseMatrix(
                    layer_sizes[i-1],
                    size,
                    density=sparsity,
                    use_gpu=self.use_gpu,
                    quantize=quantize
                ))
        
        # Simulation state
        self.current_time = 0.0
        self.dt = 1.0
    
    def step(self, input_current: np.ndarray) -> List[np.ndarray]:
        """
        Run one simulation step through all layers
        """
        self.current_time += self.dt
        
        layer_outputs = []
        current_input = input_current
        
        for i, layer in enumerate(self.layers):
            if i > 0:
                # Propagate through synapses
                current_input = self.synapses[i-1].propagate(layer_outputs[-1])
            
            # Update neurons
            spikes = layer.step(current_input, self.dt)
            layer_outputs.append(spikes)
        
        return layer_outputs
    
    def learn(self, reward: float) -> None:
        """Apply reward signal to all synapses"""
        for synapse in self.synapses:
            synapse.apply_reward(reward)
    
    def get_output(self) -> np.ndarray:
        """Get output from last layer"""
        return self.layers[-1].get_spikes_numpy()
    
    def get_stats(self) -> Dict:
        """Get network statistics"""
        xp = get_array_module(self.use_gpu)
        
        stats = {
            'n_layers': len(self.layers),
            'layer_sizes': self.layer_sizes,
            'use_gpu': self.use_gpu,
            'quantized': self.quantize,
            'current_time': self.current_time,
        }
        
        # Memory usage
        total_bytes = 0
        for layer in self.layers:
            total_bytes += layer.membrane_potential.nbytes
            total_bytes += layer.spikes.nbytes
        
        for synapse in self.synapses:
            if self.quantize:
                total_bytes += synapse.weights.nbytes
            else:
                total_bytes += synapse.weights.nbytes
        
        stats['memory_bytes'] = total_bytes
        stats['memory_mb'] = total_bytes / (1024 * 1024)
        
        if self.quantize:
            stats['compression_ratio'] = 4.0  # Approximate for int8
        else:
            stats['compression_ratio'] = 1.0
        
        return stats


def benchmark_gpu_vs_cpu(n_neurons: int = 10000, n_steps: int = 100):
    """
    Benchmark GPU vs CPU performance
    """
    import time
    
    results = {}
    
    # CPU test
    net_cpu = GPUNeuralNetwork(
        [n_neurons, n_neurons // 2, n_neurons // 4],
        use_gpu=False,
        quantize=False
    )
    
    input_data = np.random.randn(n_neurons).astype(np.float32)
    
    start = time.time()
    for _ in range(n_steps):
        net_cpu.step(input_data)
    cpu_time = time.time() - start
    results['cpu_time'] = cpu_time
    results['cpu_steps_per_sec'] = n_steps / cpu_time
    
    # GPU test (if available)
    if GPU_AVAILABLE:
        net_gpu = GPUNeuralNetwork(
            [n_neurons, n_neurons // 2, n_neurons // 4],
            use_gpu=True,
            quantize=False
        )
        
        # Warm up
        for _ in range(10):
            net_gpu.step(input_data)
        
        start = time.time()
        for _ in range(n_steps):
            net_gpu.step(input_data)
        cp.cuda.Stream.null.synchronize()  # Wait for GPU
        gpu_time = time.time() - start
        
        results['gpu_time'] = gpu_time
        results['gpu_steps_per_sec'] = n_steps / gpu_time
        results['speedup'] = cpu_time / gpu_time
    else:
        results['gpu_available'] = False
    
    return results
