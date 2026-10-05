from typing import List, Self, Tuple, override

import numpy as np

from .activation import Activation

class Perceptron():
    def __init__(self: Self, neuron_topology: List[int], activation: Activation, learning_rate: float):
        if len(neuron_topology) < 2 or any(count <= 0 for count in neuron_topology):
            raise ValueError("neuron_topology must contain at least two positive layer sizes")

        total = sum(neuron_topology)
        self.weights = np.zeros((total, total))
        self.outputs = np.zeros(total)
        self.h = np.zeros(total)
        self.deltas = np.zeros(total)
        self.activation = activation
        self.learning_rate = learning_rate
        self.last_layer_count = neuron_topology[-1]
        boundaries = np.cumsum([0, *neuron_topology])
        self.layer_slices = [
            slice(boundaries[i], boundaries[i + 1]) for i in range(len(neuron_topology))
        ]
        self._init_weights(neuron_topology)

    def _init_weights(self: Self, neuron_topology: List[int]):
        for layer_idx in range(1, len(neuron_topology)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            # TODO What is this? aka why this function specifically?
            limit = np.sqrt(6 / (neuron_topology[layer_idx - 1] + neuron_topology[layer_idx]))
            self.weights[current, previous] = np.random.uniform(
                -limit, limit, (neuron_topology[layer_idx], neuron_topology[layer_idx - 1])
            )
    def classify(self: Self, inputs: np.ndarray) -> np.ndarray:
        if len(inputs) != self.layer_slices[0].stop:
            raise ValueError("input size does not match the first layer")
        return self._calculate_outputs(inputs).copy()
    
    def _calculate_outputs(self: Self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=float).reshape(-1)

        self.outputs[:] = 0
        self.outputs[self.layer_slices[0]] = inputs
        for layer_idx in range(1, len(self.layer_slices)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            neurons = np.arange(current.start, current.stop)
            self.h[current] = (
                self.weights[current, previous] @ self.outputs[previous]
                + self.weights[neurons, neurons]
            )
            self.outputs[current] = self.activation.excite(self.h[current])
        return self.outputs[self.layer_slices[-1]]

    def _error(self: Self, error_accumulation: np.ndarray) -> float:
        return np.sum((error_accumulation[:,:,0] - error_accumulation[:,:,1])**2) / 2

    def _update_weights(self: Self, expected: np.ndarray):
        expected = np.asarray(expected, dtype=float).reshape(-1)

        self.deltas[:] = 0
        output_layer = self.layer_slices[-1]
        output_derivatives = self.activation.derivative(self.h[output_layer])
        self.deltas[output_layer] = (
            self.outputs[output_layer] - expected
        ) * output_derivatives

        for layer_idx in range(len(self.layer_slices) - 2, 0, -1):
            current = self.layer_slices[layer_idx]
            following = self.layer_slices[layer_idx + 1]
            derivatives = self.activation.derivative(self.h[current])
            self.deltas[current] = (
                self.weights[following, current].T @ self.deltas[following]
            ) * derivatives

        for layer_idx in range(1, len(self.layer_slices)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            neurons = np.arange(current.start, current.stop)
            self.weights[current, previous] -= self.learning_rate * np.outer(
                self.deltas[current], self.outputs[previous]
            )
            self.weights[neurons, neurons] -= self.learning_rate * self.deltas[current]

    def train(
        self: Self,
        training_data: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int,
        epsilon: float,
    ):
        for _ in range(epochs):
            error_accumulation = np.zeros(shape=(len(training_data), self.last_layer_count, 2))
            for i, data in enumerate(training_data):
                inputs, expected = data
                outputs = self._calculate_outputs(inputs)
                error_accumulation[i, :, 0] = outputs
                error_accumulation[i, :, 1] = expected
                self._update_weights(expected)
            if self._error(error_accumulation) < epsilon:
                return

class SimplePerceptron(Perceptron):
    def __init__(
        self: Self,
        inputs: int,
        activation: Activation,
        learning_rate: float,
        weight_range: Tuple[float, float] = (0, 1)
    ):
        self.min_weight = min(weight_range)
        self.max_weight = max(weight_range)
        super().__init__([inputs, 1], activation, learning_rate)

    @override
    def _init_weights(self: Self, neuron_topology: List[int]):
        for layer_idx in range(1, len(neuron_topology)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            self.weights[current, previous] = np.random.uniform(
                self.min_weight, self.max_weight, (neuron_topology[layer_idx], neuron_topology[layer_idx - 1])
            )

class MultiLayerPerceptron(Perceptron):
    pass
