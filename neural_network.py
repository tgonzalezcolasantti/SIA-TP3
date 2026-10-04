from typing import List, Optional, Self, Tuple

import numpy as np

from activation import Activation


class SimplePerceptron:
    def __init__(
        self: Self,
        inputs: int,
        learning_rate: float,
        activation: Activation,
        min_weight: float = 0,
        max_weight: float = 1,
    ):
        self.weights = np.random.rand(inputs + 1)
        self.weights = (self.weights + min_weight) / (max_weight - min_weight)
        self.activation = activation
        self.learning_rate = learning_rate

    def calculate_output(self: Self, inputs: np.ndarray) -> Tuple[float, float]:
        h = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation.excite(h), h

    def __update_weights(
        self: Self, inputs: np.ndarray, expected: float, output: float, h: float
    ):
        delta = self.learning_rate * (expected - output) * self.activation.derivative(h)
        self.weights[0] += delta
        self.weights[1:] += delta * inputs

    def error(self: Self, error_accumulation: List[Tuple[float, float]]) -> float:
        return sum(
            (expected - actual) ** 2 for expected, actual in error_accumulation
        ) / 2

    def train(
        self: Self,
        training_data: List[Tuple[np.ndarray, float]],
        epochs: int,
        epsilon: float,
    ):
        for _ in range(epochs):
            error_accumulation: List[Tuple[float, float]] = []
            for data, expected in training_data:
                output, h = self.calculate_output(data)
                error_accumulation.append((expected, output))
                self.__update_weights(data, expected, output, h)
            if self.error(error_accumulation) < epsilon:
                return

class MultiLayerPerceptron:
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
        self.second_to_last_layer_count = neuron_topology[-2]
        boundaries = np.cumsum([0, *neuron_topology])
        self.layer_slices = [
            slice(boundaries[i], boundaries[i + 1]) for i in range(len(neuron_topology))
        ]
        self.__init_weights(neuron_topology)

    def __init_weights(self: Self, neuron_topology: List[int]):
        for layer_idx in range(1, len(neuron_topology)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            limit = np.sqrt(6 / (neuron_topology[layer_idx - 1] + neuron_topology[layer_idx]))
            self.weights[current, previous] = np.random.uniform(
                -limit, limit, (neuron_topology[layer_idx], neuron_topology[layer_idx - 1])
            )

    def calculate_outputs(self: Self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=float).reshape(-1)
        if len(inputs) != self.layer_slices[0].stop:
            raise ValueError("input size does not match the first layer")

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
            self.outputs[current] = [self.activation.excite(value) for value in self.h[current]]
        return self.outputs[self.layer_slices[-1]].copy()

    def error(self: Self, error_accumulation: np.ndarray) -> float:
        return np.sum((error_accumulation[:,:,0] - error_accumulation[:,:,1])**2) / 2

    def __update_weights(self: Self, expected: np.ndarray):
        expected = np.asarray(expected, dtype=float).reshape(-1)
        if len(expected) != self.last_layer_count:
            raise ValueError("expected output size does not match the last layer")

        self.deltas[:] = 0
        output_layer = self.layer_slices[-1]
        output_derivatives = np.array([
            self.activation.derivative(value) for value in self.h[output_layer]
        ])
        self.deltas[output_layer] = (
            self.outputs[output_layer] - expected
        ) * output_derivatives

        for layer_idx in range(len(self.layer_slices) - 2, 0, -1):
            current = self.layer_slices[layer_idx]
            following = self.layer_slices[layer_idx + 1]
            derivatives = np.array([
                self.activation.derivative(value) for value in self.h[current]
            ])
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
                outputs = self.calculate_outputs(inputs)
                error_accumulation[i, :, 0] = outputs
                error_accumulation[i, :, 1] = expected
                self.__update_weights(expected)
            if self.error(error_accumulation) < epsilon:
                return

if __name__ == '__main__':
    MultiLayerPerceptron([3,2,1], None, 0.1)
