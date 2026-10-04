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
        h = np.dot(inputs, self.weights[1:])
        return self.activation.excite(h - self.weights[0]), h

    def __update_weights(
        self: Self, inputs: np.ndarray, expected: float, output: float, h: float
    ):
        delta_b = self.learning_rate * (expected - output) * self.activation.derivative(h)
        self.weights[0] += delta_b
        self.weights[1:] += delta_b * inputs

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
        total = sum(neuron_topology)
        self.weights = np.zeros((total, total))
        self.outputs = np.zeros(total)
        self.h = np.zeros(total)
        self.deltas = np.zeros(total)
        self.activation = activation
        self.learning_rate = learning_rate
        self.last_layer_count = neuron_topology[-1]
        self.second_to_last_layer_count = neuron_topology[-2]
        self.__init_weights(neuron_topology)

    def __init_weights(self: Self, neuron_topology: List[int]):
        for i, count in enumerate(neuron_topology):
            prev = sum(neuron_topology[:i])
            if prev == 0:
                # We reserve the first layer 'layer 0' for inputs only
                # So we can use a very simple dot product to calculate stuff
                continue
            for neuron in range(count):
                self.weights[neuron, neuron] = np.random.rand() # Bias
                # We end up with a matrix indexed by neuron holding weights for conections
                # with the previous layer
                for prev_neuron in range(sum(neuron_topology[:i-1]), prev):
                    rand = np.random.rand()
                    self.weights[neuron, prev_neuron] = rand
                    self.weights[prev_neuron, neuron] = rand

    def calculate_outputs(self: Self, inputs: List[float]) -> np.ndarray:
        self.outputs[:] = 0
        self.outputs[0:len(inputs)] = inputs
        for neuron in range(len(inputs), len(self.weights)):
            self.h[neuron] = np.dot(self.outputs, self.weights[neuron])
            self.outputs[neuron] = self.activation.excite(self.h[neuron] - self.weights[neuron, neuron])
        return self.outputs[-self.last_layer_count:]

    def error(self: Self, error_accumulation: np.ndarray) -> float:
        return np.sum((error_accumulation[:,:,0] - error_accumulation[:,:,1])**2) / 2

    def __update_weights(self: Self, expected: List[float]):
        #ALGO ESTA MAL, PERO LO REVISO MANANA
        self.deltas[:] = 0
        second_to_last_layer_count = self.second_to_last_layer_count
        for neuron in range(0, len(self.weights) - self.last_layer_count, -1):
            if second_to_last_layer_count > 0:
                second_to_last_layer_count -= 1
                self.deltas[neuron] = 
            self.deltas[neuron] = sum(self.deltas * self.weights[neuron]) * self.activation.derivative(self.h[neuron])
            self.weights[neuron] -= self.learning_rate * self.deltas[neuron] * self.outputs
