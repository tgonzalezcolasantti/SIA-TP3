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
        print(self.weights)
        print(self.weights[0])

    def __init_weights(self: Self, neuron_topology: List[int]):
        for i, count in enumerate(neuron_topology):
            prev = sum(neuron_topology[:i])
            if prev == 0:
                # We reserve the first layer 'layer 0' for inputs only
                # So we can use a very simple dot product to calculate stuff
                continue
            for neuron in range(prev, prev+count):
                self.weights[neuron, neuron] = np.random.rand() # Bias
                # We end up with a matrix indexed by neuron holding weights for conections
                # with the previous and next layer
                for prev_neuron in range(sum(neuron_topology[:i-1]), prev):
                    rand = np.random.rand()
                    self.weights[neuron, prev_neuron] = rand
                    self.weights[prev_neuron, neuron] = rand

    def calculate_outputs(self: Self, inputs: np.ndarray) -> np.ndarray:
        self.outputs[:] = 0
        self.outputs[0:len(inputs)] = inputs
        for neuron in range(len(inputs), len(self.weights)):
            # Doing it step by step like this means all outputs after this neuron are 0
            # So doing this simple dot product works!
            self.h[neuron] = np.dot(self.outputs, self.weights[neuron])
            self.outputs[neuron] = self.activation.excite(self.h[neuron] - self.weights[neuron, neuron])
        return self.outputs[-self.last_layer_count:]

    def error(self: Self, error_accumulation: np.ndarray) -> float:
        return np.sum((error_accumulation[:,:,0] - error_accumulation[:,:,1])**2) / 2

    def __update_weights(self: Self, expected: np.ndarray):
        self.deltas[:] = 0
        for output_idx in range(self.last_layer_count):
            self.deltas[-output_idx] = expected[-output_idx] * self.activation.derivative(self.h[-output_idx])
        for neuron in range(0, len(self.deltas) - self.last_layer_count, -1):
            # Doing it backwards like this means all deltas before this neuron are 0
            # So these dot products should work!
            weights_mask = self.weights[neuron] > 0
            self.weights[neuron] -= self.learning_rate * np.dot(self.deltas[weights_mask], self.outputs)
            self.weights[:,neuron] = self.weights[neuron]
            # This is technically calculating #(inputs) extra deltas, but whatever
            self.deltas[neuron] = np.dot(self.deltas, self.weights[neuron]) * self.activation.derivative(self.h[neuron])

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
