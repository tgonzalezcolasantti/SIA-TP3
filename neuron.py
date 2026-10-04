from typing import List, Self, Tuple

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
        ) / len(error_accumulation)

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
