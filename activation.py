from abc import ABC, abstractmethod
import math
from typing import List, Self, Tuple, override

import numpy as np

class Activation(ABC):
    @abstractmethod
    def excite(self: Self, value: float) -> float:
        raise NotImplementedError()

    @abstractmethod
    def derivative(self: Self, value: float) -> float:
        raise NotImplementedError()


class Step(Activation):
    @override
    def excite(self: Self, value: float) -> float:
        return 1 if value >= 0 else -1

    @override
    def derivative(self: Self, value: float) -> float:
       return 1

class Adaline(Activation):
    @override
    def excite(self: Self, value: float) -> float:
        return value

    @override
    def derivative(self: Self, value: float) -> float:
        return 1

class Tanh(Activation):
    def __init__(self: Self, beta: float = 1):
        self.beta = beta

    @override
    def excite(self: Self, value: float) -> float:
        return np.tanh(self.beta * value)

    @override
    def derivative(self: Self, value: float) -> float:
        return self.beta * (1 - self.excite(value) ** 2)

class Logistic(Activation):
    def __init__(self: Self, beta: float = 1):
        self.beta = beta

    @override
    def excite(self: Self, value: float) -> float:
        return 1 / (1 + math.exp(-2 * self.beta * value))

    @override
    def derivative(self: Self, value: float) -> float:
        excitation = self.excite(value)
        return 2 * self.beta * excitation * (1 - excitation)
