from abc import ABC, abstractmethod
import math
from typing import Self, override

import numpy as np

class Activation(ABC):
    name: str = ""
    @abstractmethod
    def excite(self: Self, value: float | np.ndarray) -> float | np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def derivative(self: Self, value: float | np.ndarray) -> float | np.ndarray:
        raise NotImplementedError()

    @classmethod
    def _from_string(cls: type[Activation], string: str):
        return cls()

    def __str__(self: Self) -> str:
        return self.name

    @classmethod
    def from_string(cls: type[Activation], string: str) -> Activation:
        for subclass in cls.__subclasses__():
            if subclass.name in string:
                return subclass._from_string(string)
        raise ValueError()


class Step(Activation):
    name = "step"
    @override
    def excite(self: Self, value: float | np.ndarray) -> float | np.ndarray:
        return np.heaviside(value, 1) * 2 - 1

    @override
    def derivative(self: Self, value: float | np.ndarray) -> float | np.ndarray:
        # Conceptually wrong, but returning this allows the multilayered perceptron
        # To simulate being a simple step perceptron
        # Because its algorithm multiplies by this derivative.
        return np.ones(value.shape) if isinstance(value, np.ndarray) else 1

class Adaline(Activation):
    name = "adaline"
    @override
    def excite(self: Self, value: float | np.ndarray) -> float | np.ndarray:
        return value

    @override
    def derivative(self: Self, value: float | np.ndarray) -> float | np.ndarray:
        return np.ones(value.shape) if isinstance(value, np.ndarray) else 1

class BetaActivation(Activation):
    def __init__(self: Self, beta: float = 1):
        super().__init__()
        self.beta = beta

    def __str__(self: Self) -> str:
        return " ".join([self.name, str(self.beta)])

    @override
    @classmethod
    def _from_string(cls: type[BetaActivation], string: str):
        beta = float(string.split()[1])
        return cls(beta)

class Tanh(BetaActivation, Activation):
    name="tanh"
    @override
    def excite(self: Self, value: float | np.ndarray) -> float | np.ndarray:
        return np.tanh(self.beta * value)

    @override
    def derivative(self: Self, value: float | np.ndarray) -> float | np.ndarray:
        return self.beta * (1 - self.excite(value) ** 2)

class Logistic(BetaActivation, Activation):
    name="logistic"
    @override
    def excite(self: Self, value: float | np.ndarray) -> float | np.ndarray:
        return 1 / (1 + math.exp(-2 * self.beta * value))

    @override
    def derivative(self: Self, value: float | np.ndarray) -> float | np.ndarray:
        excitation = self.excite(value)
        return 2 * self.beta * excitation * (1 - excitation)
