from abc import ABC, abstractmethod
from typing import Self, override

import numpy as np


class Optimization(ABC):
    name = ""
    @abstractmethod
    def apply(self: Self, learning_rate: float, derivatives: np.ndarray, last_errors: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __str__(self: Self)-> str:
        return self.name

class NoOptimization(Optimization):
    name="NoOptimization"
    @override
    def apply(self: Self, learning_rate: float, derivatives: np.ndarray, last_errors: np.ndarray) -> np.ndarray:
        return -learning_rate * derivatives

class Momentum(Optimization):
    name="Momentum"

    def __init__(self: Self, alpha: float):
        super().__init__()
        self.alpha = alpha

    @override
    def apply(self: Self, learning_rate: float, derivatives: np.ndarray, last_errors: np.ndarray) -> np.ndarray:
        return -learning_rate * derivatives + self.alpha*last_errors

    def __str__(self: Self) -> str:
        return f"{self.name}({self.alpha:.6g})"