from abc import ABC, abstractmethod
from typing import Self, override

import numpy as np


class Optimization(ABC):
    @abstractmethod
    def apply(self: Self, learning_rate: float, derivatives: np.ndarray, errors: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class NoOptimization(Optimization):
    @override
    def apply(self: Self, learning_rate: float, derivatives: np.ndarray, errors: np.ndarray) -> np.ndarray:
        return -learning_rate * derivatives