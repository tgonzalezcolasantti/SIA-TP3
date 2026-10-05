from abc import ABC, abstractmethod
from typing import Self, override

import numpy as np


class Optimization(ABC):
    name = ""
    @abstractmethod
    def apply(self: Self, learning_rate: float, derivatives: np.ndarray, errors: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    @classmethod
    def _from_string(cls: type[Optimization], string: str):
        return cls()
    @classmethod
    def from_string(cls: type[Optimization], string: str) -> Optimization:
        for subclass in cls.__subclasses__():
            if subclass.name in string:
                return subclass._from_string(string)
        raise ValueError()
    def __str__(self: Self)-> str:
        return self.name

class NoOptimization(Optimization):
    name="NoOptimization"
    @override
    def apply(self: Self, learning_rate: float, derivatives: np.ndarray, errors: np.ndarray) -> np.ndarray:
        return -learning_rate * derivatives