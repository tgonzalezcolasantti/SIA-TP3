from abc import ABC, abstractmethod
from typing import List, Self, Tuple, override

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
        return 1 if value >= 0 else 0

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
