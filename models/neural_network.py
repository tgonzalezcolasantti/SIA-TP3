from pathlib import Path
import pickle
from typing import List, Optional, Self, Tuple, override

import numpy as np

from models.optimization import NoOptimization, Optimization

from .activation import Activation


class Perceptron:
    def __init__(
        self: Self,
        neuron_topology: List[int],
        activation: Activation,
        learning_rate: float,
        save: Path | None = None,
        epoch: int = 0,
        weights: np.ndarray | None = None,
        optimization: Optimization = NoOptimization(),
        model_name: Optional[str] = None,
    ):
        self.model_name = model_name
        self.topology = neuron_topology
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimization = optimization
        self.epoch = epoch
        self.error_history = []

        if save:
            if save.is_dir():
                save = save / self._suggested_filename()
            if save.exists():
                unpickled: Perceptron = self.load(save)
                weights = unpickled.weights
                neuron_topology = unpickled.topology
                self.activation = unpickled.activation
                self.optimization = unpickled.optimization
                self.epoch = unpickled.epoch
                self.learning_rate = unpickled.learning_rate
                self.error_history = unpickled.error_history
        else:
            if len(neuron_topology) < 2 or any(count <= 0 for count in neuron_topology):
                raise ValueError(
                    "neuron_topology must contain at least two positive layer sizes"
                )

        total = sum(neuron_topology)
        self.weights = np.zeros((total, total)) if weights is None else weights
        self.outputs = np.zeros(total)
        self.h = np.zeros(total)
        self.deltas = np.zeros(total)
        self.weight_deltas = np.zeros((total, total))
        self.last_layer_count = neuron_topology[-1]
        self.save_path = save
        boundaries = np.cumsum([0, *neuron_topology])
        self.layer_slices = [
            slice(boundaries[i], boundaries[i + 1]) for i in range(len(neuron_topology))
        ]
        if weights is None:
            self._init_weights(neuron_topology)

    def _init_weights(self: Self, neuron_topology: List[int]):
        for layer_idx in range(1, len(neuron_topology)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            # TODO What is this? aka why this function specifically?
            limit = np.sqrt(
                6 / (neuron_topology[layer_idx - 1] + neuron_topology[layer_idx])
            )
            self.weights[current, previous] = np.random.uniform(
                -limit,
                limit,
                (neuron_topology[layer_idx], neuron_topology[layer_idx - 1]),
            )

    def classify(self: Self, inputs: np.ndarray) -> np.ndarray:
        if inputs.shape[-1] != self.layer_slices[0].stop:
            raise ValueError("input size does not match the first layer")
        self.outputs = np.zeros(self.weights.shape[0])
        self.h = np.zeros(self.weights.shape[0])
        return self._calculate_outputs(inputs).T.copy()

    def _calculate_outputs(self: Self, inputs: np.ndarray) -> np.ndarray:
        # inputs = np.asarray(inputs, dtype=float).reshape(-1)
        if len(inputs.shape) > 1:
            self.outputs = np.zeros((self.weights.shape[0], inputs.shape[0]))
            self.h = np.zeros((self.weights.shape[0], inputs.shape[0]))
        else:
            self.outputs[:] = 0
        self.outputs[self.layer_slices[0]] = inputs.T
        for layer_idx in range(1, len(self.layer_slices)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            neurons = np.arange(current.start, current.stop)
            self.h[current] = (
                self.weights[current, previous] @ self.outputs[previous]
                + self.weights[neurons, neurons]
            )
            self.outputs[current] = self.activation.excite(self.h[current])
        return self.outputs[self.layer_slices[-1]]

    def _error(self: Self, expected: np.ndarray, output: np.ndarray) -> float:
        return np.sum((expected - output) ** 2) / 2

    def _global_errors(
        self: Self, error_accumulation: np.ndarray
    ) -> Tuple[float, float]:
        errors = error_accumulation[:, :, 0] - error_accumulation[:, :, 1]
        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))
        print(f"MSE: {mse}\tMAE: {mae}")
        return mse, mae

    def _update_weights(self: Self, expected: np.ndarray):
        expected = np.asarray(expected, dtype=float).reshape(-1)

        self.deltas[:] = 0
        output_layer = self.layer_slices[-1]
        output_derivatives = self.activation.derivative(self.h[output_layer])
        self.deltas[output_layer] = (
            self.outputs[output_layer] - expected
        ) * output_derivatives

        for layer_idx in range(len(self.layer_slices) - 2, 0, -1):
            current = self.layer_slices[layer_idx]
            following = self.layer_slices[layer_idx + 1]
            derivatives = self.activation.derivative(self.h[current])
            self.deltas[current] = (
                self.weights[following, current].T @ self.deltas[following]
            ) * derivatives

        for layer_idx in range(1, len(self.layer_slices)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            neurons = np.arange(current.start, current.stop)
            self.weight_deltas[current, previous] = self.optimization.apply(
                self.learning_rate,
                np.outer(self.deltas[current], self.outputs[previous]),
                self.weight_deltas[current, previous],
            )
            self.weight_deltas[neurons, neurons] = self.optimization.apply(
                self.learning_rate,
                self.deltas[current],
                self.weight_deltas[neurons, neurons],
            )
        self.weights += self.weight_deltas

    def train(
        self: Self,
        training_data: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int,
        epsilon: float,
    ) -> List[Tuple[float, float]]:
        for epoch in range(self.epoch, epochs):
            print(epoch)
            self.epoch = epoch + 1
            # Array shaped like (#training samples, #outputs, (output and expected values))
            results = np.zeros(shape=(len(training_data), self.last_layer_count, 2))
            for i, data in enumerate(training_data):
                inputs, expected = data
                outputs = self._calculate_outputs(inputs)
                results[i, :, 0] = outputs
                results[i, :, 1] = expected
                self._update_weights(expected)
                if self._error(outputs, expected) < epsilon:
                    return self.error_history
            self.error_history.append(self._global_errors(results))
            self.save()
        return self.error_history

    @classmethod
    def load(cls:type[Perceptron], save_path: Path):
        with open(save_path, 'rb+') as file:
            return pickle.load(file)

    def save(self: Self):
        if self.save_path:
            with open(self.save_path, '+wb') as file:
                pickle.dump(self, file)

    def _suggested_filename(self: Self) -> str:
        return f"{self.model_name + ' ' if self.model_name else None}"+\
               f"[{','.join(str(x) for x in self.topology)}]"+\
               f" {self.learning_rate:.6g} {str(self.activation)} {str(self.optimization)}.model"


class SimplePerceptron(Perceptron):
    def __init__(
        self: Self,
        inputs: int,
        activation: Activation,
        learning_rate: float,
        weight_range: Tuple[float, float] = (0, 1),
        epoch: int = 0,
        save: Optional[Path] = None,
        model_name: Optional[str] = None
    ):
        self.min_weight = min(weight_range)
        self.max_weight = max(weight_range)
        super().__init__(
            neuron_topology=[inputs, 1],
            activation=activation,
            learning_rate=learning_rate,
            save=save,
            model_name=model_name,
            epoch=epoch
        )

    @override
    def _init_weights(self: Self, neuron_topology: List[int]):
        for layer_idx in range(1, len(neuron_topology)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            self.weights[current, previous] = np.random.uniform(
                self.min_weight,
                self.max_weight,
                (neuron_topology[layer_idx], neuron_topology[layer_idx - 1]),
            )


class MultiLayerPerceptron(Perceptron):
    pass
