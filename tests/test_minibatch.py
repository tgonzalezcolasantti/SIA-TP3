"""Regression checks for the shared perceptron's batched weight updates."""

import unittest

import numpy as np

from models.activation import Tanh
from models.neural_network import MultiLayerPerceptron
from models.optimization import NoOptimization


def model():
    np.random.seed(7)
    return MultiLayerPerceptron([2, 3, 2], Tanh(), 0.01,
                                optimization=NoOptimization())


DATA = [
    (np.array([0.2, 0.7]), np.array([1.0, -1.0])),
    (np.array([0.8, 0.1]), np.array([-1.0, 1.0])),
    (np.array([0.4, 0.5]), np.array([1.0, -1.0])),
]


class MiniBatchTests(unittest.TestCase):
    def test_full_batch_update_averages_sample_gradients(self):
        batched = model()
        batched.train(DATA, epochs=1, epsilon=-1, batch_size=-1)
        individual = []
        for sample in DATA:
            separate = model()
            separate.train([sample], epochs=1, epsilon=-1, batch_size=1)
            individual.append(separate.weights)
        np.testing.assert_allclose(batched.weights, np.mean(individual, axis=0))

    def test_last_partial_batch_is_processed_once(self):
        batched = model()
        batched.train(DATA, epochs=1, epsilon=-1, batch_size=2)
        manual = model()
        manual.train(DATA[:2], epochs=1, epsilon=-1, batch_size=2)
        manual.train(DATA[2:], epochs=1, epsilon=-1, batch_size=1)
        np.testing.assert_allclose(batched.weights, manual.weights)
        self.assertEqual(batched.classify(DATA[0][0]).shape, (1, 2))
        self.assertEqual(batched.classify(np.stack([x for x, _ in DATA])).shape,
                         (3, 2))


if __name__ == "__main__":
    unittest.main()
