from typing import List, Tuple

import numpy as np
from pandas import DataFrame

from data.digit_dataset_loader import load_dataset
from models.neural_network import MultiLayerPerceptron
from models.activation import Tanh, Adaline

def label_to_outputs(label: str) -> np.ndarray:
    outputs = np.zeros(10)
    outputs[int(label)] = 1
    return outputs

def condition_dataset(dataset: DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    #ABSOLUTELY DREADFUL, DARLING~
    images = dataset['image']
    labels = [label_to_outputs(x) for x in dataset['label']]
    return list(zip(images, labels))

def main():
    print("Loading data")
    dataset = load_dataset("data/digits.csv")
    model = MultiLayerPerceptron([28*28, int(28*28/8), 10], Tanh(), 0.0005)
    print("Conditioning data")
    training_set = condition_dataset(dataset)
    print("Training")
    model.train(training_set[:1000], 1000, 0.001)

if __name__ == '__main__':
    main()