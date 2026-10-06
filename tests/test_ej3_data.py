"""Check exercise 3 data handling before evaluating unseen digits."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ej3_mejora.run import load_learning_data, make_training_data, training_indices


def write_digits(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("label", "image"))
        writer.writeheader()
        for label, bright_pixel in rows:
            image = np.zeros(784, dtype=int)
            image[bright_pixel] = 1
            writer.writerow({"label": label, "image": str(image.tolist())})


class Exercise3DataTests(unittest.TestCase):
    def test_overlapping_files_are_deduplicated_before_split(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "old.csv"
            new = Path(directory) / "new.csv"
            write_digits(old, [(5, 100), (8, 200)])
            write_digits(new, [(8, 200), (8, 300)])
            images, labels, from_new, profile = load_learning_data(old, new)
        self.assertEqual(len(images), 3)
        self.assertEqual(profile["shared_rows"], 1)
        np.testing.assert_array_equal(labels, [8, 8, 5])
        np.testing.assert_array_equal(from_new, [True, True, False])

    def test_augmentation_keeps_labels_aligned_and_inside_training_pool(self):
        images = np.zeros((3, 784), dtype=np.float32)
        images[:, 14 * 28 + 14] = 1
        labels = np.array([5, 8, 2])
        pool = np.array([0, 1])
        indices = training_indices(pool, labels, balance_count=2, seed=2)
        data = make_training_data(images, labels, indices, shifts_per_image=2, seed=2)
        self.assertEqual(len(data), 12)
        expected = np.tile(labels[indices], 3)
        actual = np.array([np.argmax(target) for _, target in data])
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(all(np.sum(image) == 1 for image, _ in data))
        self.assertTrue(set(actual).issubset({5, 8}))


if __name__ == "__main__":
    unittest.main()
