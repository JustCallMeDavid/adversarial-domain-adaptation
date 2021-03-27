import pickle
from typing import List
import numpy as np


class Image:
    def __init__(self, image, label):
        self.image = image
        self.label = label


class DigitImage(Image):
    def __init__(self, image, label):
        super(DigitImage, self).__init__(image, label)


class CifarImage(Image):
    def __init__(self, image, label):
        super(CifarImage, self).__init__(image, label)


class Dataset:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set


class DigitDataset(Dataset):
    def __init__(self, train_set: List[DigitImage], test_set: List[DigitImage]):
        super(DigitDataset, self).__init__(train_set, test_set)


class CifarDataset(Dataset):
    def __init__(self, train_set: List[CifarImage], test_set: List[CifarImage]):
        super(CifarDataset, self).__init__(train_set, test_set)


class Cifar100Image(CifarImage):
    def __init__(self, fine_label: int, coarse_label: int, image):
        super().__init__(image, fine_label)
        self.coarse_label = coarse_label

    def __str__(self):
        return f'FineLabel {self.label} CoarseLabel {self.coarse_label}'


class Cifar10Image(CifarImage):
    def __init__(self, label, image):
        super().__init__(image, label)

    def __str__(self):
        return f'Label {self.label}'


def persist_dataset_to_pickle(dataset: DigitDataset, path):
    with open(path, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset_from_pickle(path) -> DigitDataset:
    with open(path, 'rb') as f:
        return pickle.load(f)
