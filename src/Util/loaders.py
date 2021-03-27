import torch
from .PreprocessingScripts import *
from typing import Dict
from itertools import cycle


class PtLoader:
    def __init__(self, data: DigitDataset, batch_size, drop_last):
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array([i.image for i in data.train_set])),
                                                       torch.tensor(np.array([i.label for i in data.train_set])))
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                        num_workers=0, drop_last=drop_last)
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array([i.image for i in data.test_set])),
                                                      torch.tensor(np.array([i.label for i in data.test_set])))
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                       drop_last=drop_last)


class PtLoadersByTrainClass:

    def __init__(self, data: Dataset, batch_size, drop_last):
        self.train_loader_cls = {}
        for d in data.train_set:
            if d.label not in self.train_loader_cls.keys():
                self.train_loader_cls[d.label] = Dataset([], None)
            self.train_loader_cls[d.label].train_set.append(d)

        for k in self.train_loader_cls.keys():
            self.train_loader_cls[k] = torch.utils.data.TensorDataset(
                torch.tensor(np.array([i.image for i in self.train_loader_cls[k].train_set])),
                torch.tensor(np.array([i.label for i in self.train_loader_cls[k].train_set])))
            self.train_loader_cls[k] = torch.utils.data.DataLoader(self.train_loader_cls[k], batch_size=batch_size,
                                                                   shuffle=True,
                                                                   num_workers=0, drop_last=drop_last)
            self.train_loader_cls[k] = cycle(self.train_loader_cls[k])
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array([i.image for i in data.test_set])),
                                                      torch.tensor(np.array([i.label for i in data.test_set])))
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                       drop_last=drop_last)


class MultiPtLoader:
    def __init__(self, datasets: Dict[str, DigitDataset], batch_size, drop_last):
        self.num_labels = {}

        # create numieric labels for dict items
        iter = 0
        for key in datasets.keys():
            self.num_labels[key] = iter
            iter = iter + 1

        train_sets = []
        train_sets_source = []
        test_sets = []
        test_sets_source = []
        for dataset in datasets.keys():
            train_sets.extend(datasets[dataset].train_set)
            train_sets_source.extend([self.num_labels[dataset]] * len(datasets[dataset].train_set))
        for dataset in datasets.keys():
            test_sets.extend(datasets[dataset].test_set)
            test_sets_source.extend([self.num_labels[dataset]] * len(datasets[dataset].test_set))

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array([i.image for i in train_sets])),
                                                       torch.tensor(np.array([i.label for i in train_sets])),
                                                       torch.tensor(np.array(train_sets_source)))
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                        num_workers=0, drop_last=drop_last)

        test_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array([i.image for i in test_sets])),
                                                      torch.tensor(np.array([i.label for i in test_sets])),
                                                      torch.tensor(np.array(test_sets_source)))
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                       drop_last=drop_last)
