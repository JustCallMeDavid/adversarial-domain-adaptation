from .loaders import *
import numpy as np
import copy


def prepare_mnist(mnist_data: DigitDataset):
    for instance in mnist_data.train_set + mnist_data.test_set:
        instance.image = np.repeat(np.pad(instance.image, pad_width=[(2, 2), (2, 2)],
                                          mode='constant', constant_values=0)[:, :, np.newaxis], 3, axis=2)
        assert instance.image.shape == (32, 32, 3)


def prepare_usps(usps_data: DigitDataset):
    for instance in usps_data.train_set + usps_data.test_set:
        instance.image = np.repeat(np.pad(instance.image, pad_width=[(8, 8), (8, 8)],
                                          mode='constant', constant_values=0)[:, :, np.newaxis], 3, axis=2)
        assert instance.image.shape == (32, 32, 3)


def prepare_emnist(emnist_data: DigitDataset):
    for instance in emnist_data.train_set + emnist_data.test_set:
        instance.image = np.repeat(np.pad(instance.image, pad_width=[(2, 2), (2, 2)],
                                          mode='constant', constant_values=0)[:, :, np.newaxis], 3, axis=2)
        assert instance.image.shape == (32, 32, 3)


def test_model(model, data, device, data_name):
    """takes a model and tests it on the test section of the specified dataset, prints the final accuracy"""

    # to make sure that no unforeseen side-effects occur
    model = copy.deepcopy(model)
    # sets the model in eval mode (important for dropout + batch_norm)
    model.eval()

    ignore_domain_labels = False
    if type(data) is MultiPtLoader:
        ignore_domain_labels = True

    correct = 0
    total = 0
    with torch.no_grad():
        for idx, data in enumerate(data):
            if ignore_domain_labels:
                inputs, labels, _ = data
            else:
                inputs, labels = data
            inputs = inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the {total} test images from {data_name}: {correct / total}')


def prepare_data(dataset: DigitDataset, data_name: str):
    convert_image_scale(dataset)
    if 'mnist' in data_name and not 'emnist' in data_name and not 'mnist_m' in data_name:
        prepare_mnist(dataset)
    elif 'usps' in data_name:
        prepare_usps(dataset)
    elif 'emnist' in data_name:
        prepare_emnist(dataset)
    else:
        print(f'No preparation necessary for {data_name}.')


def compute_image_mean_and_std(dataset: DigitDataset):
    train_imgs = [i.image for i in dataset.train_set]
    test_imgs = [i.image for i in dataset.test_set]

    train_means = tuple(np.squeeze(np.array(train_imgs).mean(axis=(0, -3, -2), keepdims=1)))
    test_means = tuple(np.squeeze(np.array(test_imgs).mean(axis=(0, -3, -2), keepdims=1)))
    train_stds = tuple(np.squeeze(np.array(train_imgs).std(axis=(0, -3, -2), keepdims=1)))
    test_stds = tuple(np.squeeze(np.array(test_imgs).std(axis=(0, -3, -2), keepdims=1)))

    return (train_means, train_stds), (test_means, test_stds)


def convert_image_scale(data: DigitDataset):
    for i in data.train_set:
        i.image = i.image / 255
    for i in data.test_set:
        i.image = i.image / 255


def zeroone_normalize_dataset(data: DigitDataset):
    for i in data.train_set:
        i.image = np.divide(np.subtract(i.image, 0.5), 0.5)
    for i in data.test_set:
        i.image = np.divide(np.subtract(i.image, 0.5), 0.5)


def stdmean_normalize_dataset(data: DigitDataset):
    tr, te = compute_image_mean_and_std(dataset=data)

    for i in data.train_set:
        i.image = np.divide(np.subtract(i.image, tr[0]), tr[1])
    for i in data.test_set:
        i.image = np.divide(np.subtract(i.image, te[0]), te[1])


def add_noise(inputs, factor):
    noise = torch.randn_like(inputs) * factor
    return inputs + noise