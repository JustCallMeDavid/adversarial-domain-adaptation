import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class DomainClassifier(nn.Module):
    def __init__(self, n_domains, in_size):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=in_size, out_features=3072)
        self.fc2 = nn.Linear(in_features=3072, out_features=2048)
        self.fc3 = nn.Linear(in_features=2048, out_features=n_domains)
        # self.drop = nn.Dropout2d(0.25)

    def forward(self, x):
        x = grad_reverse(x)
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FeatureExtractor(nn.Module):
    def __init__(self, dropout, feat_bn):
        super(FeatureExtractor, self).__init__()

        assert 0 <= dropout < 1, 'Dropout has to be a value between 0 and 1.'

        # 3 input channels (color image), 64 maps (i.e., output channels), 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=(2, 2))

        if feat_bn:
            self.bn_1 = nn.BatchNorm2d(64)
            self.bn_2 = nn.BatchNorm2d(64)
            self.bn_3 = nn.BatchNorm2d(128)
        else:
            self.bn_1 = None
            self.bn_2 = None
            self.bn_3 = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)

        if self.bn_1 is not None:
            x = F.max_pool2d(input=F.relu(self.bn_1(self.conv1(x))), kernel_size=(3, 3), stride=(2, 2))
            x = F.max_pool2d(input=F.relu(self.bn_2(self.conv2(x))), kernel_size=(3, 3), stride=(2, 2))
            return F.relu(self.bn_3(self.conv3(x)))
        else:
            x = F.max_pool2d(input=F.relu(self.conv1(x)), kernel_size=(3, 3), stride=(2, 2))
            x = F.max_pool2d(input=F.relu(self.conv2(x)), kernel_size=(3, 3), stride=(2, 2))
            return F.relu(self.conv3(x))


class TaskClassifier(nn.Module):
    def __init__(self, in_size):
        super(TaskClassifier, self).__init__()
        self.tc_fc1 = nn.Linear(in_features=in_size, out_features=3072)
        self.tc_fc2 = nn.Linear(in_features=3072, out_features=2048)
        self.tc_fc3 = nn.Linear(in_features=2048, out_features=10)

    def forward(self, x):
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.tc_fc1(x))
        x = F.relu(self.tc_fc2(x))
        x = self.tc_fc3(x)
        return x


class TaskPipeline(nn.Module):
    def __init__(self, feature_extractor, task_classifier):
        super(TaskPipeline, self).__init__()
        self.feature_extractor = feature_extractor
        self.task_classifier = task_classifier

    def forward(self, x):
        return self.task_classifier(self.feature_extractor(x))
