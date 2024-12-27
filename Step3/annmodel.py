import torch
import torch.nn as nn
import pygmtools
from pygmtools import sinkhorn
pygmtools.BACKEND = 'pytorch'


class CNN(nn.Module):
    def __init__(self, segment=2):
        super(CNN, self).__init__()
        self.segment, self.block = segment, segment**2
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.feature(x)


class FCN(nn.Module):
    def __init__(self, segment=2):
        super(FCN, self).__init__()
        self.segment, self.block = segment, segment**2
        self.network = nn.Sequential(
            nn.Linear(1024 * self.block, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.block ** 2),
        )

    def forward(self, x):
        return self.network(x)


class PuzzleSolver(nn.Module):
    def __init__(self, segment=2):
        super(PuzzleSolver, self).__init__()
        self.segment, self.block = segment, segment**2
        self.extractor = CNN(segment=self.segment)
        self.aggregator = FCN(segment=self.segment)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, *shape[2:])
        x = self.extractor(x)
        x = x.view(shape[0], -1)
        x = self.aggregator(x)
        x = x.view(-1, self.block, self.block)

        return x if self.training else sinkhorn(x)
