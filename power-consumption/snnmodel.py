import torch
import torch.nn as nn
import pygmtools
from pygmtools import sinkhorn
from spikingjelly.activation_based import neuron, functional, surrogate, layer

pygmtools.BACKEND = 'pytorch'


class Naive_CNN(nn.Module):
    def __init__(self, segment=2, T=8):
        super(Naive_CNN, self).__init__()
        self.segment, self.block = segment, segment**2
        self.T = T
        
        self.feature = nn.Sequential(
            layer.Conv2d(3, 32, 3, 1, 1),
            layer.BatchNorm2d(32),
            neuron.LIFNode(),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(32, 32, 3, 1, 1),
            layer.BatchNorm2d(32),
            neuron.LIFNode(),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(32, 64, 3, 1, 1),
            layer.BatchNorm2d(64),
            neuron.LIFNode(),
            layer.MaxPool2d(2, 2),
            layer.Flatten(),
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        # print("cnn:")
        # print(x.shape)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        # print(x_seq.shape)

        for i, layer in enumerate(self.feature):
            # print("layer: ", i, layer.__class__.__name__)
            x_seq = layer(x_seq)
            # print(x_seq.shape)
        out = x_seq
        return out.mean(dim=0)


class Naive_FCN(nn.Module):
    def __init__(self, segment=2, T=8):
        super(Naive_FCN, self).__init__()
        self.segment, self.block = segment, segment**2
        self.T = T
        
        self.network = nn.Sequential(
            layer.Linear(1024 * self.block, 4096),
            layer.BatchNorm1d(4096),
            neuron.LIFNode(),
            layer.Linear(4096, 1024),
            layer.BatchNorm1d(1024),
            neuron.LIFNode(),
            layer.Linear(1024, self.block ** 2),
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        # print("fcn:")
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1)
        # print(x_seq.shape)

        for i, layer in enumerate(self.network):
            # print("layer: ", i, layer.__class__.__name__)
            if layer.__class__.__name__ == 'BatchNorm1d':
                x_seq = x_seq.unsqueeze(-1)
                # print(x_seq.shape)
                x_seq = layer(x_seq)
                x_seq = x_seq.squeeze(-1)
                # print(x_seq.shape)
                continue
            x_seq = layer(x_seq)
            # print(x_seq.shape)
        out = x_seq
        return out.mean(dim=0)


class PuzzleSolver(nn.Module):
    def __init__(self, segment=2, T=8):
        super(PuzzleSolver, self).__init__()
        self.segment, self.block = segment, segment**2
        self.extractor = Naive_CNN(segment=self.segment, T=T)
        self.aggregator = Naive_FCN(segment=self.segment, T=T)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        shape = x.shape
        # print("puzzle:")
        # print(shape)
        x = x.view(-1, *shape[2:])
        x = self.extractor(x)
        x = x.view(shape[0], -1)
        x = self.aggregator(x)
        x = x.view(-1, self.block, self.block)

        return x if self.training else sinkhorn(x)
