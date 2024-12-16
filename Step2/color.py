import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from spikingjelly import visualizing

from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader


class ColorMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.mnist = MNIST(root=root, train=train, download=download)
        self.transform = transform
        self.images, self.labels = self.mnist.data, self.mnist.targets

        # Apply make_environment to the entire dataset
        env = self.make_environment(self.images, self.labels, 0.2)
        self.images, self.labels = env['images'], env['labels']

    def make_environment(self, images, labels, e):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a - b).abs()  # Assumes both inputs are either 0 or 1
        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
        return {
            'images': (images.float() / 255.),
            'labels': labels[:, None]
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# 定义保存路径
save_dir = './mnist_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 定义数据转换
transform = transforms.Compose([
    transforms.ToPILImage()
])

# 读取彩色MNIST数据集
color_mnist = ColorMNIST(root='./data', train=True, transform=transform, download=True)

# 保存图像到本地
for i, (img, label) in enumerate(color_mnist):
    img.save(os.path.join(save_dir, f'color_mnist_{i}_label_{label.item()}.png'))

print(f'Saved {len(color_mnist)} images to {save_dir}')