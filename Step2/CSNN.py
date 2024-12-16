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
import numpy as np
from PIL import Image

# 卷积脉冲神经网络定义
class CSNN(nn.Module):
    def __init__(self, T: int, channels: int, use_cupy=False):
        super().__init__()
        self.T = T  # SNN时间步长

        self.conv_fc = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),    # 普通卷积层，输入通道改为2
            layer.BatchNorm2d(channels),                                        # 普通BatchNormalization
            neuron.IFNode(surrogate_function=surrogate.ATan()),                 # IF脉冲节点
            layer.MaxPool2d(2, 2),  # 7 * 7                                     # 最大池化，输出大小改为7x7

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False), # 普通卷积层
            layer.BatchNorm2d(channels),                                        # 普通BatchNormalization
            neuron.IFNode(surrogate_function=surrogate.ATan()),                 # IF脉冲节点
            layer.MaxPool2d(2, 2),  # 3 * 3                                     # 最大池化，输出大小改为3x3

            layer.Flatten(),
            layer.Linear(channels * 3 * 3, channels * 2 * 2, bias=False),       # 普通线性层
            neuron.IFNode(surrogate_function=surrogate.ATan()),                 # IF脉冲节点

            layer.Linear(channels * 2 * 2, 10, bias=False),         
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')                           # 多步脉冲模式

        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]  # 将数据复制T份直接输入网络
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr
    
    def spiking_encoder(self):
        return self.conv_fc[0:3] # 脉冲节点即可将float输入编码为spike序列