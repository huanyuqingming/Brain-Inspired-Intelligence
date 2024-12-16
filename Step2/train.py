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

from CSNN import CSNN

def make_environment(images, labels, e):
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

def train():
    class parser():
        def __init__(self):
            self.T = 4
            self.device = 'cuda'
            self.b = 128
            self.epochs = 64
            self.j = 4
            self.data_dir = './fashionmnist'
            self.out_dir = 'logs'
            self.resume = 'None'
            self.amp = False
            self.cupy = False
            self.opt = 'Adam'
            self.momentum = 0.9
            self.lr = 0.001
            self.channels = 128
            self.save_es = None

    args = parser()
    print(args)

    net = CSNN(T=args.T, channels=args.channels, use_cupy=args.cupy)
    print(net)
    net.to(args.device)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=transform,
        download=True
    )

    train_set = make_environment(train_set.data, train_set.targets, 0.2)

    # 打印 train_set 的信息
    print(f"Number of images: {len(train_set['images'])}")
    print(f"Shape of one image: {train_set['images'][0].shape}")
    print(f"Label of first image: {train_set['labels'][0]}")

    train_data_loader = torch.utils.data.DataLoader(
        dataset=list(zip(train_set['images'], train_set['labels'])),
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    # 打印 train_data_loader 的长度
    print(f"Number of batches: {len(train_data_loader)}")

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}')

    if args.amp:
        out_dir += '_amp'

    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    for epoch in tqdm(range(start_epoch, args.epochs)):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in tqdm(train_data_loader):
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label.squeeze().long(), 10).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(img)
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label.squeeze()).float().sum().item()

            functional.reset_net(net)

        if train_samples == 0:
            raise ValueError("No training samples found. Please check your DataLoader and dataset.")

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        print(args)
        print(out_dir)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    ck_dir = "./checkpoints"
    if not os.path.exists(ck_dir):
        os.makedirs(ck_dir)
        print(f'Mkdir {ck_dir}.')
    # 保存最后一次的模型检查点
    checkpoint_path = os.path.join(ck_dir, 'checkpoint_final.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, checkpoint_path)
    print(f'Saved final checkpoint: {checkpoint_path}')

if __name__ == '__main__':
    train()