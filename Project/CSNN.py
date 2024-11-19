# %%
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
from torchvision.transforms import ToTensor, Lambda
import numpy as np

from tqdm import tqdm


class ColoredMNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(ColoredMNIST, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.color_transform = Lambda(lambda x: self.add_color(x))

    # def add_color(self, img):
    #     img = np.array(img)
    #     colored_img = np.stack([img, img, img], axis=-1)
    #     colored_img = colored_img / 255.0
    #     colored_img = torch.tensor(colored_img, dtype=torch.float32)
    #     return colored_img

    def add_color(self, img):
        img = np.array(img)
        
        # 随机生成一个 RGB 颜色
        color = np.random.rand(3)  # 生成一个 [0, 1) 范围内的随机颜色

        # 将灰度图像复制到 RGB 图像的三个通道，并乘以随机生成的颜色
        colored_img = np.stack([img, img, img], axis=-1)  # 转换为三通道
        colored_img = colored_img / 255.0  # 将像素值归一化到 [0, 1]
        
        # 为图像的每个通道乘上随机的颜色
        colored_img *= color  # 应用随机颜色

        # 转换为 PyTorch 张量
        colored_img = torch.tensor(colored_img, dtype=torch.float32)
        return colored_img

    def __getitem__(self, index):
        img, target = super(ColoredMNIST, self).__getitem__(index)
        img = self.color_transform(img)
        return img, target


#卷积脉冲神经网络定义
class CSNN(nn.Module):
    def __init__(self, T: int, channels: int, use_cupy=False):
        super().__init__()
        self.T = T  #SNN时间步长

        self.conv_fc = nn.Sequential(
        layer.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),    #普通卷积层
        layer.BatchNorm2d(channels),                                        #普通BatchNormalization
        neuron.IFNode(surrogate_function=surrogate.ATan()),                 # IF脉冲节点
        layer.MaxPool2d(2, 2),  # 14 * 14                                   # 最大池化

        layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False), #普通卷积层
        layer.BatchNorm2d(channels),                                        #普通BatchNormalization
        neuron.IFNode(surrogate_function=surrogate.ATan()),                 #IF 脉冲节点
        layer.MaxPool2d(2, 2),  # 7 * 7                                     # 最大池化

        layer.Flatten(),
        layer.Linear(channels * 7 * 7, channels * 4 * 4, bias=False),       #普通线性层
        neuron.IFNode(surrogate_function=surrogate.ATan()),                 #IF 脉冲节点

        layer.Linear(channels * 4 * 4, 10, bias=False),         
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')                       #多步脉冲模式

        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x = x.permute(0, 4, 2, 3, 1).squeeze(-1)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]  #将数据复制T份直接输入网络
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr
    
    def spiking_encoder(self):
        return self.conv_fc[0:3] #脉冲节点即可将float输入编码为spike序列

class SNNLeNet(nn.Module):
    def __init__(self, T: int, channels: int, use_cupy=False):
        super(SNNLeNet, self).__init__()
        self.T = T  #SNN时间步长
        

        self.LeNet = nn.Sequential(
            layer.Conv2d(3, 6, kernel_size=5, padding=2, bias=False),
            layer.BatchNorm2d(6),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(6, 16, kernel_size=5, padding=0, bias=False),
            layer.BatchNorm2d(16),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(16, 120, kernel_size=5, padding=0, bias=False),
            layer.Flatten(),

            layer.Linear(120, 84, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(84, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
        
        # 设置多步脉冲模式（每个脉冲节点的时间步长）
        functional.set_step_mode(self, step_mode='m')

        # 如果使用 cuPy，设置后端为 cuPy
        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x = x.permute(0, 4, 2, 3, 1).squeeze(-1)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]  #将数据复制T份直接输入网络
        x_seq = self.LeNet(x_seq)
        fr = x_seq.mean(0)
        return fr
    
    def spiking_encoder(self):
        return self.conv_fc[0:3] #脉冲节点即可将float输入编码为spike序列

class SNNAlexNet(nn.Module):
    def __init__(self, T: int, channels: int, use_cupy=False):
        super(SNNAlexNet, self).__init__()
        self.T = T  #SNN时间步长
        

        self.features = nn.Sequential(
            layer.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # 第1个卷积层，输出96个特征图
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=3, stride=2),  # 第1个最大池化层
 
            layer.Conv2d(96, 256, kernel_size=5, padding=2),  # 第2个卷积层
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=3, stride=2),  # 第2个最大池化层
 
            layer.Conv2d(256, 384, kernel_size=3, padding=1),  # 第3个卷积层
            neuron.IFNode(surrogate_function=surrogate.ATan()),
 
            layer.Conv2d(384, 384, kernel_size=3, padding=1),  # 第4个卷积层
            neuron.IFNode(surrogate_function=surrogate.ATan()),
 
            layer.Conv2d(384, 256, kernel_size=3, padding=1),  # 第5个卷积层
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=3, stride=2)  # 第3个最大池化层
        )

        self.classifier = nn.Sequential(
            layer.Dropout(0.5),  # Dropout层，防止过拟合
            layer.Linear(256 * 6 * 6, 4096),  # 全连接层，输入尺寸为 256*6*6，输出4096
            neuron.IFNode(surrogate_function=surrogate.ATan()),
 
            layer.Dropout(0.5),  # 第二个Dropout层
            layer.Linear(4096, 4096),  # 第二个全连接层
            neuron.IFNode(surrogate_function=surrogate.ATan()),
 
            layer.Linear(4096, 10),  # 最后一个全连接层，输出类别数
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
        
        # 设置多步脉冲模式（每个脉冲节点的时间步长）
        functional.set_step_mode(self, step_mode='m')

        # 如果使用 cuPy，设置后端为 cuPy
        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x = x.permute(0, 4, 2, 3, 1).squeeze(-1)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]  #将数据复制T份直接输入网络

        T, N, C, H, W = x_seq.shape
        x_seq = x_seq.view(T * N, C, H, W)
        x_seq = F.interpolate(x_seq, size=(227, 227), mode='bilinear', align_corners=False)
        x_seq = x_seq.view(T, N, C, 227, 227)

        x_seq = self.features(x_seq)
        x_seq = x_seq.view(x_seq.size(0), x_seq.size(1), -1)
        x_seq = self.classifier(x_seq)

        fr = x_seq.mean(0)
        return fr
    
    def spiking_encoder(self):
        return self.conv_fc[0:3] #脉冲节点即可将float输入编码为spike序列


def main():
    '''
    (sj-dev) wfang@Precision-5820-Tower-X-Series:~/spikingjelly_dev$ python -m spikingjelly.activation_based.examples.conv_fashion_mnist -h

    usage: conv_fashion_mnist.py [-h] [-T T] [-device DEVICE] [-b B] [-epochs N] [-j N] [-data-dir DATA_DIR] [-out-dir OUT_DIR]
                                 [-resume RESUME] [-amp] [-cupy] [-opt OPT] [-momentum MOMENTUM] [-lr LR]

    Classify Fashion-MNIST

    optional arguments:
      -h, --help          show this help message and exit
      -T T                simulating time-steps
      -device DEVICE      device
      -b B                batch size
      -epochs N           number of total epochs to run
      -j N                number of data loading workers (default: 4)
      -data-dir DATA_DIR  root dir of Fashion-MNIST dataset
      -out-dir OUT_DIR    root dir for saving logs and checkpoint
      -resume RESUME      resume from the checkpoint path
      -amp                automatic mixed precision training
      -cupy               use cupy neuron and multi-step forward mode
      -opt OPT            use which optimizer. SDG or Adam
      -momentum MOMENTUM  momentum for SGD
      -save-es            dir for saving a batch spikes encoded by the first {Conv2d-BatchNorm2d-IFNode}
    '''
    # python -m spikingjelly.activation_based.examples.conv_fashion_mnist -T 4 -device cuda:0 -b 128 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -cupy -opt sgd -lr 0.1 -j 8

    # python -m spikingjelly.activation_based.examples.conv_fashion_mnist -T 4 -device cuda:0 -b 4 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -cupy -opt sgd -lr 0.1 -j 8 -resume ./logs/T4_b256_sgd_lr0.1_c128_amp_cupy/checkpoint_latest.pth -save-es ./logs
    '''
    parser = argparse.ArgumentParser(description='Classify Fashion-MNIST')
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of Fashion-MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-save-es', default=None, help='dir for saving a batch spikes encoded by the first {Conv2d-BatchNorm2d-IFNode}')
    '''
    
    class parser():
        def __init__(self):
            self.T = 4
            self.device = 'cuda'
            self.b = 128
            self.epochs = 64
            self.j = 4
            self.data_dir='./mnist'
            self.out_dir='logs'
            self.resume = 'None'
            self.amp = False
            self.cupy = False
            self.opt = 'adam'
            self.momentum = 0.9
            self.lr = 1e-3
            self.channels = 128
            self.save_es = None
            self.mnist = "colored"
            self.net = "lenet"
    
    #args = parser.parse_args()
    args = parser()
    print(args)

    if args.net == "cnn":
        net = CSNN(T=args.T, channels=args.channels, use_cupy=args.cupy)
    elif args.net == "lenet":
        net = SNNLeNet(T=args.T, channels=args.channels, use_cupy=args.cupy)
    elif args.net == "alexnet":
        net = SNNAlexNet(T=args.T, channels=args.channels, use_cupy=args.cupy)
    else:
        raise NotImplementedError(args.net)

    print(net)

    net.to(args.device)
    
    '''
    载入数据，和普通DNN的训练步骤一样
    '''
    if args.mnist == 'normal':
        train_set = torchvision.datasets.FashionMNIST(
                root=args.data_dir,
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=True)

        test_set = torchvision.datasets.FashionMNIST(
                root=args.data_dir,
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True)

        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=args.b,
            shuffle=True,
            drop_last=True,
            num_workers=args.j,
            pin_memory=True
        )

        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.b,
            shuffle=True,
            drop_last=False,
            num_workers=args.j,
            pin_memory=True
        )
    elif args.mnist == 'colored':
        train_set = ColoredMNIST(
            root=args.data_dir,
            train=True,
            transform=ToTensor(),
            download=True
        )

        test_set = ColoredMNIST(
            root=args.data_dir,
            train=False,
            transform=ToTensor(),
            download=True
        )

        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=args.b,
            shuffle=True,
            drop_last=True,
            num_workers=args.j,
            pin_memory=True
        )

        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.b,
            shuffle=True,
            drop_last=False,
            num_workers=args.j,
            pin_memory=True
        )
    else:
        raise NotImplementedError(args.mnist)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    '''
    定义优化器，和普通DNN的训练步骤一样
    '''
    
    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
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
            optimizer.zero_grad()          #reset optimizer
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float() #one-hot encoding the label to a vector

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(img)
                    loss = F.mse_loss(out_fr, label_onehot) #calculate the MSE loss between the prediction and the vector
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
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


main()
