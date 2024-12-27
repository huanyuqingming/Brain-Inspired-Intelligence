import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import PuzzleDataset
import annmodel
import snnmodel
import utils

def execute(segment=2, batch_size=128, T=4, net_type='snn', total_epoch=10, lr=1e-3, scheduler=(10, 0.8)):
    print(f'Solving {segment}x{segment} puzzle')
    if net_type == 'snn':
        print('Using SNN')
        model = snnmodel.PuzzleSolver(segment=segment, T=T)
    elif net_type == 'ann':
        print('Using ANN')
        model = annmodel.PuzzleSolver(segment=segment)
    else:
        raise ValueError('Invalid model type.')
    model = model.cuda() if torch.cuda.is_available() else model

    # 下载cifar10数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
    

    # 加载训练和测试数据
    train_dataset = PuzzleDataset('./datasets', group='train', batch_size=batch_size, segment=segment)
    test_dataset = PuzzleDataset('./datasets', group='test', batch_size=batch_size, segment=segment)
    loader_train = DataLoader(train_dataset, batch_size=train_dataset.batch_size, shuffle=True)
    loader_test = DataLoader(test_dataset, batch_size=test_dataset.batch_size, shuffle=False)

    # 设置训练参数
    record_train, record_test = utils.train(
        model,
        (loader_train, loader_test),
        total_epoch=total_epoch,
        learning_rate=lr,
        scheduler=scheduler,
        save_name=f'{net_type}_{segment}x{segment}'
    )

    # 绘制结果
    utils.plot(record_train, record_test, save_name=f'{net_type}_{segment}x{segment}')


def main(segment=2, batch_size=128, T=4, net_type='snn', total_epoch=10, lr=1e-3, scheduler=(10, 0.8)):

    if torch.cuda.is_available():
        print('Using CUDA')
    else:
        print('Using CPU')

    execute(segment, batch_size, T, net_type, total_epoch, lr, scheduler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--net_type', type=str, default='snn')
    parser.add_argument('--total_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scheduler', type=float, nargs=2, default=(10, 0.8))

    args = parser.parse_args()
    main(args.segment, args.batch_size, args.T, args.net_type, args.total_epoch, args.lr, args.scheduler)
