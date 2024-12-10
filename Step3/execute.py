import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import PuzzleDataset
import annmodel
import snnmodel
import utils

def execute(segment, batch_size, T, net_type):
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

    # 加载训练和测试数据
    train_dataset = PuzzleDataset('./dataset', group='train', batch_size=batch_size, segment=segment)
    test_dataset = PuzzleDataset('./dataset', group='test', batch_size=batch_size, segment=segment)
    loader_train = DataLoader(train_dataset, batch_size=train_dataset.batch_size, shuffle=True)
    loader_test = DataLoader(test_dataset, batch_size=test_dataset.batch_size, shuffle=False)

    # 设置训练参数
    record_train, record_test = utils.train(
        model,
        (loader_train, loader_test),
        total_epoch=300,
        learning_rate=1e-3,
        scheduler=(10, 0.8),
        save_name=f'{segment}x{segment}'
    )

    # 绘制结果
    utils.plot(record_train, record_test, save_name=f'{segment}x{segment}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='执行模型训练和测试')
    parser.add_argument('--segment', type=int, default=2, help='拼图块分割数量')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--T', type=int, default=4, help='脉冲编码时间间隔')
    parser.add_argument('--net_type', type=str, default='snn', help='网络类型（\'snn\'/\'ann\'）')

    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        print('Using CUDA')
        torch.cuda.manual_seed(0)
    else:
        print('Using CPU')

    execute(args.segment, args.batch_size, args.T, args.net_type)
