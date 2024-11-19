import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import PuzzleDataset  # 假设PuzzleDataset已转换为PyTorch
from model import PuzzleSolver  # 假设PuzzleSolver已转换为PyTorch
import utils


class Logger:
    def __init__(self, path='./output/output.txt'):
        self.terminal = sys.stdout
        self.file = open(path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        pass


def execute(segment):
    # sys.stdout = Logger('./output/output.txt')

    print(f'<experiment> {segment}x{segment} puzzle')
    model = PuzzleSolver(segment=segment)
    model = model.cuda() if torch.cuda.is_available() else model  # 将模型加载到GPU

    # 加载训练和测试数据
    train_dataset = PuzzleDataset('./dataset', group='train', batch_size=256, segment=segment)
    test_dataset = PuzzleDataset('./dataset', group='test', batch_size=256, segment=segment)
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
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        print('CUDA is available.')
        torch.cuda.manual_seed(0)

    segment = 2
    execute(segment)
