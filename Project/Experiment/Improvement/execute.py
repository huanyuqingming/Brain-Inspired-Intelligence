import sys
import numpy as np
import jittor as jt
from dataset import PuzzleDataset
from model import PuzzleSolver
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

    # print('<experiment> 2x2 puzzle')
    # model = PuzzleSolver(segment=2)
    # loader_train = PuzzleDataset('./dataset', group='train', batch=256, segment=2)
    # loader_test = PuzzleDataset('./dataset', group='test', batch=256, segment=2)
    # record_train, record_test = utils.train(model, (loader_train, loader_test), total_epoch=100, learning_rate=1e-3, scheduler=(10, 0.8), save_name='2x2')
    # utils.plot(record_train, record_test, save_name='2x2')

    # print('<experiment> 3x3 puzzle')
    # model = PuzzleSolver(segment=3)
    # loader_train = PuzzleDataset('./dataset', group='train', batch=256, segment=3)
    # loader_test = PuzzleDataset('./dataset', group='test', batch=256, segment=3)
    # record_train, record_test = utils.train(model, (loader_train, loader_test), total_epoch=100, learning_rate=1e-3, scheduler=(10, 0.8), save_name='3x3')
    # utils.plot(record_train, record_test, save_name='3x3')

    print(f'<experiment> {segment}x{segment} puzzle')
    model = PuzzleSolver(segment=segment)
    loader_train = PuzzleDataset('./dataset', group='train', batch=256, segment=segment)
    loader_test = PuzzleDataset('./dataset', group='test', batch=256, segment=segment)
    record_train, record_test = utils.train(model, (loader_train, loader_test), total_epoch=300, learning_rate=1e-3, scheduler=(10, 0.8), save_name=f'{segment}x{segment}')
    utils.plot(record_train, record_test, save_name=f'{segment}x{segment}')


if __name__ == '__main__':
    np.random.seed(0)
    jt.set_global_seed(0)
    jt.flags.use_cuda = 1
    segment = 2
    execute(segment)
