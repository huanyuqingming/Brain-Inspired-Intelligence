import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
import torchvision.transforms.functional as F

def load_data(path, group='train'):
    image_list = []
    if group == 'train':
        for i in range(1, 6):
            filename = os.path.join(path, 'data_batch_{}'.format(i))
            with open(filename, 'rb') as file:
                data = pickle.load(file, encoding='bytes')
            image_list.append(np.array(data[b'data']))
    elif group == 'test':
        filename = os.path.join(path, 'test_batch')
        with open(filename, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
        image_list.append(np.array(data[b'data']))
    return np.concatenate(image_list, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

class PuzzleDataset(Dataset):
    def __init__(self, path, group='train', batch_size=64, segment=2):
        self.path = os.path.join(path, 'cifar-10-batches-py')
        self.group, self.batch_size, self.segment = group, batch_size, segment
        self.image = load_data(self.path, self.group)
        if group == 'test':
            k = len(self.image) // self.batch_size
            self.image = self.image[:k * self.batch_size]
        self.transform = Compose([Resize([32 * self.segment, 32 * self.segment]), ToTensor()])

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, index):
        raw = self.transform(F.to_pil_image(self.image[index]))
        cut = []
        for i in range(self.segment):
            for j in range(self.segment):
                cut.append(raw[:, 32 * i: 32 * (i + 1), 32 * j: 32 * (j + 1)])
        image = torch.stack(cut, dim=0)
        label = torch.randperm(self.segment ** 2)
        return image[label], label