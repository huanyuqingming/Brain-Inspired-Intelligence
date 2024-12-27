import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset import PuzzleDataset
import annmodel
import snnmodel
from torch.utils.data import DataLoader

def sample(model_segment, model_path, sample_height=4, sample_width=4, save_path='./figure', save_name='figure', net_type='snn'):
    # 加载数据集和模型
    dataset = PuzzleDataset(path='./datasets', group='test', batch_size=sample_height * sample_width, segment=model_segment)
    loader = DataLoader(dataset, batch_size=sample_height * sample_width, shuffle=False)
    if net_type == 'snn':
        model = snnmodel.DPN(segment=model_segment)
    elif net_type == 'ann':
        model = annmodel.DPN(segment=model_segment)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    # 获取一批测试数据
    images, _ = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(2).cpu().numpy()

    images = images.cpu().numpy()

    # 绘制并保存原始图像拼图
    plt.figure(figsize=(2 * sample_width, 2 * sample_height))
    for i in range(sample_height):
        for j in range(sample_width):
            position = i * sample_width + j
            image = np.transpose(images[position], (0, 2, 3, 1))
            pieces = []
            for u in range(model_segment):
                piece = []
                for v in range(model_segment):
                    index = u * model_segment + v
                    piece.append(image[index])
                pieces.append(np.concatenate(piece, axis=1))
            image = np.concatenate(pieces, axis=0)
            plt.subplot(sample_height, sample_width, position + 1)
            plt.imshow(image)
            plt.axis('off')
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{save_name}_sample_problem.png'))
    plt.show()
    plt.close()

    # 绘制并保存预测的拼图结果
    plt.figure(figsize=(2 * sample_width, 2 * sample_height))
    for i in range(sample_height):
        for j in range(sample_width):
            position = i * sample_width + j
            image, prediction = np.transpose(images[position], (0, 2, 3, 1)), np.argsort(predictions[position])
            pieces = []
            for u in range(model_segment):
                piece = []
                for v in range(model_segment):
                    index = u * model_segment + v
                    piece.append(image[prediction[index]])
                pieces.append(np.concatenate(piece, axis=1))
            image = np.concatenate(pieces, axis=0)
            plt.subplot(sample_height, sample_width, position + 1)
            plt.imshow(image)
            plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{save_name}_sample_solution.png'))
    plt.show()
    plt.close()