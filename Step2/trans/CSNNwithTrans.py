import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

from spikingjelly.activation_based import layer
 
# 设置设备
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
 
# 超参数
batch_size = 128
num_classes = 10
embedding_dim = 128
num_heads = 16
num_layers = 3
hidden_dim = 256
dropout_rate = 0.1
learning_rate = 0.001
num_epochs = 10
print(f"embedding_dim={embedding_dim},num_heads={num_heads}")
print(f"num_layers={num_layers},hidden_dim={hidden_dim}")

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

# 数据预处理：定义图像的转换
# 将图像尺寸变为1x28x28，并标准化至[-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1, 1]之间
])
 
# 加载 MNIST 数据集

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

train_dataset = make_environment(train_dataset.data, train_dataset.targets, 0.2)

# 打印 train_dataset 的信息
print(f"Number of images: {len(train_dataset['images'])}")
print(f"Shape of one image: {train_dataset['images'][0].shape}")
print(f"Label of first image: {train_dataset['labels'][0]}")

train_data_loader = torch.utils.data.DataLoader(
    dataset=list(zip(train_dataset['images'], train_dataset['labels'])),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=True
)

test_set = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

test_set = make_environment(test_set.data, test_set.targets, 0.2)

test_data_loader = torch.utils.data.DataLoader(
    dataset=list(zip(test_set['images'], test_set['labels'])),
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=4,
    pin_memory=True
)

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
 
 
 
# 定义 Positional Encoding 用于加入位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=28 * 28):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
 
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].to(x.device)
 
 
# 定义 Transformer 模型
class TransformerClassifier(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, hidden_dim, num_classes, dropout_rate):
        super(TransformerClassifier, self).__init__()
        self.embedding = layer.Linear(28 * 28, embedding_dim)  # 将28x28的图像像素展平并嵌入embedding_dim维度
        self.position_encoding = PositionalEncoding(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   dropout=dropout_rate,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)  # 分类器
 
    def forward(self, x):
        # 将图像展平成1D张量并进行嵌入
        x = x.view(x.size(0), -1)  # x shape: [batch_size, 28*28]
        x = self.embedding(x).unsqueeze(1)  # 添加一个长度为1的序列维度
 
        # 加入位置编码
        x = self.position_encoding(x)
 
        # 通过Transformer编码层
        x = self.transformer_encoder(x)
 
        # 取出最后一个编码后的特征向量
        x = x.mean(dim=1)  # 平均池化得到一个全局表示
 
        # 分类
        return self.fc(x)
 
 
# 实例化模型，损失函数和优化器
model = TransformerClassifier(embedding_dim, num_heads, num_layers, hidden_dim, num_classes, dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
 
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
 
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item()
 
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
 
# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
print(f'Accuracy of the model on the test images: {100 * correct / total} %')
 
# 创建checkpoint文件夹
os.makedirs('checkpoint', exist_ok=True)

# 保存模型
torch.save(model.state_dict(), 'checkpoint/transformer_mnist.pth')