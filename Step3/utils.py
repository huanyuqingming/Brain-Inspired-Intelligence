import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math
from spikingjelly.activation_based import functional

def train(model, loader, total_epoch=10, learning_rate=1e-3, loss_function=None, optimizer=None, scheduler=None, save_path='./model', save_name='model'):
    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if scheduler is not None:
        step_size, decay_rate = scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay_rate)
    
    loader_train, loader_test = loader
    record_train, record_test = [], []

    for epoch in range(1, total_epoch + 1):

        model.train()
        epoch_loss, fragment_count, puzzle_count = 0, 0, 0
        rounds = math.ceil(len(loader_train.dataset) / loader_train.batch_size)

        pbar = tqdm(loader_train, total=rounds, desc=f'Train {epoch}/{total_epoch}')
        for images, labels in pbar:
            images, labels = images.to(model.device), labels.to(model.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs.view(-1, model.block), labels.view(-1))
            loss.backward()
            optimizer.step()
            functional.reset_net(model)
            
            correct = (outputs.argmax(dim=2) == labels).sum(dim=1)
            epoch_loss += loss.item()
            fragment_count += correct.sum().item()
            puzzle_count += (correct == model.block).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss /= rounds
        fragment_accuracy = fragment_count / (model.block * len(loader_train.dataset))
        puzzle_accuracy = puzzle_count / len(loader_train.dataset)
        record_train.append([epoch_loss, fragment_accuracy, puzzle_accuracy])
        print(f'total loss: {epoch_loss:.4f}, fragment accuracy: {fragment_accuracy:.2%}, puzzle accuracy: {puzzle_accuracy:.2%}')

        model.eval()
        epoch_loss, fragment_count, puzzle_count = 0, 0, 0
        rounds = math.ceil(len(loader_test.dataset) / loader_test.batch_size)

        with torch.no_grad():
            pbar = tqdm(loader_test, total=rounds, desc=f'Test {epoch}/{total_epoch}')
            for images, labels in pbar:
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = model(images)
                loss = loss_function(outputs.view(-1, model.block), labels.view(-1))
                
                correct = (outputs.argmax(dim=2) == labels).sum(dim=1)
                epoch_loss += loss.item()
                fragment_count += correct.sum().item()
                puzzle_count += (correct == model.block).sum().item()

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss /= rounds
        fragment_accuracy = fragment_count / (model.block * len(loader_test.dataset))
        puzzle_accuracy = puzzle_count / len(loader_test.dataset)
        record_test.append([epoch_loss, fragment_accuracy, puzzle_accuracy])
        print(f'total loss: {epoch_loss:.4f}, fragment accuracy: {fragment_accuracy:.2%}, puzzle accuracy: {puzzle_accuracy:.2%}')

        if scheduler is not None:
            scheduler.step()

    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f'{save_name}_epoch_{total_epoch}.pth'))
    return np.array(record_train), np.array(record_test)


def plot(record_train, record_test, save_path='./figure', save_name='figure'):
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(record_train[:, 0], label='train')
    plt.plot(record_test[:, 0], label='test')
    plt.xlabel('epoch')
    plt.ylabel('epoch loss')
    plt.title('Epoch Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f'{save_name}_epoch_loss.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(record_train[:, 1], label='train')
    plt.plot(record_test[:, 1], label='test')
    plt.xlabel('epoch')
    plt.ylabel('fragment accuracy')
    plt.title('Fragment Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f'{save_name}_fragment_accuracy.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(record_train[:, 2], label='train')
    plt.plot(record_test[:, 2], label='test')
    plt.xlabel('epoch')
    plt.ylabel('puzzle accuracy')
    plt.title('Puzzle Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f'{save_name}_puzzle_accuracy.png'))
    plt.close()
