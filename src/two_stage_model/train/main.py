import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.models as models
import numpy as np
import os

import random
import time

from dataset import LeakingDataset
from utils import plot_images, calculate_topk_accuracy, epoch_time
from resnet import ResNet
from config import Config

from tqdm import tqdm

random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
torch.cuda.manual_seed(Config.SEED)
torch.backends.cudnn.deterministic = True

train_data = LeakingDataset(root = Config.train_dir,
                                  transforms_label = Config.transforms_label_train)

valid_data = LeakingDataset(root = Config.val_dir,
                                  transforms_label = Config.transforms_label_train)

test_data = LeakingDataset(root = Config.test_dir,
                                 transforms_label = Config.transforms_label_test)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

train_iterator = data.DataLoader(train_data,
                                 shuffle = True,
                                 batch_size = Config.BATCH_SIZE)

valid_iterator = data.DataLoader(valid_data,
                                 batch_size = Config.BATCH_SIZE)

test_iterator = data.DataLoader(test_data,
                                batch_size = Config.BATCH_SIZE)

images, labels = zip(*[(image, label) for image, label in
                           [train_data[i] for i in range(Config.start_idx_visualize, Config.start_idx_visualize+Config.n_images_visualize)]])

classes = test_data.classes
plot_images(images, labels, classes)

pretrained_model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
IN_FEATURES = pretrained_model.fc.in_features
OUTPUT_DIM = len(test_data.classes)
fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
pretrained_model.fc = fc

model = ResNet(Config.resnet50_config, OUTPUT_DIM)
model.load_state_dict(pretrained_model.state_dict())

weight = model.conv1.weight.clone()
model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    model.conv1.weight[:, :3] = weight
    model.conv1.weight[:, 3] = model.conv1.weight[:, 0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

optimizer = optim.Adam([
                        {'params': model.conv1.parameters(), 'lr': Config.init_lr / 10},
                        {'params': model.bn1.parameters(), 'lr': Config.init_lr / 10},
                        {'params': model.layer1.parameters(), 'lr': Config.init_lr / 8},
                        {'params': model.layer2.parameters(), 'lr': Config.init_lr / 6},
                        {'params': model.layer3.parameters(), 'lr': Config.init_lr / 4},
                        {'params': model.layer4.parameters(), 'lr': Config.init_lr / 2},
                        {'params': model.fc.parameters()}
                        ], lr = Config.init_lr)
MAX_LRS = [p['lr'] for p in optimizer.param_groups]
scheduler = lr_scheduler.OneCycleLR(optimizer,
                                    max_lr = MAX_LRS,
                                    total_steps = int(Config.EPOCHS * len(train_iterator)))

def calculate_topk_accuracy(y_pred, y, k=2):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k

def train(model, iterator, optimizer, criterion, scheduler, device):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0

    model.train()

    for (x, y) in tqdm(iterator):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred, _ = model(x)
        loss = criterion(y_pred, y)
        acc_1, _ = calculate_topk_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0:
          scheduler.step()

        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()

    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)

    return epoch_loss, epoch_acc_1


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)
            y_pred, _ = model(x)
            loss = criterion(y_pred, y)
            acc_1, _ = calculate_topk_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()

    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)

    return epoch_loss, epoch_acc_1

best_valid_loss = float('inf')
save_path = os.path.join(Config.save_dir, Config.model_name)
os.makedirs(Config.save_dir, exist_ok=True)

for epoch in (range(Config.EPOCHS)):
    start_time = time.monotonic()
    train_loss, train_acc_1 = train(model, train_iterator, optimizer, criterion, scheduler, device)
    valid_loss, valid_acc_1 = evaluate(model, valid_iterator, criterion, device)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), save_path)
    end_time = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | ')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | ')

model.load_state_dict(torch.load(save_path))
test_loss, test_acc_1 = evaluate(model, test_iterator, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1*100:6.2f}% | ')
