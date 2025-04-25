!pip -q install torch torchvision tqdm  

import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os, time, math, random

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

seed = 42
torch.manual_seed(seed);  random.seed(seed);  torch.cuda.manual_seed_all(seed)

mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_tfms  = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean, std)
])

train_ds = datasets.CIFAR10(root="~/data", train=True,  download=True, transform=train_tfms)
test_ds  = datasets.CIFAR10(root="~/data", train=False, download=True, transform=test_tfms)

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)



def resnet50_cifar10():
    m = models.resnet50(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(2048, 10)
    return m

model = resnet50_cifar10().to(device)
optimiser = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                      weight_decay=5e-4, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=50)
scaler = torch.cuda.amp.GradScaler()

def run(model, loader, train=False):
    if train: model.train()
    else:     model.eval()
    total, correct, running_loss = 0, 0, 0
    pbar = tqdm(loader, leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(), torch.set_grad_enabled(train):
            logits = model(x);  loss = nn.CrossEntropyLoss()(logits, y)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimiser); scaler.update(); optimiser.zero_grad()
        running_loss += loss.item() * y.size(0)
        _, pred = logits.max(1)
        correct += pred.eq(y).sum().item()
        total   += y.size(0)
        pbar.set_description(f'{"Train" if train else "Test "} '
                             f'loss {running_loss/total:.3f} '
                             f'acc {100*correct/total:.2f}%')
    return running_loss / total, 100 * correct / total

best_acc = 0
for epoch in range(1, 51):
    t0 = time.time()
    train_loss, train_acc = run(model, train_dl, train=True)
    test_loss,  test_acc  = run(model, test_dl,  train=False)
    scheduler.step()
    dt = time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))

    print(f'Epoch {epoch:02d}/50 | '
          f'train_acc {train_acc:.2f}% | test_acc {test_acc:.2f}% | {dt}')
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "resnet50_cifar10_best.pth")

print(f'Best test accuracy: {best_acc:.2f}%')

model = resnet50_cifar10().to(device)
model.load_state_dict(torch.load("resnet50_cifar10_best.pth"))
model.eval()


'''
100%|██████████| 170M/170M [00:03<00:00, 44.1MB/s]
<ipython-input-10-2ba129fcf1b8>:45: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler()
  0%|          | 0/391 [00:00<?, ?it/s]<ipython-input-10-2ba129fcf1b8>:57: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(), torch.set_grad_enabled(train):
Epoch 01/50 | train_acc 17.31% | test_acc 24.86% | 00:01:04
Epoch 02/50 | train_acc 28.80% | test_acc 36.34% | 00:01:05
Epoch 03/50 | train_acc 37.50% | test_acc 39.91% | 00:01:05
Epoch 04/50 | train_acc 44.16% | test_acc 48.24% | 00:01:04
Epoch 05/50 | train_acc 51.25% | test_acc 55.48% | 00:01:05
Epoch 06/50 | train_acc 59.02% | test_acc 63.39% | 00:01:05
Epoch 07/50 | train_acc 64.14% | test_acc 63.60% | 00:01:05
Epoch 08/50 | train_acc 67.97% | test_acc 64.14% | 00:01:05
Epoch 09/50 | train_acc 71.29% | test_acc 67.46% | 00:01:05
Epoch 10/50 | train_acc 74.88% | test_acc 67.64% | 00:01:05
Epoch 11/50 | train_acc 77.32% | test_acc 74.50% | 00:01:05
Epoch 12/50 | train_acc 78.81% | test_acc 68.84% | 00:01:05
Epoch 13/50 | train_acc 79.89% | test_acc 76.26% | 00:01:05
Epoch 14/50 | train_acc 80.83% | test_acc 76.24% | 00:01:05
Epoch 15/50 | train_acc 81.63% | test_acc 76.84% | 00:01:05
Epoch 16/50 | train_acc 82.71% | test_acc 80.08% | 00:01:04
Epoch 17/50 | train_acc 83.21% | test_acc 77.61% | 00:01:05
Epoch 18/50 | train_acc 83.94% | test_acc 79.51% | 00:01:05
Epoch 19/50 | train_acc 84.30% | test_acc 78.73% | 00:01:05
Epoch 20/50 | train_acc 84.92% | test_acc 81.87% | 00:01:05
Epoch 21/50 | train_acc 85.32% | test_acc 81.43% | 00:01:05
Epoch 22/50 | train_acc 85.97% | test_acc 83.37% | 00:01:06
Epoch 23/50 | train_acc 86.37% | test_acc 83.65% | 00:01:05
Epoch 24/50 | train_acc 87.05% | test_acc 81.14% | 00:01:05
Epoch 25/50 | train_acc 87.55% | test_acc 84.34% | 00:01:05
Epoch 26/50 | train_acc 88.08% | test_acc 82.38% | 00:01:05
Epoch 27/50 | train_acc 88.68% | test_acc 85.18% | 00:01:05
Epoch 28/50 | train_acc 89.25% | test_acc 84.76% | 00:01:05
Epoch 29/50 | train_acc 89.48% | test_acc 85.76% | 00:01:05
Epoch 30/50 | train_acc 90.27% | test_acc 87.52% | 00:01:05
Epoch 31/50 | train_acc 90.66% | test_acc 85.85% | 00:01:05
Epoch 32/50 | train_acc 91.56% | test_acc 86.62% | 00:01:05
Epoch 33/50 | train_acc 91.77% | test_acc 87.79% | 00:01:05
Epoch 34/50 | train_acc 92.38% | test_acc 88.27% | 00:01:05
Epoch 35/50 | train_acc 93.05% | test_acc 89.57% | 00:01:05
Epoch 36/50 | train_acc 93.67% | test_acc 89.09% | 00:01:05
Epoch 37/50 | train_acc 94.32% | test_acc 89.62% | 00:01:04
Epoch 38/50 | train_acc 95.07% | test_acc 90.22% | 00:01:05
Epoch 39/50 | train_acc 95.73% | test_acc 91.19% | 00:01:05
Epoch 40/50 | train_acc 96.32% | test_acc 91.14% | 00:01:05
Epoch 41/50 | train_acc 96.95% | test_acc 91.78% | 00:01:05
Epoch 42/50 | train_acc 97.57% | test_acc 91.82% | 00:01:04
Epoch 43/50 | train_acc 98.15% | test_acc 91.64% | 00:01:05
Epoch 44/50 | train_acc 98.49% | test_acc 92.92% | 00:01:05
Epoch 45/50 | train_acc 98.88% | test_acc 93.03% | 00:01:05
Epoch 46/50 | train_acc 99.10% | test_acc 93.04% | 00:01:05
Epoch 47/50 | train_acc 99.34% | test_acc 93.32% | 00:01:05
Epoch 48/50 | train_acc 99.46% | test_acc 93.33% | 00:01:05
Epoch 49/50 | train_acc 99.51% | test_acc 93.34% | 00:01:05
Epoch 50/50 | train_acc 99.58% | test_acc 93.41% | 00:01:05
Best test accuracy: 93.41%
'''
