import shutil
from torch import nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from vit import ViT
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
shutil.rmtree('./ViT/log', ignore_errors=True)

train = MNIST(root='../datasets/mnist/', train=True, transform=transforms.ToTensor(), download=True)
test = MNIST(root='../datasets/mnist/', train=False, transform=transforms.ToTensor(), download=True)
trainloader = DataLoader(train, batch_size=50, shuffle= True, num_workers=0)
testloader = DataLoader(test, batch_size=50, shuffle=False, num_workers=0)

trainwriter = SummaryWriter(log_dir='./ViT/log/train',filename_suffix='_train')
testwriter = SummaryWriter(log_dir='./ViT/log/test',filename_suffix='_test')

def train_epoch(model, dataloader, criterion, optimizer, scheduler, epoch, device):
    model.train()
    bar = tqdm(dataloader)
    bar.set_description(f'epoch: {epoch:2}')
    corr, total = 0, 0
    for X, y in bar:
        X, gt_cls = X.to(device), y.to(device)
        pred_cls = model(X)
        loss = criterion(pred_cls, gt_cls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        corr += (torch.argmax(pred_cls, axis=1)==gt_cls).sum().item()
        total += len(X)
        bar.set_postfix_str(
            f'lr={scheduler.get_last_lr()[0]:.4f}'
            f' acc={corr/total*100:.2f}'
            f' loss={loss.item():.2f}'
            )
    trainwriter.add_scalar(tag='loss', scalar_value=loss.item(), global_step=epoch)
    trainwriter.add_scalar(tag='acc', scalar_value=corr/total*100, global_step=epoch)
    scheduler.step()

def test_epoch(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for X, y in dataloader:
            X, gt_cls = X.to(device), y.to(device)
            pred_cls = model(X)
            correct += (torch.argmax(pred_cls, axis=1)==gt_cls).sum().item()
            total += len(X)
    print(f' val acc: {correct / total * 100:.2f}')
    testwriter.add_scalar(tag='acc', scalar_value=correct/total*100, global_step=epoch)

if __name__ == '__main__':
    
    model = ViT(
    image_size=28, 
    channels=1,
    patch_size=7,
    num_classes=10,
    dim=128,
    depth=2,
    heads=8,
    dim_head=16,
    mlp_dim=256,
    dropout=0.5,
    emb_dropout=0.
    ).to('cuda')
    torch.manual_seed(310)
    lr = 5e-3
    device = "cuda"
    epoch_num = 40
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)
    for epoch in range(epoch_num):
        train_epoch(model, trainloader, criterion, optimizer, scheduler, epoch, device)
        test_epoch(model, testloader, device)

