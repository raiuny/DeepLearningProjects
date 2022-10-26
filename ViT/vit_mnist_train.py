from collections import defaultdict
import shutil
from torch import nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from vit import ViT
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

def train_epoch(model, dataloader, writer, criterion, optimizer, scheduler, epoch, device, result):
    model.train()
    bar = tqdm(dataloader)
    bar.set_description(f'epoch: {epoch:2}')
    corr, total = 0, 0
    total_loss = 0
    for i, (X, y) in enumerate(bar):
        X, gt_cls = X.to(device), y.to(device)
        pred_cls = model(X)
        loss = criterion(pred_cls, gt_cls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (torch.argmax(pred_cls, axis=1)==gt_cls).sum().item()/len(X)*100
        corr += (torch.argmax(pred_cls, axis=1)==gt_cls).sum().item()
        total += len(X)
        bar.set_postfix_str(
            f'lr={scheduler.get_last_lr()[0]:.4f}'
            f' acc={corr/total*100:.2f}'
            f' loss={loss.item():.2f}'
            )
        total_loss += loss.item()
        writer.add_scalar(tag='loss_step', scalar_value=loss.item(), global_step=epoch*len(bar) + i)
        writer.add_scalar(tag='acc_step', scalar_value=acc, global_step=epoch*len(bar) + i)
    writer.add_scalar(tag='loss', scalar_value=total_loss/len(bar), global_step=epoch)
    writer.add_scalar(tag='acc', scalar_value=corr/total*100, global_step=epoch)
    result['loss'].append(total_loss/len(bar))
    result['acc'].append(corr/total*100)
    scheduler.step()

def test_epoch(model, dataloader, writer, device):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for X, y in dataloader:
            X, gt_cls = X.to(device), y.to(device)
            pred_cls = model(X)
            correct += (torch.argmax(pred_cls, axis=1)==gt_cls).sum().item()
            total += len(X)
    print(f' val acc: {correct / total * 100:.2f}')
    writer.add_scalar(tag='acc', scalar_value=correct/total*100, global_step=epoch)

if __name__ == '__main__':
    shutil.rmtree('./ViT/log', ignore_errors=True)
    train = MNIST(root='../datasets/mnist/', train=True, transform=transforms.ToTensor(), download=True)
    test = MNIST(root='../datasets/mnist/', train=False, transform=transforms.ToTensor(), download=True)
    trainloader = DataLoader(train, batch_size=50, shuffle= True, num_workers=4)
    testloader = DataLoader(test, batch_size=50, shuffle=False, num_workers=4)  
    trainwriter = SummaryWriter(log_dir='./ViT/log/train',filename_suffix='_train')
    testwriter = SummaryWriter(log_dir='./ViT/log/test',filename_suffix='_test')
    pslist = [2, 4, 7, 14]
    ps = 7
    # model = ViT(
    # image_size=28, 
    # channels=1,
    # patch_size=ps,
    # num_classes=10,
    # dim=128,
    # depth=2,
    # heads=8,
    # dim_head=16,
    # mlp_dim=256,
    # dropout=0.5,
    # emb_dropout=0.
    # ).to('cuda')
    result = {'acc':[], 'loss':[], 'patch_size': []}
    for ps in pslist:
        model = ViT(
        image_size=28, 
        channels=1,
        patch_size=ps,
        num_classes=10,
        dim=64,
        depth=6,
        heads=8,
        dim_head=16,
        mlp_dim=128,
        dropout=0.,
        emb_dropout=0.
        ).to('cuda')
        torch.manual_seed(310)
        lr = 3e-3
        device = "cuda"
        epoch_num = 25
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)
        for epoch in range(epoch_num):
            print(f'patch_size = {ps}')
            train_epoch(model, trainloader,trainwriter, criterion, optimizer, scheduler, epoch, device, result)
            test_epoch(model, testloader, testwriter, device)
            result['patch_size'].append(ps)
    pd.DataFrame(result).to_csv('ViT/result.csv',index=None)
        
