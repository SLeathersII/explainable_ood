import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def train_plain(model, device, train_loader, optimizer, epoch,
                lam=1., verbose=100):
    # lam not necessarily needed but there to ensure that the
    # learning rates on the base and the CEDA model are comparable

    criterion = nn.NLLLoss()
    model.train()

    train_loss = 0
    correct = 0

    p_in = torch.tensor(1. / (1. + lam), device=device, dtype=torch.float)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = p_in * criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
    if verbose > 0:
        print(f'Train Epoch: {epoch} [{correct}/{1000} ({correct / 10:.0f}%)]\tLoss: {train_loss:.6f}')
    return train_loss / len(train_loader.dataset), correct / len(train_loader.dataset), 0.


def test(model, device, test_loader, min_conf=.1):
    model.eval()
    test_loss = 0
    correct = 0.
    av_conf = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            c, pred = output.max(1, keepdim=True)  # get the index of the max log-probability
            correct += (pred.eq(target.view_as(pred)) * (c.exp() >= min_conf)).sum().item()
            av_conf += c.exp().sum().item()

    test_loss /= len(test_loader.dataset)
    av_conf /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)

    return correct, av_conf, test_loss