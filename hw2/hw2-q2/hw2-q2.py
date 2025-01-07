#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,         
        maxpool=True,
        batch_norm=True,
        dropout=0.1
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

        # Q2.2: Batch Normalization (2D)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        
        # Max Pooling 2x2, stride=2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if maxpool else nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        """
        Forward pass through the ConvBlock:
          1. Convolution
          2. Batch Norm
          3. ReLU
          4. Max Pooling
          5. Dropout
        """
        x = self.conv(x)        # [B, out_channels, H, W]
        x = self.bn(x)          # [B, out_channels, H, W]   # Q2.2
        x = self.relu(x)        # [B, out_channels, H, W]
        x = self.pool(x)        # [B, out_channels, H/2, W/2]
        x = self.dropout(x)     # [B, out_channels, H/2, W/2]
        return x


class CNN(nn.Module):
    def __init__(
        self,
        num_classes=6,     
        dropout_prob=0.1,
        maxpool=True,
        batch_norm=True
    ):
        super(CNN, self).__init__()

        # -- Convolutional part --
        self.conv_block1 = ConvBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1,
            maxpool=maxpool,
            batch_norm=batch_norm,
            dropout=dropout_prob
        )
        self.conv_block2 = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            maxpool=maxpool,
            batch_norm=batch_norm,
            dropout=dropout_prob
        )
        self.conv_block3 = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            maxpool=maxpool,
            batch_norm=batch_norm,
            dropout=dropout_prob
        )

        # Q2.1: After 3 blocks + pooling: 48→24→12→6 => flatten_dim = 128 * 6 * 6 = 4608
        # flatten_dim = 128 * 6 * 6

        # Q2.2: Global Average Pooling instead of flatten
        # We will adapt the final feature map to (B, 128, 1, 1) and then flatten to (B, 128).
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # -- MLP part --
        #Q2.2: The input dimension is now 128, not 4608, after global pooling
        self.fc1 = nn.Linear(128, 1024)
        
        # Q2.2: Batch normalization in the MLP block
        self.bn1 = nn.BatchNorm1d(1024) if batch_norm else nn.Identity()

        self.fc2 = nn.Linear(1024, 512)
        
        self.bn2 = nn.BatchNorm1d(512) if batch_norm else nn.Identity()

        self.fc3 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        # Q2.1 (commented out):
        # x = x.view(x.size(0), 3, 48, 48)

        # Q2.2 reshape with flat inpuit:
        x = x.view(x.size(0), 3, 48, 48)

        # Pass through ConvBlocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Q2.1 :
        # Flatten => x = x.view(x.size(0), -1)

        # Q2.2: Global Average Pooling => shape becomes [B, 128, 1, 1]
        x = self.global_pool(x)
        # Flatten => shape [B, 128]
        x = x.view(x.size(0), -1)

        # MLP: FC1 -> BN -> ReLU -> Dropout -> FC2 -> BN -> ReLU -> FC3 -> log_softmax
        x = self.fc1(x)
        x = self.bn1(x)         # Q2.2
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)         # Q2.2
        x = self.relu(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    optimizer.zero_grad()
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X, return_scores=True):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)

    if return_scores:
        return predicted_labels, scores
    else:
        return predicted_labels


def evaluate(model, X, y, criterion=None):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    with torch.no_grad():
        y_hat, scores = predict(model, X, return_scores=True)
        loss = criterion(scores, y)
        n_correct = (y == y_hat).sum().item()
        n_possible = float(y.shape[0])

    return n_correct / n_possible, loss


def plot(epochs, plottable, ylabel='', name=''):
    plt.figure()#plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def get_number_trainable_params(model):
    """
    Returns the total number of trainable parameters in the model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_file_name_sufix(opt, exlude):
    """
    opt : options from argument parser
    exlude : set of variable names to exlude from the sufix (e.g. "device")

    """
    return '-'.join([str(value) for name, value in vars(opt).items() if name not in exlude])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=40, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-no_maxpool', action='store_true')
    parser.add_argument('-no_batch_norm', action='store_true')
    parser.add_argument('-data_path', type=str, default='intel_landscapes.v2.npz',)
    parser.add_argument('-device', choices=['cpu', 'cuda', 'mps'], default='cpu')

    opt = parser.parse_args()

    # Setting seed for reproducibility
    utils.configure_seed(seed=42)

    # Load data
    data = utils.load_dataset(data_path=opt.data_path)
    dataset = utils.ClassificationDataset(data)

    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X.to(opt.device), dataset.dev_y.to(opt.device)
    test_X, test_y = dataset.test_X.to(opt.device), dataset.test_y.to(opt.device)
    

    # initialize the model
    model = CNN(
        num_classes=int(6),
        dropout_prob=opt.dropout,
        maxpool=not opt.no_maxpool,
        batch_norm=not opt.no_batch_norm
    ).to(opt.device)


    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )

    # get a loss criterion
    criterion = nn.NLLLoss()

    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('\nTraining epoch {}'.format(ii))
        model.train()
        for X_batch, y_batch in train_dataloader:
            X_batch = X_batch.to(opt.device)
            y_batch = y_batch.to(opt.device)
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        val_acc, val_loss = evaluate(model, dev_X, dev_y, criterion)
        valid_accs.append(val_acc)
        print("Valid loss: %.4f" % val_loss)
        print('Valid acc: %.4f' % val_acc)

    test_acc, _ = evaluate(model, test_X, test_y, criterion)
    test_acc_perc = test_acc * 100
    test_acc_str = '%.2f' % test_acc_perc
    print('Final Test acc: %.4f' % test_acc)
    # plot
    sufix = plot_file_name_sufix(opt, exlude={'data_path', 'device'})

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-3-train-loss-{}-{}'.format(sufix, test_acc_str))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-3-valid-accuracy-{}-{}'.format(sufix, test_acc_str))

    print('Number of trainable parameters: ', get_number_trainable_params(model))

if __name__ == '__main__':
    main()
