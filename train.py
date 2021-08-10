###############################################################
'''
Adapted from PyTorch Quickstart Tutorial: 
Tutorial: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
GitHub: https://github.com/pytorch/tutorials/blob/master/beginner_source/basics/quickstart_tutorial.py
'''
###############################################################

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import model

import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--epochs", "--ep", type=int, default=5)
    parser.add_argument("--batch_size", "--bs", type=int, default=64)
    parser.add_argument("--learning_rate", "--lr", type=float, default=0.3)
    parser.add_argument("--device", "--dv", type=str, default=None, choices=["cpu", "cuda:0", "cuda:1"])
    return parser

def train(dataloader, net, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = net(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, net, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    net.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = net(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    
    # Create data loaders.
    batch_size = args.batch_size
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Get cpu or gpu device for training.
    if not args.device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print("Using {} device".format(device))

    net = model.NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate)

    epochs = args.epochs
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, net, loss_fn, optimizer)
        test(test_dataloader, net, loss_fn)
    print("Done!")

    torch.save(net.state_dict(), "weights/checkpoint.pth")
    print("Saved PyTorch net state to weights/checkpoint.pth")