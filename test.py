import model
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import os
import utils

if __name__ == "__main__":
    if not os.path.exists("weights/checkpoint.pth"):
        print("ERROR: weights/checkpoint.pth doesn't exist")
        print("Run train.py to generate pretrained weights!")
        exit()

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    net = model.NeuralNetwork()
    net.load_state_dict(torch.load("weights/checkpoint.pth"))

    net.eval()
    for i in range(50):
        x, y = random.choice(test_data)
        with torch.no_grad():
            pred = net(x)
            result = utils.get_result(pred[0], y)
            print(result)
            