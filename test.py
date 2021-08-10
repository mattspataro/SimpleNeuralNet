import model
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import random

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

net = model.NeuralNetwork()
net.load_state_dict(torch.load("weights/checkpoint.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

net.eval()
for i in range(50):
    x, y = random.choice(test_data)
    with torch.no_grad():
        pred = net(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')