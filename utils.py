import torch 
import debugpy

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

def get_result(y_hat, y):
    predicted, actual = classes[y_hat.argmax(0)], classes[y]
    return f'Predicted: "{predicted}", Actual: "{actual}"'

def attach_debugger():
    import debugpy
    debugpy.listen(5678)
    print("Waiting for Debugger to Attach on Port 5678...")
    debugpy.wait_for_client()
    print("Attached!")