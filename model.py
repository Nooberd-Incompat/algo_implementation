import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """A simple CNN adapted for MNIST."""
    def __init__(self) -> None:
        super(Net, self).__init__()
        # Change the first Conv layer to accept 1 input channel instead of 3
        self.conv1 = nn.Conv2d(1, 6, 5) # Changed from 3 input channels to 1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # The input size to the fully connected layer changes because the
        # input image size is different (28x28 for MNIST vs 32x32 for CIFAR-10)
        # Calculation:
        # Input: 28x28 -> conv1 -> 24x24 -> pool -> 12x12
        # -> conv2 -> 8x8 -> pool -> 4x4.
        # So, the flattened size is 16 channels * 4 * 4 pixels.
        self.fc1 = nn.Linear(16 * 4 * 4, 120) # Adjusted for 28x28 input image
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # Output is 10 classes (digits 0-9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # The view needs to match the new flattened size
        x = x.view(-1, 16 * 4 * 4) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x