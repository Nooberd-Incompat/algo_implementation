import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def load_and_partition_data(num_clients: int, batch_size: int, seed: int):
    """Load MNIST and create data partitions."""
    # The transform needs to be updated for MNIST (1 channel)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))] # Update normalization for 1-channel grayscale
    )
    # Change the dataset from CIFAR10 to MNIST
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # The rest of the function remains the same
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))

    trainloaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in datasets]
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, testloader 