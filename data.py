import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from typing import List
# --- MODIFIED: Import Organization to access power weights ---
from simulation import Organization 

# --- MODIFIED: Function signature and logic are completely new ---
def load_and_partition_data(organizations: List[Organization], batch_size: int, seed: int):
    """
    Load MNIST and create data partitions proportional to organization power_weight.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # --- Proportional Splitting Logic ---
    total_train_size = len(trainset)
    power_weights = torch.tensor([org.power_weight for org in organizations])
    
    # Calculate partition sizes based on weights
    proportions = power_weights / torch.sum(power_weights)
    partition_sizes = (proportions * total_train_size).int().tolist()
    
    # Adjust for rounding errors to ensure the sum matches the total dataset size
    current_total = sum(partition_sizes)
    diff = total_train_size - current_total
    for i in range(diff):
        partition_sizes[i] += 1
    
    print(f"\nProportionally partitioning data based on power weights. Partition sizes: {partition_sizes}")

    # Split the dataset using the calculated proportional lengths
    datasets = random_split(trainset, partition_sizes, torch.Generator().manual_seed(seed))

    trainloaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in datasets]
    testloader = DataLoader(testset, batch_size=batch_size)
    
    # Return the partition sizes so we can update the Organization objects
    return trainloaders, testloader, partition_sizes