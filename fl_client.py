import torch
import flwr as fl
from collections import OrderedDict
from model import Net
# Import the necessary classes from your simulation
from simulation import Organization, Params

class FlowerClient(fl.client.NumPyClient):
    # Update __init__ to accept the full Organization object
    def __init__(self, cid, organization: Organization, trainloader, testloader):
        self.cid = cid
        self.organization = organization # Store the organization profile
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = Net()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        local_epochs = int(config["local_epochs"])
        # local_epochs = 10 
        lr = float(config["lr"])
        momentum = float(config["momentum"])
        
        print(f"[Client {self.cid}] training for {local_epochs} epochs with lr={lr}")

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        criterion = torch.nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(local_epochs):
            # In a real-world scenario, you might count batches/samples processed
            # Here we use the S_n parameter from the simulation as a proxy
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                self.model(images)
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()
        
        # --- NEW: Calculate Resource Consumption ---
        # 1. Computation
        # Total FLOPs = Samples * FLOPs_per_sample * epochs
        # We use S_n and D_n from the organization's profile
        total_flops = self.organization.S_n * (self.organization.D_n / 1e9) * local_epochs
        computation_energy_joules = total_flops * 1e9 * self.organization.energy_per_flop

        # 2. Communication (1 download + 1 upload per fit call)
        comm_energy_joules = (Params.DL_ENERGY_JOULES_PER_MBIT + Params.UL_ENERGY_JOULES_PER_MBIT) * Params.MODEL_SIZE_MBITS
        
        # 3. Total Energy
        total_energy_joules = computation_energy_joules + comm_energy_joules

        # Create a dictionary with the new metrics
        consumption_metrics = {
            "computation_gigaflops": total_flops,
            "computation_energy_joules": computation_energy_joules,
            "communication_energy_joules": comm_energy_joules,
            "total_energy_joules": total_energy_joules
        }
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), consumption_metrics

    # The evaluate method remains unchanged
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        average_loss = loss / len(self.testloader)
        return average_loss, total, {"accuracy": accuracy}