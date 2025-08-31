import flwr as fl
import numpy as np
import yaml # Import the YAML library

from simulation import run_bidding_simulation, Organization, load_params_from_config
from fl_client import FlowerClient
from fl_server import BiddingBasedStrategy
from data import load_and_partition_data

# --- 1. Load Configuration from YAML file ---
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

print("--- Configuration Loaded ---")
print(CFG)

# Set numpy random seed for simulation reproducibility
np.random.seed(0)

# --- 2. Phase 1: Run the Bidding Simulation ---
print("\n--- Starting Phase 1: Bidding Simulation ---")

# Load simulation parameters from the config file into the Params class
load_params_from_config(CFG['simulation_params'])

# Create heterogeneous valuations based on config
u_vals_hetero = [CFG['simulation_params']['u_low'] if i < CFG['simulation_params']['N'] / 2 else CFG['simulation_params']['u_high'] for i in range(CFG['simulation_params']['N'])]
organizations = [Organization(org_id=i, u_n=u_vals_hetero[i]) for i in range(CFG['simulation_params']['N'])]

# Run simulation
converged_gammas = run_bidding_simulation(organizations, max_iterations=CFG['simulation_params']['max_iterations'])
print("--- Phase 1 Finished --- \n")

# --- 3. Phase 2: Configure and Run Federated Learning with Flower ---
print("--- Starting Phase 2: Federated Learning with Flower ---")

# Load data using parameters from config
trainloaders, testloader = load_and_partition_data(
    num_clients=CFG['simulation_params']['N'],
    batch_size=CFG['data_params']['batch_size'],
    seed=CFG['data_params']['partition_seed']
)

# Define a function to create clients
def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client for a given client ID."""
    # Get the corresponding organization object from the list
    org_profile = organizations[int(cid)]
    return FlowerClient(cid, org_profile, trainloaders[int(cid)], testloader)

# Instantiate the custom strategy, passing the full model config
strategy = BiddingBasedStrategy(
    gamma_values=converged_gammas,
    fraction_fit=CFG['federated_learning_params']['fraction_fit'],
    fraction_evaluate=CFG['federated_learning_params']['fraction_evaluate'],
    min_fit_clients=int(CFG['simulation_params']['N'] * CFG['federated_learning_params']['fraction_fit']),
    min_evaluate_clients=int(CFG['simulation_params']['N'] * CFG['federated_learning_params']['fraction_evaluate']),
    min_available_clients=CFG['simulation_params']['N'],
    model_config=CFG['model_params'] # Pass model config to the strategy
)

# Start the Flower simulation
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=CFG['simulation_params']['N'],
    config=fl.server.ServerConfig(num_rounds=CFG['federated_learning_params']['total_rounds']),
    strategy=strategy,
    client_resources={"num_cpus": 8, "num_gpus": 0.5}, # Adjust based on your machine
)

print("--- FL Training Finished ---")