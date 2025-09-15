import flwr as fl
import numpy as np
import yaml

from scripts.settlement import run_settlement
from simulation import run_bidding_simulation, Organization, load_params_from_config
from fl_client import FlowerClient
from fl_server import BiddingBasedStrategy
from data import load_and_partition_data

# --- 1. Load Configuration ---
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)
print("--- Configuration Loaded ---")
print(CFG)
np.random.seed(0)

# ==============================================================================
# PHASE 1: BIDDING SIMULATION
# ==============================================================================
print("\n--- Starting Phase 1: Bidding Simulation for Local Epochs ---")
load_params_from_config(CFG['simulation_params'], CFG['federated_learning_params'])

power_weights = np.linspace(1.0, 3.0, CFG['simulation_params']['N'])
u_vals_hetero = [CFG['simulation_params']['u_low'] if i < CFG['simulation_params']['N'] / 2 else CFG['simulation_params']['u_high'] for i in range(CFG['simulation_params']['N'])]
organizations = [Organization(org_id=i, u_n=u_vals_hetero[i], power_weight=power_weights[i]) for i in range(CFG['simulation_params']['N'])]
print(f"Created {len(organizations)} organizations with power weights: {[round(w, 2) for w in power_weights]}")

trainloaders, testloader, partition_sizes = load_and_partition_data(
    organizations=organizations,
    batch_size=CFG['data_params']['batch_size'],
    seed=CFG['data_params']['partition_seed']
)

for i, org in enumerate(organizations):
    org.S_n = partition_sizes[i]

converged_epochs, final_net_balances_dollars = run_bidding_simulation(
    organizations,
    max_iterations=CFG['simulation_params']['max_iterations']
)
print("--- Phase 1 Finished --- \n")


# ==============================================================================
# PHASE 2: FEDERATED LEARNING
# ==============================================================================
print("--- Starting Phase 2: Federated Learning with Flower ---")

def client_fn(cid: str) -> fl.client.Client:
    """Create a Flower client for a given client ID."""
    org_profile = organizations[int(cid)]
    client = FlowerClient(cid, org_profile, trainloaders[int(cid)], testloader)
    return client.to_client()

strategy = BiddingBasedStrategy(
    epoch_values=converged_epochs,
    fraction_fit=CFG['federated_learning_params']['fraction_fit'],
    fraction_evaluate=CFG['federated_learning_params']['fraction_evaluate'],
    min_fit_clients=int(CFG['simulation_params']['N'] * CFG['federated_learning_params']['fraction_fit']),
    min_evaluate_clients=int(CFG['simulation_params']['N'] * CFG['federated_learning_params']['fraction_evaluate']),
    min_available_clients=CFG['simulation_params']['N'],
    model_config=CFG['model_params']
)

history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=CFG['simulation_params']['N'],
    config=fl.server.ServerConfig(num_rounds=CFG['federated_learning_params']['total_rounds']),
    strategy=strategy,
    client_resources={"num_cpus": 16, "num_gpus": 1},
)
print("--- FL Training Finished ---")


# ==============================================================================
# PHASE 3: ON-CHAIN SETTLEMENT
# ==============================================================================
print(f"\n--- Starting Phase 3: Final Settlement Process ---")

# The settlement script will handle all conversions and corrections
final_net_balances_dollars_named = {
    f"Org_{cid}": balance for cid, balance in final_net_balances_dollars.items()
}

try:
    print("Sending the following DOLLAR balances to the settlement script:")
    print(final_net_balances_dollars_named)
    
    run_settlement(final_net_balances_dollars_named)
    
    print(f"\nSettlement successful.")
except Exception as e:
    print(f"\nSettlement failed: {e}")
    print("Please ensure Ganache is running and private keys are correctly configured in scripts/settlement.py")