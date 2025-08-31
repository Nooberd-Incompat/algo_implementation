import flwr as fl
from typing import Dict, List, Tuple, Optional, Union
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

class BiddingBasedStrategy(fl.server.strategy.FedAvg):
    def __init__(self, gamma_values: Dict[str, float], model_config: dict, **kwargs):
        super().__init__(**kwargs)
        self.gamma_values = gamma_values
        self.model_config = model_config
        print(f"Strategy initialized with gamma values: {self.gamma_values}")

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        
        configured_clients = super().configure_fit(server_round, parameters, client_manager)
        
        custom_instructions = []
        for client, fit_ins in configured_clients:
            cid = client.cid
            gamma_n = self.gamma_values.get(cid, 1.0)
            local_epochs = max(1, round(gamma_n))
            
            fit_ins.config = self.model_config.copy() 
            fit_ins.config["local_epochs"] = local_epochs
            
            custom_instructions.append((client, fit_ins))
            
        return custom_instructions

    # --- NEW: Override aggregate_fit to display consumption ---
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        # Call the parent's aggregate_fit to get the aggregated weights
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Now, process and display the custom consumption metrics
        print("\n" + "="*25 + f" Round {server_round} Consumption Report " + "="*25)
        total_round_energy = 0
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            total_energy = metrics.get("total_energy_joules", 0)
            gflops = metrics.get("computation_gigaflops", 0)
            
            print(
                f"  - Client {client_proxy.cid}: "
                f"Computation: {gflops:.2f} GFLOPs, "
                f"Total Energy: {total_energy:.4f} Joules"
            )
            total_round_energy += total_energy
        
        print(f"\n  Total Energy Consumed this Round: {total_round_energy:.4f} Joules")
        print("="*75 + "\n")
        
        return aggregated_parameters, aggregated_metrics