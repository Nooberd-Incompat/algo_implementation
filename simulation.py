import numpy as np
from scipy.optimize import minimize_scalar

class Params:
    # --- MODIFIED: Added a parameter for total global rounds ---
    R_TOTAL = 10 # Total number of global rounds is now fixed
    N = 10
    K = 5
    T = 60.0
    MODEL_SIZE_MBITS = 0.16
    DL_SPEED_MBPS = 78.26
    UL_SPEED_MBPS = 42.06
    INVESTMENT_COST_PER_GHZ_HOUR = 0.22
    ELECTRICITY_RATE_KWH = 0.174
    DL_ENERGY_JOULES_PER_MBIT = 3
    UL_ENERGY_JOULES_PER_MBIT = 3
    EPSILON_0 = 9.82
    EPSILON_1 = 4.26
    RHO = 0.05   # Penalty coefficient, adjusted for epochs
    ETA = 0.2     # Step size, adjusted for epochs
    PHI = 1e-3    # Convergence threshold for epochs

def load_params_from_config(sim_config: dict, fl_config: dict):
    """Loads parameters from the config dictionary into the Params class."""
    # --- MODIFIED: Load total rounds from FL config ---
    Params.R_TOTAL = fl_config['total_rounds']
    Params.N = sim_config['N']
    Params.K = sim_config['K']
    Params.T = sim_config['T']
    Params.MODEL_SIZE_MBITS = sim_config['model_size_mbits']
    # ... (rest of the parameter loading remains the same) ...
    Params.EPSILON_1 = sim_config['epsilon_1']
    Params.RHO = sim_config['rho']
    Params.ETA = sim_config['eta']
    Params.PHI = sim_config['phi']

class Organization:
    def __init__(self, org_id, u_n, power_weight: float, D_n_giga=0.01):
        self.id = org_id
        self.u_n = float(u_n)
        self.power_weight = float(power_weight)
        
        # S_n will be set later by main.py after data partitioning
        self.S_n = 0 
        
        self.D_n = float(D_n_giga) * 1e9
        
        # The rest of the __init__ method is unchanged
        self.epochs_n = np.random.uniform(1.0, 5.0)
        self.pi_n = np.random.uniform(-0.01, 0.01)
        self.convg_flag = False
        self.C_energy_per_joule = Params.ELECTRICITY_RATE_KWH / 3.6e6
        comm_energy_joules = (Params.DL_ENERGY_JOULES_PER_MBIT + Params.UL_ENERGY_JOULES_PER_MBIT) * Params.MODEL_SIZE_MBITS
        self.comm_cost_per_round = comm_energy_joules * self.C_energy_per_joule
        self.energy_per_flop = 1e-9

    # --- MODIFIED: Utility is now a function of the average epochs (E_avg) ---
    def utility_U_n(self, E_avg):
        # Precision depends on total gradient steps: R_total * E_avg * steps_per_epoch
        # We simplify and say precision is a function of R_total * E_avg
        effective_training_effort = Params.R_TOTAL * E_avg
        initial_precision = Params.EPSILON_0 / Params.EPSILON_1
        final_precision = Params.EPSILON_0 / (Params.EPSILON_1 + effective_training_effort)
        return self.u_n * (initial_precision - final_precision)

    # --- MODIFIED: Cost is now a direct function of an org's own epochs (E_n) ---
    def cost_C_n(self, E_n):
        E_n_clamped = max(0.0, E_n)
        # Cost per round depends on local epochs
        flops_per_round = self.S_n * self.D_n * E_n_clamped
        comp_energy_per_round = flops_per_round * self.energy_per_flop
        comp_cost_per_round = comp_energy_per_round * self.C_energy_per_joule
        
        total_cost = (comp_cost_per_round + self.comm_cost_per_round) * Params.R_TOTAL
        return total_cost

    # --- MODIFIED: Payoff depends on my epochs (E_n), the average epochs (E_avg), and my payment (m_n) ---
    def payoff_V_n(self, E_n, E_avg, m_n):
        utility = self.utility_U_n(E_avg)
        cost = self.cost_C_n(E_n)
        return utility - cost + m_n

    # --- MODIFIED: Augmented payoff and penalty are redefined for epochs ---
    def augmented_payoff_V_rho(self, E_n_scalar, epoch_vector, pi_vector):
        N = len(epoch_vector)
        # Payment m_n is now per-round, so we multiply by total rounds
        idx_plus_1 = (self.id + 1) % N
        idx_plus_2 = (self.id + 2) % N
        zeta_n = pi_vector[idx_plus_1] - pi_vector[idx_plus_2]
        m_n = zeta_n * np.mean(epoch_vector) * Params.R_TOTAL

        temp_epoch_vector = epoch_vector.copy()
        temp_epoch_vector[self.id] = E_n_scalar
        E_avg = np.mean(temp_epoch_vector)
        
        payoff = self.payoff_V_n(E_n_scalar, E_avg, m_n)
        
        # Penalty encourages converging to the average effort level
        penalty = 0.0
        for i in range(N):
            diff = temp_epoch_vector[i] - E_avg
            penalty += diff ** 2
            
        return payoff - (Params.RHO * penalty)

def run_bidding_simulation(organizations, max_iterations=500, verbose=True):
    """
    Runs the convergence algorithm for EPOCHS and returns the final epoch values
    and the final net monetary balances (m_n) for settlement.
    """
    N = len(organizations)
    MAX_EPOCHS = 20 # A practical upper bound for bidding

    for t in range(max_iterations):
        epochs = np.array([o.epochs_n for o in organizations], float)
        pi = np.array([o.pi_n for o in organizations], float)

        if all(o.convg_flag for o in organizations):
            if verbose: print(f"Convergence reached at iteration {t}")
            break

        hat_epochs = np.zeros(N, float)
        for i, o in enumerate(organizations):
            def objective(e):
                e = max(0.0, min(e, MAX_EPOCHS))
                return -o.augmented_payoff_V_rho(e, epochs, pi)
            res = minimize_scalar(objective, bounds=(0.0, MAX_EPOCHS), method='bounded', options={'xatol': 1e-6})
            hat_epochs[i] = float(res.x)

        new_epochs = epochs + Params.ETA * (hat_epochs - epochs)
        
        for i, o in enumerate(organizations):
            o.convg_flag = abs(new_epochs[i] - o.epochs_n) <= Params.PHI
            o.epochs_n = new_epochs[i]

            idx_m1 = (i - 1 + N) % N
            idx_m2 = (i - 2 + N) % N
            o.pi_n = float(o.pi_n + Params.RHO * Params.ETA * (epochs[idx_m2] - epochs[idx_m1]))

        mean_pi = np.mean([o.pi_n for o in organizations])
        for o in organizations: o.pi_n -= mean_pi

    # --- Calculate Final Settlement Balances ---
    final_epochs_list = [org.epochs_n for org in organizations]
    final_pis_list = [org.pi_n for org in organizations]
    E_avg_NE = np.mean(final_epochs_list)

    final_balances = {}
    for org in organizations:
        idx_plus_1 = (org.id + 1) % N
        idx_plus_2 = (org.id + 2) % N
        zeta_n = final_pis_list[idx_plus_1] - final_pis_list[idx_plus_2]
        # Total payment is based on average effort over all rounds
        m_n = zeta_n * E_avg_NE * Params.R_TOTAL
        final_balances[str(org.id)] = m_n

    final_epochs_dict = {str(org.id): org.epochs_n for org in organizations}
    
    if verbose: 
        print(f"Final converged epochs: {final_epochs_dict}")
        print(f"Final net balances ($) for settlement: {final_balances}")

    return final_epochs_dict, final_balances