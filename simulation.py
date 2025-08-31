import numpy as np
from scipy.optimize import minimize_scalar
# --- Keep your Params and Organization classes exactly as they are ---
# (I've omitted them here for brevity, but you should copy them from algo_19_08.py)

class Params:
    N = 10  # Number of organizations
    K = 5  # Number of local updates per round
    T = 60.0  # Total training time in seconds
    MODEL_SIZE_MBITS = 0.16  # model size in Mbit
    DL_SPEED_MBPS = 78.26
    UL_SPEED_MBPS = 42.06
    INVESTMENT_COST_PER_GHZ_HOUR = 0.22   # $ per GHz-hour
    ELECTRICITY_RATE_KWH = 0.174         # $ per kWh
    DL_ENERGY_JOULES_PER_MBIT = 3        # J per Mbit (example)
    UL_ENERGY_JOULES_PER_MBIT = 3        # J per Mbit (example)
    EPSILON_0 = 9.82
    EPSILON_1 = 4.26
    RHO = 0.0007   # Penalty coefficient
    ETA = 0.1     # Step size (0 < eta <= 1)
    PHI = 1e-4    # Convergence threshold for gamma

def load_params_from_config(sim_config: dict):
    """Loads parameters from the config dictionary into the Params class."""
    Params.N = sim_config['N']
    Params.K = sim_config['K']
    Params.T = sim_config['T']
    Params.MODEL_SIZE_MBITS = sim_config['model_size_mbits']
    Params.DL_SPEED_MBPS = sim_config['dl_speed_mbps']
    Params.UL_SPEED_MBPS = sim_config['ul_speed_mbps']
    Params.INVESTMENT_COST_PER_GHZ_HOUR = sim_config['investment_cost_per_ghz_hour']
    Params.ELECTRICITY_RATE_KWH = sim_config['electricity_rate_kwh']
    Params.DL_ENERGY_JOULES_PER_MBIT = sim_config['dl_energy_joules_per_mbit']
    Params.UL_ENERGY_JOULES_PER_MBIT = sim_config['ul_energy_joules_per_mbit']
    Params.EPSILON_0 = sim_config['epsilon_0']
    Params.EPSILON_1 = sim_config['epsilon_1']
    Params.RHO = sim_config['rho']
    Params.ETA = sim_config['eta']
    Params.PHI = sim_config['phi']

class Organization:
    # --- PASTE THE FULL Organization CLASS HERE ---
    def __init__(self, org_id, u_n, S_n=600, D_n_giga=0.01):
        self.id = org_id
        self.u_n = float(u_n)
        self.S_n = float(S_n)
        self.D_n = float(D_n_giga) * 1e9
        self.f_max = 50.497 * 1e9
        self.T_DL = Params.MODEL_SIZE_MBITS / Params.DL_SPEED_MBPS
        self.T_UL = Params.MODEL_SIZE_MBITS / Params.UL_SPEED_MBPS
        self.C_invt = (Params.INVESTMENT_COST_PER_GHZ_HOUR) / (3600.0 * 1e9)
        self.C_energy_per_joule = Params.ELECTRICITY_RATE_KWH / 3.6e6
        self.C_comp = self.C_energy_per_joule
        self.energy_per_flop = 1e-9
        comm_energy_DL_joules = Params.DL_ENERGY_JOULES_PER_MBIT * Params.MODEL_SIZE_MBITS
        comm_energy_UL_joules = Params.UL_ENERGY_JOULES_PER_MBIT * Params.MODEL_SIZE_MBITS
        self.C_DL = comm_energy_DL_joules * self.C_energy_per_joule
        self.C_UL = comm_energy_UL_joules * self.C_energy_per_joule
        self.gamma_n = np.random.uniform(25.0, 50.0)
        self.pi_n = np.random.uniform(-0.001, 0.001)
        self.convg_flag = False
    # --- PASTE ALL METHODS OF Organization CLASS HERE ---
    def get_f_from_r(self, r):
        if r <= 0.0: return self.f_max
        denom = (Params.T / r) - self.T_UL - self.T_DL
        if denom <= 0.0: return self.f_max
        f_n = (self.S_n * self.D_n * Params.K) / denom
        return min(f_n, self.f_max)
    def precision_epsilon(self, r):
        r_clamped = max(0.0, r)
        return Params.EPSILON_0 / (Params.EPSILON_1 + Params.K * r_clamped)
    def utility_U_n(self, r):
        initial_precision = self.precision_epsilon(0.0)
        final_precision = self.precision_epsilon(r)
        return self.u_n * (initial_precision - final_precision)
    def cost_C_n(self, f_n, r):
        r_clamped = max(0.0, r)
        comm_cost = (self.C_UL + self.C_DL) * r_clamped
        invt_cost = self.C_invt * f_n
        total_flops = self.S_n * self.D_n * Params.K * r_clamped
        comp_cost = self.energy_per_flop * total_flops * self.C_energy_per_joule
        return comm_cost + invt_cost + comp_cost
    def payoff_V_n(self, r, m_n):
        r_clamped = max(0.0, r)
        f_n = self.get_f_from_r(r_clamped)
        utility = self.utility_U_n(r_clamped)
        cost = self.cost_C_n(f_n, r_clamped)
        return utility - cost + m_n
    def augmented_payoff_V_rho(self, gamma_n_scalar, gamma_vector, pi_vector):
        N = len(gamma_vector)
        idx_plus_1 = (self.id + 1) % N
        idx_plus_2 = (self.id + 2) % N
        zeta_n = pi_vector[idx_plus_1] - pi_vector[idx_plus_2]
        m_n = zeta_n * gamma_n_scalar
        payoff = self.payoff_V_n(gamma_n_scalar, m_n)
        temp_gamma_vector = gamma_vector.copy()
        temp_gamma_vector[self.id] = gamma_n_scalar
        penalty = 0.0
        for i in range(N):
            idx_minus_1 = (i - 1 + N) % N
            idx_minus_2 = (i - 2 + N) % N
            diff = temp_gamma_vector[idx_minus_2] - temp_gamma_vector[idx_minus_1]
            penalty += diff ** 2
        return payoff - (Params.RHO * penalty)
    def update_profile(self, gamma_vector, pi_vector, r_bar):
        def objective(g):
            g_clamped = max(0.0, g)
            return -self.augmented_payoff_V_rho(g_clamped, gamma_vector, pi_vector)
        if r_bar <= 0.0:
            hat_gamma_n = self.gamma_n
        else:
            result = minimize_scalar(objective, bounds=(0.0, r_bar), method='bounded', options={'xatol':1e-6})
            hat_gamma_n = float(result.x)
        prev_gamma_n = float(self.gamma_n)
        self.gamma_n = prev_gamma_n + Params.ETA * (hat_gamma_n - prev_gamma_n)
        N = len(gamma_vector)
        idx_minus_1 = (self.id - 1 + N) % N
        idx_minus_2 = (self.id - 2 + N) % N
        self.pi_n = float(self.pi_n + Params.RHO * Params.ETA * (gamma_vector[idx_minus_2] - gamma_vector[idx_minus_1]))
        if abs(self.gamma_n - prev_gamma_n) <= Params.PHI:
            self.convg_flag = True
        else:
            self.convg_flag = False

class SmartContract:
    # --- PASTE THE FULL SmartContract CLASS HERE, but remove plotting ---
    def __init__(self, organizations):
        self.organizations = organizations
        self.gamma_history = []
        self.t = 0
        tau_max = 0.0
        for org in self.organizations:
            tau_n = (org.S_n * org.D_n * Params.K) / org.f_max + org.T_UL + org.T_DL
            if tau_n > tau_max:
                tau_max = tau_n
        if tau_max <= 0.0:
            self.r_bar = Params.T
        else:
            self.r_bar = Params.T / tau_max
    # --- REMOVE run_simulation and plot_results ---

# NEW function that runs the simulation and returns the results
def run_bidding_simulation(organizations, max_iterations=500, verbose=True):
    """
    Runs the convergence algorithm and returns the final gamma values.
    """
    contract = SmartContract(organizations)
    N = len(organizations)

    for t in range(max_iterations):
        gamma = np.array([o.gamma_n for o in organizations], float)
        pi = np.array([o.pi_n for o in organizations], float)

        if all(o.convg_flag for o in organizations):
            if verbose: print(f"Convergence reached at iteration {t}")
            break

        hat_gamma = np.zeros(N, float)
        for i, o in enumerate(organizations):
            def obj(g):
                g = max(0.0, min(g, contract.r_bar))
                return -o.augmented_payoff_V_rho(g, gamma, pi)
            res = minimize_scalar(obj, bounds=(0.0, contract.r_bar), method='bounded', options={'xatol': 1e-6})
            hat_gamma[i] = float(res.x)

        new_gamma = gamma + Params.ETA * (hat_gamma - gamma)
        for i, o in enumerate(organizations):
            o.convg_flag = abs(new_gamma[i] - o.gamma_n) <= Params.PHI
            o.gamma_n = new_gamma[i]

        for i, o in enumerate(organizations):
            idx_m1 = (i - 1) % N
            idx_m2 = (i - 2) % N
            o.pi_n = float(o.pi_n + Params.RHO * Params.ETA * (new_gamma[idx_m2] - new_gamma[idx_m1]))

        mean_pi = np.mean([o.pi_n for o in organizations])
        for o in organizations:
            o.pi_n -= mean_pi

    final_gammas = {str(org.id): org.gamma_n for org in organizations}
    if verbose: print(f"Final converged gammas: {final_gammas}")
    return final_gammas