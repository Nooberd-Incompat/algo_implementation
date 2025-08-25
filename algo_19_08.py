import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# --- System Parameters from Table I / paper ---
class Params:
    N = 10  # Number of organizations
    K = 5  # Number of local updates per round
    T = 60.0  # Total training time in seconds
    MODEL_SIZE_MBITS = 0.16  # model size in Mbit
    DL_SPEED_MBPS = 78.26
    UL_SPEED_MBPS = 42.06
    # Cost related parameters (example numbers from your original script)
    INVESTMENT_COST_PER_GHZ_HOUR = 0.22   # $ per GHz-hour
    ELECTRICITY_RATE_KWH = 0.174         # $ per kWh
    DL_ENERGY_JOULES_PER_MBIT = 3        # J per Mbit (example)
    UL_ENERGY_JOULES_PER_MBIT = 3        # J per Mbit (example)
    # Precision function parameters
    EPSILON_0 = 9.82
    EPSILON_1 = 4.26
    # Algorithm parameters
    RHO = 0.0007   # Penalty coefficient
    ETA = 0.1     # Step size (0 < eta <= 1)
    PHI = 1e-4    # Convergence threshold for gamma

class Organization:
    """Represents a single organization in the FL system, with methods that implement
    the optimization and update steps from Algorithm 1 (page 9)."""
    def __init__(self, org_id, u_n, S_n=600, D_n_giga=0.01):
        self.id = org_id
        self.u_n = float(u_n)          # unit revenue / valuation
        self.S_n = float(S_n)          # number of data samples
        self.D_n = float(D_n_giga) * 1e9  # FLOPs per data unit (in cycles)
        # maximum processing capacity in FLOPs/s (f_max)
        self.f_max = 50.495 * 1e9  # 10 GHz => 10e9 FLOPs/s

        # Network time (seconds) to download/upload model (per-round comm times)
        self.T_DL = Params.MODEL_SIZE_MBITS / Params.DL_SPEED_MBPS
        self.T_UL = Params.MODEL_SIZE_MBITS / Params.UL_SPEED_MBPS

        # --- Cost coefficients (unit-aware approximations) ---
        # Investment cost: convert $ / (GHz * hour) -> $ / (FLOP/s)
        # 1 GHz = 1e9 FLOP/s; 1 hour = 3600 s
        # So $/(GHz * hour) divided by (3600 * 1e9) gives $ per (FLOP/s) sustained for 1 second.
        self.C_invt = (Params.INVESTMENT_COST_PER_GHZ_HOUR) / (3600.0 * 1e9)
        # Electricity price $/kWh -> $/J : divide by 3.6e6
        self.C_energy_per_joule = Params.ELECTRICITY_RATE_KWH / 3.6e6
        # Treat C_comp as the coefficient in the paper multiplying f_n^2 * S * D * K * r.
        # If you have an estimate of energy per cycle, incorporate it here. For now we use
        # the $/J baseline as a reasonable placeholder.
        self.C_comp = self.C_energy_per_joule
        self.energy_per_flop = 1e-9
        # Communication energy costs per round (used for comm cost)
        # Approximate comm energy cost converted to $ using $/J factor:
        comm_energy_DL_joules = Params.DL_ENERGY_JOULES_PER_MBIT * Params.MODEL_SIZE_MBITS
        comm_energy_UL_joules = Params.UL_ENERGY_JOULES_PER_MBIT * Params.MODEL_SIZE_MBITS
        self.C_DL = comm_energy_DL_joules * self.C_energy_per_joule
        self.C_UL = comm_energy_UL_joules * self.C_energy_per_joule

        # --- Algorithm state ---
        # Initialize with small random values (keeps deterministic runs possible if seed set externally)
        self.gamma_n = np.random.uniform(25.0, 50.0)  # candidate number-of-rounds / local variable
        self.pi_n = np.random.uniform(-0.001, 0.001)  # dual-like variable used to compute zeta
        self.convg_flag = False

    def get_f_from_r(self, r):
        """Calculates processing capacity f_n needed to complete K local updates in r rounds:
           f_n = (S_n * D_n * K) / (T/r - T_UL - T_DL) (Eq. 9 style)
           If denominator <= 0, return f_max to denote infeasible small r (bounded later by r_bar).
        """
        if r <= 0.0:
            return self.f_max
        denom = (Params.T / r) - self.T_UL - self.T_DL
        if denom <= 0.0:
            return self.f_max
        f_n = (self.S_n * self.D_n * Params.K) / denom
        # Enforce capacity limit:
        return min(f_n, self.f_max)

    def precision_epsilon(self, r):
        """Model precision epsilon(r) per Eq. (4). Guard negative r."""
        r_clamped = max(0.0, r)
        return Params.EPSILON_0 / (Params.EPSILON_1 + Params.K * r_clamped)

    def utility_U_n(self, r):
        """Organization utility U_n(r) (Eq. 5): proportional to improvement in precision."""
        initial_precision = self.precision_epsilon(0.0)
        final_precision = self.precision_epsilon(r)
        return self.u_n * (initial_precision - final_precision)

    def cost_C_n(self, f_n, r):
        r_clamped = max(0.0, r)
        comm_cost = (self.C_UL + self.C_DL) * r_clamped
        invt_cost = self.C_invt * f_n
        # total FLOPs processed across r rounds = S_n * D_n * K * r
        total_flops = self.S_n * self.D_n * Params.K * r_clamped
        # computational energy cost in $ = energy_per_flop * total_flops * $/J
        comp_cost = self.energy_per_flop * total_flops * self.C_energy_per_joule
        return comm_cost + invt_cost + comp_cost
    

    def payoff_V_n(self, r, m_n):
        """Payoff V_n(r, m_n) = U_n(r) - C_n(f_n(r), r) + m_n (Eq. 7)"""
        # r might be negative if optimizer tries; clamp
        r_clamped = max(0.0, r)
        # f required to support r_clamped rounds
        f_n = self.get_f_from_r(r_clamped)
        utility = self.utility_U_n(r_clamped)
        cost = self.cost_C_n(f_n, r_clamped)
        return utility - cost + m_n

    def augmented_payoff_V_rho(self, gamma_n_scalar, gamma_vector, pi_vector):
        """Augmented payoff V^rho_n(\gamma_n, gamma_{-n}, pi) (Eq. 20 style)
           - Uses zeta_n = pi_{n+1} - pi_{n+2}
           - m_n = zeta_n * gamma_n
           - subtract penalty term: rho * sum_i (gamma_{i-2} - gamma_{i-1})^2
        """
        N = len(gamma_vector)
        # Compute zeta_n as pi_{n+1} - pi_{n+2} (cyclic)
        idx_plus_1 = (self.id + 1) % N
        idx_plus_2 = (self.id + 2) % N
        zeta_n = pi_vector[idx_plus_1] - pi_vector[idx_plus_2]

        # Monetary transfer evaluated at candidate gamma_n_scalar
        m_n = zeta_n * gamma_n_scalar

        # Base payoff
        payoff = self.payoff_V_n(gamma_n_scalar, m_n)

        # Penalty: computed on the temporary gamma vector where this org uses candidate gamma_n_scalar
        temp_gamma_vector = gamma_vector.copy()
        temp_gamma_vector[self.id] = gamma_n_scalar

        penalty = 0.0
        for i in range(N):
            idx_minus_1 = (i - 1 + N) % N
            idx_minus_2 = (i - 2 + N) % N
            diff = temp_gamma_vector[idx_minus_2] - temp_gamma_vector[idx_minus_1]
            penalty += diff ** 2

        # Use the paper scaling: subtract rho * penalty
        return payoff - (Params.RHO * penalty)

    def update_profile(self, gamma_vector, pi_vector, r_bar):
        """Perform one iteration of Algorithm 1 for this organization:
           - compute hat_gamma_n = argmax_{gamma in [0, r_bar]} V^rho_n(...)
             (we use minimizer on negative because scipy has minimize_scalar)
           - update gamma_n and pi_n using step size ETA and RHO
           - update convergence flag
        """
        # Define objective (negative augmented payoff, since we minimize)
        def objective(g):
            # ensure g is not slightly negative due to optimizer numeric jitter
            g_clamped = max(0.0, g)
            return -self.augmented_payoff_V_rho(g_clamped, gamma_vector, pi_vector)

        # Ensure r_bar positive
        if r_bar <= 0.0:
            # degenerate case: stay at current gamma
            hat_gamma_n = self.gamma_n
        else:
            # minimize over [0, r_bar]
            result = minimize_scalar(objective, bounds=(0.0, r_bar), method='bounded', options={'xatol':1e-6})
            hat_gamma_n = float(result.x)

        # Step 7: gamma update (relaxed best-response)
        prev_gamma_n = float(self.gamma_n)
        self.gamma_n = prev_gamma_n + Params.ETA * (hat_gamma_n - prev_gamma_n)

        # Step 8: pi update (dual-like update)
        N = len(gamma_vector)
        idx_minus_1 = (self.id - 1 + N) % N
        idx_minus_2 = (self.id - 2 + N) % N
        # The paper uses pi_n <- pi_n + rho * eta * (gamma_{n-2} - gamma_{n-1})
        self.pi_n = float(self.pi_n + Params.RHO * Params.ETA * (gamma_vector[idx_minus_2] - gamma_vector[idx_minus_1]))

        # Convergence check for gamma (Step 9-11 style)
        if abs(self.gamma_n - prev_gamma_n) <= Params.PHI:
            self.convg_flag = True
        else:
            self.convg_flag = False

class SmartContract:
    """Simulates the coordinator (blockchain/smart-contract) that stores gamma/pi history
       and triggers algorithm iterations across organizations (Algorithm 1 driver).
    """
    def __init__(self, organizations):
        self.organizations = organizations
        self.gamma_history = []
        self.zeta_history = []
        self.t = 0

        # Precompute global r_bar (bar r) per the paper:
        # bar_r = T / max_n { S_n * D_n * K / f_max_n + T_UL_n + T_DL_n }
        tau_max = 0.0
        for org in self.organizations:
            tau_n = (org.S_n * org.D_n * Params.K) / org.f_max + org.T_UL + org.T_DL
            if tau_n > tau_max:
                tau_max = tau_n
        # Guard: if tau_max is zero (degenerate), set r_bar to a reasonable default
        if tau_max <= 0.0:
            self.r_bar = Params.T  # at most T rounds (degenerate)
        else:
            self.r_bar = Params.T / tau_max

    def run_simulation(self, max_iterations=500, verbose=True):
        N = len(self.organizations)
        for t in range(max_iterations):
            # snapshots
            gamma = np.array([o.gamma_n for o in self.organizations], float)
            pi    = np.array([o.pi_n    for o in self.organizations], float)

            # history for plotting
            self.gamma_history.append(gamma.copy())
            zeta = np.array([pi[(i+1)%N] - pi[(i+2)%N] for i in range(N)], float)
            self.zeta_history.append(zeta.copy())

            # convergence check
            if all(o.convg_flag for o in self.organizations):
                if verbose: print(f"Convergence reached at iteration {t}")
                break

            # ---- synchronous step ----
            # 1) compute all hat_gamma on current (gamma, pi)
            hat_gamma = np.zeros(N, float)
            for i,o in enumerate(self.organizations):
                def obj(g):  # minimize negative augmented payoff
                    g = max(0.0, min(g, self.r_bar))
                    return -o.augmented_payoff_V_rho(g, gamma, pi)
                res = minimize_scalar(obj, bounds=(0.0, self.r_bar), method='bounded', options={'xatol':1e-6})
                hat_gamma[i] = float(res.x)

            # 2) update all gamma simultaneously
            new_gamma = gamma + Params.ETA * (hat_gamma - gamma)
            for i,o in enumerate(self.organizations):
                o.convg_flag = abs(new_gamma[i] - o.gamma_n) <= Params.PHI
                o.gamma_n = new_gamma[i]

            # 3) update all pi using the **new_gamma**
            for i,o in enumerate(self.organizations):
                idx_m1 = (i - 1) % N
                idx_m2 = (i - 2) % N
                o.pi_n = float(o.pi_n + Params.RHO * Params.ETA * (new_gamma[idx_m2] - new_gamma[idx_m1]))

            # 4) optional: center duals to avoid drift
            mean_pi = np.mean([o.pi_n for o in self.organizations])
            for o in self.organizations:
                o.pi_n -= mean_pi


    def plot_results(self, title=None):
        gamma_hist = np.array(self.gamma_history)
        zeta_hist = np.array(self.zeta_history)
        N = len(self.organizations)

        if title is None:
            title = "Algorithm 1: Convergence Results"

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title, fontsize=16)

        # Plot gamma_n convergence (each org)
        for i in range(N):
            axs[0].plot(gamma_hist[:, i], label=f'$\\gamma_{{{i}}}$')
        axs[0].set_title('Convergence of $\\gamma_n$ (Number of Rounds)')
        axs[0].set_xlabel('Iteration (t)')
        axs[0].set_ylabel('$\\gamma_n$')
        axs[0].grid(True)
        axs[0].legend(loc='upper right', fontsize='small')

        # Plot zeta_n convergence
        for i in range(N):
            axs[1].plot(zeta_hist[:, i], label=f'$\\zeta_{{{i}}}$')
        axs[1].set_title('Convergence of $\\zeta_n$ (Unit Monetary Transfer)')
        axs[1].set_xlabel('Iteration (t)')
        axs[1].set_ylabel('$\\zeta_n$')
        axs[1].grid(True)
        axs[1].legend(loc='upper right', fontsize='small')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

# --- Main Execution: two scenarios as you had before ---
if __name__ == "__main__":
    # reproducible randomness (optional)
    np.random.seed(0)

    # --- Scenario 1: Homogeneous Valuations ---
    print("Running Homogeneous Scenario...")
    orgs_homo = [Organization(org_id=i, u_n=10.0) for i in range(Params.N)]
    contract_homo = SmartContract(orgs_homo)
    print(f"Global r_bar computed = {contract_homo.r_bar:.6f}")
    contract_homo.run_simulation(max_iterations=500)
    contract_homo.plot_results("Homogeneous Scenario (u_n = 10 for all)")

    # --- Scenario 2: Heterogeneous Valuations ---
    print("\nRunning Heterogeneous Scenario...")
    u_vals_hetero = [4.0 if i < Params.N / 2 else 16.0 for i in range(Params.N)]
    orgs_hetero = [Organization(org_id=i, u_n=u_vals_hetero[i]) for i in range(Params.N)]
    contract_hetero = SmartContract(orgs_hetero)
    print(f"Global r_bar computed = {contract_hetero.r_bar:.6f}")
    contract_hetero.run_simulation(max_iterations=500)
    contract_hetero.plot_results(f"Heterogeneous Scenario (u_n=4 for first {Params.N//2}, u_n=16 for last {Params.N//2})")
