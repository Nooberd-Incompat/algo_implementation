import matplotlib.pyplot as plt
import numpy as np

# --- Data extracted from the project's output log ---
# This shows the distributed loss of the global model after each round of federated training.
rounds = [1, 2, 3, 4, 5]
loss_values = [
    0.07684899800223916,  # Round 1 [cite: 177]
    0.0571388986808066,   # Round 2 [cite: 178]
    0.0523223061911322,   # Round 3 [cite: 179]
    0.04934140478503267,  # Round 4 [cite: 180]
    0.04650139436241972   # Round 5 [cite: 181]
]

# --- Plotting Logic ---

# Set up the figure
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot the loss values
plt.plot(rounds, loss_values, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8, label='Distributed Loss')

# Add labels and title
plt.xlabel('Federated Learning Round', fontweight='bold')
plt.ylabel('Distributed Loss', fontweight='bold')
plt.title('Global Model Distributed Loss per Federated Learning Round', fontsize=16, fontweight='bold')

# Set ticks to be integers for the rounds
plt.xticks(rounds)

# Adjust y-axis to better visualize the change
plt.ylim(bottom=min(loss_values) * 0.95, top=max(loss_values) * 1.05)

# Add a legend
plt.legend()

# Ensure the plot layout is tight and save the figure
plt.tight_layout()
plt.savefig('fl_loss_plot.png', dpi=300)

# Display the plot
plt.show()