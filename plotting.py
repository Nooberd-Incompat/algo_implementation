import matplotlib.pyplot as plt
import numpy as np

# --- Data extracted from the project's output log ---

# Organization identifiers
org_ids = [f'Org_{i}' for i in range(10)]

# [cite_start]The size of the data partition assigned to each organization [cite: 9432]
partition_sizes = [3001, 3667, 4334, 5000, 5667, 6333, 6999, 7666, 8333, 9000]

# [cite_start]The final net monetary balance for each organization in dollars [cite: 9435]
net_balances = [
    0.38107,   # Org_0
    -0.55698,  # Org_1
    1.07066,   # Org_2
    -0.14735,  # Org_3
    -0.48404,  # Org_4
    1.15769,   # Org_5
    -1.56746,  # Org_6
    0.05014,   # Org_7
    0.53607,   # Org_8
    -0.43981   # Org_9
]

# --- Plotting Logic ---

# Create colors for the bars: green for receivers (positive balance), red for payers (negative balance)
bar_colors = ['#2ca02c' if bal >= 0 else '#d62728' for bal in net_balances]

# Set up the figure and the primary y-axis (for balances)
fig, ax1 = plt.subplots(figsize=(12, 7))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot the bar chart for net balances
bars = ax1.bar(org_ids, net_balances, color=bar_colors, alpha=0.8, label='Net Balance ($)')

# Add labels and title for the primary axis
ax1.set_xlabel('Organization ID', fontweight='bold')
ax1.set_ylabel('Final Net Balance ($)', fontweight='bold', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.axhline(0, color='black', linewidth=0.8) # Add a zero line for reference
plt.xticks(rotation=45, ha="right")

# Create a second y-axis that shares the same x-axis (for partition sizes)
ax2 = ax1.twinx()

# Plot the line chart for partition sizes on the second axis
line = ax2.plot(org_ids, partition_sizes, color='#1f77b4', marker='o', linestyle='--', label='Assigned Data Size')
ax2.set_ylabel('Assigned Data Size (Samples)', fontweight='bold', color='#1f77b4')
ax2.tick_params(axis='y', labelcolor='#1f77b4')
ax2.set_ylim(bottom=0) # Ensure the data size axis starts at 0

# Add a comprehensive title
plt.title('Correlation of Assigned Data Size and Final Monetary Payoff', fontsize=16, fontweight='bold')

# Add a legend
# To combine legends from two axes, we collect handles and labels from both
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Manually create legend entries for bar colors
import matplotlib.patches as mpatches
receiver_patch = mpatches.Patch(color='#2ca02c', label='Receiver (Paid)')
payer_patch = mpatches.Patch(color='#d62728', label='Payer (Paid In)')
ax1.legend(handles=[receiver_patch, payer_patch, line[0]], 
           labels=['Receiver (Paid)', 'Payer (Paid In)', 'Assigned Data Size'],
           loc='upper left')

# Ensure the plot layout is tight and save the figure
fig.tight_layout()
plt.savefig('payoff_vs_data_size.png', dpi=300)

# Display the plot
plt.show()