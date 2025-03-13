import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set seaborn style for consistency with previous scripts (optional)
import seaborn as sns
sns.set_style("whitegrid")

# Simulate claim sizes (in million £)
claim_sizes = np.linspace(0, 100, 1000)  # From 0 to 100M £

# Define cost functions
def insurer_cost(claim_size):
    # Direct insurers: Lower cost per claim, high frequency, slight increase until spike
    base_cost = 0.05 * claim_size  # Linear base cost
    spike_threshold = 1  # £1M threshold where cost spikes
    if claim_size <= spike_threshold:
        return base_cost
    else:
        return base_cost + 0.5 * (claim_size - spike_threshold)**2  # Quadratic spike after threshold

def reinsurer_cost(claim_size):
    # Reinsurers: Higher cost per claim, lower frequency, steeper curve
    base_cost = 0.2 * claim_size  # Steeper linear base cost
    sustainable_threshold = 50  # £50M threshold where cost becomes unsustainable
    if claim_size <= sustainable_threshold:
        return base_cost
    else:
        return base_cost + 0.8 * (claim_size - sustainable_threshold)**2  # Steeper spike after threshold

# Vectorize cost functions for numpy array
insurer_costs = np.array([insurer_cost(x) for x in claim_sizes])
reinsurer_costs = np.array([reinsurer_cost(x) for x in claim_sizes])

# Define sustainable cost limits (hypothetical, in million £)
insurer_sustainable_limit = 1.0  # £1M sustainable cost limit for insurers
reinsurer_sustainable_limit = 20.0  # £20M sustainable cost limit for reinsurers (adjusted for realism)

# Find thresholds
insurer_threshold_idx = np.argmax(insurer_costs > insurer_sustainable_limit)
insurer_threshold = claim_sizes[insurer_threshold_idx] if insurer_threshold_idx > 0 else claim_sizes[-1]
reinsurer_threshold_idx = np.argmax(reinsurer_costs > reinsurer_sustainable_limit)
reinsurer_threshold = claim_sizes[reinsurer_threshold_idx] if reinsurer_threshold_idx > 0 else claim_sizes[-1]

# Ensure output directory exists
output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

# Plot Cost vs. Claim Size
plt.figure(figsize=(12, 6))
plt.plot(claim_sizes, insurer_costs, label="Direct Insurer Cost", color="blue")
plt.plot(claim_sizes, reinsurer_costs, label="Reinsurer Cost", color="red")

# Add sustainable limit lines
plt.axhline(y=insurer_sustainable_limit, color="blue", linestyle="--", alpha=0.5, 
            label=f"Insurer Sustainable Limit (£{insurer_sustainable_limit:.1f}M)")
plt.axhline(y=reinsurer_sustainable_limit, color="red", linestyle="--", alpha=0.5, 
            label=f"Reinsurer Sustainable Limit (£{reinsurer_sustainable_limit:.1f}M)")

# Add threshold lines
plt.axvline(x=insurer_threshold, color="green", linestyle="--", 
            label=f"Insurer Threshold (£{insurer_threshold:.1f}M)")
plt.axvline(x=reinsurer_threshold, color="purple", linestyle="--", 
            label=f"Reinsurer Threshold (£{reinsurer_threshold:.1f}M)")

plt.xlabel("Claim Size (Million £)")
plt.ylabel("Cost per Claim (Million £)")
plt.title("Break-Even Analysis: Insurers vs. Reinsurers\nThresholds for Risk Transfer")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "break_even_analysis.png"))
plt.close()

# Create Summary Table
summary_data = {
    "Entity": ["Direct Insurers", "Reinsurers", "Government"],
    "Claim Size Range (£M)": [f"0 - {insurer_threshold:.1f}", 
                              f"{insurer_threshold:.1f} - {reinsurer_threshold:.1f}", 
                              f">{reinsurer_threshold:.1f}"],
    "Max Sustainable Cost (£M)": [insurer_sustainable_limit, reinsurer_sustainable_limit, "N/A"],
    "Responsibility": ["Low to Medium Claims", "High Claims", "Extreme Claims"]
}
summary_df = pd.DataFrame(summary_data)

# Save table as CSV
table_path = os.path.join(output_dir, "break_even_summary.csv")
summary_df.to_csv(table_path, index=False)

# Print table to console for verification
print("\nBreak-Even Analysis Summary:")
print(summary_df)

# Additional diagnostics
print(f"\nInsurer Threshold: £{insurer_threshold:.1f}M")
print(f"Reinsurer Threshold: £{reinsurer_threshold:.1f}M")