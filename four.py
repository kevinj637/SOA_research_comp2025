import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Set seaborn style for better visuals
sns.set_style("whitegrid")

# Load dataset with error handling
file_path = "dam_data_navaldia.csv"  # Adjust path as needed
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found in {os.getcwd()}")
    exit(1)

try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Create Total Loss Given Failure column by summing individual losses
df["Total Loss Given Failure"] = (
    df["Loss given failure - prop (Qm)"].fillna(0) +
    df["Loss given failure - liab (Qm)"].fillna(0) +
    df["Loss given failure - BI (Qm)"].fillna(0)
)

# Ensure output directory exists
output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

# Calculate q1, q3, median for IQR and whisker limits
q1 = df["Total Loss Given Failure"].quantile(0.25)
q3 = df["Total Loss Given Failure"].quantile(0.75)
iqr = q3 - q1
median = df["Total Loss Given Failure"].median()
lower_whisker = q1 - 1.5 * iqr
upper_whisker = q3 + 1.5 * iqr

# Define additional percentiles for plotting
additional_percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
additional_percentile_probs = [p / 100 for p in additional_percentiles]
additional_percentile_values = df["Total Loss Given Failure"].quantile(additional_percentile_probs).to_dict()

# Create Boxplot
plt.figure(figsize=(10, 8))  # Increased height to accommodate more lines
sns.boxplot(y=df["Total Loss Given Failure"], color="lightblue", showfliers=True)

# Add additional percentile lines
colors = sns.color_palette("tab10", n_colors=len(additional_percentiles))
for i, (prob, value) in enumerate(additional_percentile_values.items()):
    perc = int(prob * 100)
    linestyle = "--" if perc == 50 else ":"
    plt.axhline(y=value, color=colors[i], linestyle=linestyle, alpha=0.7,
                label=f"{perc}th Percentile: £{value:.2f}M")

# Add whisker lines
plt.axhline(y=lower_whisker, color="purple", linestyle="-.", alpha=0.5,
            label=f"Lower Whisker: £{lower_whisker:.2f}M")
plt.axhline(y=upper_whisker, color="purple", linestyle="-.", alpha=0.5,
            label=f"Upper Whisker: £{upper_whisker:.2f}M")

plt.ylabel("Total Loss Given Failure (Million £)")
plt.title("Boxplot of Total Loss Given Failure\n(With Additional Percentiles, IQR, Median, and Outliers)")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside plot
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(output_dir, "navaldia_boxplot.png"), bbox_inches="tight")
plt.close()

# Print diagnostics
print("\nTotal Loss Given Failure Statistics:")
print(f"Median: £{median:.2f}M")
print(f"25th Percentile (Q1): £{q1:.2f}M")
print(f"75th Percentile (Q3): £{q3:.2f}M")
print(f"IQR: £{iqr:.2f}M")
print(f"Lower Whisker: £{lower_whisker:.2f}M")
print(f"Upper Whisker: £{upper_whisker:.2f}M")
print(f"Number of Outliers Below Lower Whisker: {len(df[df['Total Loss Given Failure'] < lower_whisker])}")
print(f"Number of Outliers Above Upper Whisker: {len(df[df['Total Loss Given Failure'] > upper_whisker])}")

# Additional percentiles
print("\nAdditional Percentiles: ")
for perc in additional_percentiles:
    prob = perc / 100
    value = df["Total Loss Given Failure"].quantile(prob)
    print(f"{perc}th Percentile: £{value:.2f}M")