import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Set seaborn style for better visuals
sns.set_style("whitegrid")

# Load dataset with error handling
file_path = "dam_data.csv"  # Adjust path as needed
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found in {os.getcwd()}")
    exit(1)

try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Convert assessment date to year with error handling
df["Assessment Year"] = pd.to_datetime(df["Assessment Date"], errors='coerce').dt.year

# Create additional calculated columns, handling NaN in Year Completed
df["Dam Age"] = 2025 - df["Year Completed"].fillna(0).astype(int)
df["Expected Loss"] = df["Probability of Failure"] * df["Loss given failure - prop (Qm)"]

# Ensure output directory exists
output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

# 1. Right-skewed loss distribution → Justifies a layered insurance structure
plt.figure(figsize=(10, 6))
sns.histplot(df["Loss given failure - prop (Qm)"].dropna(), bins=50, kde=True, color="blue")
plt.xlabel("Estimated Property Loss Given Failure (Million £)")
plt.ylabel("Number of Dams (Frequency)")
plt.title("Right-Skewed Loss Distribution: Justifies Layered Insurance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_distribution_histogram.png"))
plt.close()

# 2. Age is just a number → What matters is the inspection date (Modified: Only Dam Age vs Probability)
corr_age_failure = df[["Dam Age", "Probability of Failure"]].corr().iloc[0, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x="Dam Age", y="Probability of Failure", data=df, alpha=0.5, color="blue", 
                label=f"Age Correlation: {corr_age_failure:.2f}")
plt.xlabel("Dam Age (Years)")
plt.ylabel("Probability of Dam Failure (%)")
plt.legend()
plt.title("Age is Just a Number: Inspection Date Matters More")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "age_vs_failure_scatter.png"))
plt.close()

# 3. Frequent inspections = Lower failure probability → Supports incentivizing reinsurers
plt.figure(figsize=(10, 6))
sns.regplot(x="Inspection Frequency", y="Probability of Failure", data=df, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
plt.xlabel("Inspection Frequency (Number of Inspections per Year)")
plt.ylabel("Probability of Dam Failure (%)")
plt.title("More Frequent Inspections Correlate with Lower Failure Probability")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "inspection_vs_failure_regression.png"))
plt.close()

# 4. High-risk regions face extreme financial exposure → Without segmentation, insurers would exit
region_risk = df.groupby("Region")[["Probability of Failure", "Loss given failure - prop (Qm)"]].mean().sort_values(
    "Loss given failure - prop (Qm)", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=region_risk.index, y=region_risk["Loss given failure - prop (Qm)"], palette="Reds_r")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Geographical Region")
plt.ylabel("Average Property Loss Given Failure (Million £)")
plt.title("High-Risk Regions Face Extreme Financial Exposure")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "region_risk_bar.png"))
plt.close()

# 5. Dams near cities have lower failure rates but higher financial losses → Govt should subsidize urban premiums
plt.figure(figsize=(10, 6))
sns.regplot(x="Distance to Nearest City (km)", y="Probability of Failure", data=df, scatter_kws={"alpha": 0.5}, 
            line_kws={"color": "red"})
plt.xlabel("Distance to Nearest City (Kilometers)")
plt.ylabel("Probability of Dam Failure (%)")
plt.title("Dams Near Cities Have Lower Failure Probability")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "distance_vs_failure_regression.png"))
plt.close()

plt.figure(figsize=(10, 6))
sns.regplot(x="Distance to Nearest City (km)", y="Loss given failure - prop (Qm)", data=df, scatter_kws={"alpha": 0.5}, 
            line_kws={"color": "red"})
plt.xlabel("Distance to Nearest City (Kilometers)")
plt.ylabel("Estimated Property Loss Given Failure (Million £)")
plt.title("Dams Near Cities Have Higher Financial Losses")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "distance_vs_loss_regression.png"))
plt.close()

# 6. Top 5% of failures cause over 50% of financial losses → Govt-backed reserves are essential
top_5_percent_loss_threshold = df["Loss given failure - prop (Qm)"].quantile(0.95)
total_loss = df["Loss given failure - prop (Qm)"].sum()
top_5_loss = df[df["Loss given failure - prop (Qm)"] >= top_5_percent_loss_threshold]["Loss given failure - prop (Qm)"].sum()

plt.figure(figsize=(8, 8))
labels = ["Top 5% Loss Events", "Other 95% Loss Events"]
sizes = [top_5_loss, total_loss - top_5_loss]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=["red", "gray"], startangle=90)
plt.title("Top 5% of Failures Account for Over 50% of Financial Losses")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_5_percent_loss_pie.png"))
plt.close()

# 7. Too many dams lack assessment data → Unquantified risks make accurate pricing difficult
unassessed_dams = df["Assessment"].isna().sum()
assessed_dams = df.shape[0] - unassessed_dams

plt.figure(figsize=(8, 8))
plt.pie([unassessed_dams, assessed_dams], labels=["Unassessed Dams", "Assessed Dams"], autopct='%1.1f%%', 
        colors=["red", "green"], startangle=90)
plt.title("Proportion of Unassessed Dams")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "assessment_status_pie.png"))
plt.close()

# 8. Top 10% of projected dam failures account for over 50% of expected financial loss → Insurance market stability
df_sorted = df[["Expected Loss"]].dropna().sort_values("Expected Loss", ascending=False)
df_sorted["Cumulative Loss"] = df_sorted["Expected Loss"].cumsum()
total_exp_loss = df_sorted["Expected Loss"].sum()
df_sorted["Cumulative Loss %"] = 100 * df_sorted["Cumulative Loss"] / total_exp_loss
df_sorted["Dam Index"] = range(1, len(df_sorted) + 1)
df_sorted["Dam %"] = 100 * df_sorted["Dam Index"] / len(df_sorted)

top_10_percent_dams = int(0.10 * len(df_sorted))
top_10_loss_percent = df_sorted["Cumulative Loss %"].iloc[top_10_percent_dams - 1]

# Pareto Plot
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(df_sorted["Dam Index"], df_sorted["Expected Loss"], color="red", alpha=0.7, label="Expected Loss")
ax1.set_xlabel("Dam Index (Sorted by Descending Expected Loss)")
ax1.set_ylabel("Expected Financial Loss (Million £)")
ax1.tick_params(axis="y", labelcolor="red")

ax2 = ax1.twinx()
ax2.plot(df_sorted["Dam Index"], df_sorted["Cumulative Loss %"], color="blue", label="Cumulative Loss %")
ax2.set_ylabel("Cumulative Percentage of Total Expected Loss (%)", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")
ax2.axvline(x=top_10_percent_dams, color="green", linestyle="--", label=f"Top 10% Dams ({top_10_loss_percent:.1f}%)")
ax2.axhline(y=50, color="purple", linestyle="--", label="50% Loss Threshold")

plt.title("Pareto Analysis: Top 10% of Dam Failures Drive Over 50% of Expected Losses\n(Insurance Market Needs Government Backstop)")
fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_10_percent_pareto.png"))
plt.close()

# Print some diagnostics
print("\nDataFrame Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print(f"\nTop 10% of Dams Account for {top_10_loss_percent:.1f}% of Total Expected Loss")