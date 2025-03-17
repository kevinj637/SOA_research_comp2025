import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('dam_data.csv')

# Identify the columns that contain "failure" (case-insensitive)
failure_columns = [col for col in data.columns if 'failure' in col.lower()]

# Create a new column "total_loss" as the sum of the failure columns
data['total_loss'] = data[failure_columns].sum(axis=1)

# Use the properly capitalized column name for inspection frequency
inspection_column = 'Inspection Frequency'
if inspection_column not in data.columns:
    raise ValueError(f"Column '{inspection_column}' not found in data. Check the column name.")

# Create the scatterplot of total_loss vs. Inspection Frequency
plt.figure(figsize=(10, 6))
plt.scatter(data['total_loss'], data[inspection_column], alpha=0.7, edgecolor='k')
plt.xlabel('Total Loss (Sum of Failure Columns)', fontsize=12)
plt.ylabel('Inspection Frequency', fontsize=12)
plt.title('Scatterplot of Total Loss vs. Inspection Frequency', fontsize=14)
plt.grid(True)
plt.show()
