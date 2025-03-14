import pandas as pd
import os

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

# Verify unique values in the Region column to debug potential mismatches
print("\nUnique values in 'Region' column:")
print(df['Region'].unique())

# Define the three regions
regions = ['Flumevale', 'Lyndrassia', 'Navaldia']
region_dfs = {}

# Ensure output directory exists (optional, can remove if saving in current directory)
output_dir = "region_data"
os.makedirs(output_dir, exist_ok=True)
print(f"\nOutput directory absolute path: {os.path.abspath(output_dir)}")

# Split data by region
print("\nSplitting data by region...")
for region in regions:
    region_dfs[region] = df[df['Region'] == region]
    print(f"Rows for {region}: {len(region_dfs[region])}")

# Save each region's data to a separate CSV with an appropriate title
print("\nSaving region-specific CSVs...")
for region, region_df in region_dfs.items():
    output_file = os.path.join(output_dir, f"dam_data_{region.lower()}.csv")
    try:
        region_df.to_csv(output_file, index=False)
        absolute_path = os.path.abspath(output_file)
        print(f"Saved data for {region} to '{absolute_path}'")
    except Exception as e:
        print(f"Error saving {region} file: {e}")

# Verify the saved files
print("\nVerifying saved files:")
for region in regions:
    output_file = os.path.join(output_dir, f"dam_data_{region.lower()}.csv")
    if os.path.exists(output_file):
        print(f"File '{output_file}' exists with {len(pd.read_csv(output_file))} rows")
    else:
        print(f"File '{output_file}' was not created")

# Instructions for adding to Git
print("\nInstructions to add files to Git repository:")
print("1. Run 'git status' to check if the files are listed as untracked.")
print("2. Stage the files with: git add region_data/*.csv")
print("3. Commit the changes with: git commit -m 'Add region-specific dam data CSVs'")
print("4. Push to the remote repository with: git push origin <branch-name>")
print("   (Replace <branch-name> with your branch, e.g., 'main' or 'master')")