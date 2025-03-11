import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

print("Starting script...")

# Check current working directory
print("Current Working Directory:", os.getcwd())

# Verify input file exists
input_file = "dam_data.csv"
if not os.path.exists(input_file):
    print(f"Error: '{input_file}' not found in {os.getcwd()}")
    exit(1)
print(f"Found '{input_file}'")

# Load the dataset
try:
    df = pd.read_csv(input_file)
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Display initial info
print("Original DataFrame:")
print(df.head())
print("\nDataFrame Info:")
print(df.info())

# --- Date Handling: Convert to Numeric Years ---
def extract_year(value):
    if pd.isna(value):
        return np.nan
    value = str(value)
    try:
        if '/' in value:
            return int(value.split('/')[-1])  # Extract year from "DD/MM/YYYY"
        elif value[-1].isalpha():
            return int(value[:-1])  # Remove modifier (e.g., "1968M" -> 1968)
        else:
            return int(value)  # Plain year
    except (ValueError, IndexError):
        return np.nan

print("Converting date columns to numeric years...")
df['Last Inspection Date'] = df['Last Inspection Date'].apply(extract_year)
df['Assessment Date'] = df['Assessment Date'].apply(extract_year)
df['Years Modified'] = df['Years Modified'].apply(extract_year)
print("Date columns converted")

# --- Define Columns for Imputation ---
categorical_cols = ['Region', 'Regulated Dam', 'Primary Purpose', 'Primary Type', 
                    'Spillway', 'Hazard', 'Assessment']
numeric_cols = [
    'Height (m)', 'Length (km)', 'Volume (m3)', 'Year Completed', 'Surface (km2)',
    'Drainage (km2)', 'Inspection Frequency', 'Distance to Nearest City (km)',
    'Probability of Failure', 'Loss given failure - prop (Qm)',
    'Loss given failure - liab (Qm)', 'Loss given failure - BI (Qm)',
    'Last Inspection Date', 'Assessment Date', 'Years Modified'
]

print("Encoding categoricals...")
label_encoders = {}
df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = df_encoded[col].fillna('Missing')
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
print("Categoricals encoded")

# Combine columns for imputation
cols_to_impute = numeric_cols + categorical_cols
data_to_impute = df_encoded[cols_to_impute]

# Scale the data
print("Scaling data...")
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_to_impute)
print("Data scaled")

# --- Apply KNN Imputation with K=10 ---
print("Applying KNN imputation with K=10...")
imputer = KNNImputer(n_neighbors=10, weights='uniform')
imputed_scaled = imputer.fit_transform(data_scaled)

# Transform back to original scale
data_imputed = pd.DataFrame(scaler.inverse_transform(imputed_scaled), columns=cols_to_impute)

# --- Decode Categorical Columns ---
print("Decoding categoricals...")
for col in categorical_cols:
    imputed_values = data_imputed[col].to_numpy()
    imputed_int = np.rint(imputed_values).astype(int)
    imputed_int = np.clip(imputed_int, 0, len(label_encoders[col].classes_) - 1)
    data_imputed[col] = label_encoders[col].inverse_transform(imputed_int)
print("Categoricals decoded")

# --- Round Year Completed and Years Modified to Nearest Integer ---
print("Rounding 'Year Completed' and 'Years Modified' to nearest year...")
data_imputed['Year Completed'] = data_imputed['Year Completed'].round().astype(int)
data_imputed['Years Modified'] = data_imputed['Years Modified'].round().astype(int)

# --- Preserve Original Non-Missing Values ---
df_final = df.copy()
for col in cols_to_impute:
    df_final[col] = df_final[col].where(df_final[col].notna(), data_imputed[col])

# --- Drop 'Last Inspection Date' Column ---
print("Dropping 'Last Inspection Date' column...")
df_final = df_final.drop(columns=['Last Inspection Date'])

# Display the imputed DataFrame
print("\nDataFrame after KNN imputation, rounding years, and dropping 'Last Inspection Date':")
print(df_final.head())

# Save the imputed DataFrame
output_file = "dam_data_imputed_fixed.csv"
try:
    df_final.to_csv(output_file, index=False)
    print(f"Imputed data saved to '{output_file}' with K=10, Weights='uniform'.")
except Exception as e:
    print(f"Error saving file: {e}")