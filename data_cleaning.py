import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

# Load the UCI dataset
dataset = fetch_ucirepo(id=697)
X = dataset.data.features.copy()
y = dataset.data.targets.copy()
df = pd.concat([X, y], axis=1)

print("Original shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# Check data types
print("\nData types:")
print(df.dtypes)

# Check missing values
missing = df.isna().sum()
missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percentage": (missing / len(df) * 100).round(2)
})
print("\nMissing-value report:")
print(missing_report[missing_report["missing_count"] > 0])
if missing.sum() == 0:
    print("No missing values were found.")

# Check duplicates
duplicate_count = df.duplicated().sum()
print("\nDuplicate rows:", duplicate_count)
if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Shape after duplicate removal:", df.shape)

# Numerical summary
print("\nNumerical summary:")
print(df.describe().T)

# Target distribution
target_column = y.columns[0]
print("\nTarget distribution:")
print(df[target_column].value_counts())

print("\nTarget percentages:")
print(df[target_column].value_counts(normalize=True).mul(100).round(2))

# Check negative values in numeric columns
numeric_columns = df.select_dtypes(include=np.number).columns
negative_counts = (df[numeric_columns] < 0).sum()
negative_report = negative_counts[negative_counts > 0]
print("\nColumns containing negative values:")
print(negative_report if not negative_report.empty else "None found.")

# Separate features and target
X_clean = df.drop(columns=[target_column])
y_clean = df[target_column]
print("\nFeature shape:", X_clean.shape)
print("Target shape:", y_clean.shape)

# Save prepared data
df.to_csv("cleaned_student_data.csv", index=False)
print("\nPrepared dataset saved as: cleaned_student_data.csv")
