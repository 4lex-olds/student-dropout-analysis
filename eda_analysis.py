import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Load the UCI dataset
dataset = fetch_ucirepo(id=697)
X = dataset.data.features.copy()
y = dataset.data.targets.copy()
df = pd.concat([X, y], axis=1)

target_column = y.columns[0]

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df[target_column].value_counts())

print("\nTarget percentages:")
print(
    df[target_column]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# 1. Target distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x=target_column)
plt.title("Student Outcome Distribution")
plt.xlabel("Student Outcome")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# 2. Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age at enrollment", bins=20, kde=True)
plt.title("Age at Enrollment Distribution")
plt.xlabel("Age at Enrollment")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# 3. First-semester approved units by outcome
plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df,
    x=target_column,
    y="Curricular units 1st sem (approved)"
)
plt.title("First-Semester Approved Units by Student Outcome")
plt.xlabel("Student Outcome")
plt.ylabel("Approved Units")
plt.tight_layout()
plt.show()

# 4. First-semester grade by outcome
plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df,
    x=target_column,
    y="Curricular units 1st sem (grade)"
)
plt.title("First-Semester Grade by Student Outcome")
plt.xlabel("Student Outcome")
plt.ylabel("Grade")
plt.tight_layout()
plt.show()

# 5. Scholarship holder vs outcome
summary = pd.crosstab(
    df["Scholarship holder"],
    df[target_column],
    normalize="index"
) * 100

summary.plot(kind="bar", stacked=True, figsize=(9, 5))
plt.title("Student Outcomes by Scholarship Status")
plt.xlabel("Scholarship Holder (Encoded)")
plt.ylabel("Percentage of Students")
plt.legend(title="Student Outcome")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 6. Relationship between first- and second-semester approved units
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="Curricular units 1st sem (approved)",
    y="Curricular units 2nd sem (approved)",
    hue=target_column,
    alpha=0.6
)
plt.title("First vs Second Semester Approved Units")
plt.xlabel("1st Semester Approved Units")
plt.ylabel("2nd Semester Approved Units")
plt.legend(title="Student Outcome")
plt.tight_layout()
plt.show()

# 7. Correlation heatmap for selected numerical variables
selected = [
    "Age at enrollment",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)"
]

corr = df[selected].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Selected Numerical Variables")
plt.tight_layout()
plt.show()

# Basic numerical summary for reporting
print("\nSelected variable summary:")
print(df[selected].describe().round(2))
