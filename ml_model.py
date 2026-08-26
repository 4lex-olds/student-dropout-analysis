import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load the UCI dataset
dataset = fetch_ucirepo(id=697)

X = dataset.data.features.copy()
y = dataset.data.targets.copy()

target_column = y.columns[0]

# Convert target to a 1-D Series
y = y[target_column]

print("Dataset shape:", X.shape)
print("\nTarget distribution:")
print(y.value_counts())
print("\nTarget percentages:")
print(y.value_counts(normalize=True).mul(100).round(2))


# 2. Train/test split
# Stratification keeps the class proportions approximately similar.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 3. Define models
# Logistic Regression needs scaling for this dataset.
logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

decision_tree = DecisionTreeClassifier(
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

models = {
    "Logistic Regression": logistic_pipeline,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest
}


# 4. Stratified 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

print("\n===== CROSS-VALIDATION RESULTS =====")

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro"
    )

    results.append({
        "Model": name,
        "Mean Macro F1": scores.mean(),
        "Std Macro F1": scores.std()
    })

    print(
        f"{name}: "
        f"Mean Macro F1 = {scores.mean():.4f}, "
        f"Std = {scores.std():.4f}"
    )


# 5. Fit each model and evaluate on the held-out test set
evaluation_rows = []

print("\n===== TEST SET RESULTS =====")

for name, model in models.items():

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )
    balanced_acc = balanced_accuracy_score(y_test, predictions)

    evaluation_rows.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
        "Balanced Accuracy": balanced_acc
    })

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))


# 6. Model comparison table
results_df = pd.DataFrame(evaluation_rows)

print("\n===== MODEL COMPARISON =====")
print(results_df.round(4))

results_df.to_csv("model_comparison.csv", index=False)


# 7. Confusion matrix for the best model by Macro F1
best_model_name = results_df.loc[
    results_df["Macro F1"].idxmax(),
    "Model"
]

best_model = models[best_model_name]
best_model.fit(X_train, y_train)
best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()

print(f"\nBest model by test Macro F1: {best_model_name}")
