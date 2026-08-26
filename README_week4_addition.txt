## Week 4 ML model evaluation

`ml_model.py` contains a basic machine learning evaluation workflow for the student outcome classification project.

The script includes:
- Stratified train/test split
- Logistic Regression
- Decision Tree
- Random Forest
- Stratified 5-fold cross-validation
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Balanced Accuracy
- Classification reports
- Model comparison
- Confusion matrix

The best model is selected by the highest Macro F1 on the held-out test set in this educational example.
