# Student Dropout Analysis

A Python data cleaning and transformation project based on the **Predict Students' Dropout and Academic Success** dataset from the UCI Machine Learning Repository.

## Project objective
The project studies student academic outcomes and prepares the dataset for later exploratory data analysis and machine learning.

Target outcomes:
- Dropout
- Enrolled
- Graduate

## Dataset
Source: UCI Machine Learning Repository  
Dataset: Predict Students' Dropout and Academic Success  
DOI: https://doi.org/10.24432/C5MC89

The dataset contains 4,424 records and 36 features. The dataset itself is not stored in this repository; the Python script accesses it through `ucimlrepo`.

## Week 2 work
- Dataset loading and inspection
- Missing-value check
- Duplicate check
- Numerical range checks
- Target distribution
- Feature/target separation
- Saving the prepared dataset

## Technologies
Python, Pandas, NumPy, Scikit-learn, UCI Machine Learning Repository

## How to run
```bash
pip install -r requirements.txt
python data_cleaning.py
```
