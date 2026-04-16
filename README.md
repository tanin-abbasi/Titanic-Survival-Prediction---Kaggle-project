# Titanic-Survival-Prediction---Kaggle-project

A Machine Learning project to predict passenger survival on the Titanic using the Kaggle dataset.

---

## Overview
This project builds a complete ML pipeline including:
- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training
- Kaggle submission

---

## Dataset
- Source: Kaggle Titanic Competition  
- Train set: labeled data (`Survived`)  
- Test set: used for final prediction  

---

## Workflow

### 1. Data Preprocessing
- Handled missing values in `Age` and `Embarked`
- Encoded categorical features

### 2. Feature Engineering
- Extracted `Title` from `Name`
- Created `FamilySize = SibSp + Parch + 1`
- Created `IsAlone`
- Created interaction feature `Sex_Pclass`

### 3. Encoding
- One-Hot Encoding applied to categorical variables

---

## Model
Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
