# Titanic Survival Classification

**Arch Technologies — Data Science Internship Task 1**

## Objective

Build and evaluate multiple classification models to predict whether a Titanic passenger survived based on passenger features such as age, gender, ticket class, fare, and other information. This project demonstrates the full data science workflow from data loading to model deployment.

## Dataset

- **Source:** Kaggle Titanic Dataset
- **Training samples:** 891 passengers
- **Features:** 12 original columns (PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked)
- **Target variable:** `Survived` (0 = Did not survive, 1 = Survived)
- **Survival rate:** 38.4%

## Features Used

**Numerical:** Pclass, Age, SibSp, Parch, Fare, FamilySize, IsAlone

**Categorical:** Sex, Embarked, CabinLetter, Title

### Feature Engineering

- **FamilySize:** `SibSp + Parch + 1` — total family members including the passenger
- **IsAlone:** Binary indicator (1 if traveling alone, 0 otherwise)
- **Title:** Extracted from Name (Mr, Mrs, Miss, Master, Rare) — captures social status and age group
- **CabinLetter:** First letter of Cabin — indicates deck location on the ship

## Preprocessing

All preprocessing uses scikit-learn Pipelines to prevent data leakage:

- **No manual imputation before train/test split** — the pipeline handles this
- **Numerical features:** median imputation + StandardScaler
- **Categorical features:** most-frequent imputation + OneHotEncoder
- Missing values in Age (177) and Embarked (2) are handled by the pipeline after splitting

## Models Evaluated

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8436 | 0.8254 | 0.7536 | 0.7879 | 0.8733 |
| Decision Tree | 0.8045 | 0.7656 | 0.7101 | 0.7368 | 0.8277 |
| Random Forest | 0.8324 | 0.8421 | 0.6957 | 0.7619 | 0.8469 |
| Tuned Random Forest | 0.8156 | 0.8103 | 0.6812 | 0.7402 | 0.8457 |

## Model Selection

Models were compared using **5-fold stratified cross-validation F1-scores** on the training set. The test set was reserved for final evaluation only.

**Logistic Regression was selected because it achieved the highest mean cross-validation F1-score of 0.7715 (+/- 0.0346).**

## Final Results

The selected model was evaluated once on the held-out test set:

| Metric | Score |
|--------|------:|
| Accuracy | 0.8436 |
| Precision | 0.8254 |
| Recall | 0.7536 |
| F1-score | 0.7879 |
| ROC-AUC | 0.8733 |

### Cross-Validation Results

| Model | Mean CV F1 | Std |
|-------|------------|-----|
| Logistic Regression | 0.7715 | 0.0346 |
| Decision Tree | 0.7415 | 0.0158 |
| Random Forest | 0.7474 | 0.0291 |
| Tuned Random Forest | 0.7611 | 0.0000 |

## Feature Interpretation

Top features based on Logistic Regression coefficients:

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| Title_Mr | -1.3133 | Adult male strongly decreases survival |
| Title_Master | +1.2245 | Young boy strongly increases survival |
| CabinLetter_E | +0.8800 | Deck E location increases survival |
| CabinLetter_D | +0.7865 | Deck D location increases survival |
| Sex_female | +0.7790 | Being female increases survival |
| Sex_male | -0.7680 | Being male decreases survival |
| CabinLetter_U | -0.6738 | Unknown deck decreases survival |
| Title_Mrs | +0.6036 | Married woman increases survival |
| Pclass | -0.5847 | Lower class decreases survival |
| Age | -0.4177 | Older age decreases survival |

## Project Structure

```
Arch Technologies Task 1/
├── notebooks/
│   └── titanic_survival_classification.ipynb
├── models/
│   └── titanic_survival_model.pkl
├── data/
│   └── titanic.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the Project

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place the Kaggle Titanic dataset (`titanic.csv`) in the `data/` directory
4. Open and run the notebook: `jupyter notebook notebooks/titanic_survival_classification.ipynb`

## Saved Model

The trained Logistic Regression pipeline is saved at:

```
models/titanic_survival_model.pkl
```

The model can be loaded and used for predictions:

```python
import joblib
import pandas as pd

model = joblib.load("models/titanic_survival_model.pkl")

# Example prediction
passenger = pd.DataFrame({
    'Pclass': [1], 'Age': [25], 'SibSp': [0], 'Parch': [0],
    'Fare': [80.0], 'FamilySize': [1], 'IsAlone': [1],
    'Sex': ['female'], 'Embarked': ['S'], 'CabinLetter': ['U'], 'Title': ['Miss']
})

prediction = model.predict(passenger)  # 1 = Survived, 0 = Did not survive
probability = model.predict_proba(passenger)[:, 1]  # Survival probability
```

## Limitations

- The dataset is relatively small (891 samples)
- Many Cabin values were missing (77%), limiting the usefulness of the deck feature
- The model may not generalize perfectly to different disaster scenarios
- Some features like Name and Ticket could potentially yield more information with deeper NLP techniques

## Future Improvements

- Try gradient boosting models (XGBoost, LightGBM)
- Ensemble/stacking of multiple models
- More advanced feature engineering from Name patterns
- Use the separate test.csv for Kaggle submission
- Target encoding for high-cardinality features
- Deploy the model as a web API

## Author

**Muhammad Zeeshan** — Data Science Internship, Arch Technologies
