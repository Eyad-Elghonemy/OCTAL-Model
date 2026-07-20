# 🚗 Vehicle Price Prediction

A regression project that predicts vehicle prices from listing attributes (make, model, year, mileage, engine capacity, trim, body type, etc.) using **Linear Regression** as a baseline and **Random Forest Regression** as the main model, with hyperparameter tuning via `RandomizedSearchCV`.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Models](#-models)
- [Results](#-results)
- [Feature Importance](#-feature-importance)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Notes & Limitations](#-notes--limitations)
- [Future Work](#-future-work)

---

## 🔎 Overview

The goal of this project is to build a model that estimates a used vehicle's price based on its specifications. The pipeline covers:

1. Data cleaning and outlier handling
2. Exploratory Data Analysis (univariate, bivariate, multivariate)
3. Feature selection using correlation (Pearson / Spearman / Kendall) and ANOVA (p-value) tests
4. Feature engineering (estimating `engine_capacity` from horsepower)
5. Preprocessing (scaling, ordinal/label/one-hot encoding)
6. Model training, evaluation, and hyperparameter tuning
7. Feature importance analysis

**Target variable:** `price` (modeled as `log1p(price)` to reduce right-skew)

---

## 🗂 Dataset

- **Raw size:** 1,000,000 rows × 21 columns
- **After cleaning (removing price/mileage outliers):** ~945,955 rows × 12 columns
- **Train / Test split:** 70% / 30% (`random_state=45`)
  - `X_train`: 662,168 rows
  - `X_test`: 283,787 rows

### Features kept for modeling
| Type | Columns |
|---|---|
| Numerical | `year`, `mileage`, `engine_capacity` |
| Ordinal | `trim` (Base → LX → EX → Sport → Touring → Limited) |
| High-cardinality categorical | `make`, `model` |
| Nominal categorical | `transmission`, `fuel_type`, `drivetrain`, `body_type` |

### Columns dropped
Based on domain reasoning and weak statistical relevance to `price`:
`vehicle_age`, `engine_hp`, `owner_count`, `mileage_per_year`, `brand_popularity`, `interior_color`, `condition`, `seller_type`, `exterior_color`, `accident_history`

---

## 🔬 Project Workflow

1. **Cleaning:** stripped column names, estimated `engine_capacity` from `engine_hp`
2. **Feature relevance check:**
   - Pearson / Spearman / Kendall correlation for numerical features vs. `price`
   - One-way ANOVA (p-value) for categorical features vs. `price`
3. **Outlier handling:**
   - Removed a large spike of listings capped at `price == 1500`
   - Removed listings with `mileage >= 300,000`
4. **Transformation:** applied `log1p` on `price` to normalize its distribution
5. **EDA:** distribution plots, price-by-make/transmission/fuel/body-type/trim boxplots, price vs. mileage/engine capacity scatter plots

---

## ⚙️ Preprocessing Pipeline

Built with `ColumnTransformer` + `Pipeline`:

| Column group | Columns | Transformer |
|---|---|---|
| Numerical | `year`, `mileage`, `engine_capacity` | `StandardScaler` |
| Ordinal | `trim` | `OrdinalEncoder` (custom order) |
| Label | `make`, `model` | `OrdinalEncoder` (unknown → -1) |
| Nominal | `transmission`, `fuel_type`, `drivetrain`, `body_type` | `OneHotEncoder(drop='first')` |

Final feature matrix shape: **17 columns** after encoding (`X_train_final`: 662,168 × 17).

The fitted preprocessor is saved to `models/preprocessor.pkl`.

---

## 🤖 Models

### 1. Linear Regression (baseline)
A simple baseline to confirm the target requires a non-linear model. Residual analysis showed a curved (non-random) pattern — clear evidence of underfitting.

### 2. Random Forest Regression — Base
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=45,
    n_jobs=-1
)
```

### 3. Random Forest Regression — Tuned (RandomizedSearchCV)
A `RandomizedSearchCV` (`cv=3`, `n_iter=15`, scoring=`neg_root_mean_squared_error`) was run once on Google Colab over this search space:

```python
params_narrow = {
    'n_estimators': [150, 200, 250],
    'max_depth': [15, 20, 25],
    'min_samples_split': [3, 5, 8],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt']
}
```

**Best parameters found:**
```python
RandomForestRegressor(
    n_estimators=250,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=25,
    random_state=45,
    n_jobs=-1
)
```

> ⚠️ The tuned model (~7 GB) was trained and evaluated on Colab but could not be downloaded/committed due to its size. Only the best hyperparameters and the resulting metrics below were kept; the model itself is **not included** in this repo, and the local pipeline currently ships the **base Random Forest** (`forest_base.pkl`) instead.

---

## 📊 Results

Evaluated on `price_log` (RMSE/MAE are in log-price units):

| Model | Split | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---|---|---|
| Linear Regression | Train | 0.7005 | 0.5508 | 0.0009 |
| Linear Regression | Test | 0.7002 | 0.5502 | 0.0009 |
| Random Forest (base) | Train | 0.0886 | 0.0669 | 0.9840 |
| Random Forest (base) | Test | 0.1160 | 0.0872 | 0.9726 |
| **Random Forest (tuned)** | **Train** | **0.0565** | **0.0430** | **0.9935** |
| **Random Forest (tuned)** | **Test** | **0.1134** | **0.0859** | **0.9738** |

**Takeaways:**
- Linear Regression fails to capture the non-linear relationship between features and price (R² ≈ 0), confirming the need for a tree-based model.
- Random Forest dramatically improves performance, explaining ~97–99% of price variance.
- Tuning improves training fit further but yields only a marginal test-set gain over the base model, and the train/test gap widens slightly — suggesting the tuned model is closer to (but not clearly past) the point of overfitting. The base model remains a strong, more conservative choice.

---

## 🌟 Feature Importance

Computed from the **base Random Forest** model (`forest.feature_importances_`) after preprocessing:

![Feature Importance](assets/feature_importance.png)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `engine_capacity` | 0.3318 |
| 2 | `year` | 0.3188 |
| 3 | `mileage` | 0.2250 |
| 4 | `make` | 0.0774 |
| 5 | `model` | 0.0279 |
| 6 | `body_type_Pickup Truck` | 0.0049 |
| 7 | `body_type_Hatchback` | 0.0035 |
| 8 | `trim` | 0.0023 |
| 9 | `body_type_SUV` | 0.0015 |
| 10 | `body_type_Sedan` | 0.0013 |
| 11 | `fuel_type_Electric` | 0.0011 |
| 12 | `body_type_Minivan` | 0.0011 |
| 13 | `body_type_Wagon` | 0.0009 |
| 14 | `transmission_Manual` | 0.0008 |
| 15 | `drivetrain_FWD` | 0.0007 |
| 16 | `drivetrain_RWD` | 0.0007 |
| 17 | `fuel_type_Gasoline` | 0.0007 |

**Interpretation:** `engine_capacity`, `year`, and `mileage` together drive over **87%** of the model's predictive power, while `make` adds meaningful brand-level signal. Categorical dummy variables (body type, fuel type, drivetrain, transmission) contribute comparatively little individually.

---

## 📁 Project Structure

```
.
├── notebooks/
│   └── notebook.ipynb           # Full analysis, EDA, modeling
├── dataset/
│   └── vehicle_price_prediction.csv
├── models/
│   ├── preprocessor.pkl         # Fitted ColumnTransformer
│   └── forest_base.pkl          # Base Random Forest model
├── assets/
│   └── feature_importance.png
└── README.md
```

---

## 🛠 Installation

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

**Core dependencies:** `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, `scipy`, `joblib`

---

## ▶️ Usage

```python
import joblib
import pandas as pd

preprocessor = joblib.load('models/preprocessor.pkl')
model = joblib.load('models/forest_base.pkl')

# new_data: a DataFrame with the same raw columns used during training
X_new = preprocessor.transform(new_data)
price_log_pred = model.predict(X_new)

# Convert back from log scale
price_pred = pd.Series(price_log_pred).apply(lambda x: __import__('numpy').expm1(x))
```

---

## ⚠️ Notes & Limitations

- The tuned Random Forest was only trained once on Colab; results are recorded here but the model file itself was too large (~7 GB) to be exported and versioned.
- Metrics are reported on `price_log`, not the raw price — keep this in mind when comparing to other price-prediction benchmarks.
- `make` and `model` are label/ordinal-encoded, not one-hot encoded, due to their high cardinality — this can introduce artificial ordering; tree models handle this reasonably well, but it's worth revisiting with target encoding.

## 🚀 Future Work

- Re-run and persist the tuned Random Forest in a more storage-efficient format (e.g., compressed `joblib`, or reduce `n_estimators`)
- Try target/frequency encoding for `make` and `model` instead of label encoding
- Experiment with Gradient Boosting models (XGBoost / LightGBM / CatBoost)
- Add SHAP-based feature importance for more robust, model-agnostic interpretability
- Deploy as a simple API/web app for interactive price predictions