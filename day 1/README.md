# 📈 Linear Regression from Scratch

A hands-on implementation of **Simple Linear Regression** built from scratch using NumPy, then validated against scikit-learn. Uses a real-world Heights & Weights dataset to model the relationship between human height and body weight.

---

## 📋 Table of Contents

- [Dataset](#dataset)
- [Topics Covered](#topics-covered)
- [Mathematical Background](#mathematical-background)
- [Gradient Descent](#gradient-descent)
- [Feature Scaling](#feature-scaling)
- [Evaluation Metrics](#evaluation-metrics)
- [Sklearn Comparison](#sklearn-comparison)
- [Dependencies](#dependencies)

---

## 📦 Dataset

| Property | Detail |
|---|---|
| **Source** | Kaggle — `burnoutminer/heights-and-weights-dataset` |
| **File** | `SOCR-HeightWeight.csv` |
| **Input (X)** | Height (converted from inches → cm) |
| **Target (y)** | Weight (converted from pounds → kg) |

**Unit Conversions applied:**
```
height_cm  = Height(Inches) × 2.54
weight_kg  = Weight(Pounds) × 0.453592
```

---

## 🗂️ Topics Covered

### 1. Data Preparation
- Loading CSV with `pandas`
- Unit conversion (imperial → metric)
- Null/missing value checks (`isnull().sum()`)
- Train/Test split (70% train, 30% test) via `sklearn.model_selection.train_test_split`

### 2. Exploratory Data Analysis (EDA)
- Scatter plot of Height vs. Weight to visually inspect the linear relationship

### 3. Linear Regression from Scratch
- Manual implementation of the hypothesis function
- Gradient Descent optimization loop
- Live loss monitoring every 50,000 epochs

### 4. Feature Scaling (Standardization)
- Why a tiny learning rate was needed without scaling
- How standardization fixes the slow convergence problem

### 5. Evaluation Metrics (Manual)
- MAE, RMSE, R² — all implemented by hand

### 6. Sklearn Validation
- `LinearRegression` from scikit-learn used as ground truth
- Metrics compared between scratch implementation and library

---

## 📐 Mathematical Background

### The Hypothesis (Model)

Simple Linear Regression models the relationship between one input feature and one output:

```
ŷ = w·X + b
```

| Symbol | Name | Role |
|---|---|---|
| `X` | Feature (height) | Input variable |
| `ŷ` | Prediction (weight) | Model output |
| `w` | Weight / Slope | How much y changes per unit of X |
| `b` | Bias / Intercept | Value of y when X = 0 |

The goal is to find the **optimal values of w and b** that minimize prediction error.

---

### Loss Function — Mean Squared Error (MSE)

The loss function measures how wrong the model is. We use MSE:

```
MSE = (1/n) · Σ (yᵢ - ŷᵢ)²
```

- Squaring the errors penalizes large mistakes more heavily
- Always non-negative; **lower = better**
- Differentiable → works well with gradient descent

---

## 🔄 Gradient Descent

Gradient Descent is the optimization algorithm used to iteratively update `w` and `b` to minimize the loss.

### Partial Derivatives

```
∂L/∂w = (-2/n) · Σ (yᵢ - ŷᵢ) · Xᵢ
∂L/∂b = (-2/n) · Σ (yᵢ - ŷᵢ)
```

### Parameter Update Rule

```
w ← w - α · (∂L/∂w)
b ← b - α · (∂L/∂b)
```

Where `α` (alpha) is the **learning rate** — controls how big each update step is.

### Implementation

```python
def find_derivatives(X, y, w, b):
    y_pred = w * X + b
    error  = y - y_pred
    dw = -2 * np.sum(error * X) / n
    db = -2 * np.sum(error)     / n
    return dw, db

for epoch in range(epochs):
    dw, db = find_derivatives(X, y, w, b)
    w = w - learning_rate * dw
    b = b - learning_rate * db
```

### Why Two Different Learning Rates?

| Run | Learning Rate | Feature Scaling | Result |
|---|---|---|---|
| Run 1 | `1e-8` | None | Works, but extremely slow |
| Run 2 | `1e-4` | Standardized X | Fast, stable convergence ✅ |

Without scaling, raw height values (~150–200 cm) make gradients very large, forcing you to use a tiny learning rate to avoid overshooting.

---

## 📏 Feature Scaling — Standardization (Z-score)

```
X_scaled = (X - μ) / σ
```

| Symbol | Meaning |
|---|---|
| `μ` | Mean of X |
| `σ` | Standard deviation of X |

After scaling, all values are centered around 0 with a standard deviation of 1. This makes gradient descent converge **much faster** and allows using a larger learning rate.

> ⚠️ **Important:** When using standardized X to train, the learned `w` and `b` are in the **scaled space**, not the original cm/kg space.

---

## 📊 Evaluation Metrics

These metrics measure how well the model's predictions match the actual values. All were implemented manually and then verified with sklearn.

---

### 1. Mean Absolute Error (MAE)

```
MAE = (1/n) · Σ |yᵢ - ŷᵢ|
```

```python
def mean_absolute_error(y, y_pred):
    return np.sum(np.abs(y - y_pred)) / len(y)
```

| Property | Detail |
|---|---|
| **Unit** | Same as target variable (kg) |
| **Range** | [0, ∞) — lower is better |
| **Interpretation** | On average, predictions are off by this many kg |
| **Sensitivity to outliers** | Low — treats all errors equally |
| **When to use** | When you don't want large errors to dominate the score |

---

### 2. Mean Squared Error (MSE)

```
MSE = (1/n) · Σ (yᵢ - ŷᵢ)²
```

| Property | Detail |
|---|---|
| **Unit** | Squared target unit (kg²) — harder to interpret directly |
| **Range** | [0, ∞) — lower is better |
| **Interpretation** | Average squared deviation from true values |
| **Sensitivity to outliers** | High — squaring amplifies large errors |
| **When to use** | As a loss function during training; penalizes big mistakes |

---

### 3. Root Mean Squared Error (RMSE)

```
RMSE = √MSE = √( (1/n) · Σ (yᵢ - ŷᵢ)² )
```

```python
def rmse(y, y_pred):
    return np.sqrt(np.sum((y - y_pred)**2) / len(y))
```

| Property | Detail |
|---|---|
| **Unit** | Same as target variable (kg) |
| **Range** | [0, ∞) — lower is better |
| **Interpretation** | Similar to MAE but penalizes large errors more |
| **Sensitivity to outliers** | High |
| **When to use** | When large individual errors are especially undesirable |

> **MAE vs RMSE:** If RMSE >> MAE, the model has some large individual prediction errors (outlier predictions).

---

### 4. R² Score (Coefficient of Determination)

```
R² = 1 - (SSR / SST)

SST = Σ (yᵢ - ȳ)²       ← Total variance in y
SSR = Σ (yᵢ - ŷᵢ)²      ← Unexplained variance (residuals)
```

```python
def r_square(y, y_pred):
    y_mean = np.mean(y)
    sst = np.sum((y - y_mean)**2)
    ssr = np.sum((y - y_pred)**2)
    return 1 - (ssr / sst)
```

| Property | Detail |
|---|---|
| **Unit** | Unitless |
| **Range** | (-∞, 1] — higher is better |
| **Interpretation** | Proportion of variance in y explained by the model |
| **R² = 1.0** | Perfect predictions |
| **R² = 0.0** | Model does no better than predicting the mean of y |
| **R² < 0** | Model is worse than a simple horizontal mean line |
| **When to use** | To understand how much of the target's variation your model captures |

---

### Metrics Summary Table

| Metric | Formula | Unit | Best Value | Outlier Sensitive |
|---|---|---|---|---|
| MAE | mean of \|errors\| | kg | 0 | No |
| MSE | mean of errors² | kg² | 0 | Yes |
| RMSE | √MSE | kg | 0 | Yes |
| R² | 1 - SSR/SST | — | 1 | Moderate |

---

## 🤖 Sklearn Comparison

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train.reshape(-1, 1), y_train)

w = model.coef_[0]       # slope
b = model.intercept_     # intercept

y_pred = model.predict(X_train.reshape(-1, 1))
```

sklearn's `LinearRegression` uses the **Normal Equation** (closed-form solution) instead of gradient descent:

```
w = (XᵀX)⁻¹ Xᵀy
```

This gives the exact optimal solution in one step — no iterations needed. However, it becomes computationally expensive when the dataset has a very large number of features, where gradient descent is preferred.

---

## 🛠️ Dependencies

```
pandas
numpy
matplotlib
scikit-learn
kagglehub
```

---

## 🔑 Key Takeaways

1. **Linear regression** finds the best-fit line through data by minimizing squared errors.
2. **Gradient descent** is an iterative optimizer that nudges `w` and `b` step-by-step in the direction that reduces loss.
3. **Feature scaling** is critical for gradient descent — standardizing inputs dramatically speeds up convergence and stabilizes training.
4. **MAE / RMSE** tell you the average prediction error in the original unit (kg); **R²** tells you what fraction of the target's variance the model explains.
5. sklearn's `LinearRegression` uses an algebraically exact solution; the from-scratch version uses iterative optimization — both converge to the same result.
