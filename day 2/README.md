# 🎯 Logistic Regression — Binary Classification

A practical implementation of **Logistic Regression** for binary classification using scikit-learn. Applied to a real-world Chess Games dataset to predict whether White or Black wins a match based on player ratings.

---

## 📋 Table of Contents

- [Dataset](#dataset)
- [Topics Covered](#topics-covered)
- [Why Not Linear Regression?](#why-not-linear-regression)
- [Mathematical Background](#mathematical-background)
- [Decision Boundary](#decision-boundary)
- [Evaluation Metrics](#evaluation-metrics)
- [Confusion Matrix Deep Dive](#confusion-matrix-deep-dive)
- [Dependencies](#dependencies)

---

## 📦 Dataset

| Property | Detail |
|---|---|
| **Source** | Kaggle — `datasnaek/chess` |
| **File** | `games.csv` |
| **Features (X)** | `white_rating`, `black_rating` |
| **Target (y)** | `winner` → mapped to `white=0`, `black=1` |

**Preprocessing steps:**
- Rows where `winner == 'draw'` were dropped — only binary outcomes kept
- Target labels encoded: `white → 0`, `black → 1`
- Train/Test split: 70% train, 30% test (`random_state=42`)

---

## 🗂️ Topics Covered

### 1. Data Preparation
- Filtering out non-binary classes (`draw` removed)
- Label encoding the target variable with `.map()`
- Selecting two numerical features: player ratings

### 2. Logistic Regression with sklearn
- Fitting `LogisticRegression` model
- Inspecting learned coefficients coef_ and intercept

### 3. Prediction & Evaluation
- Generating predictions with `model.predict()`
- Computing the **Confusion Matrix**
- Manually extracting TP, TN, FP, FN from the matrix
- Computing all key classification metrics by hand

---

## ❓ Why Not Linear Regression?

Linear Regression predicts **continuous values** (e.g., weight, price). For classification, we need the output to be a **probability between 0 and 1**. Linear regression can output values like `-3` or `7.5`, which are meaningless as probabilities.

Logistic Regression solves this by passing the linear output through the **Sigmoid function**, which squashes any value into the range (0, 1).

---

## 📐 Mathematical Background

### Step 1 — Linear Combination (same as Linear Regression)

```
z = w₁·x₁ + w₂·x₂ + b
```

In this notebook:
```
z = w₁·white_rating + w₂·black_rating + b
```

### Step 2 — Sigmoid Function (the key difference)

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

| z value | σ(z) output | Meaning |
|---|---|---|
| Very large positive | → 1.0 | Almost certainly class 1 (black wins) |
| 0 | → 0.5 | Completely uncertain |
| Very large negative | → 0.0 | Almost certainly class 0 (white wins) |

The output `σ(z)` is interpreted as **P(y=1 | X)** — the probability that black wins.

### Step 3 — Decision

```
if σ(z) ≥ 0.5  →  ŷ = 1  (black wins)
if σ(z) < 0.5  →  ŷ = 0  (white wins)
```

### Loss Function — Binary Cross-Entropy (Log Loss)

Unlike MSE used in linear regression, logistic regression minimizes log loss:

```
L = -(1/n) · Σ [ yᵢ·log(ŷᵢ) + (1 - yᵢ)·log(1 - ŷᵢ) ]
```

- When `y=1` and `ŷ → 1`: loss → 0 (correct, no penalty)
- When `y=1` and `ŷ → 0`: loss → ∞ (very wrong, huge penalty)
- **Lower = better**

---

## 🔀 Decision Boundary

Since we have two features, the decision boundary is a **line** in 2D space (white_rating vs black_rating) where `σ(z) = 0.5`, i.e., where `z = 0`:

```
w₁·white_rating + w₂·black_rating + b = 0
```

Points on one side → predicted white wins; points on the other → predicted black wins.

---

## 📊 Evaluation Metrics

### The Confusion Matrix

Before understanding metrics, you need to understand the four possible prediction outcomes:

```
                    Predicted: 0        Predicted: 1
Actual: 0     |  TN (True Neg)   |  FP (False Pos)  |
Actual: 1     |  FN (False Neg)  |  TP (True Pos)   |
```

| Term | Full Name | What happened |
|---|---|---|
| **TP** | True Positive | Model said black wins → black actually won ✅ |
| **TN** | True Negative | Model said white wins → white actually won ✅ |
| **FP** | False Positive | Model said black wins → white actually won ❌ |
| **FN** | False Negative | Model said white wins → black actually won ❌ |

In sklearn, `confusion_matrix` returns:
```python
cm = [[TN, FP],
      [FN, TP]]

TN = cm[0, 0]
TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
```

---

### 1. Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

| Property | Detail |
|---|---|
| **Range** | [0, 1] — higher is better |
| **Interpretation** | Fraction of all predictions that were correct |
| **When to use** | When classes are roughly balanced |
| **Weakness** | Misleading on imbalanced datasets (e.g., 95% class 0 → model predicts all 0s → 95% accuracy, but useless) |

---

### 2. Precision

```
Precision = TP / (TP + FP)
```

```python
precision = TP / (TP + FP)
```

| Property | Detail |
|---|---|
| **Range** | [0, 1] — higher is better |
| **Question it answers** | "Of all the times the model said black wins, how often was it right?" |
| **Focuses on** | False Positives — how many positive predictions were wrong |
| **When to use** | When false positives are costly (e.g., spam filter — you don't want to delete real emails) |

---

### 3. Recall (Sensitivity / True Positive Rate)

```
Recall = TP / (TP + FN)
```

```python
recall = TP / (TP + FN)
```

| Property | Detail |
|---|---|
| **Range** | [0, 1] — higher is better |
| **Question it answers** | "Of all the actual black wins, how many did the model catch?" |
| **Focuses on** | False Negatives — how many positives the model missed |
| **When to use** | When false negatives are costly (e.g., disease detection — you don't want to miss a sick patient) |

---

### 4. Specificity (True Negative Rate)

```
Specificity = TN / (TN + FP)
```

```python
spec = TN / (TN + FP)
```

| Property | Detail |
|---|---|
| **Range** | [0, 1] — higher is better |
| **Question it answers** | "Of all the actual white wins, how many did the model correctly identify?" |
| **Focuses on** | How well the model handles the negative class |
| **Complement of** | False Positive Rate (FPR = 1 - Specificity) |
| **When to use** | Often paired with Recall in medical/security contexts |

---

### 5. F1-Score

```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```

```python
F1_score = (2 * precision * recall) / (precision + recall)
```

| Property | Detail |
|---|---|
| **Range** | [0, 1] — higher is better |
| **Interpretation** | Harmonic mean of Precision and Recall |
| **When to use** | When you need to balance Precision and Recall, especially on imbalanced data |
| **Why harmonic mean?** | Punishes extreme imbalance — if either Precision or Recall is 0, F1 = 0 |

> **Arithmetic mean vs Harmonic mean:** If Precision=1.0 and Recall=0.0, arithmetic mean = 0.5 (looks okay), but harmonic mean (F1) = 0.0 (correctly shows the model is broken).

---

### Metrics Summary Table

| Metric | Formula | Focuses On | High Value Means |
|---|---|---|---|
| Accuracy | (TP+TN) / All | Overall correctness | Most predictions are right |
| Precision | TP / (TP+FP) | False Positives | Few false alarms |
| Recall | TP / (TP+FN) | False Negatives | Few missed positives |
| Specificity | TN / (TN+FP) | Negative class | Few false alarms on negatives |
| F1-Score | Harmonic(P, R) | Balance of P and R | Good precision AND recall |

---

### Precision vs Recall Trade-off

These two metrics are in tension with each other:

- **Increase threshold** (e.g., predict black only if P > 0.7) → Precision ↑, Recall ↓
- **Decrease threshold** (e.g., predict black if P > 0.3) → Recall ↑, Precision ↓

Choose based on the cost of each error type in your problem domain.

---

## 🤖 Sklearn Usage

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

model = LogisticRegression()
model.fit(X_train, y_train)

# Learned parameters
w = model.coef_       # shape: (1, n_features)
b = model.intercept_  # shape: (1,)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)  # raw probabilities [P(class=0), P(class=1)]
```

---

## 🛠️ Dependencies

```
pandas
scikit-learn
kagglehub
```

---

## 🔑 Key Takeaways

1. **Logistic Regression** is a classification algorithm, not regression — the name is historical. It outputs probabilities via the Sigmoid function.
2. The **Sigmoid function** maps any real number to (0, 1), making the output interpretable as a probability.
3. **Accuracy alone is not enough** — on imbalanced datasets it can be highly misleading.
4. **Precision** matters when false positives are expensive; **Recall** matters when false negatives are expensive.
5. **F1-Score** is the go-to metric when you need a single number that balances both Precision and Recall.
6. Always **extract TP/TN/FP/FN from the confusion matrix** first — all other metrics are derived from these four numbers.
