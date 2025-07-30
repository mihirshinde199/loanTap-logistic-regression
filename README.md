# 💼 LoanTap Credit Risk Classification Model – Logistic Regression

LoanTap is a modern digital lending platform offering innovative personal loan products tailored for millennials and salaried professionals. This project focuses on building a robust **Logistic Regression** model to assist LoanTap’s underwriting team in assessing **loan eligibility and repayment risk**.

---

## 📊 Problem Statement

The goal is to predict whether a borrower is likely to **default** on their personal loan using demographic, credit, and employment data. This enables data-driven, safer disbursal decisions, minimizing NPAs (non-performing assets).

---

## 🗃️ Dataset Overview

The dataset contains borrower information and loan statuses. Key features include:

- `loan_amnt`, `term`, `int_rate`, `installment`
- `emp_length`, `annual_inc`, `home_ownership`
- `dti`, `revol_util`, `total_acc`
- `loan_status` (Target variable: Fully Paid / Charged Off)

---

## 🧠 ML Pipeline Highlights

### 1. Exploratory Data Analysis
- Univariate and bivariate plots
- Distribution of employment length, income, grade, loan purpose
- Correlation heatmap for numerical features

### 2. Preprocessing & Feature Engineering
- Removed irrelevant columns like address, title
- Treated missing values in `emp_length`, `mort_acc`, `pub_rec_bankruptcies`
- Created binary flags: `has_pub_rec`, `has_mortgage`, `has_bankruptcy`

### 3. Model Building – Logistic Regression
- Trained using Scikit-learn’s `LogisticRegression`
- Applied feature scaling using `StandardScaler`
- Used one-hot encoding for categorical features like `home_ownership`, `grade`

### 4. Evaluation Metrics
- ✅ ROC AUC Score
- ✅ Precision-Recall Curve
- ✅ Classification Report (F1, Precision, Recall)
- ✅ Confusion Matrix

---

## 🎯 Key Trade-Off Analysis

- To **minimize false positives**, we tuned thresholds using PR curve
- Chose **Recall** as primary metric to catch real defaulters
- Discussed how over-predicting defaulters might reduce profits but lower NPAs

---

## 📈 Results Summary

| Metric     | Value |
|------------|-------|
| Accuracy   | ~82%  |
| Precision  | ~0.76 |
| Recall     | ~0.85 |
| ROC AUC    | ~0.89 |

---

## 🔍 Business Insights

- **Loan Grade A & B** customers are significantly less risky
- High **DTI** and **low employment length** increase default probability
- **Annual income**, **home ownership**, and **installment amounts** impact approval odds
- Feature like `pub_rec_bankruptcies` is a major red flag

---

## 🚀 Tech Stack

- Python, Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn (LogisticRegression, metrics)
- Jupyter Notebook

---

## 📂 Folder Structure

```
loantap-logistic-regression/
├── data/           # Raw dataset (LoanTapData.csv)
├── notebooks/      # EDA and modeling notebook
├── src/            # Modular Python scripts
├── outputs/        # Plots, evaluation graphs
├── README.md       # This file
└── requirements.txt
```

---

## 🙋‍♂️ Author

**Mihir Shinde**  
📧 mihir.shinde18@vit.edu  
🐙 [GitHub](https://github.com/mihirshinde199)

---

⭐ _If you found this useful, please star the repo!_
