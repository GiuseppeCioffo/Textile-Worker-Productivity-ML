# Textile Worker Productivity Prediction

This project develops a complete machine learning pipeline to analyze and predict
productivity estimation accuracy in a textile manufacturing context.

The goal is to classify whether productivity is **Overestimated**, **Realistically estimated** or **Underestimated**,
based on operational, organizational, and calendar-related features.

## Data
The dataset comes from a public industrial dataset related to garment manufacturing: https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees

## Project Structure
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model comparison
- Model evaluation and interpretation

---

## 1. Exploratory Data Analysis (`01_eda.ipynb`)

- Dataset inspection and cleaning
- Target variable analysis (`DELTA`)
- Outlier detection using Tukey's rule
- Univariate and bivariate analysis
- Numerical correlations (Spearman)
- Key insights to guide feature engineering

**Outcome:** understanding data quality, distributions, and relevant relationships.

---

## 2. Feature Engineering (`02_feature_engineering.ipynb`)

- Creation of interpretable binary features (e.g. end-of-month, Saturday)
- Discretization of numerical variables to capture non-linear effects
- Construction of a clean, interpretable feature set
- Export of a modeling-ready dataset

**Output:** `dataset_model_ready.csv`

---

## 3. Machine Learning Models (`03_models.ipynb`)

- One-hot encoding via `ColumnTransformer`
- Stratified train/test split
- Training and comparison of multiple classifiers:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (RBF)
  - Multilayer Perceptron
- Model comparison using Accuracy and Macro-F1 score

**Focus:** robust and reproducible model comparison using pipelines.

---

## 4. Model Evaluation & Interpretation (`04_evaluation.ipynb`)

- Handling class imbalance using **SMOTE**
- Evaluation with Macro-F1 and confusion matrices
- Comparison of model performance under resampling
- Feature importance analysis (Random Forest)

**Key insight:** non-linear and tree-based models better capture productivity patterns,
especially for minority classes.

---

## Key Takeaways

- Class imbalance significantly affects model performance
- Macro-F1 is more informative than accuracy in this context
- Calendar and workload-related features are strong productivity drivers
- End-to-end pipelines improve reproducibility and clarity

---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn
- Seaborn, Matplotlib

---

## Notes

- Notebooks are provided **fully executed** for reproducibility and readability
- The original raw dataset is not included for size and licensing reasons. Available at the link above in the 'Data' section

---

## 👤 Author
Giuseppe Cioffo
