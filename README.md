# Textile Worker Productivity Prediction: A Critical ML Analysis

This project develops a complete machine learning pipeline to analyze and predict productivity estimation accuracy in a textile manufacturing context (Bangladesh, 2015).

The core of the study is the analysis of the **`DELTA` variable**: the discrepancy between actual productivity and management's a priori estimates. The goal is to classify whether productivity is **Overestimated**, **Realistically estimated**, or **Underestimated**.

---

## Data & Statistical Rigor
The dataset is the "Productivity Prediction of Garment Employees" from the UCI Machine Learning Repository (https://archive.ics.uci.edu).

Unlike standard "black-box" approaches, this project applies **statistical foundations** for data preparation:
- **Outlier Detection:** Used **Tukey’s Rule** (IQR) to define "physiological" estimation errors vs. significant miscalculations.
- **Association Study:** Applied **Cramér’s V** and **Spearman’s Rank Correlation** to select features, ensuring robustness against non-normal distributions.

---

## Project Structure

### 1. Exploratory Data Analysis (`01_eda.ipynb`)
- Focused on the **leptokurtic and negatively skewed** distribution of the target variable.
- Identified a strong class imbalance (~85% "Realistic" estimates).
- Business insight: Management tends to overestimate performance more often than underestimating it.

### 2. Feature Engineering (`02_feature_engineering.ipynb`)
- Created interpretable features (e.g., `FineMese`, `Sabato`) to capture temporal patterns.
- **Discretization:** Transformed numerical variables into bins to capture non-linear effects and mitigate the impact of extreme outliers.

### 3. Machine Learning Pipeline (`03_models.ipynb` & `04_evaluation.ipynb`)
- Implemented a robust comparison using `Scikit-learn` Pipelines.
- Models: Logistic Regression, Decision Tree, Random Forest, SVM (RBF), and MLP.
- **Handling Imbalance:** Integrated **SMOTE** to address the minority classes (Over/Underestimation).

---

## Results & Critical Reflection

The **Random Forest** model emerged as the most balanced performer:
- **Baseline Accuracy:** ~85% (misleading due to class prevalence).
- **Post-SMOTE Macro-F1:** ~0.495 (significant improvement in minority class recall).

### Takeaways
As a Statistician, I identified some critical constraints for future production-grade iterations:
*   **Oversampling Limits:** SMOTE was applied after numerical encoding of categorical features. While it boosted Macro-F1 (from ~0.31 to ~0.50), this linear interpolation in a discrete space can introduce noise.
*   **Curse of Dimensionality:** One-Hot Encoding on a relatively small dataset (~1200 rows) slightly degraded performance. 
*   **Future Roadmap:** To improve these results, I would prioritize **Cost-Sensitive Learning** (class weighting) or **Gradient Boosting (CatBoost/XGBoost)** with native categorical support, rather than synthetic oversampling.

---

## Technologies Used
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-learn, Seaborn, Matplotlib.
- **Workflow:** Jupyter Notebooks (fully executed for reproducibility).

---

## Author
**Giuseppe Cioffo** 
