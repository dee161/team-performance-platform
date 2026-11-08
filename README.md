# BITS Pilani - Work Integrated Learning Programmes (WILP)
## M.Tech (AIML) - Machine Learning (Assignment 2)
### End-to-End Classification & Streamlit Cloud Deployment

---

## a. Problem Statement

Customer churn represents one of the most critical operational challenges in the telecommunications industry. Telecommunications providers incur substantial customer acquisition costs (CAC)—often exceeding five to eight times the expense of retaining an existing customer. When subscribers terminate their contracts, telecommunication enterprises experience direct recurring revenue loss and diminished customer lifetime value.

The objective of this assignment is to develop an end-to-end Machine Learning classification pipeline that accurately identifies telecom subscribers at high risk of churning. By deploying five distinct machine learning classification algorithms, systematically evaluating them across six standard performance metrics on a dedicated holdout test dataset, and serving the best-performing models via an interactive Streamlit web application deployed on Streamlit Community Cloud, business decision-makers can proactively intervene with targeted retention programs (e.g., customized contract terms, service discounts, or dedicated technical support) before churn occurs.

---

## b. Dataset Description

The chosen benchmark dataset is the **Telco Customer Churn** dataset, sourced from the public **Kaggle** repository ([blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) and IBM Business Analytics.

### Dataset Specifications & Compliance
* **Repository Source**: Kaggle / IBM Business Analytics
* **Total Instances**: **7,032** valid customer records (exceeds the minimum requirement of 500 instances)
* **Total Input Features**: **19** predictive features (exceeds the minimum requirement of 12 features)
* **Target Variable**: `Churn` (Binary classification: `Yes` [churned] or `No` [retained])
* **Class Distribution**: 5,163 Retained (~73.4%) vs. 1,869 Churned (~26.6%)

### Feature Attributes Breakdown
The 19 input features encapsulate demographic attributes, subscribed telecom services, and account/contractual details:

1. **Demographic Features (4)**:
   * `gender`: Whether customer is Male or Female.
   * `SeniorCitizen`: Binary indicator if customer is a senior citizen (1, 0).
   * `Partner`: Whether customer has a partner (Yes, No).
   * `Dependents`: Whether customer has dependents (Yes, No).

2. **Telecommunications Services Subscribed (9)**:
   * `PhoneService`: Whether customer has landline phone service (Yes, No).
   * `MultipleLines`: Whether customer has multiple phone lines (Yes, No, No phone service).
   * `InternetService`: Customer’s internet service provider type (DSL, Fiber optic, No).
   * `OnlineSecurity`: Whether customer has add-on online security (Yes, No, No internet service).
   * `OnlineBackup`: Whether customer has cloud backup storage (Yes, No, No internet service).
   * `DeviceProtection`: Whether customer has device protection insurance (Yes, No, No internet service).
   * `TechSupport`: Whether customer has premium tech support (Yes, No, No internet service).
   * `StreamingTV`: Whether customer streams TV over IP (Yes, No, No internet service).
   * `StreamingMovies`: Whether customer streams movies over IP (Yes, No, No internet service).

3. **Account & Contractual Information (6)**:
   * `tenure`: Number of months the customer has stayed with the company (1 to 72 months).
   * `Contract`: Contract term agreement (Month-to-month, One year, Two year).
   * `PaperlessBilling`: Whether customer uses paperless e-billing (Yes, No).
   * `PaymentMethod`: Payment method (Electronic check, Mailed check, Bank transfer, Credit card).
   * `MonthlyCharges`: Current monthly charges billed in USD ($18.25 to $118.75).
   * `TotalCharges`: Cumulative total charges billed across tenure in USD ($18.80 to $8,684.80).

### Preprocessing and Split Strategy
1. **Data Cleaning**: 11 records containing whitespace in `TotalCharges` (customers with tenure=0) were coerced to numeric and removed, resulting in 7,032 clean rows.
2. **Identifier Removal**: Unique customer identifier (`customerID`) was removed as it carries no predictive signal.
3. **Feature Preprocessing Pipeline (`ColumnTransformer`)**:
   * **Numerical Columns** (`tenure`, `MonthlyCharges`, `TotalCharges`): Normalized using `StandardScaler` to ensure zero mean and unit variance.
   * **Categorical Columns** (16 categorical features): Transformed using `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')` to generate dense encoded arrays compatible across all estimators.
4. **Data Splitting**: Stratified 80/20 train-test split (`test_size=0.20`, `random_state=42`), producing:
   * **Training Set**: 5,625 samples (used solely for model training).
   * **Holdout Test Set (`test_data.csv`)**: 1,407 samples (used exclusively for model evaluation and Streamlit test upload).

---

## c. GitHub Repository Link

* **GitHub Repository URL**: `https://github.com/<your-github-username>/bits-wilp-ml-assignment2`
*(Please replace `<your-github-username>` with your actual GitHub username upon repository creation)*

### Repository Structure
```text
bits-wilp-ml-assignment2/
│-- app.py                    # Streamlit web application with model evaluation UI
│-- requirements.txt          # Production dependencies for Streamlit Cloud deployment
│-- README.md                 # Project documentation and evaluation report
│-- test_data.csv             # Cleaned benchmark test dataset (1,407 rows with Churn column)
│-- model/                    # Serialized models and training artifacts
│   │-- preprocessor.joblib   # Fitted ColumnTransformer (StandardScaler + OneHotEncoder)
│   │-- logistic_regression.joblib  # Trained Logistic Regression Pipeline
│   │-- decision_tree.joblib        # Trained Decision Tree Classifier Pipeline
│   │-- knn.joblib                  # Trained K-Nearest Neighbor Classifier Pipeline
│   │-- naive_bayes.joblib          # Trained Naive Bayes (GaussianNB) Pipeline
│   │-- svm.joblib                  # Trained Support Vector Machine (SVC) Pipeline
│   │-- evaluation_metrics.csv      # Precomputed benchmark test metrics
│   │-- train_models.py       # Python source code for training and export
│   └── train_models.ipynb    # Jupyter Notebook containing step-by-step experiment workflow
```

---

## d. Models Used

Five classical supervised classification algorithms were implemented, trained, and evaluated on the exact same dataset split:

1. **Logistic Regression**: Linear classifier minimizing logistic loss with L2 regularization (`C=1.0`, `max_iter=1000`). Serves as a strong benchmark for linear decision boundaries in normalized feature space.
2. **Decision Tree Classifier**: Non-parametric tree algorithm (`criterion='gini'`, `max_depth=5`, `min_samples_leaf=20`) to capture non-linear feature interactions while preventing overfitting.
3. **K-Nearest Neighbor (KNN)**: Distance-based instance classifier (`n_neighbors=15`, Euclidean metric) classifying queries based on majority vote of local neighborhood in scaled feature space.
4. **Naive Bayes Classifier (GaussianNB)**: Generative probabilistic classifier applying Bayes' theorem under the conditional feature independence assumption.
5. **Support Vector Machine (SVM)**: Maximum-margin classifier using Radial Basis Function (`kernel='rbf'`, `C=1.0`, `probability=True`) to map features into higher-dimensional space for non-linear separation.

### Evaluation Metrics Comparison Table

The six required evaluation metrics were computed on the holdout test set (`test_data.csv`, 1,407 instances):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8045** | **0.8361** | **0.6505** | 0.5722 | **0.6088** | **0.4810** |
| **Decision Tree** | 0.7783 | 0.8195 | 0.5807 | 0.5963 | 0.5884 | 0.4367 |
| **KNN** | 0.7747 | 0.8181 | 0.5789 | 0.5588 | 0.5687 | 0.4164 |
| **Naive Bayes** | 0.6446 | 0.8102 | 0.4184 | **0.8636** | 0.5637 | 0.3808 |
| **Support Vector Machine** | 0.7932 | 0.7833 | 0.6487 | 0.4840 | 0.5544 | 0.4311 |

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves the highest Accuracy (0.8045), ROC-AUC (0.8361), F1 Score (0.6088), and Matthews Correlation Coefficient (0.4810). Due to standardized features and one-hot encoding, it establishes an optimal hyperplane that balances precision (65.05%) and recall (57.22%) without overfitting to noisy churn indicators. |
| **Decision Tree** | Delivers competitive discriminative ability (AUC: 0.8195, Accuracy: 0.7783). By pruning the tree to `max_depth=5` and `min_samples_leaf=20`, the model avoids the high variance typical of unconstrained trees, producing interpretable rules centered around `Contract` type and `tenure`. |
| **KNN** | Shows moderate performance (Accuracy: 0.7747, AUC: 0.8181). While distance scaling effectively protects continuous features, the curse of dimensionality from 19 multi-category one-hot attributes creates a somewhat sparse metric space, slightly diluting neighborhood boundaries compared to linear margins. |
| **Naive Bayes** | Achieves exceptional Recall (0.8636)—capturing 323 out of 374 churned customers—at the expense of lower Precision (0.4184) and overall Accuracy (0.6446). The strong conditional independence assumption is violated by correlated telecom features (e.g., `tenure`, `MonthlyCharges`, `TotalCharges`), causing overconfident churn probability estimates. |
| **Support Vector Machine** | Attains strong Accuracy (0.7932) and solid Precision (0.6487) via its non-linear RBF kernel. However, its recall is more conservative (0.4840), as the soft-margin penalty errs on the side of the majority non-churn class to minimize overall structural risk. |
| **Overall, Winner for your dataset?** | **Logistic Regression is the decisive winner.** It dominates across virtually all key performance indicators: highest Accuracy (80.45%), highest ROC-AUC (0.8361), highest F1 Score (0.6088), and highest MCC (0.4810). For telecom retention applications, its superior probability calibration enables management to rank customer churn risk reliably and allocate retention budgets with maximum financial efficiency. |

---

## Streamlit Web Application Features

The interactive web application (`app.py`) is engineered to fulfill all rubric requirements:
1. **Dataset Upload Option (CSV)**: Users can upload any custom test CSV file containing customer attributes and the target column `Churn`. If no file is uploaded, the app automatically loads the bundled benchmark `test_data.csv` (1,407 instances). A sample download button is also provided for test convenience.
2. **Model Selection Dropdown**: Allows users to dynamically toggle between:
   * **All Models (Comparison)**: Displays the full side-by-side metric table, comparative bar charts, and multi-model ROC curve overlays.
   * **Individual Models**: Dedicated deep dive for any of the 5 classifiers.
3. **Display of Evaluation Metrics**: Renders all 6 required metrics (Accuracy, ROC-AUC, Precision, Recall, F1, MCC) using styled visual KPI summary cards.
4. **Confusion Matrix & Classification Report**: Displays a detailed confusion matrix heatmap (annotated with True Negatives, False Positives, False Negatives, True Positives) alongside the complete per-class classification report table.
5. **Interactive Single-Customer Sandbox**: A real-time simulator enabling evaluators to test churn likelihood on arbitrary customer profiles.
6. **Zero Model Retraining**: All inference runs strictly via pre-trained, serialized pipeline artifacts without retraining.

---

## Deployment Guide (Streamlit Community Cloud)

1. **Push Code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "BITS ML Assignment 2: Telco Churn Classification & Streamlit App"
   git branch -M main
   git remote add origin https://github.com/<your-github-username>/bits-wilp-ml-assignment2.git
   git push -u origin main
   ```
2. **Deploy on Streamlit Cloud**:
   * Navigate to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.
   * Click **New App**.
   * Select repository: `bits-wilp-ml-assignment2`.
   * Branch: `main`.
   * Main file path: `app.py`.
   * Click **Deploy**.
3. **Access App**: Once deployed, copy your public Streamlit URL (`https://<app-name>.streamlit.app`) for the PDF submission.
