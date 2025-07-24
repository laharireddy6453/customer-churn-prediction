# 📊 Machine Learning Project: Customer Churn Prediction – Lahari Sudhini

## 🧩 Problem Statement

In today's competitive business environment, retaining customers is essential for long-term success. The objective of this project is to develop a machine learning model that predicts whether a customer is likely to churn — i.e., discontinue using a service. Using historical data that includes customer demographics, service usage, and subscription details, this project aims to build a robust model capable of accurately identifying at-risk customers.

With this predictive insight, businesses can proactively engage high-risk customers using targeted retention strategies, thereby reducing churn rate, increasing customer satisfaction, and improving overall profitability.

---

## 📊 Data Description

The dataset used contains customer information across various attributes:

- **CustomerID**: Unique customer identifier
- **Name**: Full name of the customer
- **Age**: Customer's age
- **Gender**: Male or Female
- **Location**: City where the customer resides (e.g., Houston, Miami, Chicago, etc.)
- **Subscription_Length_Months**: Total months subscribed
- **Monthly_Bill**: Monthly subscription charge
- **Total_Usage_GB**: Total internet/data usage in GB
- **Churn**: Binary flag (1 = churned, 0 = active)

---

## 🧪 Technologies & Tools Used

- **Programming Language**: Python 3.12
- **Development Environment**: Jupyter Notebook, VS Code

### 📚 Python Libraries

- **pandas**: For data manipulation and cleaning
- **numpy**: For numerical computations
- **matplotlib** and **seaborn**: For visualizations and plots
- **scikit-learn (sklearn)**: For model training and evaluation
- **joblib**: To save and load trained ML models

### 📈 Machine Learning Algorithms Applied

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machines (SVM)**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Naive Bayes**
- **AdaBoost & Gradient Boosting**
- **XGBoost**

### 🧠 Deep Learning (optional exploration)

- **TensorFlow/Keras**: To train neural networks for better accuracy in complex cases

### 🔍 Model Improvement Techniques

- **Hyperparameter Tuning** using `GridSearchCV`
- **Cross Validation** (e.g., StratifiedKFold)
- **Principal Component Analysis (PCA)** for dimensionality reduction
- **StandardScaler** for normalization
- **Variance Inflation Factor (VIF)** to reduce multicollinearity
- **EarlyStopping** and **ModelCheckpoint** to prevent overfitting in deep learning

---

## 📌 Evaluation Metrics

- **Accuracy**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- **ROC Curve and AUC (Area Under Curve)**

---

## ✅ Project Workflow

1. **Data Exploration & Cleaning**: Handled missing values, standardized column names, and encoded categorical variables.
2. **Exploratory Data Analysis (EDA)**: Used visualizations to identify churn patterns and trends.
3. **Feature Selection**: Selected impactful features for modeling (e.g., `Subscription_Length_Months`, `Monthly_Bill`).
4. **Model Training**: Applied multiple models and tuned them using GridSearchCV.
5. **Model Evaluation**: Used confusion matrix, ROC curves, and classification reports.
6. **Model Export**: Saved the best-performing model using joblib.
7. **Model Deployment (Optional)**: App.py included for basic inference using `streamlit` or `flask`.
8. **PDF Report**: Detailed insights included in the PDF report.
9. **GitHub Hosting**: Project hosted and version-controlled with `.gitignore` and `README.md`

---

## 🚀 Results & Outcomes

- Best model: **Random Forest Classifier** with an accuracy of over **95%**
- Model helped identify key patterns in churn behavior such as:
  - Shorter subscription durations → higher churn
  - Higher monthly bill → moderate correlation to churn
  - Certain cities had higher churn concentrations

Using the churn predictions, businesses can now:

- Retain high-risk customers with targeted offers
- Optimize pricing and subscription models
- Prioritize customer satisfaction initiatives

---

## 🗂️ Project Structure

```
Customer-Churn-Prediction/
│
├── Churn_Prediction_Lahari.ipynb    # Main analysis notebook
├── app.py                           # Streamlit or Flask app for inference
├── customer_churn_large_dataset.xlsx
├── Project Report.pdf               # Final project summary report
├── README.md                        # This file
├── .gitignore                       # Ignored files for git
└── model/                           # Saved model .pkl files
```

---

## 💡 Conclusion

This project provides a complete machine learning pipeline to predict customer churn and recommend business actions. From loading and cleaning the data to model tuning and reporting, this solution is designed to be extensible, interpretable, and deployable in a real-world business context.

> Created by **Lahari Sudhini** | Submitted on **July 24, 2025**

---

## 📬 Contact

For queries or collaboration:
- 📧 laharireddy6453@email.com (placeholder)
- 🌐 [GitHub Profile](https://github.com/laharireddy6453)