```markdown
# Telecom Customer Churn Prediction

This project aims to predict customer churn in the telecom sector using machine learning techniques. By identifying customers likely to leave, telecom companies can proactively engage and retain them, reducing costs associated with acquiring new customers.

---

## Project Overview

Customer churn is a critical issue for telecom companies, as retaining existing customers is more cost-effective than acquiring new ones. This project analyzes the Telco Customer Churn Dataset from Kaggle, which contains data on 7,043 customers, to understand factors influencing churn and build predictive models.

---

## Data Processing

- **Handling Missing Values:** Dropped rows with missing `TotalCharges` values due to new customers with zero tenure.
- **Encoding Categorical Variables:** 
  - Label encoding for binary variables (e.g., Gender, Churn).
  - One-hot encoding for multi-class variables (e.g., PaymentMethod).
- **Feature Scaling:** StandardScaler applied to numerical features like MonthlyCharges and TotalCharges.
- **Feature Selection:** Correlation analysis used to remove irrelevant or redundant features such as CustomerID.
- **Class Imbalance:** Noted that only 26% of customers churn; future work may include techniques like SMOTE.
- **Train-Test Split:** 80% training (5,634 records), 20% testing (1,409 records).

---

## Models Used

- **Logistic Regression:** Baseline binary classification model.
- **K-Nearest Neighbors (KNN):** Simple parametric model for comparison.
- **Random Forest Classifier:** Non-parametric model that handles feature importance and non-linear relationships.

---

## Model Performance

| Metric      | Logistic Regression | KNN      | Random Forest  |
|-------------|---------------------|----------|----------------|
| Accuracy    | ~79.9%              | 75.5%    | ~80.4%         |
| AUC Score   | 84%                 | 74.7%    | 80%            |
| Precision (Churn Class) | 64% (LR) / 74% (RF) | - | - |
| Recall (Churn Class)    | 51% (LR) / 59% (RF) | - | - |
| F1-Score (Churn Class)  | 57% (LR) / 66% (RF) | - | - |

Random Forest and Logistic Regression models show comparable accuracy and AUC scores, with Random Forest slightly better at identifying churners.

---

## Key Insights & Feature Importance

- **Contract Type:** Month-to-month contracts have higher churn rates; longer-term contracts reduce churn.
- **Tenure:** Customers with longer tenure are less likely to churn.
- **Monthly Charges:** Higher charges increase likelihood of churn.
- **Internet Service:** Fiber optic users show higher churn compared to DSL.
- **Payment Method:** Customers paying via electronic checks churn more frequently.

---

## Recommendations

- Promote long-term contracts with incentives to reduce churn.
- Offer personalized pricing plans or discounts to customers with high monthly charges.
- Improve fiber optic service quality or pricing to enhance customer satisfaction.
- Encourage use of automated payment methods like credit cards or bank transfers with rewards.

---

## How to Use

1. Clone the repository.
2. Preprocess the dataset as described.
3. Run the provided notebooks/scripts for exploratory data analysis and model training.
4. Evaluate model performance on the test set.
5. Use the model to predict churn probabilities for new customers.

---

## Future Work

- Address class imbalance with techniques like SMOTE or cost-sensitive learning.
- Incorporate additional customer demographics or service usage data.
- Deploy model as a decision support tool for customer retention strategies.

---

## Dataset

The dataset used is publicly available from Kaggle: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/aadityabansalcodes/telecommunications-industry-customer-churn-dataset)

---

## Acknowledgments

- Dataset sourced from Kaggle.
- Inspired by business analytics practices in telecom customer retention.

---

Feel free to explore, modify, and improve the models to better suit your telecom customer churn prediction needs.
```

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/64706810/b8bd90c9-4ecd-453e-a874-ce6a7b9413e8/IBA_presentation.pdf
[2] https://github.com/KanduriR/Kaggle-Telco-Customer-Churn
[3] https://www.kaggle.com/datasets/aadityabansalcodes/telecommunications-industry-customer-churn-dataset
[4] https://www.kaggle.com/datasets/blastchar/telco-customer-churn
[5] https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset
[6] https://www.kaggle.com/datasets/blastchar/telco-customer-churn/code
[7] https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets
[8] https://www.kaggle.com/datasets/shivam131019/telecom-churn-dataset
[9] https://www.kaggle.com/datasets/denisexpsito/telco-customer-churn-ibm
[10] https://www.kaggle.com/code/praxitelisk/telco-customer-churn-eda-ml

---
Answer from Perplexity: https://www.perplexity.ai/search/go-through-attachment-and-tell-6ZDFsrVVRbWXg9Fij3iyIg?utm_source=copy_output
