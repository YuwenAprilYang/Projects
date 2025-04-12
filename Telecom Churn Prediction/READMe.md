# 📡 Telecom Churn Prediction: Identifying At-Risk Customers Using Machine Learning  
**Code:** [Telecom Churn Prediction Jupyter Notebook](https://github.com/YuwenAprilYang/Projects/blob/fb55da9e5eb18f691663ccb018458a8ae663a907/Telecom%20Churn%20Prediction/Telecom%20Churn%20Prediction%20Code.ipynb)  
**Report:** [Telecom Churn Prediction Report](https://github.com/YuwenAprilYang/Projects/blob/451a81c4c3d01b720966a6be1705995074e2f5d5/Telecom%20Churn%20Prediction/Telecom%20Churn%20Report.pdf)  

**Tools:** Python, Scikit-Learn, XGBoost, Pandas, Matplotlib, SMOTE  

## Executive Summary  
Customer churn poses a critical revenue challenge in the telecom industry, where acquiring a new customer can cost up to **5x** more than retaining an existing one. In this project, I built a predictive machine learning model to help telecom companies identify high-risk customers and take proactive retention actions.  
The XGBoost model achieved **98.2% recall**, ensuring minimal false negatives—ideal for prioritizing retention outreach. Key drivers of churn included **Monthly Charges, International Plans, and Customer Service Calls.**  

## Business Problem
Telecom companies struggle with high churn rates, leading to significant revenue loss and increased customer acquisition costs. This project addresses the question:  
>"Which customers are most likely to leave, and what actionable insights can we use to reduce churn?"

## Business Metrics Analyzed
- Churn Rate (target variable)
- Monthly Charges & Service Usage
- Customer Support Interactions
- Plan Types (International/Voicemail Plans)
  
## Data & Methodology
- Dataset: The dataset used for this project is sourced from [Kaggle Telecom Churn Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets) The dataset include features like billling, service usage, charge fee, churn, and demographics.
- Preprocessing: Categorical encoding, feature scaling, multicollinearity reduction and outlier analysis
- Imabalance Handling: Employed SMOTE and class weighting--class weighting proved more effective for preserving predictive accuracy
- Models: Logistic Regression, Decision Tree, Random Forest, and XGBoost. XGBoost was selected for deployment due to its superior recall&F1, and strong ROC AUC.

## Key Insights
**1. XGBoost Achieved Exceptional Predictive Performance**  
Among all tested models, XGBoost outperformed Logistic Regression, Decision Tree, and Random Forest—especially when class imbalance was addressed using **class weighting**.  
- **Recall:** 98.2% – The model successfully captured nearly all actual churners, critical for minimizing undetected revenue loss.  
- **Precision:** 96.9% – Most customers predicted to churn actually did, reducing unnecessary retention costs.  
- **F1 Score:** 98.0% – Balances Precision and Recall, indicating strong overall classification performance.  
- **ROC AUC:** 94.8% – Demonstrates the model’s ability to distinguish churners from non-churners effectively.


> 🔍 **Business takeaway:** The model can reliably identify at-risk customers, enabling proactive and cost-effective retention strategies.


## 🚀 Future Enhancements  
- Implement **deep learning models** for further performance improvements.  
- Deploy a **real-time churn prediction system** for dynamic customer insights.  
- Incorporate **additional customer engagement metrics** to refine predictions.  

📊 **Check out the detailed report and code to explore more insights!**  
