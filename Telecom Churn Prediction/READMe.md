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
> 📌 **Business takeaway:** These features are not only predictive but provide direct paths for improving retention.  

![Screenshot 2025-04-12 at 15 34 13](https://github.com/user-attachments/assets/eec096da-2935-4153-adc3-808de2044892)
![Screenshot 2025-04-12 at 15 34 38](https://github.com/user-attachments/assets/1f2be20d-b52e-420f-b2e4-ce41ecc74c15)


**2. Three Key Drivers of Churn Identified via SHAP analysis**  
Feature importance analysis using SHAP revealed the top contributors to churn:
- **Monthly Charges:**
High charges correlated with higher churn, suggesting dissatisfaction with pricing or perceived value.  
  *→ Actionable insight: Introduce tiered plans, discounts, or bundled packages.*

- **International Plan Subscription:**  
  Customers with international plans churned more often, likely due to unclear fees or poor experience.  
  *→ Actionable insight: Redesign or better communicate international offerings.*

- **Customer Service Calls:**  
  A strong positive correlation was found between frequent support interactions and churn.  
  *→ Actionable insight: Improve service resolution and implement first-call resolution practices.*  

![Screenshot 2025-04-12 at 15 34 48](https://github.com/user-attachments/assets/05bd7702-ca4b-4980-bc35-9563ed7300d4)


> 🔍 **Business takeaway:** The model can reliably identify at-risk customers, enabling proactive and cost-effective retention strategies.

**3. Customer Segmentation Enabled Tiered Retention Strategy**  
Customers were segmented by predicted churn probability into three risk levels:

- **High-risk (>66%)** – Immediate intervention: exclusive offers, dedicated support, contract extensions.  
- **Medium-risk (33–66%)** – Preventive strategies: proactive outreach, flexible billing, loyalty nudges.  
- **Low-risk (<33%)** – Long-term engagement: upselling opportunities, referral incentives.

> 🧩 **Business takeaway:** Tiered interventions allow businesses to optimize retention spend and focus on the most valuable customers.



## Recommendation
- **Retain High-Value Churn-Risk Customers**  
  Focus retention efforts on customers with high usage and billing but frequent complaints. Offer customized loyalty packages or rate adjustments to reduce churn.  
- **Improve International Plan Offerings**  
Churn is higher among international plan users. Introduce flat-rate options or bundled promotions tailored to frequent international callers.  

- **Enhance Customer Support Experience**  
High customer service call frequency correlates with churn. Implement first-call resolution goals, AI-based help centers, and satisfaction follow-ups to improve service quality.  

- **Target Medium-Risk Segments Before Escalation**  
Provide nudges, surveys, or minor incentives to medium-risk customers showing early signs of dissatisfaction.  

- **Embed the Model into CRM Systems**  
Allow real-time churn scoring to inform frontline staff and marketers, enabling timely and personalized outreach.

### 📊 Check out the detailed report and code to explore more insights!
