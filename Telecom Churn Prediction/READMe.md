# 📡 Telecom Churn Prediction: Identifying At-Risk Customers Using Machine Learning  
## 🎯 Purpose  
Telecom companies lose millions each year due to customer churn. Early identification of at-risk subscribers enables targeted retention campaigns, reduces revenue loss, and improves customer lifetime value.  
  
This project applies machine learning techniques to predict customer churn in the telecom industry using a real-world dataset. With customer retention being a major challenge, this study explores key factors influencing churn and builds predictive models to help telecom companies implement data-driven retention strategies.  


---
**Code:** [Telecom Churn Prediction Jupyter Notebook](https://github.com/YuwenAprilYang/Projects/blob/fb55da9e5eb18f691663ccb018458a8ae663a907/Telecom%20Churn%20Prediction/Telecom%20Churn%20Prediction%20Code.ipynb)  
**Report:** [Telecom Churn Prediction Report](https://github.com/YuwenAprilYang/Projects/blob/451a81c4c3d01b720966a6be1705995074e2f5d5/Telecom%20Churn%20Prediction/Telecom%20Churn%20Report.pdf)  

**Tools:** Python, Scikit-Learn, XGBoost, Pandas, Matplotlib, SMOTE  


## 📈 Dataset  
The dataset used for this project is sourced from [Kaggle Telecom Churn Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)  

## 🔍 Key Highlights  
- **EDA and Data Preprocessing:** Handling missing values, encoding categorical variables, and scaling numercial features  
- **Imbalance Handling:** Applying **SMOTE** and **class-weighting techniques** to address imbalance    
- **Model Evaluation:** Implementing and evaluating **Logistic Regression, Decision Tree, Random Forest, and XGBoost**  
  - **XGBoost emerged as the best model**, achieving high **Recall (98.2%)** and **F1-score (98.0%)** under a class-weighted approach.  
- **Feature Importance:** Using **SHAP values** to interpret feature importance—**Monthly Charges, International Plans, and Customer Service Calls** were the strongest predictors of churn  
- **Cusotmer Segmentation:** Categorizing customers into **low, medium, and high-risk categories** for targeted retention strategies  

## 🚀 Future Enhancements  
- Implement **deep learning models** for further performance improvements.  
- Deploy a **real-time churn prediction system** for dynamic customer insights.  
- Incorporate **additional customer engagement metrics** to refine predictions.  

📊 **Check out the detailed report and code to explore more insights!**  
