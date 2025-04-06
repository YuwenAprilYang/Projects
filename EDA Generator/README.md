# 📊 EDA Generator: Interactive Exploratory Data Analysis Tool  
## 🎯 Purpose
Exploratory Data Analysis (EDA) is the critical first step in any data-driven project—yet it often requires repetitive coding. 
  
**EDA Generator** lowers the barrier to entry by providing a no‑code, intuitive interface that lets analysts, students, and business stakeholders instantly understand their data, quickly spot trends, detect anomalies, and generate insights without writing a single line of code. Users can upload their own file to instantly generate tables and visualizations.    

---
**Code:** [EDA Generator](https://github.com/YuwenAprilYang/Projects/blob/50f54dc9d3e6740f7a4d8688e521e90f3d93d57d/EDA%20Generator/app.py)  
**Tools:** Python, Streamlit, Pandas, Matplotlib, Seaborn  


## 🔍 Key Highlights
- **Data Preview**: View the first and last 10 rows of your dataset.  
- **Dataset Info**: Inspect shape, column names, data types, and missing‑value counts.  
- **Descriptive Statistics**: Automatically compute summary stats with `df.describe()`.  
- **Correlation Heatmap**: Visualize pairwise correlations for numeric features.  
- **Interactive Distribution Plots**: Select any numeric column to display a histogram + KDE; selection persists via session state.  
- **Performance Optimizations**: Caches data loading to minimize latency on repeated interactions.

📊 **Check out the code to get started!**  
