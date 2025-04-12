# 📊 EDA Generator: Interactive Exploratory Data Analysis Tool  
**Code:** [EDA Generator](https://github.com/YuwenAprilYang/Projects/blob/50f54dc9d3e6740f7a4d8688e521e90f3d93d57d/EDA%20Generator/app.py)  
**Tools:** Python, Streamlit, Pandas, Matplotlib, Seaborn  

## Executive Summary
In today’s fast-paced business environment, decision-makers need immediate access to data insights without waiting for custom code or technical support. The **EDA Generator** bridges that gap by offering a no-code, interactive tool that empowers analysts, marketers, product managers, and executives to explore datasets, detect trends, and identify anomalies in seconds.

**EDA Generator** lowers the barrier to entry by providing a no‑code, intuitive interface that lets analysts, students, and business stakeholders instantly understand their data, quickly spot trends, detect anomalies, and generate insights without writing a single line of code. Users can upload their own file to instantly generate tables and visualizations.    

## Business Problem
Many companies struggle with delayed or inefficient exploratory data analysis (EDA) due to dependence on technical staff. This bottleneck slows down business decisions, hides critical issues in data quality, and limits early-stage hypothesis testing. The EDA Generator answers:

>“How can we reduce time-to-insight and empower any team member to perform EDA without writing code?”

## Data Context & Technical Design
This app accepts **any structured CSV dataset**, including marketing, product, sales, customer, or web analytics data. It adapts to new datasets dynamically.  

- **Python + Streamlit** – for interactive, web-based deployment
- **Pandas** – for behind-the-scenes data manipulation
- **Seaborn & Matplotlib** – for professional, customizable visualizations
- **Session State + Caching** – to retain user input and optimize performance


## Key Highlights
- **Data Preview**: View the first and last 10 rows of your dataset.  
- **Dataset Info**: Inspect shape, column names, data types, and missing‑value counts.  
- **Descriptive Statistics**: Automatically compute summary stats with `df.describe()`.  
- **Correlation Heatmap**: Visualize pairwise correlations for numeric features.  
- **Interactive Distribution Plots**: Select any numeric column to display a histogram + KDE; selection persists via session state.  
- **Performance Optimizations**: Caches data loading to minimize latency on repeated interactions.

## Recommendation
- **Marketing teams:** Quickly analyze campaign performance data and identify patterns
- **Product teams:** Validate user event tracking data before A/B testing
- **Sales analysts:** Spot trends in customer acquisition, deal sizes, or churn
- **Data science teams:** Rapidly iterate through data quality checks and EDA without setup overhead

## 🛠️ How to Get Started
1. Clone the GitHub repository

```bash
git clone git clone https://github.com/YuwenAprilYang/Projects.git
cd Projects/EDA Generator
```
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```
3. Run the Streamlit App
```bash
streamlit run app.py
```
📊 **Check out the code to get started!**  
