# 🌆 AI City Scout: The Brain of a Smart City During Crisis
  
### 🏆 First Prize Winner - Aggie Hackathon 2025

**Code:** [AI City Scout Python Code](https://github.com/YuwenAprilYang/Projects/blob/e165f48097d6a0c21692e55a89084e25b9a965e1/AI%20City%20Scout/app.py)  
**Report:** [AI City Scout Report](https://github.com/YuwenAprilYang/Projects/blob/b7e11fb1c320ab7ea49785cd6cf2bc56b0d2d94e/AI%20City%20Scout/AI%20City%20Scout%20Report.pdf)  
**Team:** Yuwen (April) Yang, Wenjun Song, Xinxin Liu  
**Tools:** Python, Streamlit, Random Forest, Isolation Forest, OpenAI GPT-4, Scikit-Learn, GeoPandas  

## Executive Summary
As natural disasters become more frequent and social misinformation spreads faster than emergency response systems can react, cities are left vulnerable.  
  
**_AI City Scout_ is a real-time, AI-powered disaster intelligence dashboard designed to predict disasters, flag fake social media posts, and generate actionable emergency plans — all within seconds.**  
  
This project was awarded **First Prize (Innovator's Summit Award)** at the 2025 Aggie Hackathon for innovation, intelligence modeling and impact potential.

## Business Problem
In crisis situations like floods or earthquakes, city leaders face overwhelming volumes of noisy data and misinformation. Key questions we address:  
> - What type of disaster is unfolding right now, and where?  
> - Is social media reporting false information?  
> - How severe are the consequences?  
> - What immediate actions should the city take?

## System Overview
AI City Scout contains **three insteractive modules:**
1. **Future Predictions**  
Uses real-time sensor + weather data to detect and predict disasters; Assesses hospital overload risk and severity; Generates AI-intergrated strategic plan
  
2. **Social Media Detector**  
Verifies tweet claiming a disaster by cross-checking sensor data near the time and location of the tweet
  
3. **Past Disasters Explorer**  
Interactive map showing historical disasters (2010-2015), allowing users to understand patterns and cascading risks

## Data Preprocessing
Dataset: 
- 50,000+ rows across sensors, weather, social media, infrastructure maps  
  
Preprocessing:  
- Outlier detection in weather via Isolation Forest
- Sensor classification via rule-based logic
- Location matching using geopy and GeoJSON infrastructure maps
- Data fusion of time-series, spatial, and text features  

## Core Models & Methods
Models:  
- Disaster Severity Prediction: Rule-based logic + weather-based mapping
- Hospital Overload Risk: Random Forest Classifier trained on severity, disaster type and hospital proximity (92% accuracy)
- Fake Tweet Detection: Proximity validation + BERT-based NLP
- Emergency Plan Generation: GPT-4 generated concise, scenario-based action prompts

## Demonstation
_1. Predict future disasters from user's time input, showing possible results (severity & hospital overload risk), generating AI-integrated strategic plans_  

https://github.com/user-attachments/assets/ee36f817-154d-48cc-bb60-cb39abcac1dc


https://github.com/user-attachments/assets/c35cedfc-f4ef-4eff-8fb1-c628a2e247f9
  
_2. Flags fake disaster-related tweets by validating against physical evidence (user's input of time, location and text)_  

https://github.com/user-attachments/assets/1fe9310c-deb2-400c-bdbb-9433a5a42c1a
  
_3. Aggregates and visualizes historical events to uncover patterns_  

https://github.com/user-attachments/assets/1e68c290-2e1d-4822-8c8f-87ea2717fdc2

## Key Insights
**1. Real-Time Disaster Prediction and Risk Scoring**
- Predict disasters before impact
- Real-time alerts up to hours
- Severity scale (1-9) calibrated across disaster types (flood, fire, earthquake, hurricane)
> 📌 **Business takeaway:** Enables proactive resource allocation before disasters  

**2. Social Media Verification Reduces Panic**
- System flagged fake tweets with >90% accuracy
- No matching sensor data within 3km or 1-hour window -> tweet flagged as fake
> 📌 **Business takeaway:** Reduces misinformation-triggered evacuations, improves public trust  

**3. Stategic Planning With AI-Generated Emergency Actions**
- Based on disaster type, predicted severity, and predicted hospital overload risk
- Each disaster instance is passed to GPT-4 model for recommendation action plan
> 📌 **Business takeaway:** Reduces decision-making lag time by 99.7%


### 📊 Check out the detailed report and code to explore more insights!
