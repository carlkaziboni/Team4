# 🌿 Renewable Energy Analysis Dashboard

An intelligent web application for analyzing and visualizing energy usage and emissions data with AI-powered insights for renewable energy opportunities.

## 🔍 Overview

This project provides a data analytics platform focused on energy usage and carbon emissions for Diageo, with particular emphasis on identifying renewable energy integration opportunities. The dashboard combines data visualization, time series analysis, and AI-powered recommendations to support sustainable energy decision-making.

## ✨ Features

- 📊 Interactive data visualization of energy metrics and carbon emissions
- 📈 Time series analysis with smoothing algorithms for trend identification
- 🤖 AI-powered analysis of visual data to identify renewable energy opportunities
- 🔎 Site-specific filtering and category selection for detailed analysis
- 📉 Scope 1 & 2 emissions tracking and analysis
- 🧠 Multiple ML models for predictive analysis (PCA, Linear Regression, MLP, Neural Networks)

## 🛠️ Technical Stack

- **Backend:** Python, Flask
- **Data Analysis:** Pandas, NumPy, Scikit-learn, TensorFlow
- **Machine Learning:** PCA, Linear Regression, MLPRegressor, Neural Networks
- **Visualization:** Matplotlib, Seaborn
- **Time Series Analysis:** StatsModels (Exponential Smoothing)
- **AI Integration:** OpenAI API (GPT-4 Turbo with vision capabilities)
- **Frontend:** HTML, CSS (Gradient template)

## 📂 Project Structure

- `app.py` - Flask application providing the web interface and API endpoints
- `agi.py` - AI analysis module for image-based insights using OpenAI's GPT-4
- `explore.py` - ML model definitions and data preprocessing operations
- `graphgen.py` - Time series visualization and graph generation functions
- `static/` - Directory for generated plots and static assets
- `templates/` - HTML templates for the web interface

## 🚀 Getting Started

1. Install required dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow flask openai statsmodels
   ```

2. Set up your OpenAI API key in `agi.py`

3. Ensure you have the dataset file `Diageo_Scotland_Full_Year_2024_Daily_Data.csv` in the project root

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000/`

## 📊 Data Analysis Process

1. Data is loaded and preprocessed with normalization and standardization
2. Principal Component Analysis (PCA) extracts key features
3. Multiple prediction models analyze Scope 1 & 2 emissions
4. Interactive visualization enables time-based analysis
5. AI provides concise recommendations based on visual data

## 🔮 Future Improvements

- Implement real-time data integration
- Add more advanced predictive models
- Develop scenario analysis tools for renewable energy planning
- Create a comprehensive dashboard with multiple visualization options
- Add user authentication and role-based access

---

*This project combines data science and AI to facilitate sustainable energy transitions for industrial operations.*
