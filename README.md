# Sleep Health & Lifestyle Analysis

## Live Demo
https://sleep-health-lifestyle-analysis-49cnedggbws6u4y7atk85s.streamlit.app/

## Project Overview
This project analyzes how lifestyle factors affect sleep quality using a health dataset. It includes data cleaning, exploratory analysis, and an interactive Streamlit dashboard.

## Dataset
The dataset contains information on age, gender, BMI, sleep hours, sleep quality, stress levels, physical activity, and heart rate. It is sourced from Kaggle and anonymized.

## Data Dictionary

| Column Name       | Description                           |
|-------------------|-------------------------------------|
| age               | Age of participant in years          |
| gender            | Gender (Male/Female)                  |
| BMI               | Body Mass Index                      |
| sleep_hours       | Hours of sleep per night              |
| sleep_quality     | Sleep quality rating (1-5)            |
| stress_level      | Stress level (1-10)                   |
| physical_activity | Daily physical activity in minutes   |
| heart_rate        | Average resting heart rate (bpm)     |

## Goals
To understand how BMI, stress, physical activity, gender, and age relate to sleep quality and duration.

## Questions
- Does BMI affect sleep duration and quality?  
- How do stress levels influence sleep?  
- Are there differences in sleep quality by gender and age?  
- How does physical activity impact sleep?

## Methods
- Loaded and cleaned data using Pandas  
- Explored data with Seaborn, Matplotlib, and Plotly  
- Created an interactive dashboard using Streamlit  

## Tools
- Python (Pandas, NumPy)  
- Seaborn, Matplotlib, Plotly Express  
- Streamlit

## Future Work
- Add machine learning models to predict sleep quality and duration  
- Deploy app on a public platform  
- Improve dashboard interactivity and UI

## How to Run
1. Clone the repo  
2. Install dependencies:  
   `pip install -r requirements.txt`  
3. Run the app:  
   `streamlit run app.py`  
4. Use sidebar filters to explore the data

## Ethics
- Dataset is anonymized to protect privacy  
- Results are for informational use only, not medical advice

## Author
Milos Visnjic
