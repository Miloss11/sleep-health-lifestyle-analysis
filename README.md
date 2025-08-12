# Sleep Health & Lifestyle Analysis

## Live Demo  
https://sleep-health-lifestyle-analysis-49cnedggbws6u4y7atk85s.streamlit.app/

![Sleep Illustration](images/sleep-illustration.png)

## Project Overview  
This project analyzes the relationship between lifestyle factors and sleep quality using a health dataset. It includes data cleaning, exploratory data analysis, and an interactive Streamlit dashboard to visualize insights.

## Dataset  
The dataset contains anonymized participant data with the following features:

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
- Investigate how BMI, stress, physical activity, gender, and age affect both sleep quality and duration.  
- Identify key lifestyle factors that contribute most significantly to healthy sleep patterns.  
- Provide clear, visual insights through an interactive dashboard to support better understanding of sleep health.  
- Lay the groundwork for predictive modeling of sleep outcomes based on lifestyle and biometric data.

## Questions  
- Does BMI affect sleep duration and quality?  
- How do stress levels influence sleep?  
- Are there differences in sleep quality by gender and age?  
- How does physical activity impact sleep?

## Methods  
- Data loading and cleaning using Pandas  
- Exploratory data analysis and visualization with Seaborn, Matplotlib, and Plotly Express  
- Interactive dashboard development using Streamlit  

## Tools  
- Python (Pandas, NumPy)  
- Seaborn, Matplotlib, Plotly Express  
- Streamlit  

## Future Work  
- Incorporate additional lifestyle and environmental variables to deepen analysis.  
- Develop advanced machine learning models for personalized sleep quality and duration prediction.  
- Enhance dashboard interactivity and customization options.  
- Improve model accuracy with more diverse and larger datasets.

## How to Run  
1. Clone the repository  
2. Install dependencies:  
   `pip install -r requirements.txt`  
3. Run the app:  
   `streamlit run app.py`  
4. Use the sidebar filters to explore the data and model outputs interactively.

## Ethics  
- The dataset is fully anonymized to ensure participant privacy.  
- This project is intended for educational and informational purposes only, not as medical advice.

## Author  
Milos Visnjic  
GitHub: https://github.com/Miloss11  
Streamlit app: https://sleep-health-lifestyle-analysis-49cnedggbws6u4y7atk85s.streamlit.app/
