# Sleep Health & Lifestyle Analysis

## Live Demo

Check out the live Streamlit app here:  
https://sleep-health-lifestyle-analysis-49cnedggbws6u4y7atk85s.streamlit.app/

## Project Overview  
This project analyzes how lifestyle factors affect sleep quality using a real-world health dataset. It includes data cleaning, exploratory analysis, and an interactive Streamlit dashboard to visualize insights.

## About the Dataset  
The dataset contains health and lifestyle information from various individuals, including age, gender, BMI, hours of sleep, sleep quality, stress levels, physical activity, and heart rate. It is publicly available and anonymized, sourced from Kaggle. The dataset is large enough to identify meaningful patterns while ensuring good performance.

## Data Dictionary  
| Column Name       | Description                                      |  
|-------------------|-------------------------------------------------|  
| age               | Age of the participant in years                  |  
| gender            | Gender of the participant (Male/Female)          |  
| BMI               | Body Mass Index                                   |  
| sleep_hours       | Number of hours slept per night                   |  
| sleep_quality    | Subjective sleep quality rating (1-5)             |  
| stress_level    | Self-reported stress level (1-10)                   |  
| physical_activity | Average daily physical activity (minutes)         |  
| heart_rate       | Average resting heart rate (bpm)                   |  

## Project Goals  
The main goal is to understand the key factors influencing sleep quality and how BMI, stress, physical activity, gender, and age relate to sleep duration and quality.

### Key Questions  
- Does BMI impact sleep duration and quality?  
- How do stress levels affect sleep?  
- Are there differences in sleep quality by gender and age?  
- How does physical activity influence sleep health?  

These insights can help guide lifestyle changes for better sleep.

## Hypotheses and Testing Methods  
- Higher BMI leads to poorer sleep quality and less sleep.  
- Increased stress causes worse sleep.  
- Physical activity improves sleep quality and duration.

Various visualizations such as scatter plots, box plots, violin plots, and heatmaps were used to explore these hypotheses.

## Approach  
- Loaded and cleaned the dataset using Pandas.  
- Filtered data by gender, stress, sleep quality, and age.  
- Conducted exploratory data analysis with Seaborn and Plotly.  
- Built an interactive dashboard with Streamlit.  
- Interpreted results and drew actionable conclusions.

Streamlit was chosen for its ease of use and interactive capabilities.

## Connecting Business Questions to Visualizations  

| Question                       | Visualization Types             | Purpose                                |  
|-------------------------------|--------------------------------|---------------------------------------|  
| BMI’s effect on sleep          | Scatter plots, violin plots     | Show trends and distribution patterns |  
| Stress levels and sleep        | Histograms, violin plots        | Display distributions and their effects |  
| Sleep differences by gender and age | Box plots, scatter plots | Compare group differences clearly      |  
| Physical activity’s impact on sleep | Scatter plots            | Show correlations                      |  

## Tools and Technologies  
- Python (Pandas, NumPy)  
- Seaborn, Matplotlib for statistical visualizations  
- Plotly Express for interactive charts  
- Streamlit for dashboard development  

## Machine Learning Integration  
This project currently focuses on exploratory data analysis and interactive visualization to understand sleep health factors. Future plans include integrating machine learning models to predict sleep quality and sleep duration based on variables like BMI, stress levels, physical activity, age, and heart rate. Implementing predictive models will enhance the dashboard by providing actionable insights and personalized recommendations for improving sleep health.

## Data Sources and Acknowledgements  
The dataset used for this analysis is publicly available on Kaggle and consists of anonymized health and lifestyle data collected from various individuals. This ensures compliance with privacy and ethical guidelines.

Dataset: Sleep Health and Lifestyle Dataset on Kaggle

Visualization methods and dashboard design were developed using resources and inspiration from Plotly, Seaborn, and Streamlit official documentation and tutorials.

## Known Issues & Future Work  
- Minor bugs fixed; a few small glitches may remain.  
- App not yet deployed online — deployment planned on Heroku or similar platforms.  
- Plan to add machine learning models to predict sleep quality and duration.  
- Improve user interface and add more interactive features.  

## Ethical Considerations  
- The dataset is anonymized to protect individual privacy.  
- Avoided biased assumptions related to gender and age.  
- Results are for informational purposes and not medical advice.  

## How to Use  
1. Clone this repository.  
2. Install the required packages listed in `requirements.txt` by running this command in your terminal:  


## How to Use

1. Clone this repository.

2. Install the required packages listed in `requirements.txt` by running this command in your terminal:

```bash
pip install -r requirements.txt

3. Run the Streamlit app with this command:

```bash
streamlit run app.py

4. Use the sidebar filters to explore different aspects of the data and see how various lifestyle factors affect sleep quality.

## Live Demo

Check out the live Streamlit app here:  
[Sleep Health Lifestyle Analysis](https://sleep-health-lifestyle-analysis-49cnedggbws6u4y7atk85s.streamlit.app/)


## Author

Milos Visnjic