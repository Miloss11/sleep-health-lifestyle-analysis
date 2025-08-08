import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv('Sleep_Health_Lifestyle_Analysis.csv')

df = load_data()

# App title
st.title("Sleep Health & Lifestyle Analysis")

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# 1. Average Sleep Hours by Gender (Bar Plot)
st.subheader("Average Sleep Hours by Gender")
fig1 = px.bar(df.groupby('gender')['sleep_hours'].mean().reset_index(),
              x='gender', y='sleep_hours',
              labels={'gender': 'Gender', 'sleep_hours': 'Average Sleep Hours'},
              title='Average Sleep Hours by Gender')
st.plotly_chart(fig1)

# 2. Distribution of Gender (Pie Chart)
st.subheader("Gender Distribution")
gender_counts = df['gender'].value_counts().reset_index()
gender_counts.columns = ['gender', 'count']
fig2 = px.pie(gender_counts, values='count', names='gender', title='Gender Distribution')
st.plotly_chart(fig2)

# 3. BMI vs Sleep Hours Colored by Sleep Quality (Scatter Plot)
st.subheader("BMI vs Sleep Hours Colored by Sleep Quality")
fig3 = px.scatter(df, x='bmi', y='sleep_hours', color='sleep_quality',
                  labels={'bmi': 'BMI', 'sleep_hours': 'Sleep Hours', 'sleep_quality': 'Sleep Quality'},
                  title='BMI vs Sleep Hours Colored by Sleep Quality')
st.plotly_chart(fig3)

# 4. Age vs Sleep Hours Colored by Gender (Scatter Plot)
st.subheader("Age vs Sleep Hours by Gender")
fig4 = px.scatter(df, x='age', y='sleep_hours', color='gender',
                  labels={'age': 'Age', 'sleep_hours': 'Sleep Hours', 'gender': 'Gender'},
                  title='Age vs Sleep Hours by Gender')
st.plotly_chart(fig4)

# 5. Physical Activity vs Sleep Hours (Scatter Plot)
st.subheader("Physical Activity vs Sleep Hours")
fig5 = px.scatter(df, x='physical_activity', y='sleep_hours',
                  labels={'physical_activity': 'Physical Activity (hours)', 'sleep_hours': 'Sleep Hours'},
                  title='Physical Activity vs Sleep Hours')
st.plotly_chart(fig5)

# 6. Sleep Quality Distribution (Bar Plot)
st.subheader("Sleep Quality Distribution")
sleep_quality_counts = df['sleep_quality'].value_counts().reset_index()
sleep_quality_counts.columns = ['sleep_quality', 'count']
fig6 = px.bar(sleep_quality_counts, x='sleep_quality', y='count',
              labels={'sleep_quality': 'Sleep Quality', 'count': 'Count'},
              title='Sleep Quality Distribution')
st.plotly_chart(fig6)

# 7. Heart Rate Distribution (Histogram)
st.subheader("Heart Rate Distribution")
fig7 = px.histogram(df, x='heart_rate', nbins=30,
                    labels={'heart_rate': 'Heart Rate (bpm)'},
                    title='Distribution of Heart Rate')
st.plotly_chart(fig7)

# 8. Stress Level Distribution (Bar Plot)
st.subheader("Stress Level Distribution")
stress_counts = df['stress_level'].value_counts().reset_index()
stress_counts.columns = ['stress_level', 'count']
fig8 = px.bar(stress_counts, x='stress_level', y='count',
              labels={'stress_level': 'Stress Level', 'count': 'Count'},
              title='Stress Level Distribution')
st.plotly_chart(fig8)

# 9. Average Sleep Hours by Stress Level (Bar Plot)
st.subheader("Average Sleep Hours by Stress Level")
fig9 = px.bar(df.groupby('stress_level')['sleep_hours'].mean().reset_index(),
              x='stress_level', y='sleep_hours',
              labels={'stress_level': 'Stress Level', 'sleep_hours': 'Average Sleep Hours'},
              title='Average Sleep Hours by Stress Level')
st.plotly_chart(fig9)

# 10. Physical Activity Distribution (Histogram)
st.subheader("Physical Activity Distribution")
fig10 = px.histogram(df, x='physical_activity', nbins=30,
                     labels={'physical_activity': 'Physical Activity (hours)'},
                     title='Distribution of Physical Activity')
st.plotly_chart(fig10)

# 11. Correlation Heatmap of Numerical Features (matplotlib + seaborn)
st.subheader("Correlation Heatmap of Numerical Features")
stress_map = {'Low': 1, 'Medium': 2, 'High': 3}
df['stress_level_num'] = df['stress_level'].map(stress_map)
corr = df[['age', 'bmi', 'sleep_hours', 'stress_level_num', 'physical_activity', 'heart_rate']].corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
st.pyplot(plt.gcf())
plt.clf()

# 12. Age Distribution (Histogram)
st.subheader("Age Distribution")
fig12 = px.histogram(df, x='age', nbins=30,
                    labels={'age': 'Age'},
                    title='Age Distribution')
st.plotly_chart(fig12)

# 13. Sleep Hours Distribution (Histogram)
st.subheader("Sleep Hours Distribution")
fig13 = px.histogram(df, x='sleep_hours', nbins=30,
                    labels={'sleep_hours': 'Sleep Hours'},
                    title='Sleep Hours Distribution')
st.plotly_chart(fig13)

# 14. Boxplot of BMI by Gender
st.subheader("BMI Distribution by Gender")
fig14 = px.box(df, x='gender', y='bmi',
               labels={'gender': 'Gender', 'bmi': 'BMI'},
               title='BMI Distribution by Gender')
st.plotly_chart(fig14)

# 15. Boxplot of Sleep Hours by Sleep Quality
st.subheader("Sleep Hours Distribution by Sleep Quality")
fig15 = px.box(df, x='sleep_quality', y='sleep_hours',
               labels={'sleep_quality': 'Sleep Quality', 'sleep_hours': 'Sleep Hours'},
               title='Sleep Hours Distribution by Sleep Quality')
st.plotly_chart(fig15)

# 16. Average Heart Rate by Sleep Quality (Bar Plot)
st.subheader("Average Heart Rate by Sleep Quality")
fig16 = px.bar(df.groupby('sleep_quality')['heart_rate'].mean().reset_index(),
               x='sleep_quality', y='heart_rate',
               labels={'sleep_quality': 'Sleep Quality', 'heart_rate': 'Average Heart Rate'},
               title='Average Heart Rate by Sleep Quality')
st.plotly_chart(fig16)

# 17. Scatter plot: Heart Rate vs BMI colored by Sleep Quality
st.subheader("Heart Rate vs BMI Colored by Sleep Quality")
fig17 = px.scatter(df, x='bmi', y='heart_rate', color='sleep_quality',
                  labels={'bmi': 'BMI', 'heart_rate': 'Heart Rate', 'sleep_quality': 'Sleep Quality'},
                  title='Heart Rate vs BMI Colored by Sleep Quality')
st.plotly_chart(fig17)

# 18. Scatter plot: Physical Activity vs Stress Level (mapped numerically)
st.subheader("Physical Activity vs Stress Level")
fig18 = px.scatter(df, x='physical_activity', y='stress_level_num',
                  labels={'physical_activity': 'Physical Activity (hours)', 'stress_level_num': 'Stress Level (Numeric)'},
                  title='Physical Activity vs Stress Level')
st.plotly_chart(fig18)

# 19. Bar plot: Count of Participants by Sleep Quality
st.subheader("Count of Participants by Sleep Quality")
sleep_quality_count = df['sleep_quality'].value_counts().reset_index()
sleep_quality_count.columns = ['sleep_quality', 'count']
fig19 = px.bar(sleep_quality_count, x='sleep_quality', y='count',
               labels={'sleep_quality': 'Sleep Quality', 'count': 'Count'},
               title='Count of Participants by Sleep Quality')
st.plotly_chart(fig19)

# 20. Bar plot: Count of Participants by Stress Level
st.subheader("Count of Participants by Stress Level")
stress_level_count = df['stress_level'].value_counts().reset_index()
stress_level_count.columns = ['stress_level', 'count']
fig20 = px.bar(stress_level_count, x='stress_level', y='count',
               labels={'stress_level': 'Stress Level', 'count': 'Count'},
               title='Count of Participants by Stress Level')
st.plotly_chart(fig20)
