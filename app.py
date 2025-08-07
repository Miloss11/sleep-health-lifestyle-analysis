import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

st.title("Sleep Health and Lifestyle Analysis")

gender_filter = st.selectbox("Select Gender", options=['All', 'M', 'F'])
stress_filter = st.selectbox("Select Stress Level", options=['All', 'Low', 'Medium', 'High'])

filtered_df = df.copy()
if gender_filter != 'All':
    filtered_df = filtered_df[filtered_df['gender'] == gender_filter]
if stress_filter != 'All':
    filtered_df = filtered_df[filtered_df['stress_level'] == stress_filter]

st.write("Filtered Data", filtered_df)

fig, ax = plt.subplots()
sns.barplot(x='gender', y='sleep_hours', data=filtered_df, ax=ax)
st.pyplot(fig)