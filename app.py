import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

st.set_page_config(page_title="Sleep Health & Lifestyle Analysis", layout="wide")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, 'Sleep_Health_Lifestyle_Analysis.csv')
    return pd.read_csv(csv_path)

df = load_data()

with st.sidebar.expander("Filter Options", expanded=True):
    genders = df['gender'].unique().tolist()
    selected_genders = st.multiselect("Select Gender(s)", genders, default=genders)
    if not selected_genders:
        selected_genders = genders

    stress_levels = df['stress_level'].unique().tolist()
    selected_stress_levels = st.multiselect("Select Stress Level(s)", stress_levels, default=stress_levels)
    if not selected_stress_levels:
        selected_stress_levels = stress_levels

    sleep_qualities = df['sleep_quality'].unique().tolist()
    selected_sleep_qualities = st.multiselect("Select Sleep Quality(ies)", sleep_qualities, default=sleep_qualities)
    if not selected_sleep_qualities:
        selected_sleep_qualities = sleep_qualities

    min_age, max_age = int(df['age'].min()), int(df['age'].max())
    age_range = st.slider("Select Age Range", min_age, max_age, (min_age, max_age))

    min_bmi, max_bmi = float(df['bmi'].min()), float(df['bmi'].max())
    bmi_range = st.slider("Select BMI Range", min_bmi, max_bmi, (min_bmi, max_bmi))

    min_pa, max_pa = int(df['physical_activity'].min()), int(df['physical_activity'].max())
    physical_activity_range = st.slider("Select Physical Activity Range", min_pa, max_pa, (min_pa, max_pa))

    min_hr, max_hr = int(df['heart_rate'].min()), int(df['heart_rate'].max())
    heart_rate_range = st.slider("Select Heart Rate Range", min_hr, max_hr, (min_hr, max_hr))

filtered_df = df[
    (df['gender'].isin(selected_genders)) &
    (df['stress_level'].isin(selected_stress_levels)) &
    (df['sleep_quality'].isin(selected_sleep_qualities)) &
    (df['age'].between(age_range[0], age_range[1])) &
    (df['bmi'].between(bmi_range[0], bmi_range[1])) &
    (df['physical_activity'].between(physical_activity_range[0], physical_activity_range[1])) &
    (df['heart_rate'].between(heart_rate_range[0], heart_rate_range[1]))
].copy()

if filtered_df.empty:
    st.warning("No data for selected filters.")
    st.stop()

st.write(f"Data after filtering: {len(filtered_df)} rows")

stress_map = {'Low': 1, 'Medium': 2, 'High': 3}
sleep_quality_map = {'Poor': 1, 'Average': 2, 'Good': 3}

filtered_df['stress_level_num'] = filtered_df['stress_level'].map(stress_map)
filtered_df['sleep_quality_num'] = filtered_df['sleep_quality'].map(sleep_quality_map)

st.title("Sleep Health & Lifestyle Analysis")

st.subheader("BMI vs Sleep Hours colored by Sleep Quality")
fig1 = px.scatter(filtered_df, x='bmi', y='sleep_hours', color='sleep_quality')
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Distribution of Sleep Quality")
fig2 = px.histogram(filtered_df, x='sleep_quality')
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Age vs Sleep Hours colored by Sleep Quality")
fig3 = px.scatter(filtered_df, x='age', y='sleep_hours', color='sleep_quality')
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Sleep Hours Distribution by Gender")
fig4 = px.box(filtered_df, x='gender', y='sleep_hours')
st.plotly_chart(fig4, use_container_width=True)

st.subheader("BMI Distribution by Sleep Quality")
fig5 = px.violin(filtered_df, x='sleep_quality', y='bmi', box=True, points='all')
st.plotly_chart(fig5, use_container_width=True)

st.subheader("Distribution of Stress Levels")
fig6 = px.histogram(filtered_df, x='stress_level')
st.plotly_chart(fig6, use_container_width=True)

st.subheader("Physical Activity vs Sleep Hours colored by Gender")
fig7 = px.scatter(filtered_df, x='physical_activity', y='sleep_hours', color='gender')
st.plotly_chart(fig7, use_container_width=True)

st.subheader("Heart Rate Distribution by Sleep Quality")
fig8 = px.box(filtered_df, x='sleep_quality', y='heart_rate')
st.plotly_chart(fig8, use_container_width=True)

st.subheader("Age vs BMI colored by Sleep Quality")
fig9 = px.scatter(filtered_df, x='age', y='bmi', color='sleep_quality')
st.plotly_chart(fig9, use_container_width=True)

st.subheader("Distribution of Gender")
fig10 = px.histogram(filtered_df, x='gender')
st.plotly_chart(fig10, use_container_width=True)

st.subheader("Correlation Heatmap of Numerical Features")
corr = filtered_df[['age', 'bmi', 'sleep_hours', 'stress_level_num', 'physical_activity', 'heart_rate']].corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
st.pyplot(plt.gcf())
plt.clf()

st.header("Machine Learning Models: Predicting Sleep Hours")

filtered_df_ml = filtered_df.dropna(subset=['age', 'bmi', 'stress_level_num', 'physical_activity', 'heart_rate', 'sleep_quality_num', 'sleep_hours'])

st.write(f"Data for ML after dropna: {len(filtered_df_ml)} rows")

if filtered_df_ml.empty:
    st.warning("No data for ML models after filtering.")
    st.stop()

X = filtered_df_ml[['age', 'bmi', 'stress_level_num', 'physical_activity', 'heart_rate', 'sleep_quality_num']]
y = filtered_df_ml['sleep_hours']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    st.subheader(name)
    st.write(f"Mean Squared Error: {mse:.3f}")
    st.write(f"R-squared: {r2:.3f}")
    fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, title=f'{name}: Actual vs Predicted')
    fig.add_shape(type="line", line=dict(dash='dash'), x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
    st.plotly_chart(fig, use_container_width=True)
