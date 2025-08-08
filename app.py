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

@st.cache_data
def filter_data(df, genders, stress_levels, sleep_qualities, age_range):
    filtered = df[
        (df['gender'].isin(genders)) &
        (df['stress_level'].isin(stress_levels)) &
        (df['sleep_quality'].isin(sleep_qualities)) &
        (df['age'].between(age_range[0], age_range[1]))
    ].copy()
    return filtered

df = load_data()

with st.sidebar.expander("Filter Options", expanded=True):
    genders = df['gender'].unique().tolist()
    selected_genders = st.multiselect("Select Gender(s)", genders, default=genders)

    stress_levels = df['stress_level'].unique().tolist()
    selected_stress_levels = st.multiselect("Select Stress Level(s)", stress_levels, default=stress_levels)

    sleep_qualities = df['sleep_quality'].unique().tolist()
    selected_sleep_qualities = st.multiselect("Select Sleep Quality(ies)", sleep_qualities, default=sleep_qualities)

    min_age, max_age = int(df['age'].min()), int(df['age'].max())
    age_range = st.slider("Select Age Range", min_age, max_age, (min_age, max_age))

filtered_df = filter_data(df, selected_genders, selected_stress_levels, selected_sleep_qualities, age_range)

st.title("Sleep Health & Lifestyle Analysis")

# 1. BMI vs Sleep Hours colored by Sleep Quality
st.subheader("1. BMI vs Sleep Hours colored by Sleep Quality")
fig1 = px.scatter(filtered_df, x='bmi', y='sleep_hours', color='sleep_quality',
                  labels={'bmi': 'BMI', 'sleep_hours': 'Sleep Hours', 'sleep_quality': 'Sleep Quality'},
                  title='BMI vs Sleep Hours Colored by Sleep Quality')
st.plotly_chart(fig1, use_container_width=True, key='fig1')

# 2. Distribution of Sleep Quality
st.subheader("2. Distribution of Sleep Quality")
fig2 = px.histogram(filtered_df, x='sleep_quality', title='Sleep Quality Count')
st.plotly_chart(fig2, use_container_width=True, key='fig2')

# 3. Age vs Sleep Hours colored by Sleep Quality
st.subheader("3. Age vs Sleep Hours Colored by Sleep Quality")
fig3 = px.scatter(filtered_df, x='age', y='sleep_hours', color='sleep_quality',
                  labels={'age': 'Age', 'sleep_hours': 'Sleep Hours'},
                  title='Age vs Sleep Hours Colored by Sleep Quality')
st.plotly_chart(fig3, use_container_width=True, key='fig3')

# 4. Sleep Hours Distribution by Gender (Box plot)
st.subheader("4. Sleep Hours Distribution by Gender")
fig4 = px.box(filtered_df, x='gender', y='sleep_hours',
              labels={'gender': 'Gender', 'sleep_hours': 'Sleep Hours'},
              title='Sleep Hours Distribution by Gender')
st.plotly_chart(fig4, use_container_width=True, key='fig4')

# 5. BMI Distribution by Sleep Quality (Violin plot)
st.subheader("5. BMI Distribution by Sleep Quality")
fig5 = px.violin(filtered_df, x='sleep_quality', y='bmi', box=True, points='all',
                 labels={'sleep_quality': 'Sleep Quality', 'bmi': 'BMI'},
                 title='BMI Distribution by Sleep Quality')
st.plotly_chart(fig5, use_container_width=True, key='fig5')

# 6. Distribution of Stress Levels
st.subheader("6. Distribution of Stress Levels")
fig6 = px.histogram(filtered_df, x='stress_level', title='Stress Level Count')
st.plotly_chart(fig6, use_container_width=True, key='fig6')

# 7. Physical Activity vs Sleep Hours colored by Gender
st.subheader("7. Physical Activity vs Sleep Hours Colored by Gender")
fig7 = px.scatter(filtered_df, x='physical_activity', y='sleep_hours', color='gender',
                  labels={'physical_activity': 'Physical Activity', 'sleep_hours': 'Sleep Hours'},
                  title='Physical Activity vs Sleep Hours Colored by Gender')
st.plotly_chart(fig7, use_container_width=True, key='fig7')

# 8. Heart Rate Distribution by Sleep Quality (Box plot)
st.subheader("8. Heart Rate Distribution by Sleep Quality")
fig8 = px.box(filtered_df, x='sleep_quality', y='heart_rate',
              labels={'sleep_quality': 'Sleep Quality', 'heart_rate': 'Heart Rate'},
              title='Heart Rate Distribution by Sleep Quality')
st.plotly_chart(fig8, use_container_width=True, key='fig8')

# 9. Age vs BMI colored by Sleep Quality
st.subheader("9. Age vs BMI Colored by Sleep Quality")
fig9 = px.scatter(filtered_df, x='age', y='bmi', color='sleep_quality',
                  labels={'age': 'Age', 'bmi': 'BMI'},
                  title='Age vs BMI Colored by Sleep Quality')
st.plotly_chart(fig9, use_container_width=True, key='fig9')

# 10. Distribution of Gender
st.subheader("10. Distribution of Gender")
fig10 = px.histogram(filtered_df, x='gender', title='Gender Count')
st.plotly_chart(fig10, use_container_width=True, key='fig10')

# 11. Correlation Heatmap of Numerical Features
st.subheader("11. Correlation Heatmap of Numerical Features")
stress_map = {'Low': 1, 'Medium': 2, 'High': 3}
filtered_df['stress_level_num'] = filtered_df['stress_level'].map(stress_map)
corr = filtered_df[['age', 'bmi', 'sleep_hours', 'stress_level_num', 'physical_activity', 'heart_rate']].corr()

plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
st.pyplot(plt.gcf())
plt.clf()

# 12. Sleep Hours Distribution by Gender (Box plot)
st.subheader("12. Sleep Hours Distribution by Gender")
fig12 = px.box(filtered_df, x='gender', y='sleep_hours',
               labels={'gender': 'Gender', 'sleep_hours': 'Sleep Hours'},
               title='Sleep Hours Distribution by Gender')
st.plotly_chart(fig12, use_container_width=True, key='fig12')

# 13. Stress Level Distribution by Sleep Quality (Violin plot)
st.subheader("13. Stress Level Distribution by Sleep Quality")
fig13 = px.violin(filtered_df, x='sleep_quality', y='stress_level', box=True, points='all',
                  labels={'sleep_quality': 'Sleep Quality', 'stress_level': 'Stress Level'},
                  title='Stress Level Distribution by Sleep Quality')
st.plotly_chart(fig13, use_container_width=True, key='fig13')

# 14. Pair Plot of Key Health Variables
st.subheader("14. Pair Plot of Key Health Variables")
pair_data = filtered_df[['age', 'bmi', 'sleep_hours', 'stress_level_num', 'physical_activity', 'heart_rate']]
sns.pairplot(pair_data)
st.pyplot(plt.gcf())
plt.clf()

# 15. Sleep Hours Distribution
st.subheader("15. Sleep Hours Distribution")
fig15 = px.histogram(filtered_df, x='sleep_hours', nbins=20,
                     labels={'sleep_hours': 'Sleep Hours'},
                     title='Sleep Hours Distribution')
st.plotly_chart(fig15, use_container_width=True, key='fig15')

# 16. Physical Activity vs Sleep Hours
st.subheader("16. Physical Activity vs Sleep Hours")
fig16 = px.scatter(filtered_df, x='physical_activity', y='sleep_hours',
                   labels={'physical_activity': 'Physical Activity', 'sleep_hours': 'Sleep Hours'},
                   title='Physical Activity vs Sleep Hours')
st.plotly_chart(fig16, use_container_width=True, key='fig16')

# 17. Heart Rate Distribution by Gender
st.subheader("17. Heart Rate Distribution by Gender")
fig17 = px.histogram(filtered_df, x='heart_rate', color='gender', barmode='overlay',
                     labels={'heart_rate': 'Heart Rate', 'gender': 'Gender'},
                     title='Heart Rate Distribution by Gender')
st.plotly_chart(fig17, use_container_width=True, key='fig17')

# 18. Stress Level Distribution by Gender (Box plot)
st.subheader("18. Stress Level Distribution by Gender")
fig18 = px.box(filtered_df, x='gender', y='stress_level',
               labels={'gender': 'Gender', 'stress_level': 'Stress Level'},
               title='Stress Level Distribution by Gender')
st.plotly_chart(fig18, use_container_width=True, key='fig18')

# 19. BMI Distribution
st.subheader("19. BMI Distribution")
fig19 = px.histogram(filtered_df, x='bmi', nbins=20,
                     labels={'bmi': 'BMI'},
                     title='BMI Distribution')
st.plotly_chart(fig19, use_container_width=True, key='fig19')

# 20. Counts of Sleep Quality Categories
st.subheader("20. Counts of Sleep Quality Categories")
sleep_quality_counts = filtered_df['sleep_quality'].value_counts().reset_index()
sleep_quality_counts.columns = ['sleep_quality', 'count']
fig20 = px.bar(sleep_quality_counts, x='sleep_quality', y='count',
               labels={'sleep_quality': 'Sleep Quality', 'count': 'Count'},
               title='Counts of Sleep Quality Categories')
st.plotly_chart(fig20, use_container_width=True, key='fig20')

# Machine Learning Section

st.header("Machine Learning Models: Predicting Sleep Hours")

filtered_df_ml = filtered_df.copy()

# Map categorical variables to numeric
filtered_df_ml['stress_level_num'] = filtered_df_ml['stress_level'].map({'Low': 1, 'Medium': 2, 'High': 3})
filtered_df_ml['sleep_quality_num'] = filtered_df_ml['sleep_quality'].map({'Poor': 1, 'Average': 2, 'Good': 3})

filtered_df_ml = filtered_df_ml.dropna(subset=['age', 'bmi', 'stress_level_num', 'physical_activity', 'heart_rate', 'sleep_quality_num', 'sleep_hours'])

X = filtered_df_ml[['age', 'bmi', 'stress_level_num', 'physical_activity', 'heart_rate', 'sleep_quality_num']]
y = filtered_df_ml['sleep_hours']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = (mse, r2, preds)

st.subheader("Model Evaluation Metrics")

for name, (mse, r2, _) in results.items():
    st.write(f"**{name}**")
    st.write(f"Mean Squared Error (MSE): {mse:.3f}")
    st.write(f"R-squared (R²): {r2:.3f}")
    st.write("---")

st.subheader("Actual vs Predicted Sleep Hours")

for name, (_, _, preds) in results.items():
    fig = px.scatter(x=y_test, y=preds,
                     labels={'x': 'Actual Sleep Hours', 'y': 'Predicted Sleep Hours'},
                     title=f'{name}: Actual vs Predicted')
    fig.add_shape(
        type="line",
        line=dict(dash='dash'),
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max()
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ml_{name}")
