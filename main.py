import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
import plotly.graph_objects as go
from scipy.stats import rankdata

from daos import LocalSixMwtSummary

DEBUG = False

cardio_disease_mapping = {
    "Heart Attack/Myocardial Infarction": "Heart_Attack_Myocardial_Infarction",
    "Heart Bypass Surgery": "Heart_Bypass_Surgery",
    "Coronary Blockage/Stenosis": "Coronary_Blockage_Stenosis",
    "Coronary Stent/Angioplasty": "Coronary_Stent_Angioplasty",
    "Angina (heart chest pains)": "Angina_Heart_Chest_Pains",
    "High Coronary Calcium Score": "High_Coronary_Calcium_Score",
    "Heart Failure or CHF": "Heart_Failure_or_CHF",
    "Atrial fibrillation (Afib)": "Atrial_Fibrillation_Afib",
    "Congenital Heart Defect": "Congenital_Heart_Defect",
    "Pulmonary Hypertension": "Pulmonary_Hypertension",
}

vascular_disease_mapping = {
    "Stroke": 'Stroke',
    "Transient Ischemic Attack (TIA)": 'Transient_Ischemic_Attack',
    "Carotid Artery Blockage/Stenosis": "Carotid_Artery_Blockage_Stenosis",
    "Carotid Artery Surgery or Stent": "Carotid_Artery_Surgery_or_Stent",
    "Peripheral Vascular Disease (Blockage/Stenosis, Surgery, or Stent)": "Peripheral_Vascular_Disease",
    "Abdominal Aortic Aneurysm": "Abdominal_Aortic_Aneurysm",
    "Pulmonary Arterial Hypertension": "Pulmonary_Arterial_Hypertension",
}

@st.cache_resource
def get_summary_data():
    return _summary_dao.get_full_df()

@st.cache_data
def loading_csv(df: pd.DataFrame):
    return df.to_csv(index=False)

TITLE = "The Stanford My Heart Counts (MHC) Cardiovascular Health Study: Six-Minute Walk Test (6MWT) Data Visualization"
K = 200

# CSS styles for the title
title_styles = """
<style>
.title {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 36px;
    font-weight: 700;
    text-align: center;
    color: white;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    padding: 20px 0;
}
</style>
"""

# Use columns to center the title
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown(f"{title_styles}<div class='title'>{TITLE}</div>", unsafe_allow_html=True)
    st.markdown('<h4 style="text-align:center;"><i>The Largest Crowd-Sourced Digital Walk Repository with over 30,000 '
                'Smartphone-Recorded 6-Minute Walk Tests from more than 5,000 Users.</i></h4>',
                unsafe_allow_html=True)

_summary_dao = LocalSixMwtSummary(path="~/Downloads/18k_6mwts_anonymized.parquet")
df = get_summary_data()

button_styles = """
<style>
.button {
    background-color: #4CAF50; /* Green background */
    border: none; /* Remove border */
    color: white; /* White text */
    padding: 12px 24px; /* Add padding */
    text-align: center; /* Center text */
    text-decoration: none; /* Remove underline */
    display: inline-block; /* Make it an inline element */
    font-size: 16px; /* Increase font size */
    border-radius: 4px; /* Rounded corners */
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); /* Add a subtle box shadow */
    margin: 10px; /* Add some margin for spacing */
}

/* Change background color on hover */
.button:hover {
    background-color: #45a060;
}
</style>
"""

# Create a centered div with the buttons
html_code = f"""
{button_styles}
<div style="display: flex; justify-content: center; margin-top:50px; margin-bottom:75px;">
    <a href="https://www.google.com" download="data.csv">
        <button class="button">Download Dataset</button>
    </a>
    <button class="button">To the iOS App</button>
</div>
"""

st.write(html_code, unsafe_allow_html=True)
st.header('Where do you stand (or walk)?', divider='rainbow')
st.write(f'Give it a try, we compare your data against the {K} most similar walks!')

pretty_print_mapping = {
    "BiologicalSex": "Biological Sex",
    "age": "Age [years]",
    "HeightCentimeters": "Height [cm]",
    "WeightKilograms": "Weight [kg]",
}

cardio_disease_options = (
    "Heart Attack/Myocardial Infarction", 
    "Heart Bypass Surgery", 
    "Coronary Blockage/Stenosis", 
    "Coronary Stent/Angioplasty", 
    "Angina (heart chest pains)", 
    "High Coronary Calcium Score", 
    "Heart Failure or CHF", 
    "Atrial fibrillation (Afib)", 
    "Congenital Heart Defect", 
    "Pulmonary Hypertension"
)

vascular_disease_options = (
    "Stroke", 
    "Transient Ischemic Attack (TIA)",
    "Carotid Artery Blockage/Stenosis",
    "Carotid Artery Surgery or Stent",
    "Peripheral Vascular Disease (Blockage/Stenosis, Surgery, or Stent)",
    "Abdominal Aortic Aneurysm",
    "Pulmonary Arterial Hypertension"
)

disease_options = ["None"] + list(cardio_disease_options) + list(vascular_disease_options)

# User input fields
sixmwd = st.number_input("Enter your 6-minute walk distance (meters)", min_value=300, value=600, max_value=1500, step=1)
sex = st.selectbox("Select your sex", ["Undefined", "Male", "Female"])
age = st.number_input("Enter your age (years)", min_value=18, value=None, step=1, max_value=90)
height = st.number_input("Enter your height (cm)", min_value=120, value=None, max_value=220, step=1)
weight = st.number_input("Enter your weight (kg)", min_value=30, value=None, max_value=200, step=1)
disease = st.selectbox("Do you have any of the following cardiovascular diseases?", disease_options, index=0)

# Transform sex to 0 and 1
if sex == "Male":
    sex_code = 1
elif sex == "Female":
    sex_code = 0
else:
    sex_code = None


values = [sex_code, age, height, weight]
selected_values = []
selected_columns = []
for i in range(4):
    val = values[i]
    if val is not None:
        selected_values.append(val)
        selected_columns.append(list(pretty_print_mapping.keys())[i])


# Filter the dataframe based on selected disease
if disease != "None":
    if DEBUG:
        st.write(len(df))
    if disease in cardio_disease_mapping:
        disease_code = cardio_disease_mapping[disease]
        df = df[df[disease_code] == 1]
    elif disease in vascular_disease_mapping:
        disease_code = vascular_disease_mapping[disease]
        df = df[df[disease_code] == 1]


if selected_columns:
    if len(df) < K:
        st.warning(f"Only {len(df)} walks found for the selected demographics, results may be biased.")
        K = len(df)
    filtered_df = df[selected_columns]
    scaler = RobustScaler()
    filtered_df_scaled = scaler.fit_transform(filtered_df.dropna())
    query_point_scaled = scaler.transform([selected_values])
    nbrs = NearestNeighbors(n_neighbors=K).fit(filtered_df_scaled)
    distances, indices = nbrs.kneighbors(query_point_scaled)
    original_indices = filtered_df.dropna().index[indices[0]]
    nearest_neighbors_df = df.loc[original_indices]
else:
    nearest_neighbors_df = df


cfgs = [
    {
        "var": "6mwt_total_distance",
        "title": "6-Minute Walk Test Distance (m)",
        "xlim": (300, 1500),
    },
    {
        "var": "6mwt_total_steps",
        "title": "6-Minute Walk Test Steps (count)",
        "xlim": (300, 1500),
    },
    {
        "var": "walk_hr_mean",
        "title": "Mean Walk Heart Rate (bpm)",
        "xlim": (50, 180),
    }
]

# Assuming 'sixmwd' is the user's 6-minute walk distance
user_walk_distance = sixmwd
walk_distances = nearest_neighbors_df["6mwt_total_distance"].dropna()

# Calculate ranks
ranks = rankdata(walk_distances, method='average')
ecdf = ranks / len(walk_distances)

# Find the user's quantile position
user_rank = rankdata(np.append(walk_distances, user_walk_distance), method='average')[-1]
user_quantile = user_rank / len(walk_distances)

for cfg in cfgs:
    if cfg["var"] == "6mwt_total_distance":  # Adjust for walk distance
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=walk_distances, name='Walk Distances'))
        fig.add_vline(x=user_walk_distance, line=dict(color="Red", width=3), name='Your Position')
        fig.update_layout(
            title='Distribution and Your Position in 6-Minute Walk Test Distance',
            xaxis_title=cfg["title"],
            yaxis_title='Frequency',
            bargap=0.2,
            template="plotly_white",
            xaxis=dict(range=cfg["xlim"]),
        )

        st.plotly_chart(fig)
        st.write(f"Your 6-Minute Walk Distance: {user_walk_distance} meters puts you at the {user_quantile:.2%} quantile.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=nearest_neighbors_df[cfg["var"]].dropna()))
        fig.update_layout(
            title=f'Distribution of {cfg["title"]}',
            xaxis_title=cfg["title"],
            yaxis_title='Frequency',
            bargap=0.2,
            template="plotly_white",
            xaxis=dict(range=cfg["xlim"]),
        )
        st.plotly_chart(fig)

hr_timeseries = np.vstack(nearest_neighbors_df.hr_at_n_seconds.dropna().values)
hr_timeseries_mean = hr_timeseries.mean(axis=0)

# Calculate SEM
n = hr_timeseries.shape[0]  # Number of observations
std_dev = hr_timeseries.std(axis=0)  # Standard deviation for each time point
sem = std_dev / np.sqrt(n)  # Standard Error of the Mean

# Time points for plotting
time_points = np.arange(0, 360, 30)

# Create figure and add the mean heart rate trace
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_points, y=hr_timeseries_mean, mode='lines', name='Mean HR',
                         line=dict(color='blue')))

# Add error bars for SEM
fig.add_trace(go.Scatter(x=np.concatenate([time_points, time_points[::-1]]),
                         y=np.concatenate([hr_timeseries_mean - sem, (hr_timeseries_mean + sem)[::-1]]),
                         fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'),
                         hoverinfo="skip", name='SEM'))

# Update the layout
fig.update_layout(
    title='Average Heart Rate Over 6-Minute Walk Test Duration with SEM',
    xaxis_title='Time [seconds]',
    yaxis_title='Heart Rate [bpm]',
    template="plotly_white",
)

st.plotly_chart(fig)

st.warning(f'Treat this estimations with caution. We look for the {K} walks that most closely match your demographics '
           'data and this may result in biased estimates if you differ too much from our current users.', icon="⚠️")
