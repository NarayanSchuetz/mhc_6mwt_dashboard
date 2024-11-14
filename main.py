import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
import plotly.graph_objects as go
from scipy.stats import rankdata

from daos import LocalSixMwtSummary
from mappings import cardio_disease_mapping, vascular_disease_mapping, pretty_print_mapping
from css import title_styles, button_styles
from options import cardio_disease_options, vascular_disease_options
from plot_config import cfgs


DEBUG = False
TITLE = "The Stanford My Heart Counts (MHC) Cardiovascular Health Study: Six-Minute Walk Test (6MWT) Data Visualization"
k = 200

_summary_dao = LocalSixMwtSummary(path="30k_6mwts_anonymized_filtered_and_winsored.parquet")


@st.cache_resource
def get_summary_data():
    return _summary_dao.get_full_df()


# Sidebar for unit toggle button
st.sidebar.markdown("## Unit System")
use_metric = st.sidebar.checkbox("Metric Units", value=True)

col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown(f"{title_styles}<div class='title'>{TITLE}</div>", unsafe_allow_html=True)
    st.markdown('<h4 style="text-align:center;"><i>The Largest Crowd-Sourced Digital Walk Repository with over 30,000 '
                'Smartphone-Recorded 6-Minute Walk Tests from more than 8,000 Users.</i></h4>',
                unsafe_allow_html=True)


# Create a centered div with the buttons
html_code = f"""
{button_styles}
<div style="display: flex; justify-content: center; margin-top:50px; margin-bottom:75px;">
    <a href="https://www.synapse.org/Synapse:syn61843472/wiki/629116">
        <button class="button">To the Dataset</button>
    </a>
    <a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://apps.apple.com/us/app/myheart-counts/id972189947&ved=2ahUKEwiH-Km9ouKGAxV7KEQIHTIrBz4QFnoECBIQAQ&usg=AOvVaw2KKeG75JTdl-8FA17ZPzOv">
        <button class="button">To the iOS App</button>
    </a>
</div>
"""


df = get_summary_data()

st.write(html_code, unsafe_allow_html=True)
st.header('Where do you stand (or walk)?', divider='rainbow')
st.write(f'Give it a try, we compare your data against the {k} most similar walks!')


disease_options = ["None"] + list(cardio_disease_options) + list(vascular_disease_options)

# User input fields
sixmwd = st.number_input("Enter your 6-minute walk distance (meters)", min_value=300, value=None, max_value=1500, step=1)
sex = st.selectbox("Select your sex", ["Undefined", "Male", "Female"])
age = st.number_input("Enter your age (years)", min_value=18, value=None, step=1, max_value=90)

height, weight = None, None

if use_metric:
    height = st.number_input("Enter your height (cm)", min_value=120, value=None, max_value=220, step=1)
    weight = st.number_input("Enter your weight (kg)", min_value=30, value=None, max_value=200, step=1)
else:
    height_ft = st.number_input("Enter your height (feet)", min_value=4, value=None, max_value=7, step=1)
    height_in = st.number_input("Enter your height (inches)", min_value=0, value=None, max_value=11, step=1)
    weight_lbs = st.number_input("Enter your weight (lbs)", min_value=66, value=None, max_value=440, step=1)
    
    if height_ft is not None and height_in is not None:
        height = (height_ft * 30.48) + (height_in * 2.54)
    elif height_ft is not None and height_in is None:
        height = height_ft * 30.48
    if weight_lbs is not None:
        weight = weight_lbs * 0.453592

disease = st.selectbox("Do you have any of the following cardiovascular diseases?", disease_options, index=0)

# Transform sex to 0 and 1
if sex == "Male":
    sex_code = 1
elif sex == "Female":
    sex_code = 0
else:
    sex_code = None

# Calculate expected walking distance
expected_distance = None
if sex_code is not None and age is not None and height is not None and weight is not None:
    bmi = weight / ((height / 100) ** 2)
    age_squared = age ** 2
    expected_distance = (
        sex_code * 57.3225873156172 +
        age * 6.50473291068942 +
        age_squared * -0.0782335499384106 +
        height * 3.91707348474492 +
        bmi * -1.97874151686983 +
        -275.910433171369
    )
    st.markdown(f"### Your expected 6-minute walk distance based on your demographics: <span style='color: blue;'>{expected_distance:.2f} meters</span>", unsafe_allow_html=True)

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
    filtered_df = df[selected_columns].dropna()

    if len(filtered_df) < k:
        st.warning(f"Only {len(filtered_df)} walks found for the selected demographics, results may be biased.")
        k = len(filtered_df) - 1

    scaler = RobustScaler()
    filtered_df_scaled = scaler.fit_transform(filtered_df)
    query_point_scaled = scaler.transform([selected_values])
    nbrs = NearestNeighbors(n_neighbors=k).fit(filtered_df_scaled)
    distances, indices = nbrs.kneighbors(query_point_scaled)
    original_indices = filtered_df.dropna().index[indices[0]]
    nearest_neighbors_df = df.loc[original_indices]
else:
    nearest_neighbors_df = df


user_walk_distance = sixmwd
walk_distances = nearest_neighbors_df["6mwt_total_distance"].dropna()

# Calculate ranks and user's quantile position only if user_walk_distance is provided
if user_walk_distance is not None:
    ranks = rankdata(walk_distances, method='average')
    ecdf = ranks / len(walk_distances)
    user_rank = rankdata(np.append(walk_distances, user_walk_distance), method='average')[-1]
    user_quantile = user_rank / len(walk_distances)

for cfg in cfgs:
    if cfg["var"] == "6mwt_total_distance":  # Adjust for walk distance
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=walk_distances, name='Walk Distances'))
        if user_walk_distance is not None:
            fig.add_vline(x=user_walk_distance, line=dict(color="Red", width=3), name='Your Position')
        if expected_distance is not None:
            fig.add_vline(x=expected_distance, line=dict(color="Blue", width=2, dash='dash'), name='Expected Distance')
        fig.update_layout(
            title='Distribution and Your Position in 6-Minute Walk Test Distance',
            xaxis_title=cfg["title"],
            yaxis_title='Frequency',
            bargap=0.2,
            template="plotly_white",
            xaxis=dict(range=cfg["xlim"]),
        )

        st.plotly_chart(fig)
        if user_walk_distance is not None:
            st.write(f"Your 6-Minute Walk Distance: {user_walk_distance} meters puts you at the {user_quantile:.2%} quantile.")
        if expected_distance is not None:
            st.markdown(f"Your expected distance: <span style='color: blue;'>{expected_distance:.2f} meters</span>", unsafe_allow_html=True)
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

st.warning(f'Treat this estimations with caution. We look for the {k} walks that most closely match your demographics '
           'data and this may result in biased estimates if you differ too much from our current users.', icon="⚠️")
