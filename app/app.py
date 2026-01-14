import streamlit as st
import requests
import os
import pandas as pd
import folium
from streamlit_folium import folium_static

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.title("Next Location Prediction Using the Microsoft GeoLife Dataset")
st.write("""
This app allows users to conduct a basic level pattern-of-life analysis and leverage semi-supervised machine learning techniques
to conduct next location prediction given a previous location and datetime information.
""")

st.sidebar.title("User Inputs")
st.sidebar.subheader("Select A User and Specify Parameters")
with st.sidebar:
    with st.form('Modeling Inputs'):
        approved_uid = ['000', '002', '003', '004', '011', '014']
        st.subheader("Clustering Inputs")
        uid = st.selectbox(label='User ID', options=approved_uid)
        distance = st.slider(label='Max Distance Between Two Points (in kms)', min_value=.01, max_value=.5, value=.2)
        min_k = st.slider(label='Min. Number of Observations', min_value=1, max_value=5, value=1)
        cluster_submit = st.form_submit_button("Cluster", type="primary")
        st.subheader("Modeling Inputs")
        thresh = st.slider(label="Threshold", min_value=3, max_value=10, value=6)
        min_samlpes = st.slider(label='Min. # of Samples', min_value=2, max_value=9)
        col1, col2 = st.columns(2)
        with col1:
            model_submit = st.form_submit_button("Model", type="primary")
        with col2:
            model_save = st.form_submit_button('Save', type='secondary')
        
        st.subheader("Prediction Inputs")
        lat = st.number_input(label='Last Lat', step=0.000001, format='%.6f')
        lng = st.number_input(label='Last Lon', step=0.000001, format='%.6f')
        date = st.date_input(label='Date')
        time = st.time_input(label='Time')
        timedelta = st.number_input(label='Time Spent (in Minutes) at Last Location', min_value=20.00, step=0.01, format='%.2f')
        predict_submit = st.form_submit_button(label='Predict', type='primary')

if cluster_submit:

    payload = {
        "uid": uid,
        "distance": distance,
        "min_k": min_k
    }

    with st.spinner("Clustering..."):
        try:
            cluster_response = requests.post(f"{BACKEND_URL}/cluster", json=payload)
            cluster_response.raise_for_status()  # Raise exception for HTTP errors
            cluster_api_dict = cluster_response.json()
            cluster_df = pd.DataFrame().from_dict(cluster_api_dict['df'], orient='columns')
            cluster_scores = pd.DataFrame().from_dict(cluster_api_dict['scores'], orient='index').T        
            st.dataframe(cluster_scores, hide_index=True)

            m = folium.Map(
                location=[cluster_df['lat_origin'].median(), cluster_df['lng_origin'].median()], 
                zoom_start=12,
                tiles='OpenStreetMap'
                )

            for index, row in cluster_df.iterrows():
                folium.CircleMarker(
                    location=[row['lat_origin'], row['lng_origin']],
                    popup=row['cluster_origin'],
                    tooltip=row['cluster_origin'],
                    color="#061C80FF",
                    fill=True,
                    radius=7,
                    fill_color="#061C80FF",
                    fill_opacity=1
                    ).add_to(m)

            for index, row in cluster_df.iterrows():
                folium.CircleMarker(
                    location=[row['cluster_dest_lat'], row['cluster_dest_lng']],
                    popup=row['cluster_dest'],
                    tooltip=row['cluster_dest'],
                    color="#A51D1DFF",
                    fill=True,
                    radius=1,
                    fill_color="#A51D1DFF",
                    fill_opacity=1
                    ).add_to(m)
            m.add_child(folium.ClickForLatLng(format_str='lat + " , " + lng'))
            folium_static(m, width=1000, height=600)
            st.dataframe(cluster_df, hide_index=True, width=1600)

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to prediction service: {str(e)}")
            st.warning(f"Make sure the backend service is running at {BACKEND_URL}")

if model_submit:

    payload = {
        "uid": uid,
        "distance": distance,
        "min_k": min_k,
        "thresh": thresh,
        'min_samples': min_samlpes
    }

    with st.spinner("Modeling..."):
        try:
            model_response = requests.post(f"{BACKEND_URL}/model", json=payload)
            model_response.raise_for_status()  # Raise exception for HTTP errors
            model_api_dict = model_response.json()
            cluster_df = pd.DataFrame().from_dict(model_api_dict['df'], orient='columns')
            cluster_scores = pd.DataFrame().from_dict(model_api_dict['model_scores'], orient='index').T        
            st.dataframe(cluster_scores, hide_index=True)

            m = folium.Map(
                location=[cluster_df['lat_origin'].median(), cluster_df['lng_origin'].median()], 
                zoom_start=12,
                tiles='OpenStreetMap'
                )

            for index, row in cluster_df.iterrows():
                folium.CircleMarker(
                    location=[row['lat_origin'], row['lng_origin']],
                    popup=row['cluster_origin'],
                    tooltip=row['cluster_origin'],
                    color="#061C80FF",
                    fill=True,
                    radius=7,
                    fill_color="#061C80FF",
                    fill_opacity=1
                    ).add_to(m)

            for index, row in cluster_df.iterrows():
                folium.CircleMarker(
                    location=[row['cluster_dest_lat'], row['cluster_dest_lng']],
                    popup=row['cluster_dest'],
                    tooltip=row['cluster_dest'],
                    color="#A51D1DFF",
                    fill=True,
                    radius=1,
                    fill_color="#A51D1DFF",
                    fill_opacity=1
                    ).add_to(m)
            m.add_child(folium.ClickForLatLng())
            folium_static(m, width=1000, height=600)
            st.dataframe(cluster_df, hide_index=True, width=1600)

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to prediction service: {str(e)}")
            st.warning(f"Make sure the backend service is running at {BACKEND_URL}")

if model_save:

    payload = {
        "uid": uid,
        "distance": distance,
        "min_k": min_k,
        "thresh": thresh,
        'min_samples': min_samlpes
    }

    with st.spinner("Saving..."):
        try:
            save_response = requests.post(f"{BACKEND_URL}/save", json=payload)
            save_response.raise_for_status()  # Raise exception for HTTP errors
            save_api_dict = save_response.json()
            cluster_df = pd.DataFrame().from_dict(save_api_dict['df'], orient='columns')
            message = save_api_dict['Message']        
            st.write(message)

            m = folium.Map(
                location=[cluster_df['lat_origin'].median(), cluster_df['lng_origin'].median()], 
                zoom_start=12,
                tiles='OpenStreetMap'
                )

            for index, row in cluster_df.iterrows():
                folium.CircleMarker(
                    location=[row['lat_origin'], row['lng_origin']],
                    popup=row['cluster_origin'],
                    tooltip=row['cluster_origin'],
                    color="#061C80FF",
                    fill=True,
                    radius=7,
                    fill_color="#061C80FF",
                    fill_opacity=1
                    ).add_to(m)

            for index, row in cluster_df.iterrows():
                folium.CircleMarker(
                    location=[row['cluster_dest_lat'], row['cluster_dest_lng']],
                    popup=row['cluster_dest'],
                    tooltip=row['cluster_dest'],
                    color="#A51D1DFF",
                    fill=True,
                    radius=1,
                    fill_color="#A51D1DFF",
                    fill_opacity=1
                    ).add_to(m)
            m.add_child(folium.ClickForLatLng())
            folium_static(m, width=1000, height=600)
            st.dataframe(cluster_df, hide_index=True, width=1600)

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to prediction service: {str(e)}")
            st.warning(f"Make sure the backend service is running at {BACKEND_URL}")

if predict_submit:
    month = date.month
    day = date.day
    day_of_week = date.weekday()
    hour = time.hour
    minute = time.minute

    payload = {
        "uid": uid,
        'lat_origin': lat,
        'lng_origin': lng,
        'timedelta': timedelta,
        'month': month,
        'day': day,
        'day_of_week': day_of_week,
        'hour_in_day': hour,
        'minute_in_hour': minute
    }

    with st.spinner("Predicting..."):
        try:
            predict_response = requests.post(f"{BACKEND_URL}/predict", json=payload)
            predict_response.raise_for_status()  # Raise exception for HTTP errors
            predict_api_dict = predict_response.json()
            cluster_df = pd.DataFrame().from_dict(predict_api_dict['df'], orient='columns')
            prediction = predict_api_dict['Prediction']        
            st.write(f"Predicted Cluster Destination: {prediction}")

            m = folium.Map(
                location=[cluster_df['lat_origin'].median(), cluster_df['lng_origin'].median()], 
                zoom_start=12,
                tiles='OpenStreetMap'
                )

            for index, row in cluster_df.iterrows():
                folium.CircleMarker(
                    location=[row['lat_origin'], row['lng_origin']],
                    popup=row['cluster_origin'],
                    tooltip=row['cluster_origin'],
                    color="#061C80FF",
                    fill=True,
                    radius=7,
                    fill_color="#061C80FF",
                    fill_opacity=1
                    ).add_to(m)

            for index, row in cluster_df.iterrows():
                folium.CircleMarker(
                    location=[row['cluster_dest_lat'], row['cluster_dest_lng']],
                    popup=row['cluster_dest'],
                    tooltip=row['cluster_dest'],
                    color="#A51D1DFF",
                    fill=True,
                    radius=1,
                    fill_color="#A51D1DFF",
                    fill_opacity=1
                    ).add_to(m)
            folium.Marker(
                location=[
                    cluster_df[cluster_df['cluster_dest'] == prediction]['cluster_dest_lat'].unique(), 
                    cluster_df[cluster_df['cluster_dest'] == prediction]['cluster_dest_lng'].unique()
                    ]
                ).add_to(m)
            m.add_child(folium.ClickForLatLng())
            folium_static(m, width=1000, height=600)
            st.dataframe(cluster_df, hide_index=True, width=1600)

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to prediction service: {str(e)}")
            st.warning(f"Make sure the backend service is running at {BACKEND_URL}")
# # Add information about the app
# st.sidebar.header("about")
# st.sidebar.write("""
# Uses a random forest regressor to predict car mpg

# """)

# st.sidebar.header("Feature Impact on MPG")
# feature_importance = pd.DataFrame({
#     'Feature': ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin'],
#     'Importance': [0.12, 0.18, 0.15, 0.25, 0.05, 0.15, 0.10]  # Example values
# })

# st.sidebar.bar_chart(feature_importance.set_index('Feature'))