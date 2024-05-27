import streamlit as st
import pickle
from joblib import load
import datetime
import geocoder
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load the models
model = load('best_model.pkl')
modelxgb = load('xgb_model.pkl')

# Define the Streamlit app
def main():
    st.image("C:/Users/Ale/OneDrive/Desktop/CAPSTONE/tbD.png", width=200)
    st.title('Predictive Models')

    # Initialize a LabelEncoder
    le = LabelEncoder()

    # Load the species data from output.csv
    species_data = pd.read_csv('C:/Users/Ale/OneDrive/Desktop/CAPSTONE/output.csv')
    le.fit(species_data['rank_species'])

    # Add input options for user
    col1, col2, col3, col4, col5 = st.columns(5)
    with col3:
        current_date = datetime.date.today()
        g = geocoder.ip('me')
        your_lat = g.latlng[0] if g.latlng else 0.0
        your_lon = g.latlng[1] if g.latlng else 0.0
        location = g.city if g.city else "Unknown Location"
        lat = st.number_input('Latitude', value=your_lat, format="%.6f")
        lon = st.number_input('Longitude', value=your_lon, format="%.6f")
        st.markdown(f"<p class='center-text'>Location: {location}</p>", unsafe_allow_html=True)

    with col2:
        year = st.number_input('Year', value=current_date.year)
        month = st.number_input('Month', value=current_date.month)
        day = st.number_input('Day', value=current_date.day)
        if st.button('Will Jellyfish and Humans be Present?'):
            input_data = [[lat, lon, year, month, day]]
            try:
                prediction = model.predict(input_data)
                prediction = prediction[0]
                st.write('Dual Presence:', prediction)
                if prediction == 1:
                    st.write('Yes, both jellyfish and human presence is expected at this location.')
                    data_merged = pd.read_csv('C:/Users/Ale/OneDrive/Desktop/CAPSTONE/data_merged_human_jellyfish.csv')
                    location_data = data_merged[(abs(data_merged['lat'] - lat) <= 0.1) & (abs(data_merged['lon'] - lon) <= 0.1)]
                    if not location_data.empty:
                        st.write('Historical Data:')
                        st.write(location_data) 
                else:
                    st.write('No, either jellyfish or human presence is not expected at this location.')
                    data_merged = pd.read_csv('C:/Users/Ale/OneDrive/Desktop/CAPSTONE/data_merged_human_jellyfish.csv')
                    location_data = data_merged[(abs(data_merged['lat'] - lat) <= 0.1) & (abs(data_merged['lon'] - lon) <= 0.1)]
                    if not location_data.empty:
                        st.write('Historical Data:')
                        st.write(location_data)
            except Exception as e:
                st.write("Error making prediction:", e)

    with col4:
        try:
            species_data = pd.read_csv('C:/Users/Ale/OneDrive/Desktop/CAPSTONE/output.csv')
            unique_species = species_data['rank_species'].unique()
            options = np.insert(unique_species, 0, '')
            rank_species = st.selectbox('Select Species', options)
            if st.button('Predict Biovolume for Jellyfish'):
                if rank_species == '':
                    st.write('Please select a species.')
                else:
                    encoded_species = le.transform([rank_species])
                    input_data = xgb.DMatrix([[year, month, lat, lon, encoded_species[0], day]], feature_names=['year', 'month', 'lat', 'lon', 'rank_species', 'day'])
                    biovolume_prediction = modelxgb.predict(input_data)
                    st.write('Biovolume Prediction in mm3:', biovolume_prediction)
        except Exception as e:
            st.write("Error during biovolume prediction:", e)

if __name__ == '__main__':
    main()
