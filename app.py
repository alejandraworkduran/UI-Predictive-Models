import streamlit as st
import pickle
from joblib import load
import datetime
import geocoder
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


# Load the model
model = load('best_model.pkl')
modelxgb = load('xgb_model.pkl')

# Define the Streamlit app
def main():
    st.title('Predictive Models')

     # Initialize a LabelEncoder
    le = LabelEncoder()

    # Load the species data from output.csv
    species_data = pd.read_csv('C:/Users/Ale/OneDrive/Desktop/CAPSTONE/output.csv')

    # Fit the LabelEncoder to the species data
    le.fit(species_data['rank_species'])

    
    # Add input options for user
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        # Get current date
        current_date = datetime.date.today()
        # Set default values
        # Get current location coordinates
        g = geocoder.ip('me')
        your_lat = g.latlng[0]
        your_lon = g.latlng[1]
        # Get city name or closest beach name based on latitude and longitude
        location = g.city if g.city else g.address.split(',')[0]   
        lat = st.number_input('Latitude', value=your_lat)
        lon = st.number_input('Longitude', value=your_lon)
        st.write('')  # Add an empty line for spacing
        st.write('')  # Add an empty line for spacing
        st.write('Location:', location) 



        # # Function to find the closest beach based on latitude and longitude
        # def find_closest_beach(lat, lon):
        #     # Make a request to the beach API or database
        #     response = requests.get(f'https://api.example.com/beaches?lat={lat}&lon={lon}')

        #     # Parse the response and extract the closest beach name
        #     closest_beach_name = response.json()['closest_beach']

        #     return closest_beach_name

        # # Get the closest beach name
        # lat = st.number_input('Latitude', value=your_lat)
        # lon = st.number_input('Longitude', value=your_lon)

        # closest_beach = find_closest_beach(lat, lon)

        # # Display the closest beach name
        # st.write('Closest Beach:', closest_beach)
    # with col2:

    with col3:
        year = st.number_input('Year', value=current_date.year)
        month = st.number_input('Month', value=current_date.month)
        day = st.number_input('Day', value=current_date.day)

    with col5:
        # Load the species data from output.csv
        # species_data = pd.read_csv('C:/Users/Ale/OneDrive/Desktop/CAPSTONE/output.csv')
        # st.write(species_data['rank_species'].unique())
        # Load the species data from output.csv
        species_data = pd.read_csv('C:/Users/Ale/OneDrive/Desktop/CAPSTONE/output.csv')

        # Get unique species ranks
        unique_species = species_data['rank_species'].unique()

        # Add an empty option at the beginning
        options = np.insert(unique_species, 0, '')

        # Get the rank of the selected species
        rank_species = st.selectbox('Select Species', options)


    # Use the model to make predictions
    st.write('')  # Add an empty line for spacing
    st.write('')  # Add an empty line for spacing
    if st.button('Will Jellyfish and Humans be Present?'):
        # Use the model to make predictions
        prediction = model.predict([[lat, lon, year, month, day]])
        # Display the prediction
        st.write('Dual Presence:', prediction)
        if prediction == 1:
            st.write('Yes, both jellyfish and human presence is expected at this location.')
            # Filter the merged data for the specified location
            # Find approximate matches based on latitude and longitude
            st.write('')
            st.write('')
            data_merged = pd.read_csv('C:/Users/Ale/OneDrive/Desktop/CAPSTONE/data_merged_human_jellyfish.csv')
            location_data = data_merged[(abs(data_merged['lat'] - lat) <= 0.1) & (abs(data_merged['lon'] - lon) <= 0.1)]
           
            # Display the filtered data if there are matching records
            if not location_data.empty:
                st.write('Historical Data:')
                st.write(location_data)
        else:
            st.write('No, either jellyfish or human presence is not expected at this location.')
    if st.button('Predict Biovolume'):
        # Encode the selected species
        if rank_species == '':
            st.write('Please select a species.')
        else:
            encoded_species = le.transform([rank_species])
            # Convert the input data into a DMatrix object
            input_data = xgb.DMatrix([[year, month, lat, lon, encoded_species[0], day]], feature_names=['year', 'month', 'lat', 'lon', 'rank_species', 'day'])
            
            # Use the second model to make predictions
            biovolume_prediction = modelxgb.predict(input_data)
            
            # Display the biovolume prediction
            st.write('Biovolume Prediction in mm3:', biovolume_prediction)

if __name__ == '__main__':
    main()
