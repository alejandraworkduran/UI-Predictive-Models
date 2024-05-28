import streamlit as st
import pandas as pd
import datetime
import geocoder
import joblib
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
from geopy.geocoders import Nominatim
import osmnx as ox


# Load the models
model = joblib.load('best_model.pkl')
modelxgb = joblib.load('xgb_model.pkl')

def get_nearest_ocean_sea(lon):
    ocean_sea = "Unknown Ocean/Sea"
    if -180 <= lon <= -80:
        ocean_sea = "Southern Ocean"
    elif -80 < lon <= -20:
        ocean_sea = "Atlantic Ocean"
    elif -20 < lon <= 30:
        ocean_sea = "Indian Ocean"
    elif 30 < lon <= 160:
        ocean_sea = "Pacific Ocean"
    elif 160 < lon <= 180:
        ocean_sea = "Pacific Ocean"  # Close to International Date Line, might be in the Pacific or the Arctic Ocean
    return ocean_sea

def get_nearest_beach(lat, lon):
    # Define the tags to search for beaches
    tags = {"landuse": "beach"}

    # Use osmnx to retrieve the nearest beach
    try:
        beach = ox.geometries_from_point((lat, lon), tags=tags, dist=1000)
        if len(beach) > 0:
            return beach.iloc[0]['name']
    except Exception as e:
        print("Error retrieving beach information:", e)
    
    # Return "Unknown Beach" if no beach is found within the specified distance
    return "Unknown Beach"

from geopy.geocoders import Nominatim

# def get_location_name(lat, lon):
#     geolocator = Nominatim(user_agent="geoapiExercises")
#     location = geolocator.reverse((lat, lon), language="en")
#     return location.address if location else "Unknown Location"

def set_background_image(url, size="cover"):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({url});
            background-size: {size};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    
    # Set the background image
    set_background_image("https://cosmosmagazine.com/wp-content/uploads/2019/12/170510_Larvacean_Thumb.jpg", size='115% 115%')
    st.image("tbD.png", width=200)
    st.title('Predictive Models')

    le = LabelEncoder()
    species_data = pd.read_csv('output.csv')
    le.fit(species_data['rank_species'])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col3:
        current_date = datetime.date.today()
        g = geocoder.ip('me')
        your_lat = g.latlng[0] if g.latlng else 0.0
        your_lon = g.latlng[1] if g.latlng else 0.0
        lat = st.number_input('**Latitude**', value=your_lat, format="%.6f")
        lon = st.number_input('**Longitude**', value=your_lon, format="%.6f")
        
        # Fetch dynamic location name using geocoder.ip('me')
        location = g.city if g.city else "Unknown Location"
        
        # Get nearest beach
        nearest_beach = get_nearest_beach(lat, lon)
        
        # Update location name when latitude and longitude change
        if (lat != your_lat) or (lon != your_lon):
            g = geocoder.osm([lat, lon], method='reverse')
            location = g.city if g.city else "Unknown Location"
        
        nearest_ocean_sea = get_nearest_ocean_sea(lon)
        st.write('')
        st.markdown(f"<p style='text-align: center; font-size: 20px;'><b>Location: {location}</b></p>", unsafe_allow_html=True)
        # st.markdown(f"<p style='text-align: center;'>Nearest Bgeach: {nearest_beach}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 20px;'><b>Nearest Ocean/Sea: {nearest_ocean_sea}<b></p>", unsafe_allow_html=True)
    with col2:
        year = st.number_input('**Year**', value=current_date.year)        
        month = st.number_input('**Month**', value=current_date.month)
        day = st.number_input('**Day**', value=current_date.day)
        st.write('')
        # st.write('')
        st.write('')
        if st.button('Will Jellyfish and Humans be Present?'):
            input_data = pd.DataFrame({'lat': [lat], 'lon': [lon], 'year': [year], 'month': [month], 'day': [day]})

            try:
                prediction = model.predict(input_data)
                st.write('Prediction:', prediction)
                if prediction[0] == 1:
                    st.write('**Yes, both jellyfish and human presence is expected at this location.**')
 
                if prediction[0] == 0:
                    st.write('**No, either jellyfish or human presence is not expected at this location.**')
            except Exception as e:
                st.write("Error making prediction:", e)

    with col4:
        unique_species = species_data['rank_species'].unique()
        options = np.insert(unique_species, 0, '')
        rank_species = st.selectbox('**Select Species**', options, key='species_select')

        # st.write('')
        st.write('')
        if st.button('Predict Biovolume for Jellyfish'):
            if rank_species == '':
                st.write('Please select a species.')
            else:
                encoded_species = le.transform([rank_species])
                input_data = xgb.DMatrix([[year, month, lat, lon, encoded_species[0], day]], feature_names=['year', 'month', 'lat', 'lon', 'rank_species', 'day'])
                biovolume_prediction = modelxgb.predict(input_data)
                st.write('Biovolume Prediction in mm3:', biovolume_prediction)

    # Display historical data outside of the columns
    data_merged = pd.read_csv('C:/Users/Ale/OneDrive/Desktop/CAPSTONE/data_merged_human_jellyfish.csv')
    location_data = data_merged[(abs(data_merged['lat'] - lat) <= 0.1) & (abs(data_merged['lon'] - lon) <= 0.1)]
    if not location_data.empty:
        st.write('')
        st.write('')
        st.write('')

        st.write('Historical Data for Location:')
        selected_columns = ['year', 'day', 'month', 'lat', 'lon', 'count_actual_jellyfish', 'count_actual_humans', 'both_present']  # Replace with the actual column names you want to display
        st.write(location_data[selected_columns])

if __name__ == '__main__':
    main()
