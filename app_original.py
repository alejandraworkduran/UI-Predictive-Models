import streamlit as st
import pickle
from joblib import load
import datetime
import geocoder
import requests

# Load the model
model = load('best_model.pkl')

# # # Load the model
# with open('best_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# Define the Streamlit app
def main():
    st.title('Predictive Models')
    
    # Add input options for user
    col1, col2, col3 = st.columns(3)
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
        st.write('Location:', location) 
        lat = st.number_input('Latitude', value=your_lat)
        lon = st.number_input('Longitude', value=your_lon)


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
    # with col3:
    #     net_mesh = st.number_input('Net Mesh')
    #     net_opening = st.number_input('Net Opening')
    #     time_seconds_x = st.number_input('Time Seconds X')
    #     time_seconds_y = st.number_input('Time Seconds Y')
    
     # Add a button to make predictions
    if st.button('Will Jellyfish and Humans be Present?'):
        # Use the model to make predictions
        # prediction = model.predict([[lat, lon, year, month, day, net_mesh, net_opening, time_seconds_x, time_seconds_y]])
        prediction = model.predict([[lat, lon, year, month, day]])
        
        # Display the prediction
        st.write('Are both_present:', prediction)
        if prediction == 1:
            st.write('Yes, both jellyfish and human presence are expected at this location.')
        else:
            st.write('No, either jellyfish or human presence is not expected at this location.')
if __name__ == '__main__':
    main()