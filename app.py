import streamlit as st
import pickle
from joblib import load

# Load the model
model = load('best_model.pkl')

# # # Load the model
# with open('best_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# Define the Streamlit app
def main():
    st.title('Predictive Models')
    
    # Add input options for user
    lat = st.number_input('Latitude')
    lon = st.number_input('Longitude')
    year = st.number_input('Year')
    month = st.number_input('Month')
    day = st.number_input('Day')
    both_present = st.number_input('Both Present')  # Add this line
    
    # Use the model to make predictions
    prediction = model.predict([[lat, lon, year, month, day, both_present]])  # Include both_present here
    
    # Display the prediction
    st.write('Latitude:', lat)
    st.write('Longitude:', lon)
    st.write('Year:', year)
    st.write('Month:', month)
    st.write('Day:', day)
    st.write('Both Present:', both_present)  # Add this line


if __name__ == '__main__':
    main()