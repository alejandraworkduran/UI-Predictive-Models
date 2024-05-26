import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# default_data_path= '/content/drive/MyDrive/JeDI.csv'
default_data_path = "C:\\Users\\Ale\\OneDrive\\Desktop\\CAPSTONE\\JeDI.csv"

data= pd.read_csv(default_data_path)



data.info()

data.head()

# Assuming your data is in a DataFrame called data_dropped_columns
distinct_count = data['rank_species'].nunique()
print("Distinct count:", distinct_count)

data_dropped_columns = data.drop(columns=['weight_wet','weight_dry','project_title', 'sub_project_title', 'owner_dataset', 'contact', 'date', 'study_type','location_name', 'accompanying_ancillary_data','catch_per_effort','taxon','rank_phylum','rank_genus','rank_class','rank_order','rank_family','data_type','collection_method','depth_upper','depth_lower','density_integrated','biovolume_integrated'])

data_dropped_columns.head()

# # Display distinct values of the 'time_local' column
# distinct_times = data_dropped_columns['rank_species'].unique()

# # Print the distinct values
# print(distinct_times)

# Replace 'nd' with 'Unknown' in the 'rank_species' column
data_dropped_columns['rank_species'] = data_dropped_columns['rank_species'].replace('nd', 'Unknown')

# Convert the predictor column to categorical data type (if not already)
data_dropped_columns['presence_absence'] = data_dropped_columns['presence_absence'].astype('category')


data_dropped_columns.head()

edited_columns = ['net_opening', 'net_mesh', 'depth', 'count_actual', 'density', 'biovolume']

# Loop through each selected column
for col in edited_columns:
    # Convert 'nd' to NaN (missing values) in the current column
    data_dropped_columns[col] = data_dropped_columns[col].replace('nd', pd.NA)

    # Convert the column to numeric (optional, in case it's not already numeric)
    data_dropped_columns[col] = pd.to_numeric(data_dropped_columns[col], errors='coerce')

    # Calculate the mean excluding missing values for the current column
    mean_value = data_dropped_columns[col].mean()

    # Replace missing values with the calculated mean for the current column
    data_dropped_columns[col].fillna(mean_value, inplace=True)


data_dropped_columns.head()

# Display distinct values of the 'time_local' column
distinct_times = data_dropped_columns['time_local'].unique()

# Print the distinct values
print(distinct_times)

# Columns to process
date_columns = ['year', 'month', 'day']
time_columns = ['time_local']

# Replace 'nd' with NaN in the date and time columns
data_dropped_columns[date_columns] = data_dropped_columns[date_columns].replace('nd', pd.NaT)
data_dropped_columns[time_columns] = data_dropped_columns[time_columns].replace('nd', pd.NaT)

# Impute missing values for date columns using the mode
for col in date_columns:
    most_frequent_date = data_dropped_columns[col].mode()[0]
    data_dropped_columns[col].fillna(most_frequent_date, inplace=True)

# Replace 'nd' with NaN in the 'time_local' column
data_dropped_columns['time_local'] = data_dropped_columns['time_local'].replace('nd', pd.NaT)

# Calculate the median time
median_time = pd.to_timedelta(data_dropped_columns['time_local']).median()

# Fill missing values in the 'time_local' column with the median time
data_dropped_columns['time_local'].fillna(median_time, inplace=True)

# Now, the 'time_local' column should have missing values imputed with the median time.

data_dropped_columns['time_local'] = data_dropped_columns['time_local'].apply(lambda x: str(x).split(' ')[-1])

data_dropped_columns.head()

#convert data types for use in later models

# Convert data type on month and day
data_dropped_columns['month'] = data_dropped_columns['month'].astype(int)
data_dropped_columns['day'] = data_dropped_columns['day'].astype(int)

# Convert 'time_local' column to datetime format
data_dropped_columns['time_local'] = pd.to_datetime(data_dropped_columns['time_local'])

# Calculate the number of seconds elapsed since midnight for each time value
data_dropped_columns['time_seconds'] = data_dropped_columns['time_local'].dt.hour * 3600 + \
                                       data_dropped_columns['time_local'].dt.minute * 60 + \
                                       data_dropped_columns['time_local'].dt.second

# Now, the 'time_seconds' column contains the numeric representation of time values

#  'time_local' is the column you want to drop since we created new column time_seconds
data_dropped_columns.drop(columns=['time_local'], inplace=True)


data_dropped_columns.head()

data_dropped_columns.info()

# Insert to output file
output_file_path = "C:\\Users\\Ale\\OneDrive\\Desktop\\CAPSTONE\\output.csv"
data_dropped_columns.to_csv(output_file_path, index=False)

# Create a new DataFrame to store the encoded data
data_encoded = data_dropped_columns.copy()

# Create a LabelEncoder object
le = LabelEncoder()

# Fit the LabelEncoder to the categorical variable
le.fit(data_dropped_columns['rank_species'])

# Transform the categorical variable to numeric values
data_encoded['rank_species'] = le.transform(data_dropped_columns['rank_species'])

# Encode the target variable 'presence_absence'
data_encoded['presence_absence'] = data_dropped_columns['presence_absence'].map({'present': 1, 'absent': 0})

# Convert the target variable to a categorical variable
data_encoded['presence_absence'] = data_encoded['presence_absence'].astype('category')

#after this cell exectuion . i was able to confirm all columns are finally in data types appropriate for use in models
data_encoded.info()

data_encoded.head()

data_encoded.to_csv('data_encoded.csv', index=False)

# Get the unique categories (species names)
species_names = le.classes_

# Print the species names and their corresponding indices
for index, species_name in enumerate(species_names):
    print(f"Index: {index}, Species name: {species_name}")

#after this you can you data_encoded to make your own x and y

# **Presence Prediction**

# Species Presence Prediction

# Split the data into training and testing sets
X = data_encoded.drop('presence_absence', axis=1)
y = data_encoded['presence_absence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Presence LR Classification Report:')
print(report)

from sklearn.ensemble import RandomForestClassifier
# tried a different model
# random forest classifier

# Create a random forest classifier
model = RandomForestClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Presence RF Classification Report:')
print(report)

# **Biovolume Estimation**

## **Biovolume Estimation**


# # Split the data into training and testing sets
# X2 = data_encoded.drop('biovolume', axis=1)
# y2 = data_encoded['biovolume']
# X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# # Rename the variable that overwrites the r2_score function
# my_r2_score = r2_score

# # Train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# print("Biovolume Linear Regression Model")
# r2_score_ = my_r2_score(y_test, y_pred)
# print(f'R2 Score: {r2_score_}')

# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean squared error: {mse}')

# #TRY TO IMPROVE MODEL

# number_of_features = X_train.shape[1]

# print(f"Number of features: {number_of_features}")

#feature importance
#so we can limit model

import statsmodels.api as sm

# # Fit a linear regression model to the data
# model = sm.OLS(y_train, X_train).fit()

# # Extract the coefficients from the model
# coefficients = model.params

# # Calculate the R^2 for each feature
# r_squared_values = {}
# for feature in X_train.columns:
#     coefficient = coefficients[feature]
#     standard_error = model.bse[feature]
#     r_squared = (coefficient**2) / (coefficient**2 + standard_error**2)
#     r_squared_values[feature] = r_squared

# # Rank the features in order of importance
# ranked_features = sorted(r_squared_values, key=r_squared_values.get, reverse=True)

# # Print the ranked features
# for feature in ranked_features:
#     print(f"{feature}: {r_squared_values[feature]}")

# # Split the data into training and testing sets
# X2 = data_encoded.drop('biovolume', axis=1)
# y2 = data_encoded['biovolume']
# X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# # Rename the variable that overwrites the r2_score function
# my_r2_score = r2_score

# # Train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = model.predict(X_test)

# print("Biovolume Linear Regression Model Limited Features???")

# # Evaluate the model
# r2_score_ = my_r2_score(y_test, y_pred)
# print(f'R2 Score: {r2_score_}')

# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean squared error: {mse}')

#TRY TO IMPROVE MODEL

# number_of_features = X_train.shape[1]

# print(f"Number of features: {number_of_features}")

# #feature importance
# #so we can limit model

# import statsmodels.api as sm

# # Fit a linear regression model to the data
# model = sm.OLS(y_train, X_train).fit()

# # Extract the coefficients from the model
# coefficients = model.params

# # Calculate the R^2 for each feature
# r_squared_values = {}
# for feature in X_train.columns:
#     coefficient = coefficients[feature]
#     standard_error = model.bse[feature]
#     r_squared = (coefficient**2) / (coefficient**2 + standard_error**2)
#     r_squared_values[feature] = r_squared

# # Rank the features in order of importance
# ranked_features = sorted(r_squared_values, key=r_squared_values.get, reverse=True)

# # Print the ranked features
# for feature in ranked_features:
#     print(f"{feature}: {r_squared_values[feature]}")

#New model using new selected 'imporatant' features

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data_encoded2 = data_encoded.copy()

# Convert the biovolume column to a binary variable
data_encoded2['biovolume'] = (data_encoded2['biovolume'] > 0).astype(int)

# Prepare the data
# X_important = data_encoded2[['year', 'presence_absence', 'time_seconds', 'net_mesh', 'net_opening', 'lon', 'rank_species', 'day']]
X_important = data_encoded2[['year', 'month', 'lat', 'lon', 'rank_species', 'day']]
y = data_encoded2[['biovolume']]

X_train, X_test, y_train, y_test = train_test_split(X_important, y, test_size=0.20, random_state=42)


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

# Set parameters for XGBoost
params = {
    'objective': 'reg:squarederror',  # Regression
    'max_depth': 5,
    'eta': 0.1,
}

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=100)

# Predict on test data
y_pred = bst.predict(dtest)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Biovolume XGB Model")
print(f"Mean Squared Error: {mse}")

# Optional: Save model
# bst.save_model('xgb_model.json')

# Optional: Load model
# bst = xgb.Booster()
# bst.load_model('xgb_model.json')

# # **SPECIES CLASSIFICATION**

# # SPECIES CLASSIFICATION

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score

# # Separating features and target variable
# X = data_encoded.drop(columns='rank_species')
# y = data_encoded['rank_species']

# # Normalizing numerical features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# X_train.shape, X_test.shape, y_train.shape, y_test.shape

# # Training the Random Forest classifier
# rf_classifier = RandomForestClassifier(random_state=1)
# rf_classifier.fit(X_train, y_train)

# # Predicting on the test set
# y_pred = rf_classifier.predict(X_test)

# # Evaluating the model
# accuracy = accuracy_score(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)

# print(accuracy)
# print( classification_rep)

#need new model
#try to improve accuracy

# from xgboost import XGBClassifier

# # Training the XGBoost classifier

# xgb_classifier = XGBClassifier(random_state=1)
# xgb_classifier.fit(X_train, y_train)

# # Predicting on the test set
# y_pred = xgb_classifier.predict(X_test)

# # Evaluating the model
# accuracy = accuracy_score(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)

# print(f'Accuracy: {accuracy}')
# # print('Classification Report:')
# # print(classification_rep)



# BRING IN HUMAN DATA**

# default_data_path= '/content/drive/MyDrive/JeDI.csv'
default_data_path2 = r"C:\Users\Ale\OneDrive\Desktop\CAPSTONE\data_encoded_human.csv"

data_human= pd.read_csv(default_data_path2)

data_human.info()

data_human.head()

# Drop the specified columns
columns_to_drop = ['age', 'height', 'weight', 'gender', 'occupation', 'income', 'education_level']
human_data = data_human.drop(columns=columns_to_drop)

human_data.head()

# count rows in df
num_rows = len(human_data)
print("Number of rows:", num_rows)

# **JOINT DATA**

import pandas as pd
import numpy as np

# Sample data - replace with your actual data
# data_encoded = ...  # Your encoded jellyfish DataFrame
# human_data = ...    # Your human DataFrame

# Convert 'presence_absence' to numerical values if it is categorical
data_encoded['presence_absence'] = data_encoded['presence_absence'].astype(int)

# Aggregate jellyfish data by date and location
jellyfish_agg = data_encoded.groupby(['year', 'month', 'day', 'lat', 'lon']).agg({
    'presence_absence': lambda x: int(any(x)),  # Convert counts to binary indicators
    'net_mesh': 'mean',  # Example: You can aggregate other columns if needed
    'net_opening': 'mean',
    'count_actual': 'sum',  # Summing up actual counts of jellyfish
    'density': 'mean',
    'biovolume': 'mean',
    'time_seconds': 'mean'
}).reset_index()

# Aggregate human data by date and location
human_agg = human_data.groupby(['year', 'month', 'day', 'lat', 'lon']).agg({
    'count_actual': 'sum',  # Summing up actual counts of humans
    'time_seconds': 'mean'
}).reset_index()

# Merge the aggregated data on date and location
merged_data = pd.merge(jellyfish_agg, human_agg, on=['year', 'month', 'day', 'lat', 'lon'], how='inner')

# Create a target variable indicating both are present
merged_data['both_present'] = np.where((merged_data['presence_absence'] > 0) & (merged_data['count_actual_x'] > 0), 1, 0)

# Rename columns for clarity
merged_data.rename(columns={'count_actual_x': 'count_actual_jellyfish', 'count_actual_y': 'count_actual_humans'}, inplace=True)

# Display the first few rows of the merged data
print(merged_data.head())


# Verify unique values
print("Unique values in presence_absence:", merged_data['presence_absence'].unique())
print("Unique values in both_present:", merged_data['both_present'].unique())

merged_data.head()

print(merged_data['both_present'].value_counts())

print(merged_data['presence_absence'].value_counts())

# Insert to output file
merged_data_output_file_path = "C:\\Users\\Ale\\OneDrive\\Desktop\\CAPSTONE\\data_merged_human_jellyfish.csv"
merged_data.to_csv(merged_data_output_file_path, index=False)

# **RISK ANALYSIS  - HUMAN & JELLYFISH PRESENCE**

# Features and target variable
# X = merged_data[['year', 'month', 'day', 'lat', 'lon', 'net_mesh', 'net_opening', 'time_seconds_x', 'time_seconds_y']]
X = merged_data[['year', 'month', 'day', 'lat', 'lon']]
y = merged_data['both_present']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Evaluate the model
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

#IMPROVE MODEL PERFORMANCE

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint

# Define the Gradient Boosting classifier
gradient_boosting = GradientBoostingClassifier()

# Define the hyperparameters and their distributions to search
param_dist = {
    'n_estimators': randint(100, 1000),      # Number of trees in the ensemble
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Learning rate (shrinkage parameter)
    'max_depth': randint(3, 10),              # Maximum depth of the trees
    'min_samples_split': randint(2, 10),      # Minimum number of samples required to split an internal node
    'min_samples_leaf': randint(1, 5)         # Minimum number of samples required to be at a leaf node
}

# Perform random search cross-validation
random_search = RandomizedSearchCV(estimator=gradient_boosting, param_distributions=param_dist, n_iter=10, cv=2, scoring='accuracy', n_jobs=-1)
random_search.fit(X_train, y_train)  # X_train and y_train are your training data and labels

# Get the best hyperparameters and the best model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Evaluate the best model on the test data
y_pred = best_model.predict(X_test)  # X_test is your test data
accuracy = accuracy_score(y_test, y_pred)  # y_test is your test labels
print("Best hyperparameters:", best_params)
print("Gradient Boosting Test accuracy:", accuracy)

# Print the classification report
print(classification_report(y_test, y_pred))

import sklearn
print(sklearn.__version__)

# Specify the latitude and longitude of the location you want to check
target_lat = 53.95
target_lon = -166.52

# Filter the merged data for the specified location
location_data = merged_data[(merged_data['lat'] == target_lat) & (merged_data['lon'] == target_lon)]

# Display the filtered data
print(location_data)

# **SAVE PREDICTIVE MODELS FOR LATER USE**
import joblib
import os

# Specify the directory to save the models
save_dir = 'C:/Users/Ale/source/repos/UI-Predictive-Models/'

# Save the model to a file
joblib.dump(best_model, os.path.join(save_dir, 'best_model.pkl'))

# Save XGBoost biovolume prediction model
joblib.dump(bst, os.path.join(save_dir, 'xgb_model.pkl'))

# Save the presence prediction model to a file
joblib.dump(model, os.path.join(save_dir, 'random_forest_model.pkl'))

print("Models saved successfully.")



print("the end")