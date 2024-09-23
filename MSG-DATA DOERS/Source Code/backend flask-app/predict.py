import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB

# Load and preprocess your dataset
df = pd.read_csv('yield_df.csv').drop(columns=['Unnamed: 0'], errors='ignore')

# Ensure no categorical values remain in the data for scaling
X = df.drop(columns=['hg/ha_yield', 'category'], errors='ignore')
y = df['hg/ha_yield']  # Target (Yield)

# Encode categorical variables (Area and Item)
label_encoder_area = LabelEncoder()
label_encoder_item = LabelEncoder()

X['Area'] = label_encoder_area.fit_transform(X['Area'])
X['Item'] = label_encoder_item.fit_transform(X['Item'])

# Ensure avg_temp is numeric
X['avg_temp'] = pd.to_numeric(X['avg_temp'], errors='coerce')
X = X.dropna(subset=['avg_temp'])  # Drop rows with missing 'avg_temp'

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
svm_model = SVR(kernel='rbf')
nb_model = GaussianNB()

rf_model.fit(X_train_scaled, y_train)
svm_model.fit(X_train_scaled, y_train)
nb_model.fit(X_train_scaled, y_train)

# Function to categorize predicted yield into 'Low', 'Medium', 'High'
def categorize_yield(yield_value, bins):
    if yield_value < bins[1]:
        return 'Low'
    elif yield_value < bins[2]:
        return 'Medium'
    else:
        return 'High'

# Helper function to handle unseen labels in new data
def handle_unseen_labels(data, encoder, column_name):
    for val in data[column_name]:
        if val not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, val)
    return encoder.transform(data[column_name])

# Decode function to convert encoded labels back to original strings
def decode_labels(encoded_value, encoder):
    return encoder.inverse_transform([encoded_value])[0]

# Prediction function
def predict_yield(area, item, year, avg_rainfall, pesticides, avg_temp):
    # Prepare the new data
    new_data = pd.DataFrame({
        'Area': [area],
        'Item': [item],
        'Year': [year],
        'average_rain_fall_mm_per_year': [avg_rainfall],
        'pesticides_tonnes': [pesticides],
        'avg_temp': [avg_temp]
    })

    # Handle unseen labels in 'Area' and 'Item'
    new_data['Area'] = handle_unseen_labels(new_data, label_encoder_area, 'Area')
    new_data['Item'] = handle_unseen_labels(new_data, label_encoder_item, 'Item')

    # Standardize the new data
    new_data_scaled = scaler.transform(new_data)

    # Make predictions with the models
    rf_prediction = rf_model.predict(new_data_scaled)[0].item()
    svm_prediction = svm_model.predict(new_data_scaled)[0].item()
    nb_prediction = nb_model.predict(new_data_scaled)[0].item()

    # Define bins for categorizing the yield predictions
    min_value = y.min()
    max_value = y.max()
    bins = [min_value, (min_value + max_value) / 3, 2 * (min_value + max_value) / 3, max_value]

    # Categorize the predictions
    rf_category = categorize_yield(rf_prediction, bins)
    svm_category = categorize_yield(svm_prediction, bins)
    nb_category = categorize_yield(nb_prediction, bins)

    # Decode the labels back to original strings
    decoded_area = decode_labels(new_data['Area'][0], label_encoder_area)
    decoded_item = decode_labels(new_data['Item'][0], label_encoder_item)

    # Load the existing CSV to get the last index
    try:
        df = pd.read_csv('yield_df.csv')
        last_index = df.index[-1]  # Get the last index number
    except (FileNotFoundError, IndexError):
        last_index = -1  # Start from -1 if the file does not exist or is empty

    # Prepare the new row to save
    new_row = {
        '': [last_index + 1],
        'Area': [decoded_area],
        'Item': [decoded_item],
        'Year': [year],
        'hg/ha_yield': [int(rf_prediction)],
        'average_rain_fall_mm_per_year': [avg_rainfall],
        'pesticides_tonnes': [pesticides],
        'avg_temp': [avg_temp],
        'category': [rf_category]
    }

    # Append the new row to the original CSV
    new_data_df = pd.DataFrame(new_row)
    new_data_df.to_csv('yield_df.csv', mode='a', header=False, index=False)

    # Return predictions and categories for all models
    return {
        'Random Forest': {'yield': int(rf_prediction), 'category': rf_category},
        'SVM': {'yield': int(svm_prediction), 'category': svm_category},
        'Naive Bayes': {'yield': int(nb_prediction), 'category': nb_category}
    }

