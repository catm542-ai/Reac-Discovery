"""
Reac-Discovery: Machine Learning Model M2
@author: Cristopher Tinajero
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model 
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# 1. Load Data
# Load the dataset from an Excel file. Ensure the file is placed in the correct directory.
data_path = 'Data_set_M2.xlsx'
sheet_name = 'Hoja1'
df = pd.read_excel(data_path, sheet_name=sheet_name)

# 2. Descriptive Statistics
# Compute mean and standard deviation for each variable in the dataset
stats = df.describe()
means = stats.loc['mean']
stds = stats.loc['std']

# Geometry Descriptors
mean_Size = means['Size']
stds_Size = stds['Size']
mean_Level = means['Level']
stds_Level = stds['Level']

# Process Descriptors
mean_fr_gas = means['fr_gas']
stds_fr_gas = stds['fr_gas']
mean_fr_liq = means['fr_liq']
stds_fr_liq = stds['fr_liq']
mean_concentration = means['concentration_(M)']
stds_concentration = stds['concentration_(M)']
mean_temperature = means['temperature']
stds_temperature = stds['temperature']

mean_rT = means['r_toruosity']
stds_rT = stds['r_toruosity']
mean_SA = means['Surface_Area']
stds_SA = stds['Surface_Area']
mean_FV = means['Free_Volume']
stds_FV = stds['Free_Volume']
mean_PP = means['Perce_P']
stds_PP = stds['Perce_P']

# Target
mean_STY = means['STY']
stds_STY = stds['STY']

# 3. Data Scaling
# Standardize numerical columns using precomputed means and standard deviations
scaler = StandardScaler()
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Prepare feature and target matrices
# Drop 'cod_prueba' and 'STY' columns from the feature set
data_features = df.drop(['cod_prueba', 'STY'], axis=1)
data_array_features = data_features.to_numpy()
scaled_inputs_all = data_array_features

# Prepare target matrix
# Isolate the target variable 'STY'
data_targets = df[['STY']]
data_array_targets = data_targets.to_numpy()
scaled_targets_all = data_array_targets

# Define features (X) and targets (y)
X = scaled_inputs_all
y = scaled_targets_all

# 4. K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

# Save training and validation data for reproducibility
np.savez('AutoOptimization_data_train', inputs=X_train, targets=y_train)
np.savez('AutoOptimization_data_validation', inputs=X_val, targets=y_val)

# 5. Load Training and Validation Data
train_data = np.load('AutoOptimization_data_train.npz')
X_train, y_train = train_data['inputs'].astype(float), train_data['targets'].astype(float)
val_data = np.load('AutoOptimization_data_validation.npz')
X_val, y_val = val_data['inputs'].astype(float), val_data['targets'].astype(float)

# 6. Model Training
input_size = X_train.shape[1]  # Number of input features
output_size = 1  # Target is a single value
hidden_layer_size = 15  # Number of neurons in hidden layers

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size)
])

# Compile the model
model.compile(optimizer='Adam', loss='mean_squared_error')

# Training Configuration
NUM_EPOCHS = 500
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)

# Train the Model
history = model.fit(
    X_train, y_train,
    epochs=NUM_EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=2
)

# Save the trained model
model.save('modelo_M2')

# 7. Plot Training and Validation Loss
sns.set()
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], 'bo-', label='Training Loss')
plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], 'ro-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# 8. Multiple Configuration Testing of Reac-Gen (480)
# Load dataset for structure validation
data_path_MS = 'Multiple_Structure_Dataset.xlsx'
sheet_name_MS = 'Hoja1'
df_MS = pd.read_excel(data_path_MS, sheet_name=sheet_name_MS)

# Apply standardization using precomputed means and standard deviations
df_MS['Size'] = (df_MS['Size'] - mean_Size) / stds_Size
df_MS['Level'] = (df_MS['Level'] - mean_Level) / stds_Level

df_MS['fr_gas'] = (df_MS['fr_gas'] - mean_fr_gas) / stds_fr_gas
df_MS['fr_liq'] = (df_MS['fr_liq'] - mean_fr_liq) / stds_fr_liq
df_MS['concentration_(M)'] = (df_MS['concentration_(M)'] - mean_concentration) / stds_concentration
df_MS['temperature'] = (df_MS['temperature'] - mean_temperature) / stds_temperature

df_MS['r_toruosity'] = (df_MS['r_toruosity'] - mean_rT) / stds_rT
df_MS['Surface_Area'] = (df_MS['Surface_Area'] - mean_SA) / stds_SA
df_MS['Free_Volume'] = (df_MS['Free_Volume'] - mean_FV) / stds_FV
df_MS['Perce_P'] = (df_MS['Perce_P'] - mean_PP) / stds_PP

# Convert features to numpy array
data_array_features_MS = df_MS.to_numpy()
scaled_inputs_all_MS = data_array_features_MS

# Save standardized dataset
np.savez('Multi_Structure_Dataset', inputs=scaled_inputs_all_MS)
npz_MS = np.load('Multi_Structure_Dataset.npz')

# Convert dataset to float type
test_inputs_MS = npz_MS['inputs'].astype(float)

# Load trained model
model = load_model('modelo_M2')

# Prepare data for prediction
X_predict = test_inputs_MS  # Ensure format matches training data

# Run predictions
predictions = model.predict(X_predict)

# Convert predictions to DataFrame for better visualization
predictions_df = pd.DataFrame(predictions, columns=['STY'])

# Sort predictions from highest to lowest
predictions_df_sorted = predictions_df.sort_values(by='STY', ascending=False)

# Select top 480 configurations
top_480_predictions = predictions_df_sorted.head(480)

# Retrieve indices of the top 480 predictions
top_480_indices = top_480_predictions.index

# Extract the corresponding input values for these top predictions
top_480_X_predict = X_predict[top_480_indices]

# Create a DataFrame combining the inputs and predictions
top_480_X_predict_df = pd.DataFrame(
    top_480_X_predict, 
    columns=['Size', 'Level', 'fr_gas', 'fr_liq', 'concentration_(M)', 'temperature', 'r_toruosity', 'Surface_Area', 'Free_Volume', 'Perce_P']
)

# Merge inputs and predictions
top_480_combined = pd.concat([top_480_X_predict_df.reset_index(drop=True), top_480_predictions.reset_index(drop=True)], axis=1)

# 9. 3D Printability Validation
# Standardize initial values for printability validation
df_M2P = top_480_combined
df_M2P['Free_Volume'] = (df_M2P['Free_Volume'] - 39.0121) / 11.9696
df_M2P['Surface_Area'] = (df_M2P['Surface_Area'] - 2742.663) / 1720.279
df_M2P['r_toruosity'] = (df_M2P['r_toruosity'] - 7.9895) / 4.4124

# Prepare data for printability validation
data_targets_M2P = df_M2P.drop(columns=['Size', 'fr_gas', 'fr_liq', 'concentration_(M)', 'temperature', 'Level', 'Perce_P', 'STY'])

# Convert to numpy array
data_array_targets_M2P = data_targets_M2P.to_numpy()
scaled_targets_all_M2P = data_array_targets_M2P

# Save printability validation dataset
np.savez('M2_Prediction', inputs=scaled_targets_all_M2P)
npz_M2P = np.load('M2_Prediction.npz')

# Convert dataset to float type
test_inputs_M2P = npz_M2P['inputs'].astype(float)

# Load printability model
model_M2P = load_model('modelo_Printability')

# Prepare data for prediction
X_predict_M2P = test_inputs_M2P 

# Run predictions
predictions_M2P = model_M2P.predict(X_predict_M2P)  # Output shape (480,2)

# Convert predictions to binary classification (0 or 1)
predicted_classes_M2P = np.argmax(predictions_M2P, axis=1)  # Returns 0 or 1 per sample

# Map to labels "YES" or "NO"
predicted_labels_M2P = np.where(predicted_classes_M2P == 1, "YES", "NO")

# Create DataFrame with features and predicted printability results
df_resultado_M2P = pd.DataFrame(X_predict_M2P, columns=['r_toruosity', 'Surface_Area', 'Free_Volume','Perce_P'])
df_resultado_M2P['Printability'] = predicted_labels_M2P  # Add printability predictions

# Reverse standardization for better interpretation
df_resultado_M2P['Free_Volume'] = (df_resultado_M2P['Free_Volume'] * 11.9696) + 39.0121
df_resultado_M2P['Surface_Area'] = (df_resultado_M2P['Surface_Area'] * 1720.279) + 2742.663
df_resultado_M2P['r_toruosity'] = (df_resultado_M2P['r_toruosity'] * 4.4124) + 7.9895

# Export results to Excel
file_path_M2P = "Validated_results.xlsx"
df_resultado_M2P.to_excel(file_path_M2P, index=False)

print(f"File exported successfully: {file_path_M2P}")
