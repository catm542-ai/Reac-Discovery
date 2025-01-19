"""
Reac-Discovery: Machine Learning Model M1
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
# Replace 'Data_set_M1.xlsx' with the path to your data file.
# Replace 'Hoja1' with the sheet name containing your data.
data_path = 'Data_set_M1.xlsx'
sheet_name = 'Hoja1'
df = pd.read_excel(data_path, sheet_name=sheet_name)

# 2. Descriptive Statistics
# Calculate mean and standard deviation for each variable in the dataset
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

mean_yield = means['yield']
stds_yield = stds['yield']

mean_STY = means['STY']
stds_STY = stds['STY']

# 3. Data Scaling
# Standardize numerical columns using StandardScaler
scaler = StandardScaler()
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Prepare feature and target matrices
# Drop 'cod_prueba' and 'yield and STY' for features
data_features_wo_cod_prueba = df.drop('cod_prueba', axis=1)
data_features_yield = data_features_wo_cod_prueba.drop('yield', axis=1)
data_features = data_features_yield.drop('sty', axis=1)

# Convert features to numpy array
data_array_features = data_features.to_numpy()
scaled_inputs_all = data_array_features

# Prepare target matrix
# Drop all feature-related columns to isolate target 'yield and sty'
data_targets_wo_cod_prueba = df.drop('cod_prueba', axis=1)
data_targets1 = data_targets_wo_cod_prueba.drop('Size', axis=1)
data_targets2 = data_targets1.drop('fr_gas', axis=1)
data_targets3 = data_targets2.drop('fr_liq', axis=1)
data_targets4 = data_targets3.drop('concentration_(M)', axis=1)
data_targets5 = data_targets4.drop('temperature', axis=1)
data_targets = data_targets5.drop('Level', axis=1)

# Convert targets to numpy array
data_array_targets = data_targets.to_numpy()
scaled_targets_all = data_array_targets

# Define features (X) and targets (y)
X = scaled_inputs_all
y = scaled_targets_all

# 4. K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over each fold for splitting data
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

# Save training and validation data
np.savez('AutoOptimization_data_train', inputs=X_train, targets=y_train)
np.savez('AutoOptimization_data_validation', inputs=X_val, targets=y_val)

# 5. Load Training and Validation Data
npz = np.load('AutoOptimization_data_train.npz')
train_inputs = npz['inputs'].astype(float)
train_targets = npz['targets'].astype(float)

npz = np.load('AutoOptimization_data_validation.npz')
validation_inputs = npz['inputs'].astype(float)
validation_targets = npz['targets'].astype(float)

# 6. Model Training
input_size = 6
output_size = 2
hidden_layer_size = 15

# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size)
])

# Compile model
model.compile(optimizer='Adam', loss='mean_squared_error')

# Define training parameters
NUM_EPOCHS = 500
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=NUM_EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=2
)

# Save trained model
model.save('modelo_M1')

# 7. Plot Training and Validation Loss
sns.set()
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

plt.figure(figsize=(12, 8))
plt.plot(range(1, len(training_loss) + 1), training_loss, 'bo-', label='Training Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 8. Generate All Combinations for Prediction
def standardize_values(values, mean, std):
    return [(x - mean) / std for x in values]

def generate_all_combinations():
    Size_range = np.arange(10, 23, 1)
    Level_range = np.arange(250, 601, 5)
    fr_gas_range = np.arange(50, 501, 10)
    fr_liq_range = np.arange(50, 301, 10)
    concentration_range = np.arange(0,1, 13.1, 0.5)
    temperature_range = np.arange(25, 121, 10)

    # Standardize ranges
    Size_std = standardize_values(Size_range, mean_Size, stds_Size)
    Level_std = standardize_values(Level_range, mean_Level, stds_Level)
    fr_gas_std = standardize_values(fr_gas_range, mean_fr_gas, stds_fr_gas)
    fr_liq_std = standardize_values(fr_liq_range, mean_fr_liq, stds_fr_liq)
    concentration_std = standardize_values(concentration_range, mean_concentration, stds_concentration)
    temperature_std = standardize_values(temperature_range, mean_temperature, stds_temperature)

    # Generate combinations
    base_combinations = itertools.product(
        Size_std, Level_std, fr_gas_std, fr_liq_std, concentration_std, temperature_std
    )

    # Split combinations into 4 parts
    all_combinations = [list(comb) for comb in base_combinations]
    part_size = len(all_combinations) // 4
    return (
        all_combinations[:part_size],
        all_combinations[part_size:2 * part_size],
        all_combinations[2 * part_size:3 * part_size],
        all_combinations[3 * part_size:]
    )

all_combinations1, all_combinations2, all_combinations3, all_combinations4 = generate_all_combinations()

# Save combinations
np.savez('AutoOptimization_data_test1', inputs=all_combinations1)
np.savez('AutoOptimization_data_test2', inputs=all_combinations2)
np.savez('AutoOptimization_data_test3', inputs=all_combinations3)
np.savez('AutoOptimization_data_test4', inputs=all_combinations4)

npz1 = np.load('AutoOptimization_data_test1.npz')
npz2 = np.load('AutoOptimization_data_test2.npz')
npz3 = np.load('AutoOptimization_data_test3.npz')
npz4 = np.load('AutoOptimization_data_test4.npz')

# Usar float o np.float64
test_inputs1 = npz1['inputs'].astype(float)  # or np.float64 
test_inputs2 = npz2['inputs'].astype(float)  # or np.float64 
test_inputs3 = npz3['inputs'].astype(float)  # or np.float64 
test_inputs4 = npz4['inputs'].astype(float)  # or np.float64 

# 9. Load and Predict / This process will be repeated for all four data blocks (test_inputs1, test_inputs2, test_inputs3, test_inputs4)

# Load the trained model
model = load_model('modelo_M1')

# Prepare data for prediction
# Ensure that test_inputs1 has the same format as the training data
X_predict1 = test_inputs1

# Perform predictions
predicciones = model.predict(X_predict1)

# Convert predictions to a DataFrame for better visualization
predicciones_df = pd.DataFrame(predicciones, columns=['yield']) #yield or STY

# Sort predictions in descending order
predicciones_df_sorted = predicciones_df.sort_values(by='yield', ascending=False) #yield or STY

# Select the top 20 predictions
top_20_predicciones = predicciones_df_sorted.head(20)

# Get the indices of the top 20 predictions
top_20_indices = top_20_predicciones.index

# Retrieve the corresponding input values for the top predictions
top_20_X_predict = X_predict1[top_20_indices]

# Create a DataFrame with the input values that generated the top predictions
top_20_X_predict_df = pd.DataFrame(
    top_20_X_predict,
    columns=['Size', 'Level', 'fr_gas', 'fr_liq', 'concentration', 'temperature']
)

# Combine the input values and their predictions into a single DataFrame
top_20_combined = pd.concat(
    [top_20_X_predict_df.reset_index(drop=True), top_20_predicciones.reset_index(drop=True)],
    axis=1
)

# Display the combined DataFrame
print(top_20_combined)

# Re-standardize values
# Adjust the top 20 combined DataFrame back to their original scale using the stored mean and standard deviation.

# Create a copy of the DataFrame to apply re-standardization
standardize_values = top_20_combined.copy()

# Re-standardize the 'Size' column
standardize_values['Size'] *= stds_Size
standardize_values['Size'] += mean_Size

# Re-standardize the 'Level' column
standardize_values['Level'] *= stds_Level
standardize_values['Level'] += mean_Level

# Re-standardize the 'fr_gas' column
standardize_values['fr_gas'] *= stds_fr_gas
standardize_values['fr_gas'] += mean_fr_gas

# Re-standardize the 'fr_liq' column
standardize_values['fr_liq'] *= stds_fr_liq
standardize_values['fr_liq'] += mean_fr_liq

# Re-standardize the 'temperature' column
standardize_values['temperature'] *= stds_temperature
standardize_values['temperature'] += mean_temperature

# Re-standardize the 'concentration' column
standardize_values['concentration'] *= stds_concentration
standardize_values['concentration'] += mean_concentration

# Re-standardize the 'yield' column
standardize_values['yield'] *= stds_yield
standardize_values['yield'] += mean_yield

# Re-standardize the 'yield' column
standardize_values['STY'] *= stds_STY
standardize_values['STY'] += mean_STY

# Print the re-standardized DataFrame
print(standardize_values)