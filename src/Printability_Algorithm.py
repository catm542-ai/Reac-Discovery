"""
Printability Algorithm
@author: Cristopher Tinajero
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# 1. Load Data
# Load the dataset from an Excel file. Update 'data_path' with the actual file path.
data_path = 'data/Analisis_Estructuras_Imprimibilidad.xlsx'
sheet_name = 'Hoja1'  # Update with the correct sheet name if necessary.
df = pd.read_excel(data_path, sheet_name=sheet_name)

# 2. Descriptive Statistics
# Calculate mean and standard deviation for each variable.
stats = df.describe()
means = stats.loc['mean']
stds = stats.loc['std']

mean_Volume_F = means['volume_fraction (%)']
stds_Volume_F = stds['volume_fraction (%)']

mean_SA = means['surface_area (sq mm)']
stds_SA = stds['surface_area (sq mm)']

mean_tortuosity = means['tortuosity']
stds_tortuosity = stds['tortuosity']

mean_size = means['size']
stds_size = stds['size']

mean_level = means['level']
stds_level = stds['level']

mean_Min_h_D_mm = means['Min_h_D_mm']
stds_Min_h_D_mm = stds['Min_h_D_mm']

mean_Std_h_D_mm = means['Std_h_D_mm']
stds_Std_h_D_mm = stds['Std_h_D_mm']

mean_Min_E_N_D = means['Min_Equivalent_Neck_Diameter']
stds_Min_E_N_D = stds['Min_Equivalent_Neck_Diameter']

mean_Max_Angle = means['Max_Angle_Variation_rad']
stds_Max_Angle = stds['Max_Angle_Variation_rad']

mean_Euler = means['Euler_Characteristic']
stds_Euler = stds['Euler_Characteristic']

# 3. Data Scaling
# Normalize numerical data using StandardScaler.
scaler = StandardScaler()
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Apply One-Hot Encoding to the 'printable' categorical variable.
one_hot_encoder = OneHotEncoder(sparse=False)
solvent_encoded = one_hot_encoder.fit_transform(df[['printable']])
solvent_encoded_df = pd.DataFrame(solvent_encoded, columns=one_hot_encoder.get_feature_names_out(['printable']))
df = df.drop('printable', axis=1).join(solvent_encoded_df)

# 4. Feature Preparation
# Remove irrelevant features step-by-step.
data_features_1 = df.drop('Nº', axis=1)
data_features_4 = data_features_1.drop('volume', axis=1)
data_features_5 = data_features_4.drop('specific_surface (cm-1)', axis=1)
data_features_6 = data_features_5.drop('geometry_name', axis=1)
data_features_7 = data_features_6.drop('teorical_weight', axis=1)
data_features_8 = data_features_7.drop('weight', axis=1)
data_features_9 = data_features_8.drop('%error', axis=1)
data_features_10 = data_features_9.drop('printable_NO', axis=1)
data_features_11 = data_features_10.drop('printable_SI', axis=1)
data_features = data_features_11

# Convert feature DataFrame to a NumPy array.
data_array_features = data_features.to_numpy()
scaled_inputs_all = data_array_features

# Prepare target matrix.
# Drop irrelevant columns to isolate the target variables.
data_targets_1 = df.drop('Nº', axis=1)
data_targets_2 = data_targets_1.drop('size', axis=1)
data_targets_3 = data_targets_2.drop('level', axis=1)
data_targets_4 = data_targets_3.drop('volume', axis=1)
data_targets_5 = data_targets_4.drop('specific_surface (cm-1)', axis=1)
data_targets_6 = data_targets_5.drop('geometry_name', axis=1)
data_targets_7 = data_targets_6.drop('teorical_weight', axis=1)
data_targets_8 = data_targets_7.drop('weight', axis=1)
data_targets_9 = data_targets_8.drop('%error', axis=1)
data_targets_10 = data_targets_9.drop('volume_fraction (%)', axis=1)
data_targets_11 = data_targets_10.drop('surface_area (sq mm)', axis=1)
data_targets_12 = data_targets_11.drop('tortuosity', axis=1)
data_targets_13 = data_targets_12.drop('Min_h_D_mm', axis=1)
data_targets_14 = data_targets_13.drop('Std_h_D_mm', axis=1)
data_targets_15 = data_targets_14.drop('Min_Equivalent_Neck_Diameter', axis=1)
data_targets_16 = data_targets_15.drop('Max_Angle_Variation_rad', axis=1)
data_targets_17 = data_targets_16.drop('Euler_Characteristic', axis=1)
data_targets = data_targets_17

data_array_targets = data_targets.to_numpy()
scaled_targets_all = data_array_targets

# 5. Define Features (X) and Targets (y)
X = scaled_inputs_all
y = scaled_targets_all

# 6. K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Save data for each fold.
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

# Save training and validation datasets for reproducibility.
np.savez('Printability_data_trainV2', inputs=X_train, targets=y_train)
np.savez('Printability_data_validationV2', inputs=X_val, targets=y_val)

# Load the saved data.
train_data = np.load('Printability_data_trainV2.npz')
X_train = train_data['inputs']
y_train = train_data['targets']

val_data = np.load('Printability_data_validationV2.npz')
X_val = val_data['inputs']
y_val = val_data['targets']

# 7. Model Construction
input_size = X_train.shape[1]  # 3 features
output_size = y_train.shape[1]  # 1 salida binaria
hidden_layer_size = 10  # Pequeño para evitar overfitting

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Regularización
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(output_size, activation='sigmoid')  # Cambio a sigmoide para clasificación
])

# 8. Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',  # Función de pérdida para clasificación binaria
    metrics=['accuracy']  # Evaluación basada en precisión
)


# 9. Training Configuration
NUM_EPOCHS = 500
BATCH_SIZE = 10  # Pequeño para datasets pequeños
early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
# 10. Train the Model
history = model.fit(
    X_train, y_train,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=2
)

# Save the trained model.
model.save('models/modelo_Printability')


