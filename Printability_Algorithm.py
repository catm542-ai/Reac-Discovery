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

# 3. Data Scaling
# Normalize numerical data using StandardScaler.
scaler = StandardScaler()
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Apply One-Hot Encoding to the 'printable' categorical variable.
one_hot_encoder = OneHotEncoder(sparse=False)
printable_encoded = one_hot_encoder.fit_transform(df[['printable']])
printable_encoded_df = pd.DataFrame(printable_encoded, columns=one_hot_encoder.get_feature_names_out(['printable']))
df = df.drop('printable', axis=1).join(printable_encoded_df)

# 4. Feature Preparation
# Remove irrelevant features step-by-step.
data_features = df.drop(
    ['Nº', 'size', 'level', 'volume', 'specific_surface (cm-1)', 
     'geometry_name', 'teorical_weight', 'weight', '%error', 
     'printable_NO', 'printable_SI'], 
    axis=1
)

# Convert feature DataFrame to a NumPy array.
scaled_inputs_all = data_features.to_numpy()

# Prepare target matrix.
# Drop irrelevant columns to isolate the target variables.
data_targets = df.drop(
    ['Nº', 'size', 'level', 'volume', 'specific_surface (cm-1)', 
     'geometry_name', 'teorical_weight', 'weight', '%error', 
     'volume_fraction (%)', 'surface_area (sq mm)', 'tortuosity'], 
    axis=1
)
scaled_targets_all = data_targets.to_numpy()

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
np.savez('data/Printability_data_train', inputs=X_train, targets=y_train)
np.savez('data/Printability_data_validation', inputs=X_val, targets=y_val)

# Load the saved data.
train_data = np.load('data/Printability_data_train.npz')
X_train = train_data['inputs']
y_train = train_data['targets']

val_data = np.load('data/Printability_data_validation.npz')
X_val = val_data['inputs']
y_val = val_data['targets']

# 7. Model Construction
input_size = X_train.shape[1]  # Number of features
output_size = y_train.shape[1]  # Number of target variables
hidden_layer_size = 8  # Small size to avoid overfitting

# Define the neural network model.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Regularization to prevent overfitting
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(output_size, activation='sigmoid')  # Sigmoid for binary classification
])

# 8. Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',  # Loss function for binary classification
    metrics=['accuracy']  # Track accuracy during training
)

# 9. Training Configuration
NUM_EPOCHS = 500
BATCH_SIZE = 10  # Small batch size for small datasets
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


