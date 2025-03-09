import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import r2_score

# Define the folder containing datasets
data_folder = "E:\\3rd_Year_College_Project\\6th Sem\\Datasets"

# List all files in the folder and filter for .nc files
all_files = os.listdir(data_folder)
nc_files = [os.path.join(data_folder, file) for file in all_files if file.endswith('.nc')]

# Check if there are any .nc files in the folder
if not nc_files:
    raise FileNotFoundError("No .nc files found in the specified folder. Check the path or file extensions.")

# Load all datasets
xr_datasets = [xr.open_dataset(ds) for ds in nc_files]

# Combine the datasets along the time dimension
combined_dataset = xr.concat(xr_datasets, dim='time')

# Select the U10 component
u10_data = combined_dataset['u10']

# Calculate the spatial mean (averaged across latitude and longitude)
u10_mean = u10_data.mean(dim=['latitude', 'longitude'])

# Convert to numpy array and reshape
u10_values = u10_mean.values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
u10_scaled = scaler.fit_transform(u10_values)

# Create sequences for LSTM
look_back = 24  # Use past 24 hours to predict next 6 hours
forecast_horizon = 6

def create_sequences(data, look_back, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back:i + look_back + forecast_horizon])
    return np.array(X), np.array(y)

X, y = create_sequences(u10_scaled, look_back, forecast_horizon)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(50, activation='tanh', return_sequences=True, input_shape=(look_back, 1)),
    LSTM(50, activation='tanh'),
    Dense(forecast_horizon)
])

# Compile the model
learning_rate = 0.001
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()

# Make predictions
y_pred = model.predict(X_test)

# Rescale predictions back to original scale
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled[:100, 0], label='Actual', color='black')
plt.plot(y_pred_rescaled[:100, 0], label='Predicted', color='red', linestyle='dashed')
plt.xlabel('Time Steps')
plt.ylabel('U10 Wind Speed')
plt.title('Actual vs Predicted Wind Speed')
plt.legend()
plt.grid()
plt.show()


# Compute R² score
r2 = r2_score(y_test_rescaled.flatten(), y_pred_rescaled.flatten())
print(f"R² Score: {r2}")