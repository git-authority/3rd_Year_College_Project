import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tqdm import tqdm  # For progress bar

# Define dataset path
data_folder = "E:\\3rd_Year_College_Project\\6th Sem\\Datasets"

# Load all .nc files
all_files = os.listdir(data_folder)
nc_files = [os.path.join(data_folder, file) for file in all_files if file.endswith('.nc')]

if not nc_files:
    raise FileNotFoundError("No .nc files found in the specified folder.")

# Load and merge datasets
xr_datasets = [xr.open_dataset(ds) for ds in nc_files]
combined_dataset = xr.concat(xr_datasets, dim='time')

# Select U10 component and compute mean across lat/lon
u10_data = combined_dataset['u10'].mean(dim=['latitude', 'longitude']).values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
u10_scaled = scaler.fit_transform(u10_data)

# Sequence parameters
look_back = 30 * 24  # 30 days input
forecast_horizon = 7 * 24  # 7 days output

def create_sequences(data, look_back, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back:i + look_back + forecast_horizon])
    return np.array(X), np.array(y)

X, y = create_sequences(u10_scaled, look_back, forecast_horizon)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# PyTorch Dataset and DataLoader
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
train_loader = DataLoader(WeatherDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(WeatherDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# **Encoder-Decoder LSTM Model**
class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(EncoderDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Decoder
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]

        # Encoder
        _, (hidden, cell) = self.encoder_lstm(x)

        # Decoder
        decoder_input = torch.zeros(batch_size, 1, self.hidden_size).to(x.device)
        decoder_outputs = []

        for _ in range(forecast_horizon):
            out, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            prediction = self.fc(out.squeeze(1))
            decoder_outputs.append(prediction.unsqueeze(1))
            decoder_input = out  # Feeding the output back as input

        return torch.cat(decoder_outputs, dim=1)

# Initialize Model
input_size = 1
hidden_size = 64
output_size = 1
num_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderDecoderLSTM(input_size, hidden_size, output_size, num_layers).to(device)

# Define Loss, Optimizer & Early Stopping
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
early_stopping_patience = 5  # Stop training if val loss doesn't improve for 5 epochs

# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=early_stopping_patience):
    best_val_loss = float('inf')
    patience_counter = 0

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(train_loss=loss.item())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses

# Train Model
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)

# Plot Train vs Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss", color='blue')
plt.plot(val_losses, label="Validation Loss", color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid()
plt.show()

# Evaluate Model
model.eval()
y_pred = []
with torch.no_grad():
    for inputs, _ in val_loader:
        inputs = inputs.to(device)
        y_pred.append(model(inputs).cpu().numpy())

y_pred = np.vstack(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

# Plot Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled[:100, 0], label='Actual', color='black')
plt.plot(y_pred_rescaled[:100, 0], label='Predicted', color='red', linestyle='dashed')
plt.xlabel("Time Steps")
plt.ylabel("U10 Wind Speed")
plt.title("Actual vs Predicted Wind Speed")
plt.legend()
plt.grid()
plt.show()

# Compute R² score
r2 = r2_score(y_test_rescaled.flatten(), y_pred_rescaled.flatten())
print(f"R² Score: {r2:.6f}")
