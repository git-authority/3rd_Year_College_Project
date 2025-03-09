import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset folder
data_folder = "E:\\3rd_Year_College_Project\\6th Sem\\Datasets"

# Load all .nc files
nc_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.nc')]
if not nc_files:
    raise FileNotFoundError("No .nc files found!")

# Load and combine NetCDF files
xr_datasets = [xr.open_dataset(ds) for ds in nc_files]
combined_dataset = xr.concat(xr_datasets, dim='time')

# Select U10 component (wind speed) and compute spatial mean
u10_mean = combined_dataset['u10'].mean(dim=['latitude', 'longitude']).values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
u10_scaled = scaler.fit_transform(u10_mean)

# Define sequence parameters
look_back = 30  # Input: past 30 days
forecast_horizon = 7  # Output: next 7 days

# Create sequences for training
def create_sequences(data, look_back, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back:i + look_back + forecast_horizon])
    return np.array(X), np.array(y)

X, y = create_sequences(u10_scaled, look_back, forecast_horizon)

# Train-test split (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Define LSTM Encoder-Decoder Model
class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output 1 value per timestep

    def forward(self, x):
        batch_size = x.size(0)
        h, c = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

        _, (h, c) = self.encoder_lstm(x, (h, c))  # Encode input sequence

        decoder_input = torch.zeros(batch_size, 1, 1).to(device)  # Start with zero tensor

        outputs = []
        for _ in range(forecast_horizon):
            decoder_output, (h, c) = self.decoder_lstm(decoder_input, (h, c))
            prediction = self.fc(decoder_output).squeeze(-1)  # Ensure shape (batch_size, 1)
            outputs.append(prediction)
            decoder_input = prediction.unsqueeze(-1)  # Make it (batch_size, 1, 1)

        return torch.cat(outputs, dim=1).unsqueeze(-1)  # Final shape (batch_size, 7, 1)

# Initialize model
input_size = 1
hidden_size = 64
num_layers = 2
model = EncoderDecoderLSTM(input_size, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

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

        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    return train_losses, val_losses

# Train the model
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)

# Load best model
model.load_state_dict(torch.load("best_model.pth"))

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.grid()
plt.show()

# Make predictions
model.eval()
with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()

# Rescale to original values
y_test_rescaled = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).reshape(y_test.shape)
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
print(f"R² Score: {r2:.4f}")
