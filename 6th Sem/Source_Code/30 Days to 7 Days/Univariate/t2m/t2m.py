import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# ----------------- CONFIG ----------------- #
DATA_DIR = "E:/3rd_Year_College_Project/6th Sem/Datasets"
MODEL_PATH = "lstm_teacher_forcing_updated.pth"

LOOKBACK = 30 * 24  # 30 days = 720 hours
HORIZON  = 7 * 24   # 7 days  = 168 hours
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
HIDDEN_SIZE = 256
NUM_LAYERS = 2      # Must be ≥2 for built-in LSTM dropout to apply
DROPOUT = 0.2
PATIENCE = 5
TEACHER_FORCING_RATIO = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== 1. LOAD NETCDF DATA (t2m) ==========
def load_netcdf_data(data_dir):
    """Load multiple NetCDF files (2013 to 2023), average lat/lon of 't2m'."""
    nc_files = [os.path.join(data_dir, f"{year}.nc") for year in range(2013, 2024)]
    if not nc_files:
        raise FileNotFoundError("No .nc files found in the specified folder.")

    ds_list = [xr.open_dataset(f) for f in nc_files]
    ds_merged = xr.concat(ds_list, dim='time')

    # Extract t2m, average lat/lon
    t2m_data = ds_merged["t2m"].mean(dim=["latitude", "longitude"]).values  # shape (N,)

    for ds in ds_list:
        ds.close()
    ds_merged.close()

    return t2m_data

# ========== 2. CREATE SLIDING WINDOWS ==========
def create_sequences(data, lookback, horizon):
    """Sliding window approach: for each hour, gather the previous `lookback` hours and next `horizon` hours."""
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon])
    return np.array(X), np.array(y)

# ========== 3. DATASET CLASS ==========
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========== 4. ENCODER-DECODER LSTM MODEL ==========
class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder with built-in dropout (applies between layers)
        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, y=None, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        """
        x: shape (batch_size, lookback)
        y: shape (batch_size, horizon)
        """
        batch_size = x.shape[0]

        # Reshape for LSTM
        x = x.unsqueeze(-1)  # (batch_size, lookback, 1)
        if y is not None:
            y = y.unsqueeze(-1)  # (batch_size, horizon, 1)

        # Encoder
        _, (h, c) = self.encoder_lstm(x)
        # h,c shape: (num_layers, batch_size, hidden_size)

        # Decoder
        dec_input = x[:, -1, :]  # (batch_size, 1)
        outputs = []

        horizon = y.size(1) if y is not None else HORIZON
        for t in range(horizon):
            dec_input = dec_input.unsqueeze(1)  # (batch_size,1,1)
            dec_out, (h, c) = self.decoder_lstm(dec_input, (h, c))
            # dec_out shape: (batch_size,1,hidden_size)
            pred = self.fc(dec_out.squeeze(1))  # (batch_size,1)
            outputs.append(pred)

            # Teacher forcing
            if y is not None and np.random.rand() < teacher_forcing_ratio:
                dec_input = y[:, t, :]
            else:
                dec_input = pred

        return torch.cat(outputs, dim=1)  # (batch_size, horizon)

# ========== 5. TRAIN FUNCTION ==========
def train_model(model, train_loader, val_loader, criterion, optimizer):
    best_val_loss = float("inf")
    early_stop_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

        for X_batch, y_batch in loop:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch, y_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_preds = model(X_val, teacher_forcing_ratio=0)
                loss_val = criterion(val_preds, y_val)
                val_loss += loss_val.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train={avg_train_loss:.5f}, Val={avg_val_loss:.5f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses

# ========== 6. EVALUATE FUNCTION ==========
def evaluate(model, X_test, y_test, scaler):
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    preds_list = []
    with torch.no_grad():
        for i in range(len(X_test_t)):
            x_inp = X_test_t[i].unsqueeze(0)
            pred = model(x_inp, teacher_forcing_ratio=0).cpu().numpy()
            preds_list.append(pred.flatten())

    preds_arr = np.concatenate(preds_list)
    actuals_arr = y_test.reshape(-1)

    preds_unscaled   = scaler.inverse_transform(preds_arr.reshape(-1,1)).flatten()
    actuals_unscaled = scaler.inverse_transform(actuals_arr.reshape(-1,1)).flatten()

    mse_val = mean_squared_error(actuals_unscaled, preds_unscaled)
    mae_val = mean_absolute_error(actuals_unscaled, preds_unscaled)
    r2_val  = r2_score(actuals_unscaled, preds_unscaled)
    return preds_unscaled, actuals_unscaled, mse_val, mae_val, r2_val

# ========== 7. MAIN SCRIPT ==========
def main():
    # 1. Load raw data (t2m)
    raw_data = load_netcdf_data(DATA_DIR)
    print(f"t2m data length: {len(raw_data)}")

    # 2. Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(raw_data.reshape(-1,1)).flatten()

    # 3. Create sequences
    X, Y = create_sequences(data_scaled, LOOKBACK, HORIZON)
    print(f"Total samples (sliding windows): {len(X)}")

    # 4. Split: 80% train, 10% val, 10% test
    total_len = len(X)
    train_len = int(0.8 * total_len)
    val_len   = int(0.1 * total_len)
    test_len  = total_len - train_len - val_len

    X_train, y_train = X[:train_len], Y[:train_len]
    X_val,   y_val   = X[train_len:train_len+val_len], Y[train_len:train_len+val_len]
    X_test,  y_test  = X[train_len+val_len:],         Y[train_len+val_len:]

    print(f"Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 5. Create DataLoaders
    train_ds = WeatherDataset(X_train, y_train)
    val_ds   = WeatherDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # 6. Define model
    model = EncoderDecoderLSTM(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. Train
    print("\n===== Training LSTM for t2m with Built-in Dropout =====")
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer)

    # Load best checkpoint
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("\n✅ Model loaded from best checkpoint.")

    # 8. Evaluate
    preds_unscaled, actuals_unscaled, mse_val, mae_val, r2_val = evaluate(model, X_test, y_test, scaler)
    print("\n===== Final Test Metrics =====")
    print(f"MSE: {mse_val:.5f}, MAE: {mae_val:.5f}, R²: {r2_val:.5f}")

    # 9. Plot Train vs Validation Loss
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses,   label="Val Loss",   color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Train vs Validation Loss (t2m Forecast)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 10. Plot last 168 hours
    num_hours = 168
    if len(preds_unscaled) < num_hours:
        print(f"Warning: <168 test predictions. Plotting all {len(preds_unscaled)} hours.")
        num_hours = len(preds_unscaled)

    actual_subset = actuals_unscaled[-num_hours:]
    pred_subset   = preds_unscaled[-num_hours:]

    plt.figure(figsize=(10,5))
    plt.plot(actual_subset, label="Actual (t2m)", color="black")
    plt.plot(pred_subset,   label="Predicted (t2m)", color="red", linestyle="--")
    plt.xlabel("Time Steps (Hours)")
    plt.ylabel("Temperature (K)")  # or your desired unit
    plt.title(f"Last 7 Days: Actual vs Predicted t2m\nMSE={mse_val:.5f}, MAE={mae_val:.5f}, R²={r2_val:.5f}")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
