import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------
# 1) CONFIG & HYPERPARAMETERS
# ----------------------------
DATA_DIR = "E:/3rd_Year_College_Project/6th Sem/Datasets"
LOOKBACK = 24   # 24-hour input sequence
HORIZON  = 6    # 6-hour forecast
HIDDEN_SIZE = 192
NUM_LAYERS  = 2
DROPOUT     = 0.3
LEARNING_RATE = 1e-4
EPOCHS        = 50
BATCH_SIZE    = 64
PATIENCE      = 5  # Early-stopping patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ----------------------------
# 2) LOAD & PREPROCESS DATA
# ----------------------------
def load_u10_data(directory):
    """
    Loads multiple .nc files from 2013–2023, extracts 'u10',
    averages lat/lon, concatenates into one array.
    """
    all_data = []
    for file in sorted(os.listdir(directory)):
        if file.endswith(".nc"):
            fpath = os.path.join(directory, file)
            ds = xr.open_dataset(fpath)
            # Mean across latitude & longitude
            arr = ds["u10"].mean(dim=["latitude", "longitude"]).values
            all_data.append(arr)
            ds.close()

    data = np.concatenate(all_data)  # shape (N,)
    return data.reshape(-1, 1)       # for MinMaxScaler

raw_data = load_u10_data(DATA_DIR)
print("Raw data shape:", raw_data.shape)

# Scale to (-1,1)
scaler = MinMaxScaler(feature_range=(-1,1))
data_scaled = scaler.fit_transform(raw_data)

# ----------------------------
# 3) CREATE SLIDING WINDOW
# ----------------------------
class WindDataset(Dataset):
    def __init__(self, data, lookback, horizon):
        """
        data   : normalized data shape (N,1)
        lookback: how many past hours as input
        horizon : how many hours to forecast
        """
        self.data = data
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.lookback - self.horizon

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.lookback]             # shape (lookback,1)
        y = self.data[idx + self.lookback : idx + self.lookback + self.horizon]  # shape (horizon,1)
        # Convert to float32 tensors
        x_t = torch.tensor(x, dtype=torch.float32)  # (lookback,1)
        y_t = torch.tensor(y, dtype=torch.float32)  # (horizon,1)
        return x_t, y_t

dataset = WindDataset(data_scaled, LOOKBACK, HORIZON)
n_total = len(dataset)
n_train = int(0.8 * n_total)
n_val   = int(0.1 * n_total)
n_test  = n_total - n_train - n_val

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

# ----------------------------
# 4) ENCODER-DECODER LSTM MODEL
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout)

    def forward(self, x):
        """
        x => (batch, lookback, 1)
        Returns final (hidden, cell)
        hidden, cell => shape (num_layers, batch, hidden_size)
        """
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, hidden, cell):
        """
        A simple decoder that uses the last output as the next input.
        We'll feed a zero initial input for the first step.
        """
        # We'll do the loop externally in the forward pass of Seq2Seq
        # so this method just does a single step.
        # input shape => (batch,1,1)
        # hidden, cell => from the encoder
        dec_out, (hidden, cell) = self.lstm(self.input_step, (hidden, cell))
        pred = self.fc(dec_out)  # => (batch,1,1)
        return pred, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, hidden_size=192, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size, num_layers, dropout)

    def forward(self, x):
        """
        x => (batch, lookback,1)
        Returns => (batch, horizon) or (batch, horizon,1)
        """
        # 1) Encode
        hidden, cell = self.encoder(x)   # shape => (num_layers,batch,hidden_size)

        # 2) Decode autoregressively
        # We'll start with a zero input for the first step
        batch_size = x.size(0)
        dec_input = torch.zeros((batch_size, 1, 1), device=x.device)

        outputs = []
        for _ in range(HORIZON):
            # Single-step decode
            dec_out, (hidden, cell) = self.decoder.lstm(dec_input, (hidden, cell))
            # dec_out => (batch,1,hidden_size)
            pred = self.decoder.fc(dec_out)  # => (batch,1,1)
            outputs.append(pred.squeeze(1))  # => shape (batch,1)
            # next input
            dec_input = pred  # shape => (batch,1,1)

        # shape => (batch,horizon)
        return torch.cat(outputs, dim=1)

# ----------------------------
# 5) TRAINING & EARLY STOPPING
# ----------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(batch_x)  # shape => (batch,horizon)
            loss = criterion(y_pred.unsqueeze(-1), batch_y)  # compare (batch,horizon,1)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                y_pred = model(batch_x)
                vloss = criterion(y_pred.unsqueeze(-1), batch_y)
                val_loss += vloss.item()
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} => Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss - 1e-7:
            best_val_loss = avg_val_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses

# ----------------------------
# 6) EVALUATION & PLOTTING
# ----------------------------
def evaluate_and_plot(model, test_loader, scaler):
    model.eval()
    preds_list, actuals_list = [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            y_pred = model(batch_x)   # shape => (batch,horizon)
            preds_list.append(y_pred.cpu().numpy())
            actuals_list.append(batch_y.numpy().squeeze(-1))  # shape => (batch,horizon)

    preds_arr   = np.concatenate(preds_list,   axis=0)  # (N, horizon)
    actuals_arr = np.concatenate(actuals_list, axis=0)  # (N, horizon)

    # Flatten => (N*horizon,)
    preds_flat   = preds_arr.reshape(-1)
    actuals_flat = actuals_arr.reshape(-1)

    # Inverse scaling
    preds_unscaled   = scaler.inverse_transform(preds_flat.reshape(-1,1)).flatten()
    actuals_unscaled = scaler.inverse_transform(actuals_flat.reshape(-1,1)).flatten()

    # Compute metrics
    mse_val = mean_squared_error(actuals_unscaled, preds_unscaled)
    mae_val = mean_absolute_error(actuals_unscaled, preds_unscaled)
    r2_val  = r2_score(actuals_unscaled, preds_unscaled)

    print("\n=== Final Test Metrics ===")
    print(f"MSE={mse_val:.4f}, MAE={mae_val:.4f}, R²={r2_val:.4f}")

    # Plot first 30 sequences => 30*horizon
    num_plot = 30 * HORIZON
    if len(preds_unscaled) < num_plot:
        num_plot = len(preds_unscaled)

    plt.figure(figsize=(10,6))
    plt.plot(actuals_unscaled[:num_plot], label="Actual", color="black")
    plt.plot(preds_unscaled[:num_plot],   label="Predicted", color="red", linestyle="--")
    plt.xlabel(f"First 30 sequences × {HORIZON} hours")
    plt.ylabel("Wind Speed (m/s)")
    plt.title(f"{HORIZON}hr Forecast: MSE={mse_val:.4f}, MAE={mae_val:.4f}, R²={r2_val:.4f}")
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------------------
# 7) MAIN
# ----------------------------
def main():
    model = Seq2Seq()
    print(model)

    print(f"\n=== Training LSTM (Lookback={LOOKBACK} → Horizon={HORIZON}) ===")
    train_losses, val_losses = train_model(model, train_loader, val_loader)

    # Plot Train vs Validation Loss
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses,   label="Val Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate on Test Set
    evaluate_and_plot(model, test_loader, scaler)

if __name__ == "__main__":
    main()
