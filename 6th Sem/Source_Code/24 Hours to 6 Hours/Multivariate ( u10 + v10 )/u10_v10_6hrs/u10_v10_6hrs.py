import os
import math
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# ================================
# 1) CONFIG & HYPERPARAMETERS
# ================================
DATA_DIR       = "E:/3rd_Year_College_Project/6th Sem/Datasets"   # Adjust to your .nc files directory
LOOKBACK       = 24              # 24-hour input
HORIZON        = 6               # 6-hour forecast
HIDDEN_SIZE    = 256
NUM_LAYERS     = 2
DROPOUT        = 0.3
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
EPOCHS         = 30
BATCH_SIZE     = 32
PATIENCE       = 5               # early stopping patience
TRAIN_RATIO    = 0.8             # first 80% => train, last 20% => test
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print("=== Multi-Input Multi-Output LSTM (24→6) for (u10,v10), Seq Split 80–20 ===")

# ================================
# 2) LOAD NetCDF => u10, v10
# ================================
def load_nc_u10_v10(folder):
    """
    Loads multiple .nc files, returns concatenated arrays for u10, v10.
    Averages lat/lon if needed.
    """
    all_u, all_v = [], []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".nc"):
            path = os.path.join(folder, fname)
            ds = xr.open_dataset(path)
            # average lat/lon
            uvals = ds["u10"].mean(dim=["latitude","longitude"]).values
            vvals = ds["v10"].mean(dim=["latitude","longitude"]).values
            ds.close()
            all_u.append(uvals)
            all_v.append(vvals)
    u_arr = np.concatenate(all_u, axis=0)
    v_arr = np.concatenate(all_v, axis=0)
    return u_arr, v_arr

u_vals, v_vals = load_nc_u10_v10(DATA_DIR)
print("Total data length:", len(u_vals), len(v_vals))

# ================================
# 3) SEQUENTIAL SPLIT (80–20)
# ================================
split_idx = int(TRAIN_RATIO * len(u_vals))
u_train, u_test = u_vals[:split_idx], u_vals[split_idx:]
v_train, v_test = v_vals[:split_idx], v_vals[split_idx:]
print(f"Train={len(u_train)}, Test={len(u_test)}")

# ================================
# 4) SCALING
# ================================
scaler_u = MinMaxScaler(feature_range=(-1,1))
scaler_v = MinMaxScaler(feature_range=(-1,1))
u_train_scaled = scaler_u.fit_transform(u_train.reshape(-1,1)).flatten()
v_train_scaled = scaler_v.fit_transform(v_train.reshape(-1,1)).flatten()

# For the test set, use the same scaler (don't fit again)
u_test_scaled = scaler_u.transform(u_test.reshape(-1,1)).flatten()
v_test_scaled = scaler_v.transform(v_test.reshape(-1,1)).flatten()

# ================================
# 5) SLIDING WINDOW DATASET
# ================================
class WindDataset(Dataset):
    """
    X => shape (lookback,2) => (u10,v10)
    Y => shape (horizon,2)
    """
    def __init__(self, u_array, v_array, lookback=24, horizon=6):
        self.u = u_array
        self.v = v_array
        self.lookback = lookback
        self.horizon  = horizon
        self.length = len(self.u) - lookback - horizon

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_u = self.u[idx : idx+self.lookback]
        x_v = self.v[idx : idx+self.lookback]
        y_u = self.u[idx+self.lookback : idx+self.lookback+self.horizon]
        y_v = self.v[idx+self.lookback : idx+self.lookback+self.horizon]
        # combine => X => (lookback,2), Y => (horizon,2)
        X = np.column_stack([x_u, x_v])  # shape (lookback,2)
        Y = np.column_stack([y_u, y_v])  # shape (horizon,2)
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32))

train_ds = WindDataset(u_train_scaled, v_train_scaled, LOOKBACK, HORIZON)
test_ds  = WindDataset(u_test_scaled,  v_test_scaled,  LOOKBACK, HORIZON)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
print(f"Train DS={len(train_ds)}, Test DS={len(test_ds)}")

# ================================
# 6) ENCODER-DECODER LSTM
# ================================
class Encoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)

    def forward(self, x):
        # x => (batch,lookback,2)
        outputs, (h, c) = self.lstm(x)
        return (h, c)

class Decoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, 2)

    def forward(self, dec_in, h, c):
        # dec_in => (batch,1,2)
        out, (h_out, c_out) = self.lstm(dec_in, (h, c))
        pred = self.fc(out.squeeze(1))  # => (batch,2)
        return pred.unsqueeze(1), (h_out, c_out)

class Seq2Seq(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, num_layers=2, dropout=0.3, horizon=6):
        super().__init__()
        self.horizon = horizon
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(input_size, hidden_size, num_layers, dropout)

    def forward(self, x, y=None, teacher_forcing_ratio=0.0):
        batch_size = x.size(0)
        # 1) encode
        h, c = self.encoder(x)
        # 2) decode step by step
        outputs = []
        dec_in = torch.zeros(batch_size, 1, 2, device=x.device)
        for t in range(self.horizon):
            pred, (h, c) = self.decoder(dec_in, h, c)
            outputs.append(pred)
            if (y is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
                dec_in = y[:, t].unsqueeze(1)  # teacher forcing
            else:
                dec_in = pred
        return torch.cat(outputs, dim=1)  # (batch,horizon,2)

# ================================
# 7) TRAINING LOOP
# ================================
def train_model(model, train_loader, test_loader, epochs=30, lr=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    patience_cnt  = 0
    train_losses, test_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        # teacher forcing ratio decays from 0.6 -> 0
        tf_ratio = max(0.0, 0.6 - 0.6*(epoch/epochs))

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch_x, batch_y in loop:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(batch_x, y=batch_y, teacher_forcing_ratio=tf_ratio)
            loss  = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                preds = model(batch_x, teacher_forcing_ratio=0.0)
                vloss = criterion(preds, batch_y)
                val_loss += vloss.item()
        avg_val_loss = val_loss / len(test_loader)
        test_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch}/{epochs} => TF={tf_ratio:.2f} | Train={avg_train_loss:.4f}, Test={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss - 1e-7:
            best_val_loss = avg_val_loss
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping triggered.")
                break

    return train_losses, test_losses

# ================================
# 8) EVALUATION & PLOT
# ================================
def evaluate_and_plot(model, test_loader, scaler_u, scaler_v):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            preds = model(batch_x, teacher_forcing_ratio=0.0)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(batch_y.numpy())

    preds_arr = np.concatenate(all_preds, axis=0)  # => (N, horizon,2)
    trues_arr = np.concatenate(all_trues, axis=0) # => (N, horizon,2)

    # Flatten => (N*horizon,2)
    preds_flat = preds_arr.reshape(-1,2)
    trues_flat = trues_arr.reshape(-1,2)

    # Inverse transform
    p_u = scaler_u.inverse_transform(preds_flat[:,0].reshape(-1,1)).flatten()
    p_v = scaler_v.inverse_transform(preds_flat[:,1].reshape(-1,1)).flatten()

    t_u = scaler_u.inverse_transform(trues_flat[:,0].reshape(-1,1)).flatten()
    t_v = scaler_v.inverse_transform(trues_flat[:,1].reshape(-1,1)).flatten()

    # Metrics
    def compute_metrics(true_vals, pred_vals):
        mse = mean_squared_error(true_vals, pred_vals)
        mae = mean_absolute_error(true_vals, pred_vals)
        r2  = r2_score(true_vals, pred_vals)
        return mse, mae, r2

    u_mse, u_mae, u_r2 = compute_metrics(t_u, p_u)
    v_mse, v_mae, v_r2 = compute_metrics(t_v, p_v)

    print("\n=== Final Test Metrics ===")
    print(f"u10: MSE={u_mse:.4f}, MAE={u_mae:.4f}, R²={u_r2:.4f}")
    print(f"v10: MSE={v_mse:.4f}, MAE={v_mae:.4f}, R²={v_r2:.4f}")

    # Plot first 30 sequences => 30*horizon => 180 points
    num_seq = 30
    total_len = preds_flat.shape[0]
    max_plot = num_seq * HORIZON
    if max_plot > total_len:
        max_plot = total_len

    s_true_u = t_u[:max_plot]
    s_pred_u = p_u[:max_plot]
    s_true_v = t_v[:max_plot]
    s_pred_v = p_v[:max_plot]

    fig, axes = plt.subplots(2,1, figsize=(10,8), sharex=True)

    # u10 subplot
    axes[0].plot(s_true_u, label="Actual u", color="black")
    axes[0].plot(s_pred_u, label="Pred u", color="red", linestyle="--")
    axes[0].set_ylabel("u10 (m/s)")
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_title(f"u10: MSE={u_mse:.4f}, MAE={u_mae:.4f}, R²={u_r2:.4f}")

    # v10 subplot
    axes[1].plot(s_true_v, label="Actual v", color="black")
    axes[1].plot(s_pred_v, label="Pred v", color="red", linestyle="--")
    axes[1].set_ylabel("v10 (m/s)")
    axes[1].set_xlabel(f"First 30 sequences × {HORIZON} hours = {max_plot} points")
    axes[1].grid(True)
    axes[1].legend()
    axes[1].set_title(f"v10: MSE={v_mse:.4f}, MAE={v_mae:.4f}, R²={v_r2:.4f}")

    fig.suptitle(f"{HORIZON}-Hour Forecast: (u10, v10)")
    plt.tight_layout()
    plt.show()

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses,   label="Val Loss",   color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 1) Build model
    model = Seq2Seq(
        input_size=2,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON
    ).to(DEVICE)

    # 2) Train model
    print(f"\n=== Training Seq2Seq LSTM (24→6) for (u10,v10), hidden={HIDDEN_SIZE}, dropout={DROPOUT} ===")
    train_losses, val_losses = train_model(model, train_loader, test_loader,
                                           epochs=EPOCHS, lr=LEARNING_RATE)

    # 3) Evaluate + Plot
    evaluate_and_plot(model, test_loader, scaler_u, scaler_v)
    plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()
