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

# =========================================
# 1) CONFIG & HYPERPARAMETERS
# =========================================
DATA_DIR       = "E:/3rd_Year_College_Project/6th Sem/Datasets"  # your .nc folder
LOOKBACK       = 24            # 24-hr input
HORIZON        = 6             # 6-hr forecast
HIDDEN_SIZE    = 256
NUM_LAYERS     = 2
DROPOUT        = 0.3
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
EPOCHS         = 30
BATCH_SIZE     = 32
PATIENCE       = 5
TRAIN_RATIO    = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print("=== Multi-Input (u,v) => Multi-Output (u,v,dir), 24→6, only 6th hour metrics & plot ===")

# =========================================
# 2) LOAD NetCDF => u10, v10
# =========================================
def load_nc_u10_v10(folder):
    """
    Loads .nc files => returns concatenated arrays for u10, v10.
    Averages lat/lon if needed.
    """
    all_u, all_v = [], []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".nc"):
            path = os.path.join(folder, fname)
            ds = xr.open_dataset(path)
            uvals = ds["u10"].mean(dim=["latitude","longitude"]).values
            vvals = ds["v10"].mean(dim=["latitude","longitude"]).values
            ds.close()
            all_u.append(uvals)
            all_v.append(vvals)
    u_arr = np.concatenate(all_u, axis=0)
    v_arr = np.concatenate(all_v, axis=0)
    return u_arr, v_arr

u_vals, v_vals = load_nc_u10_v10(DATA_DIR)
print("Total length after loading:", len(u_vals), len(v_vals))

# Compute direction => (0..360)
dir_vals = (np.degrees(np.arctan2(-v_vals, -u_vals)) + 360) % 360

# Seq split => 80% train, 20% test
split_idx = int(TRAIN_RATIO * len(u_vals))
u_train, u_test = u_vals[:split_idx], u_vals[split_idx:]
v_train, v_test = v_vals[:split_idx], v_vals[split_idx:]
d_train, d_test = dir_vals[:split_idx], dir_vals[split_idx:]
print(f"Train={len(u_train)}, Test={len(u_test)}")

# We'll scale input (u,v) and output (u,v,dir) separately
scaler_in  = MinMaxScaler(feature_range=(-1,1))  # for (u,v)
scaler_out = MinMaxScaler(feature_range=(-1,1))  # for (u,v,dir)

# training arrays
in_train_2d = np.column_stack([u_train, v_train])     # shape(N,2)
out_train_3d= np.column_stack([u_train, v_train, d_train])  # shape(N,3)

# Fit scalers on training
in_train_scaled  = scaler_in .fit_transform(in_train_2d)
out_train_scaled = scaler_out.fit_transform(out_train_3d)

# test arrays => transform
in_test_2d  = np.column_stack([u_test, v_test])
out_test_3d = np.column_stack([u_test, v_test, d_test])
in_test_scaled  = scaler_in.transform(in_test_2d)
out_test_scaled = scaler_out.transform(out_test_3d)

# =========================================
# 3) DATASET => 24->6
# =========================================
class WindDir6thDataset(Dataset):
    """
    Input => (lookback,2)
    Output => (horizon,3)
    """
    def __init__(self, in_data, out_data, lookback=24, horizon=6):
        self.in_data  = in_data
        self.out_data = out_data
        self.lookback = lookback
        self.horizon  = horizon
        self.length   = len(in_data) - lookback - horizon

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X = self.in_data [idx : idx+self.lookback]        # shape(lookback,2)
        Y = self.out_data[idx+self.lookback : idx+self.lookback+self.horizon] # shape(horizon,3)
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32))

train_ds = WindDir6thDataset(in_train_scaled, out_train_scaled, LOOKBACK, HORIZON)
test_ds  = WindDir6thDataset(in_test_scaled,  out_test_scaled,  LOOKBACK, HORIZON)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
print(f"Train DS={len(train_ds)}, Test DS={len(test_ds)}")

# =========================================
# 4) ENCODER-DECODER LSTM
# =========================================
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
    def __init__(self, input_size=3, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, 3)

    def forward(self, dec_in, h, c):
        # dec_in => (batch,1,3)
        out, (h_out, c_out) = self.lstm(dec_in, (h, c))
        pred_3 = self.fc(out.squeeze(1))  # =>(batch,3)
        return pred_3.unsqueeze(1), (h_out, c_out)

class Seq2Seq(nn.Module):
    def __init__(self, in_size=2, out_size=3, hidden_size=256, num_layers=2, dropout=0.3, horizon=6):
        super().__init__()
        self.horizon = horizon
        self.encoder = Encoder(in_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(out_size, hidden_size, num_layers, dropout)

    def forward(self, x, y=None, teacher_forcing_ratio=0.0):
        batch_size = x.size(0)
        # encode
        h, c = self.encoder(x)
        # decode => produce 6 steps => shape(batch,horizon,3)
        outputs = []
        dec_in = torch.zeros(batch_size, 1, 3, device=x.device)
        for t in range(self.horizon):
            pred, (h, c) = self.decoder(dec_in, h, c)
            outputs.append(pred)
            # teacher forcing
            if (y is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
                dec_in = y[:, t].unsqueeze(1)
            else:
                dec_in = pred
        return torch.cat(outputs, dim=1)

# =========================================
# 5) TRAINING LOOP
# =========================================
def train_model(model, train_loader, test_loader, epochs=30, lr=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    patience_cnt  = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        # teacher forcing decays
        tf_ratio = max(0.0, 0.6 - 0.6*(epoch/epochs))
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for bx, by in loop:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            preds = model(bx, y=by, teacher_forcing_ratio=tf_ratio)
            loss  = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                vpred = model(bx, teacher_forcing_ratio=0.0)
                vloss = criterion(vpred, by)
                val_loss += vloss.item()
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch}/{epochs} => TF={tf_ratio:.2f} | Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss - 1e-7:
            best_val_loss = avg_val_loss
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses

# =========================================
# 6) EVALUATION => only 6th hour
# =========================================
def evaluate_and_plot(model, test_loader, scaler_out):
    """
    We produce 6-step forecasts => (u,v,dir).
    We'll only use the 6th hour => index -1 => shape (N,3).
    We'll compute metrics for that hour & also plot only that hour for the first 180 samples.
    """
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            preds = model(bx, teacher_forcing_ratio=0.0)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(by.numpy())

    preds_arr = np.concatenate(all_preds, axis=0)  # => (N,6,3)
    trues_arr = np.concatenate(all_trues, axis=0) # => (N,6,3)

    # Only 6th hour => preds_6 => shape (N,3)
    preds_6 = preds_arr[:, -1, :]
    trues_6 = trues_arr[:, -1, :]

    # Inverse transform
    unscaled_preds_6 = scaler_out.inverse_transform(preds_6)
    unscaled_trues_6 = scaler_out.inverse_transform(trues_6)

    p_u6, p_v6, p_d6 = unscaled_preds_6[:,0], unscaled_preds_6[:,1], unscaled_preds_6[:,2]
    t_u6, t_v6, t_d6 = unscaled_trues_6[:,0], unscaled_trues_6[:,1], unscaled_trues_6[:,2]

    # Metrics for the 6th hour
    def comp_metrics(a, b):
        mse = mean_squared_error(a,b)
        mae = mean_absolute_error(a,b)
        r2  = r2_score(a,b)
        return mse, mae, r2

    u_mse, u_mae, u_r2 = comp_metrics(t_u6, p_u6)
    v_mse, v_mae, v_r2 = comp_metrics(t_v6, p_v6)
    d_mse, d_mae, d_r2 = comp_metrics(t_d6, p_d6)

    print("\n=== Final Test Metrics (6th Hour Only) ===")
    print(f"u:   MSE={u_mse:.4f}, MAE={u_mae:.4f}, R²={u_r2:.4f}")
    print(f"v:   MSE={v_mse:.4f}, MAE={v_mae:.4f}, R²={v_r2:.4f}")
    print(f"Dir: MSE={d_mse:.4f}, MAE={d_mae:.4f}, R²={d_r2:.4f}")

    # For the plot => first 180 sequences => shape(180,3). We'll do separate subplots for (u,v,dir).
    # That means we only have 180 points => x-axis is sample index 0..179
    preds_6_all = unscaled_preds_6
    trues_6_all = unscaled_trues_6

    max_plot = 180
    if max_plot > len(preds_6_all):
        max_plot = len(preds_6_all)

    p_u_plot = preds_6_all[:max_plot, 0]
    p_v_plot = preds_6_all[:max_plot, 1]
    p_d_plot = preds_6_all[:max_plot, 2]
    t_u_plot = trues_6_all[:max_plot,0]
    t_v_plot = trues_6_all[:max_plot,1]
    t_d_plot = trues_6_all[:max_plot,2]

    fig, axes = plt.subplots(3,1, figsize=(10,9), sharex=True)

    # u subplot
    axes[0].plot(t_u_plot, label="Actual u (6th hr)", color="black")
    axes[0].plot(p_u_plot, label="Pred u (6th hr)", color="red", linestyle="--")
    axes[0].set_ylabel("u10 (m/s)")
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_title(f"u(6th hr): MSE={u_mse:.4f}, MAE={u_mae:.4f}, R²={u_r2:.4f}")

    # v subplot
    axes[1].plot(t_v_plot, label="Actual v (6th hr)", color="black")
    axes[1].plot(p_v_plot, label="Pred v (6th hr)", color="red", linestyle="--")
    axes[1].set_ylabel("v10 (m/s)")
    axes[1].grid(True)
    axes[1].legend()
    axes[1].set_title(f"v(6th hr): MSE={v_mse:.4f}, MAE={v_mae:.4f}, R²={v_r2:.4f}")

    # dir subplot
    axes[2].plot(t_d_plot, label="Actual Dir (6th hr)", color="black")
    axes[2].plot(p_d_plot, label="Pred Dir (6th hr)", color="red", linestyle="--")
    axes[2].set_ylabel("Direction (°)")
    axes[2].set_xlabel(f"First {max_plot} sequences => 6th hour")
    axes[2].grid(True)
    axes[2].legend()
    axes[2].set_title(f"Dir(6th hr): MSE={d_mse:.4f}, MAE={d_mae:.4f}, R²={d_r2:.4f}")

    fig.suptitle("6th-Hour Forecast: (u, v, direction)", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses,   label="Val Loss",   color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (u,v,dir) across 6 steps")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Build model => in_size=2, out_size=3
    model = Seq2Seq(
        in_size=2,
        out_size=3,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON
    ).to(DEVICE)

    print(f"\n=== Training Seq2Seq => (u,v)->(u,v,dir), 6th hr only for final metrics & plot ===")
    train_losses, val_losses = train_model(model, train_loader, test_loader,
                                           epochs=EPOCHS, lr=LEARNING_RATE)

    evaluate_and_plot(model, test_loader, scaler_out)
    plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()
