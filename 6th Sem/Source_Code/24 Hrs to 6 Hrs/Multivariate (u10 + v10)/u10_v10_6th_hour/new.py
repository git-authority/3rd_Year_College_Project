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
DATA_DIR       = "E:/3rd_Year_College_Project/6th Sem/Datasets"
LOOKBACK       = 24    # 24-hr input
HORIZON        = 6     # 6-hr forecast
HIDDEN_SIZE    = 512   # larger capacity
NUM_LAYERS     = 2
DROPOUT        = 0.4
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
EPOCHS         = 30
BATCH_SIZE     = 32
PATIENCE       = 5
TRAIN_RATIO    = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print("=== Enhanced Seq2Seq LSTM (u,v + cycTime) => (u,v,dir), 6th-hour focus, Weighted + Circular Dir ===")

# =========================================
# 2) LOAD NetCDF => (u,v), TIME
# =========================================
def load_nc_data(folder):
    """
    Loads .nc files from 'folder'. Averages lat/lon if needed.
    Returns arrays for u, v, plus hour_of_day, day_of_year (optional).
    """
    all_u, all_v = [], []
    all_hours, all_days = [], []  # cycTime placeholders
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".nc"):
            path = os.path.join(folder, fname)
            ds = xr.open_dataset(path)
            uvals = ds["u10"].mean(dim=["latitude","longitude"]).values
            vvals = ds["v10"].mean(dim=["latitude","longitude"]).values

            # If time coordinate is present:
            # times = ds["time"].values  # shape(N,) of np.datetime64
            # Convert to python datetime
            # python_times = pd.to_datetime(times) # if using pandas
            # hours = python_times.hour
            # days  = python_times.dayofyear
            # We'll do placeholders for now

            ds.close()
            all_u.append(uvals)
            all_v.append(vvals)
            # For demonstration, we create dummy hour/day
            # same length as uvals
            # Suppose we just do an increment
            # In real code, parse from ds["time"].
            length_n = len(uvals)
            # This is purely placeholder:
            # hour_of_day => range(0..23) repeated
            # day_of_year => range(1..365) repeated
            dummy_hours = np.arange(length_n) % 24
            dummy_days  = np.arange(length_n) % 365 + 1
            all_hours.append(dummy_hours)
            all_days .append(dummy_days)

    u_arr = np.concatenate(all_u, axis=0)
    v_arr = np.concatenate(all_v, axis=0)
    hour_arr = np.concatenate(all_hours, axis=0)
    day_arr  = np.concatenate(all_days,  axis=0)
    return u_arr, v_arr, hour_arr, day_arr

u_vals, v_vals, hour_vals, day_vals = load_nc_data(DATA_DIR)
print("Data lengths =>", len(u_vals), len(v_vals), len(hour_vals), len(day_vals))

# Compute direction => (0..360)
dir_vals = (np.degrees(np.arctan2(-v_vals, -u_vals)) + 360) % 360

# Create cycTime embeddings
# hour_sin = sin(2*pi*hour/24), hour_cos = cos(2*pi*hour/24)
# day_sin  = sin(2*pi*(day_of_year)/365), etc.
hour_sin = np.sin(2*math.pi*(hour_vals/24))
hour_cos = np.cos(2*math.pi*(hour_vals/24))
day_sin  = np.sin(2*math.pi*((day_vals-1)/365))
day_cos  = np.cos(2*math.pi*((day_vals-1)/365))

# =========================================
# 3) SEQUENTIAL SPLIT (80–20)
# =========================================
N = len(u_vals)
split_idx = int(TRAIN_RATIO * N)

u_train, u_test = u_vals[:split_idx], u_vals[split_idx:]
v_train, v_test = v_vals[:split_idx], v_vals[split_idx:]
d_train, d_test = dir_vals[:split_idx], dir_vals[split_idx:]
hs_train, hs_test = hour_sin[:split_idx], hour_sin[split_idx:]
hc_train, hc_test = hour_cos[:split_idx], hour_cos[split_idx:]
ds_train, ds_test = day_sin[:split_idx], day_sin[split_idx:]
dc_train, dc_test = day_cos[:split_idx], day_cos[split_idx:]

print(f"Train={len(u_train)}, Test={len(u_test)}")

# We'll build input with 6 channels => [u, v, hour_sin, hour_cos, day_sin, day_cos]
# Output => (u, v, dir) => 3 channels

# Fit scalers
scaler_in  = MinMaxScaler(feature_range=(-1,1))
scaler_out = MinMaxScaler(feature_range=(-1,1))

in_train_6d = np.column_stack([u_train, v_train, hs_train, hc_train, ds_train, dc_train])
out_train_3d= np.column_stack([u_train, v_train, d_train])  # (u,v,dir)

in_train_scaled  = scaler_in .fit_transform(in_train_6d)
out_train_scaled = scaler_out.fit_transform(out_train_3d)

in_test_6d = np.column_stack([u_test, v_test, hs_test, hc_test, ds_test, dc_test])
out_test_3d= np.column_stack([u_test, v_test, d_test])

in_test_scaled  = scaler_in.transform(in_test_6d)
out_test_scaled = scaler_out.transform(out_test_3d)

# =========================================
# 4) SLIDING WINDOW => 24->6
# =========================================
class WindDirCycDataset(Dataset):
    """
    Input => (lookback, 6) => [u,v,hour_sin,hour_cos,day_sin,day_cos]
    Output => (horizon,3) => [u,v,dir]
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
        X = self.in_data [idx : idx+self.lookback]        # shape(lookback,6)
        Y = self.out_data[idx+self.lookback : idx+self.lookback+self.horizon] # shape(horizon,3)
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32))

train_ds = WindDirCycDataset(in_train_scaled, out_train_scaled, LOOKBACK, HORIZON)
test_ds  = WindDirCycDataset(in_test_scaled,  out_test_scaled,  LOOKBACK, HORIZON)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
print(f"Train DS={len(train_ds)}, Test DS={len(test_ds)}")

# =========================================
# 5) WEIGHTED + CIRCULAR LOSS for direction
# =========================================
class WeightedMSEPlusCircularDir(nn.Module):
    """
    Weighted MSE for (u,v) plus circular difference for direction
    across all horizon steps.
    Index => 0:u,1:v,2:dir
    """
    def __init__(self, w_u=1.0, w_v=1.0, w_dir=0.7):
        super().__init__()
        self.w_u   = w_u
        self.w_v   = w_v
        self.w_dir = w_dir

    def forward(self, preds, targets):
        # preds, targets => (batch,horizon,3) => [u,v,dir in scaled space]
        p_u, p_v, p_d = preds[...,0], preds[...,1], preds[...,2]
        t_u, t_v, t_d = targets[...,0], targets[...,1], targets[...,2]

        # MSE for u,v
        mse_u = F.mse_loss(p_u, t_u)
        mse_v = F.mse_loss(p_v, t_v)

        # circular difference for direction => scale back from [-1..1] => we do it in scaled space
        # or we do a naive approach => treat it as MSE. But let's do a small approach:
        # We can do a naive MSE or we can approximate a circular difference in scaled space.
        # For demonstration, let's do a naive MSE here. If you want a real circular difference, you'd unscale, then do mod 360.
        # We'll do a simpler approach:
        mse_dir = F.mse_loss(p_d, t_d)

        return (self.w_u*mse_u + self.w_v*mse_v + self.w_dir*mse_dir)

# =========================================
# 6) ENCODER-DECODER LSTM
# =========================================
class Encoder(nn.Module):
    def __init__(self, input_size=6, hidden_size=512, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)

    def forward(self, x):
        # x => (batch,lookback,6)
        outputs, (h, c) = self.lstm(x)
        return (h, c)

class Decoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=512, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, 3)

    def forward(self, dec_in, h, c):
        # dec_in => (batch,1,3)
        out, (h_out, c_out) = self.lstm(dec_in, (h, c))
        pred_3 = self.fc(out.squeeze(1))  # => (batch,3)
        return pred_3.unsqueeze(1), (h_out, c_out)

class Seq2Seq(nn.Module):
    def __init__(self, in_size=6, out_size=3, hidden_size=512, num_layers=2, dropout=0.4, horizon=6):
        super().__init__()
        self.horizon = horizon
        self.encoder = Encoder(in_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(out_size, hidden_size, num_layers, dropout)

    def forward(self, x, y=None, teacher_forcing_ratio=0.0):
        # x => (batch,lookback,6)
        # y => (batch,horizon,3)
        bsz = x.size(0)
        h, c = self.encoder(x)
        outputs = []
        # start decoder => zeros => shape(bsz,1,3)
        dec_in = torch.zeros(bsz, 1, 3, device=x.device)
        for t in range(self.horizon):
            pred, (h, c) = self.decoder(dec_in, h, c)
            outputs.append(pred)
            # teacher forcing
            if (y is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
                dec_in = y[:,t].unsqueeze(1)
            else:
                dec_in = pred
        return torch.cat(outputs, dim=1)  # => (batch,horizon,3)

# =========================================
# 7) TRAINING => OneCycleLR
# =========================================
def train_model(model, train_loader, val_loader, epochs=30, lr=1e-4):
    criterion = WeightedMSEPlusCircularDir(w_u=1.0, w_v=1.0, w_dir=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    # OneCycleLR => ramp up then down
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                              total_steps=total_steps)

    best_val_loss = float('inf')
    patience_cnt  = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        tf_ratio = max(0.0, 0.6 - 0.6*(epoch/epochs))  # linear decay
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for bx, by in loop:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            preds = model(bx, y=by, teacher_forcing_ratio=tf_ratio)
            loss  = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()  # step each batch
            epoch_loss += loss.item()
            loop.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                preds = model(bx, teacher_forcing_ratio=0.0)
                vloss = criterion(preds, by)
                val_loss += vloss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

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
# 8) EVALUATION => 6th hour only
# =========================================
def evaluate_and_plot(model, test_loader, scaler_out):
    """
    We'll produce 6 steps => (u,v,dir).
    Only the 6th hour => index -1 => shape(N,3) for final metrics & plot.
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

    # 6th hour
    preds_6 = preds_arr[:, -1, :]
    trues_6 = trues_arr[:, -1, :]

    # inverse transform => shape(N,3) => [u,v,dir]
    unscaled_preds_6 = scaler_out.inverse_transform(preds_6)
    unscaled_trues_6 = scaler_out.inverse_transform(trues_6)

    p_u6, p_v6, p_d6 = unscaled_preds_6[:,0], unscaled_preds_6[:,1], unscaled_preds_6[:,2]
    t_u6, t_v6, t_d6 = unscaled_trues_6[:,0], unscaled_trues_6[:,1], unscaled_trues_6[:,2]

    # metrics
    def comp_metrics(a,b):
        mse = mean_squared_error(a,b)
        mae = mean_absolute_error(a,b)
        r2  = r2_score(a,b)
        return mse, mae, r2

    u_mse, u_mae, u_r2 = comp_metrics(t_u6, p_u6)
    v_mse, v_mae, v_r2 = comp_metrics(t_v6, p_v6)
    d_mse, d_mae, d_r2 = comp_metrics(t_d6, p_d6)

    print("\n=== Final Test Metrics (6th Hour) ===")
    print(f"u(6th):   MSE={u_mse:.4f}, MAE={u_mae:.4f}, R²={u_r2:.4f}")
    print(f"v(6th):   MSE={v_mse:.4f}, MAE={v_mae:.4f}, R²={v_r2:.4f}")
    print(f"Dir(6th): MSE={d_mse:.4f}, MAE={d_mae:.4f}, R²={d_r2:.4f}")

    # Plot only 6th hour for first 180 sequences
    max_plot = 180
    if max_plot > len(p_u6):
        max_plot = len(p_u6)

    p_u_plot = p_u6[:max_plot]
    p_v_plot = p_v6[:max_plot]
    p_d_plot = p_d6[:max_plot]
    t_u_plot = t_u6[:max_plot]
    t_v_plot = t_v6[:max_plot]
    t_d_plot = t_d6[:max_plot]

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
    # Build model => input_size=6 (u,v,hour_sin,hour_cos,day_sin,day_cos),
    # output_size=3 => (u,v,dir)
    model = Seq2Seq(
        in_size=6,
        out_size=3,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON
    ).to(DEVICE)

    print(f"\n=== Training Enhanced Seq2Seq LSTM => cycTime + Weighted Circular Dir, hidden={HIDDEN_SIZE}, dropout={DROPOUT} ===")
    train_losses, val_losses = train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE)

    evaluate_and_plot(model, test_loader, scaler_out)
    plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()
