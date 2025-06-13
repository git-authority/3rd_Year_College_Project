# Wind & Temperature Time Series Forecasting

This repository contains all the resources, data, and code related to my undergraduate project on wind and temperature time series forecasting using deep learning.

---

## **5th Semester**

This directory contains a `Plots` subdirectory with the following visualizations:

- **2m air temperature.png**: Plot of 2-meter air temperature.
- **2m dew point temperature.png**: Plot of 2-meter dew point temperature.
- **10m u-component of wind.png**: Plot of 10-meter u-component (zonal) wind.
- **10m v-component of wind.png**: Plot of 10-meter v-component (meridional) wind.

---

## **6th Semester**

This directory is organized into the following sections:

### 1. **Datasets**
Contains all the hourly datasets from **2013 to 2023 (11 Years)**.

---

### 2. **Plots**

Organized as follows:

- **24 Hours to 6 Hours**
  - **Multivariate (u10 + v10)**
    - **u10_v10_6hrs/**
      - `u10_v10_6hrs_loss.png`: Loss curve for 6-hour forecasts of u10 and v10.
      - `u10_v10_6hrs_plot.png`: Predicted vs. actual plots for 6-hour forecasts of u10 and v10.
    - **u10_v10_6th_hour/**
      - `u10_v10_6th_hour_loss.png`: Loss curve specifically for the 6th forecast hour.
      - `u10_v10_6th_hour_plot.png`: Predicted vs. actual plots for the 6th forecast hour.
  - **Univariate (u10)**
    - `u10_Loss_24h_6h.png`: Loss curve for univariate 6-hour u10 forecasts.
    - `u10_Plot_24h_6h.png`: Predicted vs. actual plots for univariate 6-hour u10 forecasts.

- **30 Days to 7 Days / Univariate**
  - **t2m/**
    - `t2m_loss_30d_7d.png`: Loss curve for 7-day t2m forecasts.
    - `t2m_plot_30d_7d.png`: Predicted vs. actual plots for 7-day t2m forecasts.
  - **u10/**
    - `u10_Loss_30d_7d.png`: Loss curve for 7-day u10 forecasts.
    - `u10_Plot_30d_7d.png`: Predicted vs. actual plots for 7-day u10 forecasts.

- **Visualizing u10 Plots**
  - `Plot for u10 (Hourly).png`: Hourly spatial mean time series for selected days.

---

### 3. **Source Code**

Organized as follows:

- **24 Hours to 6 Hours**
  - **Multivariate (u10 + v10)**
    - **u10_v10_6hrs/**
      - `u10_v10_6hrs.py`: Script for forecasting the next 6 hours of u10 and v10 wind components using the past 24 hours of spatially averaged data.
    - **u10_v10_6th_hour/**
      - `u10_v10_6th_hour.py`: Script for generating visualizations and metrics for the 6th forecasted hour using the outputs from the multivariate model.
  - **Univariate (u10)**
    - `u10_24h_6h.py`: Script for forecasting the next 6 hours of the u10 wind component using the past 24 hours of spatially averaged data.

- **30 Days to 7 Days / Univariate**
  - **t2m/**
    - `t2m_30d_7d.py`: Script for forecasting the next 7 days (168 hours) of spatially averaged 2-meter temperature (t2m) using the past 30 days (720 hours) of data.
  - **u10/**
    - `u10_30d_7d.pth`: Trained model weights for 7-day u10 forecasting.
    - `u10_30d_7d.py`: Script for forecasting the next 7 days (168 hours) of spatially averaged u10 wind component using the past 30 days (720 hours) of data.

- **Visualizing u10 Plots**
  - `u10_Plots.py`: Script for generating hourly spatial mean plots for a specific day of each year (2013–2023).

---
