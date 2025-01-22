import matplotlib.pyplot as plt
import xarray as xr
import os
import matplotlib.dates as mdates
import numpy as np

# Define the folder containing datasets
data_folder = "E:\\3rd_Year_College_Project\\6th Sem\\Datasets"

# List all files in the folder and filter for .nc files
all_files = os.listdir(data_folder)
nc_files = [os.path.join(data_folder, file) for file in all_files if file.endswith('.nc')]

# Debugging: Print the list of detected .nc files
print("\nDetected .nc files:", nc_files)

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

# Plot the U10 component
plt.figure(figsize=(15, 6))  # Extend width (x-axis) and height (y-axis)
u10_mean.plot(color='blue', label='10-meter U-component of Wind')

# Customize the plot
plt.title('10-meter U-component of Wind Over Time (Hourly Data)', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('U10 (m/s)', fontsize=12)

# Adjust y-axis ticks to include more points
y_min, y_max = u10_mean.min().values, u10_mean.max().values  # Get min and max values
y_ticks = np.linspace(y_min, y_max, num=20)  # Generate 20 evenly spaced ticks
plt.yticks(y_ticks)

# Format the x-axis to better handle hourly data
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Major ticks every year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format major ticks as year
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks every month

# Rotate x-axis labels for better readability
plt.gcf().autofmt_xdate()

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
