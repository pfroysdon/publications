import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# 1. Generate Synthetic Daily Temperature Data
days = np.arange(1, 366)  # Days 1 to 365
# Seasonal temperature: mean 15°C with amplitude 10°C and phase shift of 80 days.
seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (days - 80) / 365)
# Add Gaussian noise (std=2°C)
noise = np.random.randn(len(days)) * 2
temperature = seasonal_temp + noise

# 2. Define Sliding Window Filter (Moving Average)
def sliding_window_filter(y, window_size):
    # Ensure window_size is odd for symmetry.
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    N = len(y)
    y_filtered = np.zeros_like(y)
    for i in range(N):
        start_idx = max(0, i - half_window)
        end_idx = min(N, i + half_window + 1)
        y_filtered[i] = np.mean(y[start_idx:end_idx])
    return y_filtered

# Apply sliding window filters
window_size_7 = 7
window_size_30 = 30
smoothed_temperature7 = sliding_window_filter(temperature, window_size_7)
smoothed_temperature30 = sliding_window_filter(temperature, window_size_30)

# 3. Plot the Original and Smoothed Temperature Data
plt.figure(figsize=(10,5))
plt.plot(days, temperature, 'b-', linewidth=1.5, label='Original Temperature')
plt.plot(days, smoothed_temperature7, 'r-', linewidth=2, label='7-Day Moving Average')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.title('Daily Temperature and 7-Day Moving Average')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(days, temperature, 'b-', linewidth=1.5, label='Original Temperature')
plt.plot(days, smoothed_temperature7, 'r-', linewidth=2, label='7-Day Moving Average')
plt.plot(days, smoothed_temperature30, 'k-', linewidth=2, label='30-Day Moving Average')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.title('Daily Temperature with 7-Day and 30-Day Moving Averages')
plt.legend()
plt.grid(True)
plt.xlim([130, 230])
plt.show()
