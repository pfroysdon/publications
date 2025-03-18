import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load and Parse the Data
filename = 'data/jena_climate_2009_2016.csv'
df = pd.read_csv(filename)
# Assume the first column is date and the third column is temperature.
df['Date'] = pd.to_datetime(df.iloc[:,0])
temp = df.iloc[:,2].values  # temperature in °C
print(f"Loaded {len(temp)} samples from {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")

# 2. Plot Temperature Time-Series
plt.figure(figsize=(10,4))
plt.plot(df['Date'], temp)
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Time-Series")
plt.show()

# 3. Prepare Data: windowing function.
def create_dataset(data, lookback, delay, step, start_index, end_index):
    num_samples = (end_index - start_index - lookback - delay) + 1
    X = np.zeros((num_samples, lookback // step))
    y = np.zeros((num_samples,))
    for i in range(num_samples):
        indices = np.arange(start_index+i, start_index+i+lookback, step)
        X[i] = data[indices]
        y[i] = data[start_index+i+lookback+delay-1]
    return X, y

lookback = 720   # past 720 timesteps
delay = 144      # predict 144 timesteps ahead
step = 6         # sample every 6 timesteps

num_data = len(temp)
train_end = 200000
val_end = 300000
if num_data < train_end + lookback + delay:
    raise ValueError("Not enough data for these parameters.")

X_train, y_train = create_dataset(temp, lookback, delay, step, 0, train_end)
X_val, y_val = create_dataset(temp, lookback, delay, step, train_end, val_end)
X_test, y_test = create_dataset(temp, lookback, delay, step, val_end, num_data)

print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])
print("Testing samples:", X_test.shape[0])

# 5. Normalize the Data using training statistics.
train_mean = X_train.mean()
train_std = X_train.std()
X_train = (X_train - train_mean) / train_std
X_val = (X_val - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

y_train = (y_train - train_mean) / train_std
y_val = (y_val - train_mean) / train_std
y_test = (y_test - train_mean) / train_std

# 6. Baseline: predict last value.
y_pred_baseline = X_val[:,-1]
baseline_mae = np.mean(np.abs(y_pred_baseline - y_val))
print(f"Baseline MAE (normalized): {baseline_mae:.4f}")

# 7. Build models using Keras.
input_shape = (X_train.shape[1],)
def build_dense_model():
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model

def build_rnn_model():
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1], 1)),
        layers.SimpleRNN(32, return_sequences=False),
        layers.Dense(1)
    ])
    model.compile(optimizer=optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model

def build_lstm_model():
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1], 1)),
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer=optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model

# Reshape data for RNN/LSTM (samples, timesteps, 1)
X_train_rnn = X_train[..., np.newaxis]
X_val_rnn = X_val[..., np.newaxis]
X_test_rnn = X_test[..., np.newaxis]

# Train Dense Model
dense_model = build_dense_model()
es = EarlyStopping(monitor='val_loss', patience=3)
history_dense = dense_model.fit(X_train, y_train, epochs=10, batch_size=32,
                                validation_data=(X_val, y_val), callbacks=[es], verbose=2)
y_pred_dense = dense_model.predict(X_val).flatten()
dense_mae = np.mean(np.abs(y_pred_dense - y_val))
print(f"Dense model MAE (normalized): {dense_mae:.4f}")

# Train RNN Model
rnn_model = build_rnn_model()
history_rnn = rnn_model.fit(X_train_rnn, y_train, epochs=10, batch_size=32,
                            validation_data=(X_val_rnn, y_val), callbacks=[es], verbose=2)
y_pred_rnn = rnn_model.predict(X_val_rnn).flatten()
rnn_mae = np.mean(np.abs(y_pred_rnn - y_val))
print(f"RNN model MAE (normalized): {rnn_mae:.4f}")

# Train LSTM Model
lstm_model = build_lstm_model()
history_lstm = lstm_model.fit(X_train_rnn, y_train, epochs=10, batch_size=32,
                              validation_data=(X_val_rnn, y_val), callbacks=[es], verbose=2)
y_pred_lstm = lstm_model.predict(X_val_rnn).flatten()
lstm_mae = np.mean(np.abs(y_pred_lstm - y_val))
print(f"LSTM model MAE (normalized): {lstm_mae:.4f}")

# 10. Plot predictions on validation set.
t = np.arange(len(y_val))
plt.figure(figsize=(10,6))
plt.plot(t, y_val, 'k', linewidth=1.5, label='True')
plt.plot(t, y_pred_baseline, 'b', label='Baseline')
plt.plot(t, y_pred_dense, 'r', label='Dense')
plt.plot(t, y_pred_rnn, 'g', label='RNN')
plt.plot(t, y_pred_lstm, 'm', label='LSTM')
plt.xlabel("Validation Sample Index")
plt.ylabel("Normalized Temperature")
plt.legend()
plt.title("Model Predictions on Validation Set")
plt.grid(True)
plt.show()
