import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# CPU→GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ticker = "7203.T"

toyota_finance_data = yf.download(ticker, start='2020-04-01', end='2025-03-31')
toyota_finance_data.to_csv("toyota_stock_data")

#データの前処理

# 不要なカラムを削除
df = toyota_finance_data[['Open', 'High', 'Low', 'Close']]
df = df.dropna()

# 線形補完で欠損値を補完
df = df.fillna(method='ffill')


# LSTMに最適化するためスケーリング(0~1)する
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])

# 検証用データとテストデータに分割
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# シーケンス長
sequence_length = 40

#シーケンス作成関数(過去80日分)
def create_sequence(data, sequence_length):
    x, y = [], []
    for i in range(sequence_length, len(data)):
        x.append(data[i - sequence_length: i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# シーケンス作成
x_train, y_train = create_sequence(train_data, sequence_length)
x_test, y_test = create_sequence(test_data, sequence_length)

x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


class StockPricePredicter(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(StockPricePredicter, self).__init__()
        
        self.lstm_in = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm_in(x)  # (batch, seq_len, hidden_size)
        predictions = self.lstm_out(lstm_out[:, -1, :])  # 最後の出力だけ全結合に通す
        return predictions

    
# モデルをインスタンス化
models = StockPricePredicter(input_size=1, hidden_size=50, output_size=1)

lr = 0.001
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(models.parameters(), lr=lr)


#学習
epoch_num = 20
for epoch in range(epoch_num):
    models.train()
    
    x_train_seq = x_train.unsqueeze(2)
    y_train_seq = y_train.unsqueeze(1)
    
    optimizer.zero_grad()
    y_pred = models(x_train_seq)
    
    loss = loss_function(y_pred, y_train_seq)
    loss.backward()
    optimizer.step()
    

    # 途中経過の表示
    if (epoch + 1) % 1 == 0:
        print(f'Epoch {epoch+1}/{epoch_num}, Loss: {loss.item():.4f}')


#======================================================================

# モデルの評価
models.eval()

# テストデータに対する予測
x_test_seq = x_test.unsqueeze(2)  # (batch_size, sequence_length, input_size)
predictions = models(x_test_seq)

# 予測結果を元のスケールに戻す
predictions = predictions.detach().numpy()
y_test = y_test.detach().numpy()

predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 結果をプロット
plt.figure(figsize=(10, 6))
plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

