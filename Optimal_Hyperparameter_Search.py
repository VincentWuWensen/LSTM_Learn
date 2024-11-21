# Import necessary libraries
import warnings  # To suppress unnecessary warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch import optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from efficient_kan.src.efficient_kan import KAN
import time
import random
import os
from matplotlib.ticker import MultipleLocator
from datetime import datetime
from sklearn.model_selection import ParameterGrid  # Import for hyperparameter search

# Set device for GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Set random seeds
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load and preprocess the dataset
filepath = 'D:\\pycharm\\pytorch_learn\\dataset\\archive\\employment23_cn.csv'
data = pd.read_csv(filepath)
data = data.sort_values('year')
data['year'] = pd.to_datetime(data['year'], format='%Y')

# Visualize the target variable
sns.set_style("darkgrid")
plt.figure(figsize=(15, 9))
plt.plot(data['year'], data[['Ind']])
ax = plt.gca()
ax.xaxis.set_major_locator(YearLocator(2))
ax.xaxis.set_major_formatter(DateFormatter("%Y"))
plt.title("Employment Population Over Years", fontsize=18, fontweight='bold')
plt.xlabel('Year', fontsize=18)
plt.ylabel('Employment Population', fontsize=18)
plt.show()

# Extract and scale predictors
predictors = ['Agr_Output', 'Sec_Output', 'Thi_Output', 'Pop', 'Agr', 'Thi', 'Inv', 'Ene', 'Coal', 'Oil', 'Gas', 'Ren']
scaler = MinMaxScaler(feature_range=(-1, 1))
data[predictors] = scaler.fit_transform(data[predictors])
data_filtered_2015 = data.loc[data['Ind'] > 1]
data_filtered_2015['Ind'] = scaler.fit_transform(data_filtered_2015['Ind'].values.reshape(-1, 1))
data_filtered_2025 = data.loc[data['Ind'] <= 1]
data_filtered_2025['Ind'] = 0
data['Ind'] = np.concatenate((data_filtered_2015['Ind'], data_filtered_2025['Ind']), axis=0)

# Split data into training, validation, and test sets
def split_data(stock, lookback, shuffle=True, train_pct=0.63, val_pct=0.1):
    data_raw = stock.to_numpy()
    data = []
    for index in range(len(data_raw) - lookback + 1):
        data.append(data_raw[index: index + lookback])
    data = np.array(data)
    train_size = int(np.round(train_pct * data.shape[0]))
    val_size = int(np.round(val_pct * data.shape[0]))
    x_train = data[:train_size, :-1, 1:]
    y_train = data[:train_size, -1, 0:1]
    x_val = data[train_size:train_size + val_size, :-1, 1:]
    y_val = data[train_size:train_size + val_size, -1, 0:1]
    x_test = data[train_size + val_size:, :-1, 1:]
    y_test = data[train_size + val_size:, -1, 0:1]
    if shuffle:
        train_indices = np.arange(len(x_train))
        np.random.shuffle(train_indices)
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]
    if x_test.shape[0] == 0 or stock.iloc[-1, 0] not in y_test:
        x_test = np.concatenate((x_test, [data[-1, :-1, 1:]]), axis=0)
        y_test = np.concatenate((y_test, [data[-1, -1, 0:1]]), axis=0)
    return [x_train, y_train, x_val, y_val, x_test, y_test]

lookback = 2
price = data[['Ind', 'Agr_Output', 'Sec_Output', 'Thi_Output', 'Pop', 'Agr', 'Thi', 'Inv', 'Ene', 'Coal', 'Oil', 'Gas', 'Ren']]
x_train, y_train, x_val, y_val, x_test, y_test = split_data(price, lookback, shuffle=False)

# Convert to tensors and move to device
X_train = torch.from_numpy(x_train).type(torch.Tensor).to(device)
Y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)
X_val = torch.from_numpy(x_val).type(torch.Tensor).to(device)
Y_val = torch.from_numpy(y_val).type(torch.Tensor).to(device)
X_test = torch.from_numpy(x_test).type(torch.Tensor).to(device)
Y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)

# Define Attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # 线性层将 LSTM 输出转换为新的注意力评分表示法
        #self.attention = nn.Linear(hidden_dim , hidden_dim )  #非双向LSTM
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2) # 双向LSTM两层
        # 线性层为每个时间步长创建一个分数（权重)
        #self.context_vector = nn.Linear(hidden_dim , 1, bias=False) #非双向LSTM
        self.context_vector = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, lstm_outputs):
        # 第 1 步：转换 LSTM 输出并应用 tanh 来处理非线性问题
        attention_weights = torch.tanh(self.attention(lstm_outputs))
        # 第 2 步：计算非规范化注意力分数，然后挤压以去除单子维度
        attention_weights = self.context_vector(attention_weights).squeeze(-1)
        # 第 3 步：使用 softmax 对分数进行归一化处理，以获得各时间步长总和为 1 的注意力权重
        attention_weights = torch.softmax(attention_weights, dim=1)
        # 第 4 步： 将注意力权重应用于 LSTM 输出，在各时间步长内对加权输出求和
        weighted_output = torch.sum(lstm_outputs * attention_weights.unsqueeze(-1), dim=1)
        return weighted_output, attention_weights

# Define CNN-LSTM-Attention model
class CNN_LSTM_Attention_Model(nn.Module):
    def __init__(self, conv_input, input_dim, hidden_dim, num_layers, output_dim):
        super(CNN_LSTM_Attention_Model, self).__init__()
        self.conv = nn.Conv1d(conv_input, conv_input, 1) #卷积层的激活函数
        self.conv_activation = nn.LeakyReLU(0.01)
        #self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) #非双向LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.attention_activation = nn.Tanh()  #注意力的激活函数
        #self.fc = nn.Linear(hidden_dim , output_dim)  #非双向LSTM
        #self.fc = nn.Linear(hidden_dim * 2, output_dim) # 双向LSTM两层
        self.fc = KAN([hidden_dim * 2, output_dim]) #KAN设定函数-全连接-双向LSTM两层
        #self.fc = KAN([hidden_dim, output_dim])  # KAN设定函数-全连接-非双向LSTM两层
        self.dropout = nn.Dropout(p=0.3)
        #self.output_activation = nn.Identity() #输出的激活函数

    def forward(self, x):
        x = self.conv(x) #卷积层的激活函数
        x = self.conv_activation(x)
        #h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device) # 非双向LSTM
        #c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device) # 非双向LSTM
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(device) # 双向LSTM两层
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(device) # 双向LSTM两层
        lstm_outputs, _ = self.lstm(x, (h0, c0))         # 通过注意力机制传递 LSTM 输出，计算加权输出
        attn_output, _ = self.attention(lstm_outputs)
        attn_output = self.attention_activation(attn_output)  #注意力的激活函数
        #将 dropout 应用于注意力加权输出，以实现正则化
        #out = self.fc(attn_output) #无dropout正则化输出
        out = self.dropout(attn_output) # dropout正则化输出
        out = self.fc(out) # dropout正则化输出
        #out = self.output_activation(out)  #输出的激活函数
        return out


input_dim = 12
output_dim = 1
conv_input = 1
num_epochs = 4500
# Define hyperparameter grid for search
param_grid = {
    'hidden_dim': [64, 128, 256],
    'num_layers': [1, 2],
    'lr': [0.0001, 0.01],
    'batch_size': [64, 128, 256]
}

# Perform grid search over the hyperparameters
best_model = None
best_params = None
best_val_loss = float('inf')

for params in ParameterGrid(param_grid):
    print(f"Training model with params: {params}")

    # Initialize model with current hyperparameters
    model = CNN_LSTM_Attention_Model(conv_input, input_dim,
                                     params['hidden_dim'], params['num_layers'], output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], betas=(0.5, 0.999), weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Training loop
    train_losses = []
    val_losses = []
    hist = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Ensure batch_size does not exceed the number of training samples
        batch_size = min(params['batch_size'], len(X_train))

        # Randomly sample a batch from the training set
        batch_indices = np.random.choice(len(X_train), batch_size, replace=False)  # Sampling without replacement
        X_batch = X_train[batch_indices]
        Y_batch = Y_train[batch_indices]

        # Forward pass
        output = model(X_batch)
        train_loss = criterion(output, Y_batch)

        # Backward pass and optimization
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)  # Use X_val directly on GPU
            val_loss = criterion(val_outputs, Y_val)
            val_losses.append(val_loss.item())

        hist[epoch] = train_loss.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}")

    # Check if current model is the best
    if np.min(val_losses) < best_val_loss:
        best_val_loss = np.min(val_losses)
        best_model = model
        best_params = params

# Use best model and parameters
print(f"Best parameters found: {best_params}")



