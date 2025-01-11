# Import necessary libraries
import warnings
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
sns.set_style("ticks")
plt.figure(figsize=(15, 9))
plt.plot(data['year'], data[['Ind']])
ax = plt.gca()
ax.xaxis.set_major_locator(YearLocator(2))
ax.xaxis.set_major_formatter(DateFormatter("%Y"))
plt.title("Employment Population Over Years", fontsize=18, fontweight='bold')
plt.xlabel('Year', fontsize=18)
plt.ylabel('Employment Population', fontsize=18)
sns.despine()
plt.show()

# Extract and scale predictors
predictors = ['Agr_Output', 'Sec_Output', 'Thi_Output', 'Pop', 'Agr', 'Thi', 'Inv', 'Ene', 'Coal', 'Oil', 'Gas', 'Ren']
scaler_predicators = MinMaxScaler(feature_range=(-1, 1))
scaler  = MinMaxScaler(feature_range=(-1, 1))
data[predictors] = scaler_predicators.fit_transform(data[predictors])
data_filtered_2014 = data.loc[data['Ind'] > 1]
data_filtered_2014_train =  data.loc[data['year'].dt.year <=2009]
data_filtered_2014_val =  data.loc[(data['year'].dt.year >=2010) & (data['year'].dt.year <=2013)]
data_filtered_2014_train['Ind'] = scaler.fit_transform(data_filtered_2014_train['Ind'].values.reshape(-1, 1))
data_filtered_2014_val['Ind'] = scaler.transform(data_filtered_2014_val['Ind'].values.reshape(-1, 1))
data_filtered_2025 = data.loc[data['Ind'] <= 1]
data_filtered_2025['Ind'] = 0
data['Ind'] = np.concatenate((data_filtered_2014_train['Ind'],data_filtered_2014_val['Ind'], data_filtered_2025['Ind']), axis=0)



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
        self.dropout = nn.Dropout(p=0.3) # dropout正则化输出
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

# Model parameters
input_dim = 12
hidden_dim = 128
conv_input = 1
num_layers = 2
output_dim = 1
num_epochs = 4500
batch_size = 128

# Initialize model, loss, and optimizer
model = CNN_LSTM_Attention_Model(conv_input, input_dim, hidden_dim, num_layers, output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=0.0001) #lr=0.001 验证集R方0.67

# Training loop
train_losses, val_losses = [], []
hist = np.zeros(num_epochs)
hist2 = np.zeros(num_epochs)
start_time = time.time()
l1_lambda = 0.001

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    batch_size = min(batch_size, len(X_train))
    batch_indices = np.random.choice(len(X_train), batch_size, replace=False)  #replace=False--#batch_size = min(batch_size, len(X_train))
    X_batch = X_train[batch_indices]
    Y_batch = Y_train[batch_indices]
    output = model(X_batch)
    # Calculate the original loss
    train_loss = criterion(output, Y_batch)
    # Calculate L1 penalty (sum of absolute values of parameters)
    l1_penalty = sum(param.abs().sum() for param in model.parameters())
    # Combine the loss with L1 regularization
    total_loss = train_loss + l1_lambda * l1_penalty
    train_loss = criterion(output, Y_batch)
    train_loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, Y_val)
            hist[epoch] = val_loss.item()
            hist2[epoch] = train_loss.item()
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        print(f"Epoch {epoch}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}")

# Final evaluations
train_pred = model(X_train).detach().cpu().numpy()
val_pred = model(X_val).detach().cpu().numpy()
test_pred = model(X_test).detach().cpu().numpy()

#基于训练集+验证集+预测集的预测
#反归一化
pred_y=np.concatenate((train_pred,val_pred,test_pred))
pred_y=scaler.inverse_transform(pred_y).T[0]
true_y=np.concatenate((y_train, y_val, y_test))
true_y=scaler.inverse_transform(true_y).T[0]
# Plotting results
plt.title("LSTM model + CNN model + KAN model")
#x = [i for i in range(len(true_y))]
x = pd.date_range(start='1985', end='2023', freq='Y')
plt.plot(x, true_y, marker="x", markersize=1, label="true_y")
plt.plot(x, pred_y, marker="o", markersize=1, label="pred_y")
#一般化思路
#x_major_locator = MultipleLocator(2) # 把x轴的刻度间隔设置为2，并存在变量里
#ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为2的倍数
#ax = plt.gca() # ax为两条坐标轴的实例
#日期型设置
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(YearLocator(4))  # Major ticks every 2 years
ax.xaxis.set_major_formatter(DateFormatter("%Y"))  # Format ticks as years only
plt.axvline(pd.to_datetime('2014-01-01'), color='red', linestyle='--', label='Year 2014')
plt.axvline(pd.to_datetime('2010-01-01'), color='red', linestyle='--', label='Year 2010')
plt.legend()
plt.show()

ax = sns.lineplot(data=hist, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Valid Loss", size = 14, fontweight='bold')
plt.show()

ax = sns.lineplot(data=hist2, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Training Loss", size = 14, fontweight='bold')
plt.show()



#R方计算验证
pred_y_train_val =np.concatenate((train_pred,val_pred))
pred_y_train_val =scaler.inverse_transform(pred_y_train_val).T[0]
true_y_train_val=np.concatenate((y_train, y_val))
true_y_train_val=scaler.inverse_transform(true_y_train_val).T[0]

pred_y_train =scaler.inverse_transform(train_pred).T[0]
true_y_train =scaler.inverse_transform(y_train).T[0]

pred_y_val =scaler.inverse_transform(val_pred).T[0]
true_y_val=scaler.inverse_transform(y_val).T[0]

print('以下是验证集模型的误差')
print('R^2 Score:', r2_score(true_y_val, pred_y_val))
print('RMSE:', np.sqrt(mean_squared_error(true_y_val, pred_y_val)))
print('MAPE:', (abs(pred_y_val - true_y_val) / pred_y_val).mean())
print('以下是训练集集模型的误差')
print('R^2 Score:', r2_score(true_y_train, pred_y_train))
print('RMSE:', np.sqrt(mean_squared_error(true_y_train, pred_y_train)))
print('MAPE:', (abs(pred_y_train - true_y_train) / pred_y_train).mean())
print('以下是LSTM模型的误差')
print('R^2 Score:', r2_score(true_y_train_val, pred_y_train_val))
print('RMSE:', np.sqrt(mean_squared_error(true_y_train_val, pred_y_train_val)))
print('MAPE:', (abs(pred_y_train_val - true_y_train_val) / pred_y_train_val).mean())



#基于预测集的预测
# Final predictions and evaluations
# Load and preprocess the dataset
filepath_test = 'D:\\pycharm\\pytorch_learn\\dataset\\Ind_em_cn_true_value.csv'
data_test = pd.read_csv(filepath_test)
data_test['year'] = pd.to_datetime(data_test['year'], format='%Y')
true_test =  data_test.loc[data_test['year'].dt.year >=2014]
true_y_test = true_test['Ind']
pred_y_test=scaler.inverse_transform(test_pred).T[0]
#true_y_test=scaler.inverse_transform(y_test).T[0]

print('以下是测试集模型的误差')
print('R^2 Score:', r2_score(true_y_test, pred_y_test))
print('RMSE:', np.sqrt(mean_squared_error(true_y_test, pred_y_test)))
print('MAPE:', (abs(pred_y_test - true_y_test) / pred_y_test).mean())

# Plot true vs predicted values for the test set
plt.figure(figsize=(10, 6))
x0 = pd.date_range(start='2013', end='2023', freq='Y')
plt.plot(x0, true_y_test, label="True Values", color='blue', marker="x", markersize=5)
plt.plot(x0, pred_y_test, label="Predicted Values", color='orange', marker="o", markersize=5)
#一般化思路
#x_major_locator = MultipleLocator(2) # 把x轴的刻度间隔设置为2，并存在变量里
#ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为2的倍数
#ax = plt.gca() # ax为两条坐标轴的实例
#日期型设置
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(YearLocator(1))  # Major ticks every 2 years
ax.xaxis.set_major_formatter(DateFormatter("%Y"))  # Format ticks as years only
plt.title("True vs Predicted Values on Test Set")
plt.xlabel("Sample Index")
plt.ylabel("Original Values")
plt.legend()
plt.show()

# Export test results to a CSV file
output_df = pd.DataFrame({
    'True Values': true_y_test,
    'Predicted Values': pred_y_test
})
output_filepath = 'D:\\pycharm\\pytorch_learn\\dataset\\test_predictions.csv'
output_df.to_csv(output_filepath, index=False)
print(f"Test predictions exported to {output_filepath}")