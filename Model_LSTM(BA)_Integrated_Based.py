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


#可视化预测变量
sns.set_style("darkgrid")
plt.figure(figsize=(15, 9))
plt.plot(data['year'], data[['Ind']])
# Set major ticks to show each year and format them as year only
#一般化思路
#x_major_locator = MultipleLocator(2) # 把x轴的刻度间隔设置为2，并存在变量里
#ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为2的倍数
#ax = plt.gca() # ax为两条坐标轴的实例
#日期型设置
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(YearLocator(2))  # Major ticks every 2 years
ax.xaxis.set_major_formatter(DateFormatter("%Y"))  # Format ticks as years only
# Plot labels and title
plt.title("Employment Population Over Years", fontsize=18, fontweight='bold')
plt.xlabel('Year', fontsize=18)
plt.ylabel('Employment Population', fontsize=18)
plt.show()


# Extracting only the predictor variables and scaling them---测试分离归一化
predictors = ['Agr_Output','Sec_Output','Thi_Output','Pop','Agr','Thi','Inv','Ene','Coal','Oil','Gas','Ren']
# Initialize the scaler for normalization (to be used for both normalization and denormalization)
scaler = MinMaxScaler(feature_range=(-1, 1))
# Normalize predictors
data[predictors] = scaler.fit_transform(data[predictors])
data_filtered_2015 = data.loc[data['Ind'] > 1]
data_filtered_2015['Ind']=scaler.fit_transform(data_filtered_2015['Ind'].values.reshape(-1,1))
data_filtered_2025 = data.loc[data['Ind'] <= 1]
data_filtered_2025['Ind'] = 0
data['Ind'] = np.concatenate((data_filtered_2015['Ind'],data_filtered_2025['Ind']), axis=0)


#分离数据
def split_data(stock, lookback, shuffle=True, train_pct=0.63, val_pct=0.1):
    data_raw = stock.to_numpy()
    data = []

    # Create a sequence of data based on the backtracking period
    for index in range(len(data_raw) - lookback + 1):  # Include the last sequence exactly
        data.append(data_raw[index: index + lookback])

    data = np.array(data)

    # Split into training, validation, and test sets
    train_size = int(np.round(train_pct * data.shape[0]))
    val_size = int(np.round(val_pct * data.shape[0]))

    # Training and validation data (do not shuffle these yet)
    x_train = data[:train_size, :-1, 1:]
    y_train = data[:train_size, -1, 0:1]
    x_val = data[train_size:train_size + val_size, :-1, 1:]
    y_val = data[train_size:train_size + val_size, -1, 0:1]

    # Test data (ensure it includes the last sequence)
    x_test = data[train_size + val_size:, :-1, 1:]
    y_test = data[train_size + val_size:, -1, 0:1]

    # Shuffle the training set if shuffle=True
    if shuffle:
        train_indices = [i for i in range(len(x_train))]
        np.random.shuffle(train_indices)
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]

    # If the test set is empty or the last year is missing, add it manually
    if x_test.shape[0] == 0 or stock.iloc[-1, 0] not in y_test:
        x_test = np.concatenate((x_test, [data[-1, :-1, 1:]]), axis=0)
        y_test = np.concatenate((y_test, [data[-1, -1, 0:1]]), axis=0)

    return [x_train, y_train, x_val, y_val, x_test, y_test]



lookback = 2
price = data[['Ind','Agr_Output','Sec_Output','Thi_Output','Pop','Agr','Thi','Inv','Ene','Coal','Oil','Gas','Ren']]
# Shuffle while maintaining original dataset distribution
x_train, y_train, x_val, y_val, x_test, y_test = split_data(price, lookback, shuffle=False, train_pct=0.63, val_pct=0.1)
print(x_train.__len__(),x_val.__len__(),x_test.__len__())


# Convert to tensors
X_train, Y_train = torch.from_numpy(x_train).type(torch.Tensor), torch.from_numpy(y_train).type(torch.Tensor)
X_val, Y_val = torch.from_numpy(x_val).type(torch.Tensor), torch.from_numpy(y_val).type(torch.Tensor)
X_test, Y_test = torch.from_numpy(x_test).type(torch.Tensor), torch.from_numpy(y_test).type(torch.Tensor)


# Define CNN-LSTM-Attention model
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


class CNN_LSTM_Attention_Model(nn.Module):
    def __init__(self, conv_input, input_dim, hidden_dim, num_layers, output_dim):
        super(CNN_LSTM_Attention_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv = nn.Conv1d(conv_input, conv_input, 1)
        self.conv_activation = nn.LeakyReLU(negative_slope=0.01)  #卷积层的激活函数
        #self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) #非双向LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.attention_activation = nn.Tanh()  #注意力的激活函数
        #self.fc = nn.Linear(hidden_dim , output_dim)  #非双向LSTM
        #self.fc = nn.Linear(hidden_dim * 2, output_dim) # 双向LSTM两层
        self.fc = KAN([hidden_dim * 2, output_dim])  #KAN设定函数-全连接-双向LSTM两层
        #self.fc = KAN([hidden_dim, output_dim])  # KAN设定函数-全连接-非双向LSTM两层
        self.dropout = nn.Dropout(p=0.3)
        #self.output_activation = nn.Identity() #输出的激活函数

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_activation(x)  #卷积层的激活函数
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_() # 非双向LSTM
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_() # 非双向LSTM
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_() # 双向LSTM两层
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_() # 双向LSTM两层
        lstm_outputs, _ = self.lstm(x, (h0.detach(), c0.detach()))
        # 通过注意力机制传递 LSTM 输出，计算加权输出
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
conv_input = 1 #和步长一致(如果训练维度减一，则卷积维度减一）
num_layers = 2
output_dim = 1
num_epochs = 4500 #650/800-50 epoch；4500-100 epoch
batch_size = 128

# Initialize model
model = CNN_LSTM_Attention_Model(conv_input, input_dim, hidden_dim, num_layers, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
criterion = nn.MSELoss()

# Training loop
train_losses = []
val_losses = []
hist = np.zeros(num_epochs)
hist2 = np.zeros(num_epochs)
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    batch_size = min(batch_size, len(X_train))
    batch_indices = np.random.choice(len(X_train), batch_size, replace=False) #replace=False与#batch_size = min对应
    X_batch = X_train[batch_indices]
    Y_batch = Y_train[batch_indices]
    output = model(X_batch)
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
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"epoch:{epoch}, train_loss:{train_loss}, val_loss:{val_loss}")

# Final predictions and evaluations
train_pred = model(X_train).detach().numpy()
val_pred = model(X_val).detach().numpy()
test_pred = model(X_test).detach().numpy()
print(X_train.__len__(), X_val.__len__(), X_test.__len__())

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
pred_test_y=scaler.inverse_transform(test_pred).T[0]
true_test_y=scaler.inverse_transform(y_test).T[0]

# Plot true vs predicted values for the test set
plt.figure(figsize=(10, 6))
x0 = pd.date_range(start='2013', end='2023', freq='Y')
plt.plot(x0, true_test_y, label="True Values", color='blue', marker="x", markersize=5)
plt.plot(x0, pred_test_y, label="Predicted Values", color='orange', marker="o", markersize=5)
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
    'True Values': true_test_y,
    'Predicted Values': pred_test_y
})
output_filepath = 'D:\\pycharm\\pytorch_learn\\dataset\\test_predictions.csv'
output_df.to_csv(output_filepath, index=False)
print(f"Test predictions exported to {output_filepath}")