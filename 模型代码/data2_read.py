import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 

df = pd.read_csv("E:/各类文档/深度视觉寒假培训/task2.csv")
#提取样本
x = df["x"].values
y = df["y"].values
#调整维度
x = x.reshape(2000,-1)
y = y.reshape(2000,-1)
#转化为pytorch张量
device = torch.device("cuda")
x_tensor = torch.tensor(x,dtype = torch.float32).to(device)
y_tensor = torch.tensor(y,dtype = torch.float32).to(device)
#验证是否分离成功
#print(x[:5])
#print(y[:5])

#数据可视化，画散点图
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False 
#plt.figure(figsize = (16,12),dpi = 200)
#plt.scatter(x,y,s = 5,alpha = 0.5,color = "#1f77b4")
#plt.grid(True,linestyle = "--",alpha = 0.6)
#plt.xticks(fontsize = 12)
#plt.yticks(fontsize = 12)
#plt.title("Scatter Plot of x vs y")
#plt.xlabel("x")
#plt.ylabel("y")
#plt.tight_layout()
#plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader

#数据标准化，标准化器为StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()
#划分训练集和测试集
#训练集占80%，测试集占20%，随机种子为42
x_train, x_test, y_train, y_test = train_test_split(scaler_x.fit_transform(x), scaler_y.fit_transform(y), test_size = 0.2, random_state = 42)
#把数据转化为CUDA张量
x_train = torch.tensor(x_train,dtype = torch.float32).to(device)
y_train = torch.tensor(y_train,dtype = torch.float32).to(device)
x_test = torch.tensor(x_test,dtype = torch.float32).to(device)
y_test = torch.tensor(y_test,dtype = torch.float32).to(device)
#为了增强效果，使用数据增强，将y>0的样本复制一份加入训练集中
mask_pos = (y_train > 0).reshape(-1)
x_pos = x_train[mask_pos]
y_pos = y_train[mask_pos]
#将正样本添加到训练集中
x_train_aug = torch.cat([x_train, x_pos], dim=0)
y_train_aug = torch.cat([y_train, y_pos], dim=0)
#创建TensorDataset和DataLoader
train_dataset = TensorDataset(x_train_aug,y_train_aug)
test_dataset = TensorDataset(x_test,y_test)
#这里TensorDataset将x_train和y_train转换为张量格式，再组合成数据集
#x_train是特征张量，y_train是标签张量
train_loader = DataLoader(train_dataset,batch_size = 64,shuffle = True)
test_loader = DataLoader(test_dataset,batch_size = 64,shuffle = False)

#定义单隐藏层MLP模型
class TwoHiddenMLP(nn.Module):
    def __init__(self,input_dim = 1,hidden_dim = 512,output_dim = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.leaky_relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.leaky_relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_dim,output_dim)
        
    def forward(self,x):
        #输入层进入隐藏层进行线形变换
        out = self.fc1(x)
        #经过LeakyReLU激活函数
        out = self.leaky_relu1(out)
        out = self.fc2(out)
        out = self.leaky_relu2(out) 
        out = self.fc3(out)
        #输出当前结果
        return out 
    
    

#训练参数设置
#输入x的维度为1（input_dim = 1），隐藏层神经元数量为512（hidden_dim = 512），输出y的维度为1（output_dim = 1）
model = TwoHiddenMLP(input_dim = 1, hidden_dim = 512, output_dim = 1).to(device)
#这是一个回归任务，选用均方误差损失函数
criterion = nn.MSELoss()
#优化器选用Adam。学习率为5e-4，权重衰减为1e-4   
#学习率调度器，每800轮学习率衰减为0.9倍，当前为1.0  
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.00001,weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.9)
#训练轮数
epochs = 3000

#切换为训练模式
train_losses = []
test_lossses = []

#训练循环
for epoch in range(epochs):
    #初始化当前轮次的总损失
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:     #内层循环
        optimizer.zero_grad()          #优化器梯度清零，避免与上一轮梯度；累积
        outputs = model(inputs)    #将批次数据传入模型训练，前向传播，计算模型的预测输出
        loss = criterion(outputs, targets)  #利用损失函数计算预测值和标签值的损失
        loss.backward()      #反向传播，计算参数的梯度
        optimizer.step()        #根据梯度更新模型参数
        train_loss += loss.item() * inputs.size(0)  #累计当前批次的损失值，乘以批次大小（nputs.size），得到当前批次的总损失
    #计算每一轮的平均训练损失，总损失除以训练集样本数量
    train_loss_avg = train_loss / len(train_loader.dataset)
    #把平均损失存入列表，以便后续绘图分析
    train_losses.append(train_loss_avg)    

#切换为评估模式，用测试集进行评估
    model.eval()
    #初始化测试集损失
    test_loss = 0.0
    #禁用梯度计算，在评估模式下不需要计算梯度，节省内存和计算资源
    with torch.no_grad():
        for inputs, targets in test_loader:   #内层循环，遍历测试集数据
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
    test_loss_avg = test_loss / len(test_loader.dataset)
    test_lossses.append(test_loss_avg)   #把当前轮次的测试损失加入列表，后续分析泛化能力
    
    scheduler.step()  
    #每100轮打印一次训练和测试损失的日志
    if (epoch + 1) % 200 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss_avg:.6f}, Test Loss: {test_loss_avg:.6f}") 

#绘制拟合曲线
model.eval()
with torch.no_grad():
    #在原始输入数据x的范围内生成2000个均匀分布的点，作为预测的输入区间
    x_range = torch.linspace(x.min(),x.max(),2000).reshape(-1,1).to(device)
    #用数据标准化器StandardScaler对生成的区间数据标准化
    x_range_scaled = scaler_x.transform(x_range.cpu().numpy())
    #把标准化后的数据转化为pytorch张量
    x_range_tensor = torch.tensor(x_range_scaled,dtype = torch.float32).to(device)
    #用模型将标准化的数据进行预测，得到预测值
    y_pred_scaled = model(x_range_tensor)
    #用scaler_y将预测值反标准化，转换回原始尺度
    y_pred = scaler_y.inverse_transform(y_pred_scaled.cpu().numpy())

#拟合曲线
plt.figure(figsize = (16,12),dpi = 200)
plt.scatter(x,y,s = 5,alpha = 0.5,color = "#1f77b4",label = "原始数据")
plt.plot(x_range.cpu().numpy(),y_pred,color = "red",linewidth = 2,label = "MLP拟合曲线")
plt.grid(True,linestyle = "--",alpha = 0.6)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.title("原始尺度下的MLP拟合曲线",fontsize = 16)
plt.xlabel("x",fontsize = 14)
plt.ylabel("y",fontsize = 14)
plt.legend(fontsize = 12)
plt.tight_layout()
plt.show()
#损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='训练损失')
plt.plot(test_lossses, label='测试损失')
plt.title('训练和测试损失曲线')
plt.xlabel('轮数')
plt.ylabel('损失')
plt.legend()
plt.grid(True)
plt.show()