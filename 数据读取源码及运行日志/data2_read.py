import pandas as pd
import numpy as np
import torch as torch

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
print(x[:5])
print(y[:5])


