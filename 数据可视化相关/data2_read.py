import pandas as pd
import numpy as np
import torch 

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
plt.figure(figsize = (16,12),dpi = 200)
plt.scatter(x,y,s = 5,alpha = 0.5,color = "#1f77b4")
plt.grid(True,linestyle = "--",alpha = 0.6)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.title("Scatter Plot of x vs y")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

