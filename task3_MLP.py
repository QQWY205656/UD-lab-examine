import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision import transforms
import sys
import os
import numpy as np

#设置微软雅黑中文字体，解决-号显示为方块的问题
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
#取消密集打印下载进度
class HiddenPrints:  #屏蔽掉后文的所有输出
    def __enter__(self):    #enter：保存原水的标准输出和错误输出，定向到空设备丢弃
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        #stdout标准输出，stderr标准错误。
        #os.devnull表示系统的空设备，写入的数据会被丢弃
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')  #以写入模式（'w'）打开空设备，指定编码utf-8
        sys.stderr = open(os.devnull, 'w', encoding='utf-8')

    def __exit__(self, exc_type, exc_val, exc_tb):  #exit：关闭定向文件，恢复标准输出和错误输出
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

#数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  #把PIL图像转化为张量Tensor
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  #归一化  
])                 #mean：三原色通道都分别减去均值0.5    std：三原色通道都分别除以标准差0.5  归一化将像素值从[0,1]映射到[-1,1]，数据中心移到0
                   #归一化作用：加速模型收敛，提升模型稳定性
#下载加载数据集，加入预处理
print(f"开始下载")
with HiddenPrints():
    train_dataset = datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)  #train=True加载训练集
    test_dataset = datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)  #train=False加载测试集
print(f"下载完成")

print("训练集样本数：", len(train_dataset))
print("测试集样本数：", len(test_dataset))
print(f"样本类别列表：{train_dataset.classes}")  

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
fig = plt.figure(figsize=(10, 5))
num_classes = 10

"""
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
    ax.set_title(class_names[i])
    img = next(img for img, label in train_dataset if label == i)
    #反归一化,  
    img = img / 2 + 0.5 
    #调整维度顺序，适应plt.imshow 
    plt.imshow(img.permute(1, 2, 0))
plt.suptitle("CIFAR10.png")  
plt.show()
"""


#创建Dataloader
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

#检查数据维度
data_iter = iter(train_loader)
images, labels = next(data_iter)
print(f"图像张量维度: {images.shape}")
print(f"标签张量维度: {labels.shape}")

#搭建模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(32*32*3,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,10)

    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
device = torch.device("cuda")
model = MLP().to(device)   #将模型移动到GPU
#定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
#打印模型结构
print(model)

#开始训练循环
train_losses = []
test_accuracies = []
for epochs in range(num_epochs):
    #模型进入训练阶段
    model.train()
    epoch_loss = 0.0
    for i,(inputs,labels) in enumerate(train_loader):
        #将输入移动到GPU
        inputs,labels = inputs.to(device),labels.to(device)
        #梯度清零
        optimizer.zero_grad()
        #正向传播
        outputs = model(inputs)
        #计算损失
        loss = criterion(outputs,labels)
        #反向传播
        loss.backward()
        #更新权重
        optimizer.step()
        #累积损失
        epoch_loss += loss.item()
        #每100个批次打印一次损失
        if (i + 1) % 100 == 0:
            print(f"轮数 [{epochs+1}/{num_epochs}], 批次：[{i+1}/{len(train_loader)}], 损失: {loss.item():.4f}")
           
    #记录平均损失
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    #模型评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"轮数 [{epochs+1}/{num_epochs}], 测试集准确率: {accuracy:.2f}%")
print("训练结束")

#画训练损失曲线和测试集准确率曲线
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='训练损失', marker='o')
plt.title("训练损失曲线")
plt.xlabel('轮数')
plt.ylabel('损失')
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='测试集准确率', marker='o', color='orange')
plt.title("测试集准确率曲线")
plt.xlabel('轮数')
plt.ylabel('准确率')
plt.grid(True)
plt.legend()

plt.suptitle("MLP模型训练损失和测试集准确率")
plt.show()

#展示分类错误和正确的图片
correct_images = []
incorrect_images = []
correct_labels = []
incorrect_labels = []
incorrect_preds = []

model.eval()
with torch.no_grad():
    for images,labels in test_loader:
        if len(correct_images) >= 5 and len(incorrect_images) >= 5:
            break
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)

        #筛选正确和错误的预测
        correct_mask = (predicted == labels)
        incorrect_mask = ~correct_mask

        #分别收集5个正确和错误的分类样本
        for i in range(images.size(0)):
            if correct_mask[i] and len(correct_images) < 5:
                correct_images.append(images[i].cpu())
                correct_labels.append(labels[i].cpu())
            elif incorrect_mask[i] and len(incorrect_images) < 5:
                incorrect_images.append(images[i].cpu())
                incorrect_labels.append(labels[i].cpu())
                incorrect_preds.append(predicted[i].cpu())
        
# 创建一个2x5的子图网格
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
fig.suptitle("分类结果示例", fontsize=16)

# 反归一化并显示图片的函数
def imshow_result(ax, img, title):
    img = img / 2 + 0.5  # 反归一化 (mean=0.5, std=0.5)
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_title(title, fontsize=9)
    ax.axis('off')

# 可视化正确分类的图片，放在第一行
for i in range(5):
    ax = axes[0, i]
    if i < len(correct_images):
        title = f"真实: {class_names[correct_labels[i]]}"
        imshow_result(ax, correct_images[i], title)
    else:
        ax.axis('off') # 如果没有足够的图片，则隐藏坐标轴
axes[0, 0].text(-0.3, 0.5, '正确分类', transform=axes[0, 0].transAxes,
                fontsize=14, va='center', ha='right', rotation='vertical')


# 可视化错误分类的图片，放在第二行
for i in range(5):
    ax = axes[1, i]
    if i < len(incorrect_images):
        title = f"真实: {class_names[incorrect_labels[i]]}\n预测: {class_names[incorrect_preds[i]]}"
        imshow_result(ax, incorrect_images[i], title)
    else:
        ax.axis('off')
axes[1, 0].text(-0.3, 0.5, '错误分类', transform=axes[1, 0].transAxes,
                fontsize=14, va='center', ha='right', rotation='vertical')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
