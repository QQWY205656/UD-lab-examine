import torch
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision import transforms
import sys
import os

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
    transforms.ToTensor(),  #转化为张量Tensor
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  #归一化
])

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