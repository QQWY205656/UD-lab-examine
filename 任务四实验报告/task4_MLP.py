import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision import transforms
import sys
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from thop import profile

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
#单进程,避免在win上运行出错
num_workers = 0 if os.name == 'nt' else 2

#屏蔽下载时冗余打印
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
        sys.stderr = open(os.devnull, 'w', encoding='utf-8')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

#设置训练集：只保留对小模型有效的增强，移除冗余的旋转/色彩抖动
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # 轻量擦除
])

#测试集：仅标准化
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
])

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#mixup数据增强
def mixup_data(x, y, alpha=0.1, use_cuda=True):
    """alpha=0.1更适合小模型，避免增强过度"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    device = torch.device("cuda")
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam   

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#轻量化，高性能MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        #1x1卷积+批归一化（提升特征提取）
        self.input_compress = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=1, stride=1, padding=0),  # 3→2通道，比3→1更保留信息
            nn.BatchNorm2d(2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        #小维度+批归一化+残差连接（关键提升准确率）
        self.fc1 = nn.Linear(32*32*2, 192)  # 2048→192，参数量仅0.38M
        self.bn1 = nn.BatchNorm1d(192)
        self.fc2 = nn.Linear(192, 96)
        self.bn2 = nn.BatchNorm1d(96)
        self.fc3 = nn.Linear(96, 10)
        #加入残差连接：弥补小模型的特征损失
        self.residual = nn.Linear(192, 96)
        #轻量正则化，低dropout+inplace激活（减少计算）
        self.dropout = nn.Dropout(0.2)  #0.2比0.3更适合小模型
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        #输入压缩
        x = self.input_compress(x)
        x = x.view(x.size(0), -1)  #展平：32*32*2=2048
        
        #主干网络+残差连接
        x1 = self.act(self.bn1(self.fc1(x)))
        x1 = self.dropout(x1)
        x2 = self.act(self.bn2(self.fc2(x1)))
        #残差，x1→96维，和x2相加
        x2 = x2 + self.residual(x1)  #残差连接提升特征复用
        x2 = self.dropout(x2)
        
        out = self.fc3(x2)
        return out

def train_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.1):
    model.train()
    total_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        #Mixup数据增强
        inputs, y_a, y_b, lam = mixup_data(inputs, labels, mixup_alpha, device.type=="cuda")
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        #梯度裁剪，防止小模型梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

#下载并训练
if __name__ == "__main__":
    device = torch.device("cuda")
    #加载数据集
    with HiddenPrints():
        train_dataset = datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform_test)
    print(f"训练集: {len(train_dataset)} 样本 | 测试集: {len(test_dataset)} 样本")

    #数据加载器，单进程+合理批次
    batch_size = 96  #64→96，平衡显存和训练稳定性
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    #初始化模型
    model = MLP().to(device)
    #损失函数，标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    #优化器，SGD+Nesterov
    lr = 0.1
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-5,nesterov=True)

    #学习率调度，精准预热+余弦退火
    num_epochs = 40  
    warmup_epochs = 5
    scheduler_warmup = LinearLR(optimizer, start_factor=0.05, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, [scheduler_warmup, scheduler_cosine], [warmup_epochs])

    #训练循环
    train_losses = []
    test_losses = []
    test_accs = []
    best_acc = 0.0
    print("\n开始训练（轻量化+高准确率）...")
    for epoch in range(num_epochs):
        #训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        #评估
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        #更新学习率
        scheduler.step()
        
        #记录结果
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        #保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "lightweight_high_acc_mlp.pth")

        #打印日志
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.2f}% | "
              f"LR: {current_lr:.6f} | "
              f"Best Acc: {best_acc:.2f}%")

    print(f"\n训练完成！最佳测试准确率: {best_acc:.2f}%")

    #可视化结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    #损失曲线
    ax1.plot(range(1, num_epochs+1), test_losses, label='测试损失', marker='s')
    ax1.set_title("测试损失曲线")
    ax1.set_xlabel('轮数')
    ax1.set_ylabel('损失')
    ax1.grid(True)
    ax1.legend()
    #准确率曲线
    ax2.plot(range(1, num_epochs+1), test_accs, label='测试准确率', marker='o', color='orange')
    ax2.set_title("测试集准确率曲线")
    ax2.set_xlabel('轮数')
    ax2.set_ylabel('准确率(%)')
    ax2.grid(True)
    ax2.legend()
    plt.suptitle("MLP模型训练损失曲线和准确率曲线")
    plt.show()

    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"\n模型轻量化指标：")
    print(f"参数量：{params/1e6:.4f} M ")
    print(f"计算量：{macs/1e9:.4f} G ")
    print(f"最佳准确率：{best_acc:.2f}% ")