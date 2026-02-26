import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from thop import profile
import platform  
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR,SequentialLR,LinearLR
from torchvision import datasets,transforms

#设置中文字体
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

#VGG配置字典
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

#定义squeeze-and-excitation模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.5), #正则化
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:  
                #每两个3*3卷积为一组，替换为一个深度可分离卷积模块
                out_channels = x
                #深度可分离卷积模块
                layers.extend([
                    #深度卷积 Depthwise Convolution
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    #逐点卷积 Pointwise Convolution
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ])
                #把SE模块放到深度可分离卷积后，让通道注意力更精细
                layers.append(SELayer(out_channels))
                in_channels = out_channels
        

        return nn.Sequential(*layers)

#定义mixup数据增强函数
def mixup_data(x,y,alpha = 1.0):
    if alpha > 0:
        lam = np.random.beta(alpha,alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_dataloaders(num_workers=0):
    """封装数据加载逻辑的函数"""
    print("数据加载预处理")
    #训练数据增强和归一化
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    #加载CIFAR10数据集
    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transforms_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )

    testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transforms_test)
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=100, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return trainloader, testloader

#反归一化并显示图片的函数（抽离成全局函数，方便复用）
def imshow_result(ax, img, title):
    #CIFAR10的均值和标准差
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    img = img * std + mean  #反归一化
    img = torch.clamp(img, 0, 1)  #确保像素值在[0, 1]范围内
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_title(title, fontsize=9)
    ax.axis('off')

if __name__ == '__main__':
    #根据系统设置num_workers（Windows下设为0避免报错）
    num_workers = 0 if platform.system() == "Windows" else 2
    #调用数据加载函数
    trainloader, testloader = get_dataloaders(num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    #初始化模型
    VGG13 = VGG('VGG13')
    VGG13 = VGG13.to(device)

    #初始化权重
    for m in VGG13.modules():
        if isinstance(m, nn.Conv2d):  #卷积层参数
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):  #全连接层参数
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    #定义损失函数和优化器
    criterion = nn.CrossEntropyLoss() 
    LR = 0.1
    optimizer = optim.SGD(VGG13.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    #加入Warmup学习率预热
    #设置预热参数
    warmup_epochs = 5
    num_epochs = 20
    #创建带有预热的学习率调度器
    #主调度器
    main_scheduler = CosineAnnealingLR(optimizer, T_max = num_epochs - warmup_epochs)
    #预热调度器
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    #组合调度器
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    #加入早停机制
    #参数
    patience = 3
    best_accuracy = 0.0
    epochs_no_improve = 0
    best_model_state = None

    #训练模型
    epoch_losses = []
    test_accuracies = []
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for epoch in range(num_epochs):
        #训练部分
        VGG13.train()
        running_loss = 0.0
        with tqdm(trainloader, desc=f'Epoch {epoch + 1}/{num_epochs}[训练]', unit='batch') as tepoch:
            for i, data in enumerate(tepoch, 0):
                #获取训练数据
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                #mixup数据增强
                inputs, y_a, y_b, lam = mixup_data(inputs, labels, alpha=1.0)

                #权重参数梯度清零
                optimizer.zero_grad()

                #正向及反向传播
                outputs = VGG13(inputs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                loss.backward()
                optimizer.step()

                #累加损失值
                running_loss += loss.item()

                #实时更新进度条的损失信息
                tepoch.set_postfix(
                    avg_loss=running_loss / (i + 1)  #当前epoch的平均损失
                )
        
        #更新学习率
        scheduler.step()
        
        #计算并打印该轮 epoch 的平均损失
        epoch_avg_loss = running_loss / len(trainloader)
        epoch_losses.append(epoch_avg_loss)
        print(f'Epoch {epoch + 1} 平均损失: {epoch_avg_loss:.4f}')

        #测试部分   
        VGG13.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            with tqdm(testloader, desc=f'Epoch {epoch + 1}/{num_epochs}[测试]', unit='batch') as tepoch:
                for data in tepoch:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = VGG13(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)  #返回最大值和其索引
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        epoch_accuracy = correct / total
        test_accuracies.append(epoch_accuracy * 100)
        avg_test_loss = test_loss / len(testloader)
        print(f'Epoch {epoch + 1} 测试集准确率: {epoch_accuracy * 100:.2f}%, 测试损失: {avg_test_loss:.4f}')
    
        #早停机制判断
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            epochs_no_improve = 0
            best_model_state = VGG13.state_dict()  #保存当前最优模型参数
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'连续 {patience} 轮没有提升，早停训练。')
                break

    print("训练完成。")

    #加载最优模型参数
    if best_model_state:
        VGG13.load_state_dict(best_model_state)
        print(f'加载最优模型参数，最佳测试集准确率为: {best_accuracy * 100:.2f}%')    

    #画训练损失曲线和测试集准确率曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    actual_epochs = len(epoch_losses)  #实际训练的轮数,可能早停少于num_epochs
    ax1.plot(range(1, actual_epochs+1), epoch_losses, label='训练损失', marker='o')
    ax1.set_title("训练损失曲线")
    ax1.set_xlabel('轮数')
    ax1.set_ylabel('损失')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(range(1, actual_epochs + 1), test_accuracies, label='测试集准确率', marker='o', color='orange')
    ax2.set_title("测试集准确率曲线")
    ax2.set_xlabel('轮数')
    ax2.set_ylabel('准确率%')
    ax2.grid(True)
    ax2.legend()

    plt.suptitle("VGG13训练损失和测试集准确率")
    plt.show()

    #最终模型在整个测试集上的准确率
    print(f'最终模型在测试集上的准确率: {best_accuracy * 100:.2f}%')

    #展示分类错误和正确的图片
    correct_images = []
    incorrect_images = []
    correct_labels = []
    incorrect_labels = []
    incorrect_preds = []

    VGG13.eval()
    with torch.no_grad():
        #循环遍历所有testloader的数据
        for images, labels in testloader:
            #收集够5个正确和5个错误样本
            if len(correct_images) >= 5 and len(incorrect_images) >= 5:
                break
            
            images, labels = images.to(device), labels.to(device)
            outputs = VGG13(images)
            _, predicted = torch.max(outputs.data, 1)

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
        
    #创建一个2x5的子图网格
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle("分类结果示例", fontsize=16)

    #可视化正确分类的图片，放在第一行
    for i in range(5):
        ax = axes[0, i]
        if i < len(correct_images):
            title = f"真实: {class_names[correct_labels[i]]}"
            imshow_result(ax, correct_images[i], title)
        else:
            ax.axis('off')  #如果没有足够的图片，则隐藏坐标轴
    axes[0, 0].text(-0.3, 0.5, '正确分类', transform=axes[0, 0].transAxes,
                    fontsize=14, va='center', ha='right', rotation='vertical')

    #可视化错误分类的图片，放在第二行
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

    #计算模型参数数量和计算量
    input_tensor = torch.randn(1, 3, 32, 32).to(device)
    flops, params = profile(VGG13, inputs=(input_tensor,))
    print(f"模型参数数量: {params/1e6:.2f} M")
    print(f"模型计算量 (FLOPs): {flops/1e9:.2f} G")