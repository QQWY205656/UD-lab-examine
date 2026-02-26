import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from thop import profile


plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

# VGG配置字典
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':# 如果是M就是池化层
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:# 否则是卷积层，每个卷积层跟一个ReLu激活函数(还可以加一个BN层优化)
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)] #可加可不加
        return nn.Sequential(*layers)

def get_dataloaders(num_workers=2):
    """封装数据加载逻辑的函数"""
    print("数据加载预处理")
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

    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transforms_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transforms_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return trainloader, testloader


if __name__ == '__main__':
    # 调用数据加载函数
    trainloader, testloader = get_dataloaders(num_workers=2)

    device = torch.device("cuda")
    VGG13 = VGG('VGG13')
    VGG13=VGG13.to(device)

    #print(VGG13)

    #初始化权重
    for m in VGG13.modules():
        if isinstance(m, nn.Conv2d):# 卷积层参数
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):# 全连接层参数
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(VGG13.parameters(), lr=0.001)

    #训练模型
    epoch_losses = []
    test_accuracies = []
    test_losses = []
    num_epochs = 10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for epoch in range(num_epochs):
        #训练部分
        VGG13.train()
        running_loss = 0.0
        with tqdm(trainloader, desc=f'Epoch {epoch + 1}/{num_epochs}[训练]', unit='batch') as tepoch:
            for i, data in enumerate(tepoch, 0):
                # 获取训练数据
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # 权重参数梯度清零
                optimizer.zero_grad()

                # 正向及反向传播
                outputs = VGG13(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 累加损失值
                running_loss += loss.item()

                # 实时更新进度条的损失信息
                tepoch.set_postfix(
                    avg_loss=running_loss / (i + 1)  # 当前epoch的平均损失
                )
        # 计算并打印该轮 epoch 的平均损失
        epoch_avg_loss = running_loss / len(trainloader)
        epoch_losses.append(epoch_avg_loss)
        print(f'Epoch {epoch + 1} 平均损失: {epoch_avg_loss:.4f}')


        #测试部分
        VGG13.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(testloader, desc=f'Epoch {epoch + 1}/{num_epochs}[测试]', unit='batch') as tepoch:
                for data in tepoch:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = VGG13(images)
                    _, predicted = torch.max(outputs.data, 1)# 返回最大值和其索引
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        epoch_accuracy = correct / total
        test_accuracies.append(epoch_accuracy * 100)            
        print(f'Epoch {epoch + 1} 测试集准确率: {epoch_accuracy * 100:.2f}%')
    print("训练完成。")

    #画训练损失曲线和测试集准确率曲线
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), epoch_losses, label='训练损失', marker='o')
    plt.title("训练损失曲线")
    plt.xlabel('轮数')
    plt.ylabel('损失')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='测试集准确率', marker='o', color='orange')
    plt.title("测试集准确率曲线")
    plt.xlabel('轮数')
    plt.ylabel('准确率%')
    plt.grid(True)
    plt.legend()

    plt.suptitle("VGG13训练损失和测试集准确率")
    plt.show()

    #最终模型在整个测试集上的准确率
    print(f'最终模型在测试集上的准确率: {test_accuracies[-1]:.2f}%')


    #展示分类错误和正确的图片
    correct_images = []
    incorrect_images = []
    correct_labels = []
    incorrect_labels = []
    incorrect_preds = []

    VGG13.eval()
    with torch.no_grad():
        for images,labels in testloader:
            if len(correct_images) >= 5 and len(incorrect_images) >= 5:
                break
            images,labels = images.to(device),labels.to(device)
            outputs = VGG13(images)
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
        #CIFAR10的均值和标准差
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = img * std + mean  # 反归一化 (mean=0.5, std=0.5)
        img = torch.clamp(img, 0, 1)  # 确保像素值在[0, 1]范围内
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

    #计算模型参数数量和计算量
    input_tensor = torch.randn(1, 3, 32, 32).to(device)
    flops, params = profile(VGG13, inputs=(input_tensor,))
    print(f"模型参数数量: {params}")
    print(f"模型计算量 (FLOPs): {flops}")