"""
RTX 5070 Ti可用版SNN - CIFAR-10（最终修复版）
"""
import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import math
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ===================== 初始化：创建charts文件夹 =====================
os.makedirs('./charts', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows中文支持
plt.rcParams['axes.unicode_minus'] = False


# ===================== 设备设置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 使用设备: {device}")
if device.type == 'cuda':
    print(f"✅ GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    cudnn.benchmark = True
    cudnn.enabled = True

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ===================== 参数解析 =====================
parser = argparse.ArgumentParser(description='RTX 5070 Ti SNN CIFAR10 (Final Working Version)')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-b', '--batch-size', default=32, type=int)
parser.add_argument('-T', '--timesteps', default=10, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)
parser.add_argument('--epochs', default=10, type=int)
args = parser.parse_args()


# ===================== 正确的SNN神经元（膜电位为动态状态，不参与训练） =====================
class IFNode(nn.Module):
    def __init__(self, v_threshold=0.5, v_reset=0.0):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        # 膜电位是动态状态，每次前向重新初始化（保留梯度）
        self.v = None

    def forward(self, dv: torch.Tensor):
        # 初始化膜电位（与输入同形状、同设备，保留梯度）
        if self.v is None:
            self.v = torch.full_like(dv, self.v_reset, device=device, requires_grad=True)
        
        # 膜电位累积（保留梯度）
        self.v = self.v + dv
        # 脉冲发放（用可微分的近似阶跃函数）
        spike = torch.sigmoid(10 * (self.v - self.v_threshold))  # 近似阶跃，确保梯度
        # 膜电位重置（非in-place操作，保留梯度）
        self.v = torch.where(spike > 0.5, torch.tensor(self.v_reset, device=device), self.v)
        return spike

    def reset(self):
        # 重置膜电位（训练下一个batch前清空）
        self.v = None


# ===================== 轻量SNN模型 =====================
class LightSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 卷积层1
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            IFNode(),
            nn.MaxPool2d(2),
            # 卷积层2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            IFNode(),
            nn.MaxPool2d(2),
            # 卷积层3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            IFNode(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128, bias=False),
            IFNode(),
            nn.Linear(128, 10, bias=False),
            IFNode()
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset_(self):
        # 重置所有神经元的膜电位
        for module in self.modules():
            if isinstance(module, IFNode):
                module.reset()


# ===================== 图表生成函数 =====================
def plot_accuracy_curve(epochs, train_accs, test_accs):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, label='训练精度', linewidth=2.5, marker='o', markersize=6)
    plt.plot(epochs, test_accs, label='测试精度', linewidth=2.5, marker='s', markersize=6, color='red')
    plt.xlabel('训练轮次（Epoch）', fontsize=12)
    plt.ylabel('精度（%）', fontsize=12)
    plt.title('SNN模型训练/测试精度变化曲线（RTX 5070 Ti）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.savefig('./charts/accuracy_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_curve(epochs, train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='训练损失', linewidth=2.5, marker='o', color='orange')
    plt.xlabel('训练轮次（Epoch）', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.title('SNN模型训练损失变化曲线（RTX 5070 Ti）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('./charts/loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_spike_heatmap(net, test_img, T):
    net.eval()
    net.reset_()
    spike_records = []
    with torch.no_grad():
        for t in range(T):
            spike = net.features[1](net.features[0](test_img))  # 第一层神经元脉冲
            spike_records.append(spike.cpu().numpy()[0, :50])  # 取第1个样本的前50个神经元
            net.reset_()  # 重置当前层神经元
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(np.stack(spike_records), cmap='binary', cbar_kws={'label': '脉冲发放（1=发放，0=未发放）'})
    plt.xlabel('神经元编号', fontsize=12)
    plt.ylabel('时间步（Timestep）', fontsize=12)
    plt.title('SNN第一层神经元脉冲发放热力图（T=10）', fontsize=14, fontweight='bold')
    plt.savefig('./charts/spike_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(all_labels, all_preds):
    class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.title('SNN模型CIFAR-10混淆矩阵（测试集）', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig('./charts/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


# ===================== 主训练函数 =====================
def main():
    # 数据加载
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test),
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # 模型、损失、优化器
    model = LightSNN().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # 训练记录
    train_accs = []
    test_accs = []
    train_losses = []
    start_time = time.time()

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            model.reset_()  # 重置所有神经元

            # SNN前向（累积时间步脉冲）
            output = torch.zeros((inputs.shape[0], 10), device=device)
            for t in range(args.timesteps):
                output += model(inputs)
                model.reset_()  # 重置神经元，准备下一个时间步

            # 计算损失与梯度
            loss = criterion(output / args.timesteps, labels)
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = output.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(acc=f"{100*train_correct/train_total:.2f}%", loss=f"{train_loss/train_total:.4f}")

        # 测试
        model.eval()
        test_correct = 0
        test_total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Test]')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                model.reset_()
                output = torch.zeros((inputs.shape[0], 10), device=device)
                for t in range(args.timesteps):
                    output += model(inputs)
                    model.reset_()
                _, predicted = output.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                pbar.set_postfix(acc=f"{100*test_correct/test_total:.2f}%")

        # 记录
        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total
        train_loss_avg = train_loss / train_total
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_losses.append(train_loss_avg)
        scheduler.step()

        # 打印
        print(f"\nEpoch {epoch+1} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Loss: {train_loss_avg:.4f}")

    # 生成图表
    plot_accuracy_curve(list(range(1, args.epochs+1)), train_accs, test_accs)
    plot_loss_curve(list(range(1, args.epochs+1)), train_losses)
    plot_confusion_matrix(all_labels, all_preds)
    # 生成脉冲热力图（取测试集第一个样本）
    test_img = next(iter(test_loader))[0][0:1].to(device)
    plot_spike_heatmap(model, test_img, args.timesteps)

    print(f"\n训练完成！总耗时: {(time.time()-start_time)/60:.2f} 分钟 | 最佳测试精度: {max(test_accs):.2f}%")
    print("所有图表已保存到 ./charts 文件夹")


if __name__ == '__main__':
    main()