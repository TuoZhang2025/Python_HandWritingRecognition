import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# ----------------------
# 数据加载与处理
# ----------------------

class HandwrittenCharsDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

        # 创建字符到索引的映射
        self.class_map = {}
        # 添加数字0-9
        for i in range(10):
            self.class_map[str(i)] = i
        # 添加大写字母A-Z (10-35)
        for i in range(26):
            self.class_map[chr(ord('A') + i)] = 10 + i
        # 添加小写字母a-z (36-61)
        for i in range(26):
            self.class_map[chr(ord('a') + i)] = 36 + i

        # 创建索引到字符的映射
        self.index_to_class = {v: k for k, v in self.class_map.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 转为灰度图

        # 从文件名提取标签（例如：img043-002.png → 标签为043对应的类别）
        filename = os.path.basename(img_path)
        parts = filename.split('-')
        if len(parts) < 1:
            raise ValueError(f"无法从文件名 {filename} 提取标签")

        # 提取类别编号（如043）
        class_code = parts[0][3:]  # 移除"img"前缀

        # 将类别编号映射到实际标签
        # 注意：这里假设你的类别编号是连续的，从001开始
        # 如果不是，请根据实际情况修改映射逻辑
        if len(class_code) == 1:
            class_code = '0' + class_code  # 确保两位数格式

        # 将编号转换为对应的字符
        # 001-010 → 0-9, 011-036 → A-Z, 037-062 → a-z
        class_num = int(class_code)
        if 1 <= class_num <= 10:
            label = str(class_num - 1)  # 0-9
        elif 11 <= class_num <= 36:
            label = chr(ord('A') + class_num - 11)  # A-Z
        elif 37 <= class_num <= 62:
            label = chr(ord('a') + class_num - 37)  # a-z
        else:
            raise ValueError(f"无效的类别编号: {class_code}")

        # 将标签转换为数字索引
        label_idx = self.class_map[label]

        if self.transform:
            image = self.transform(image)

        return image, label_idx


def create_datasets_from_json(json_path):
    # 从JSON文件加载分割数据
    with open(json_path, 'r', encoding='utf-8') as f:
        split_data = json.load(f)

    # 获取训练集、验证集和测试集的图像路径
    train_paths = split_data.get('train', [])
    val_paths = split_data.get('val', [])
    test_paths = split_data.get('test', [])

    # 打印数据集大小
    print(f"训练集大小: {len(train_paths)}")
    print(f"验证集大小: {len(val_paths)}")
    print(f"测试集大小: {len(test_paths)}")

    # 定义数据转换
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整图像大小
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 随机平移和缩放
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 创建数据集
    train_dataset = HandwrittenCharsDataset(train_paths, train_transform)
    val_dataset = HandwrittenCharsDataset(val_paths, test_transform)
    test_dataset = HandwrittenCharsDataset(test_paths, test_transform)

    return train_dataset, val_dataset, test_dataset


# ----------------------
# 模型定义
# ----------------------

class CharCNN(nn.Module):
    def __init__(self, num_classes=62):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x)
        x = x.view(-1, 256)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ----------------------
# 训练和评估函数
# ----------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)

        # 学习率调度
        scheduler.step()

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {history["train_loss"][-1]:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {history["val_loss"][-1]:.4f} | Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved best model with acc: {best_val_acc:.2f}%')

    return history


def evaluate_model(model, test_loader, device):
    model.eval()
    test_correct = 0
    test_total = 0
    class_correct = list(0. for i in range(62))
    class_total = list(0. for i in range(62))

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            # 计算每类的准确率
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += predicted[i].eq(label).sum().item()
                class_total[label] += 1

    test_acc = 100.0 * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')

    # 打印每类的准确率
    dataset = test_loader.dataset
    for i in range(62):
        if class_total[i] > 0:
            print(f'Accuracy of {dataset.index_to_class[i]}: {100.0 * class_correct[i] / class_total[i]:.2f}%')
        else:
            print(f'Accuracy of {dataset.index_to_class[i]}: N/A (no test samples)')

    return test_acc


# ----------------------
# 主函数
# ----------------------

def main():
    # 设置参数
    json_path = "splits.json"  # 数据分割的JSON文件
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001
    weight_decay = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets_from_json(json_path)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型
    print("Initializing model...")
    model = CharCNN(num_classes=62).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 训练模型
    print("Training model...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    # 评估模型
    print("Evaluating model...")
    evaluate_model(model, test_loader, device)

    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


if __name__ == "__main__":
    main()