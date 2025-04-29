import numpy as np
import sys
import os
import urllib.request
import zipfile
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

sys.path.append(
    r"D:\\PycharmProjects\\pythonProject\\Time-series-classification-and-clustering-with-Reservoir-Computing")
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from reservoir_computing.modules import RC_model
from reservoir_computing.utils import compute_test_scores

np.random.seed(0)  # Fix the seed for reproducibility

# 创建自定义卷积前端类
class CustomConvFrontend:
    """
    实现自定义卷积前端，使用20个3x9的随机卷积核，取值范围为[-0.5, 0.5]

    参数:
    -----------
    n_kernels: int, 默认20
        卷积核数量（输出通道数）
    kernel_size: int, 默认3
        卷积核在时间维度上的大小
    in_channels: int, 默认9
        输入通道数
    """

    def __init__(self, n_kernels=20, kernel_size=3, in_channels=12):
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        # 检查是否有GPU可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化卷积层 - 使用手动设置的权重
        self.conv = nn.Conv1d(
            in_channels=in_channels,  # 输入通道为9
            out_channels=n_kernels,  # 20个卷积核（输出通道）
            kernel_size=kernel_size,  # 卷积核大小为3（在时间维度上）
            stride=1,
            padding='valid'  # 不使用padding，使输出时间维度减小
        )

        # 手动设置卷积核权重为[-0.5, 0.5]范围内的随机值
        with torch.no_grad():
            # 初始化卷积核为正态分布
            nn.init.normal_(self.conv.weight, mean=0.0, std=0.25)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

        # 将模型移至设备
        self.conv = self.conv.to(self.device)
        # 设置为评估模式
        self.conv.eval()

        print(f"自定义卷积前端已初始化: {n_kernels}个卷积核，卷积核大小={kernel_size}，输入通道={in_channels}")

    def transform(self, X):
        """
        使用随机卷积核处理输入数据

        参数:
        -----------
        X : np.ndarray
            形状为 [N, T, V] 的多变量时间序列数据
            N = 样本数, T = 时间步数, V = 变量数（通道数）

        返回:
        -----------
        transformed_X : np.ndarray
            卷积处理后的数据，形状为 [N, T-kernel_size+1, n_kernels]
        """
        N, T, V = X.shape

        # 将数据转换为PyTorch张量并移至设备
        X_torch = torch.from_numpy(X).float().to(self.device)

        # 重塑张量为[N, V, T]以适应1D卷积（Conv1d要求通道在第二维）
        X_torch = X_torch.permute(0, 2, 1)

        # 应用卷积层
        with torch.no_grad():
            conv_output = self.conv(X_torch)

        # 当前输出形状是 [N, n_kernels, T-kernel_size+1]
        # 转置为 [N, T-kernel_size+1, n_kernels]
        conv_output = conv_output.permute(0, 2, 1)

        # 转换回NumPy数组
        transformed_X = conv_output.cpu().numpy()

        return transformed_X

# 创建数据目录
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(data_path, exist_ok=True)
print(f"数据将被下载到: {os.path.abspath(data_path)}")

# 设置MotionSense数据集路径
dataset_path = r"D:\PycharmProjects\pythonProject\Time-series-classification-and-clustering-with-Reservoir-Computing\data\motion-sense-master"

# 定义加载MotionSense数据的函数
def load_motion_sense_data():
    data_dir = os.path.join(dataset_path, 'A_DeviceMotion_data')
    activities = ['dws', 'ups', 'sit', 'std', 'wlk', 'jog']
    all_data = []
    all_labels = []
    
    # 加载受试者信息
    subjects_info = pd.read_csv(os.path.join(dataset_path, 'data_subjects_info.csv'))
    
    for activity in activities:
        # 遍历16次实验
        for trial_num in range(1, 17):
            trial_name = f"{activity}_{trial_num}"
            trial_dir = os.path.join(data_dir, trial_name)
            
            if not os.path.exists(trial_dir):
                # print(f"跳过不存在的实验目录: {trial_dir}")
                continue
                
            print(f"处理实验: {trial_name}")
            
            # 处理每个受试者的数据
            for subject_file in os.listdir(trial_dir):
                if not subject_file.endswith('.csv'):
                    continue
                    
                file_path = os.path.join(trial_dir, subject_file)
                df = pd.read_csv(file_path)
                
                # 提取传感器数据（12个特征）
                sensor_data = df[['attitude.roll', 'attitude.pitch', 'attitude.yaw',
                                'gravity.x', 'gravity.y', 'gravity.z',
                                'rotationRate.x', 'rotationRate.y', 'rotationRate.z',
                                'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z']].values
                
                # 将数据分割成固定长度的序列
                sequence_length = 128
                for i in range(0, len(sensor_data) - sequence_length + 1, sequence_length // 2):
                    sequence = sensor_data[i:i + sequence_length]
                    if len(sequence) == sequence_length:
                        all_data.append(sequence)
                        all_labels.append(activities.index(activity))
    
    return np.array(all_data), np.array(all_labels)

def evaluate_model(classifier, X, y, action_classes, dataset_name="数据集"):
    """
    评估模型在指定数据集上的性能
    
    参数:
    classifier : RC_model
        训练好的模型
    X : np.ndarray
        输入数据
    y : np.ndarray
        真实标签（one-hot编码）
    action_classes : list
        动作类别名称列表
    dataset_name : str
        数据集名称，用于打印输出
    """
    # 计算预测结果
    print(f"\n计算{dataset_name}的预测...")
    pred = classifier.predict(X)
    accuracy, f1 = compute_test_scores(pred, y)
    print(f"{dataset_name}总体准确率 = {accuracy:.3f}, F1 = {f1:.3f}")
    
    # 将one-hot编码转换回类别标签
    y_labels = np.argmax(y, axis=1)
    
    # 确保pred是二维数组
    if len(pred.shape) == 1:
        pred_one_hot = np.zeros((len(pred), len(action_classes)))
        pred_one_hot[np.arange(len(pred)), pred] = 1
        pred_labels = np.argmax(pred_one_hot, axis=1)
    else:
        pred_labels = np.argmax(pred, axis=1)
    
    # 获取实际出现的类别
    unique_classes = np.unique(np.concatenate([y_labels, pred_labels]))
    actual_action_classes = [action_classes[i] for i in unique_classes]
    
    # 打印每个类别的详细指标
    print(f"\n{dataset_name}每个类别的详细指标:")
    print(classification_report(y_labels, pred_labels, 
                              target_names=actual_action_classes,
                              labels=unique_classes))
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_labels, pred_labels, labels=unique_classes)
    # 将混淆矩阵转换为0-1范围
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    heatmap = sns.heatmap(cm_normalized, annot=False, cmap='Blues',
                xticklabels=actual_action_classes,
                yticklabels=actual_action_classes,
                annot_kws={'size': 12})
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.title(f'{dataset_name}', fontsize=20)
    plt.xlabel('Prediction', fontsize=18)
    plt.ylabel('Ground Truth', fontsize=18)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(rotation=45, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_confusion_matrix.png')
    plt.close()
    
# 加载数据
print("正在加载MotionSense数据...")
X, y = load_motion_sense_data()

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"数据形状:\n  X_train: {X_train.shape}\n  y_train: {y_train.shape}\n  X_test: {X_test.shape}\n  y_test: {y_test.shape}")

# 标准化每个传感器通道数据
print("标准化传感器数据...")
for i in range(X_train.shape[2]):
    scaler = StandardScaler()
    X_train[:, :, i] = scaler.fit_transform(X_train[:, :, i])
    X_test[:, :, i] = scaler.transform(X_test[:, :, i])

# 对数据进行稀疏采样 - 每4个时间步采样一次，从128减少到32个时间步
print("进行稀疏采样，每4个时间步采样一次...")
X_train_sampled = X_train[:, ::1, :]  # 从索引0开始，步长为4
X_test_sampled = X_test[:, ::1, :]

print(f"采样后数据形状:\n  X_train_sampled: {X_train_sampled.shape}\n  X_test_sampled: {X_test_sampled.shape}")

action_classes = ['DOWNSTAIRS', 'UPSTAIRS', 'SITTING', 'STANDING', 'WALKING', 'JOGGING']

# 用采样后的数据替换原始数据
X_train = X_train_sampled
X_test = X_test_sampled

# 应用自定义卷积前端
print("应用自定义卷积前端处理...")
conv_frontend = CustomConvFrontend()
X_train_conv = conv_frontend.transform(X_train)
X_test_conv = conv_frontend.transform(X_test)
print(f"卷积处理后的数据形状: X_train_conv: {X_train_conv.shape}, X_test_conv: {X_test_conv.shape}")

# One-hot encoding for labels
onehot_encoder = OneHotEncoder(sparse_output=False)
y_train_oh = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
y_test_oh = onehot_encoder.transform(y_test.reshape(-1, 1))

# 为储备池计算初始化RC模型
classifier = RC_model(
    n_internal_units=20,
    nonlinearity='relu',
    spectral_radius=0.95,
    leak=[0.9, 0.95, 1.0],  # 使用单个leak值
    w_l2=0.001,
    mlp_layout=(100, 100),
    num_epochs=200,
    mts_rep='concat',
)

# 训练模型
print("训练RC模型...")
start_time = time.time()

classifier.fit(X_train_conv, y_train_oh)

training_time = time.time() - start_time
print(f"训练时间: {training_time:.3f}秒")
# 加载预训练的权重
# weights_path = r"D:\PycharmProjects\pythonProject\Time-series-classification-and-clustering-with-Reservoir-Computing\UCI_HAR_model_weights.joblib"
# print(f"正在加载预训练权重: {weights_path}")
# classifier.load_weights(weights_path)

# 在测试数据上进行预测
print("计算测试数据的预测...")
pred_class = classifier.predict(X_test_conv)
accuracy, f1 = compute_test_scores(pred_class, y_test_oh)
print(f"准确率 = {accuracy:.3f}, F1 = {f1:.3f}")

evaluate_model(classifier, X_train_conv, y_train_oh, action_classes, "train_MotionSense")
    
    # 评估测试集
evaluate_model(classifier, X_test_conv, y_test_oh, action_classes, "test_MotionSense")

# 可视化传感器数据（可选）
# try:
#     import matplotlib.pyplot as plt
#
#     # 选择一个样本进行可视化
#     sample_idx = 100
#     activity_name = ['Downstairs', 'Upstairs', 'Sitting', 'Standing', 'Walking', 'Jogging'][y_train[sample_idx]]
#
#     plt.figure(figsize=(15, 10))
#     for i in range(12):
#         plt.subplot(4, 3, i+1)
#         plt.plot(range(128), X_train[sample_idx, :, i])
#
#         # 确定传感器类型和轴
#         sensor_type = ['Attitude', 'Gravity', 'Rotation Rate', 'User Acceleration'][i // 3]
#         axis = ['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z'][i % 3]
#         plt.title(f'{sensor_type} - {axis}')
#
#     plt.suptitle(f'Activity: {activity_name}', fontsize=16)
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.savefig('motion_sense_visualization.png')
#     print("已生成传感器数据可视化图像: motion_sense_visualization.png")
# except ImportError:
#     print("未安装matplotlib，跳过可视化步骤")