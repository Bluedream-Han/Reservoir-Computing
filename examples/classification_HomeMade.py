import numpy as np
import sys
import os
import torch
import torch.nn as nn
import time
import scipy.io as sio
import h5py
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(
    r"D:\\PycharmProjects\\pythonProject\\Time-series-classification-and-clustering-with-Reservoir-Computing")
from reservoir_computing.modules import RC_model
from reservoir_computing.utils import compute_test_scores

np.random.seed(0)  # 固定随机种子以确保可重复性

class CustomConvFrontend:
    """
    实现自定义卷积前端，用于处理3D动作数据
    使用30个5x1的随机卷积核，取值范围为[-0.5, 0.5]
    """
    def __init__(self, n_kernels=30, kernel_size=5, in_channels=75):
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_kernels,
            kernel_size=kernel_size,
            stride=1,
            padding='valid'
        )

        with torch.no_grad():
            nn.init.normal_(self.conv.weight, mean=0.0, std=0.25)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

        self.conv = self.conv.to(self.device)
        self.conv.eval()

    def transform(self, X):
        """
        使用随机卷积核处理输入数据
        
        参数:
        X : np.ndarray
            形状为 [N, T, V] 的3D动作数据
            N = 样本数, T = 时间步数, V = 变量数（75个关节坐标）
        """
        N, T, V = X.shape
        X_torch = torch.from_numpy(X).float().to(self.device)
        X_torch = X_torch.permute(0, 2, 1)
        
        with torch.no_grad():
            conv_output = self.conv(X_torch)
        
        conv_output = conv_output.permute(0, 2, 1)
        return conv_output.cpu().numpy()

class DataAugmentation:
    def __init__(self, noise_level=0.02, scale_range=(0.85, 1.15), rotation_range=(-15, 15)):
        self.noise_level = noise_level
        self.scale_range = scale_range
        self.rotation_range = rotation_range
    
    def add_gaussian_noise(self, data):
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_level, data.shape)
        return data + noise
    
    def random_scaling(self, data):
        """随机缩放"""
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return data * scale
    
    def random_rotation(self, data):
        """随机旋转（3D骨架数据）"""
        # 生成随机旋转角度
        angles = np.random.uniform(self.rotation_range[0], self.rotation_range[1], 3)
        # 构建旋转矩阵
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(np.radians(angles[0])), -np.sin(np.radians(angles[0]))],
                      [0, np.sin(np.radians(angles[0])), np.cos(np.radians(angles[0]))]])
        Ry = np.array([[np.cos(np.radians(angles[1])), 0, np.sin(np.radians(angles[1]))],
                      [0, 1, 0],
                      [-np.sin(np.radians(angles[1])), 0, np.cos(np.radians(angles[1]))]])
        Rz = np.array([[np.cos(np.radians(angles[2])), -np.sin(np.radians(angles[2])), 0],
                      [np.sin(np.radians(angles[2])), np.cos(np.radians(angles[2])), 0],
                      [0, 0, 1]])
        R = Rz @ Ry @ Rx
        
        # 重塑数据为25个关节点，每个点3个坐标
        n_frames = data.shape[0]
        data_reshaped = data.reshape(n_frames, -1, 3)
        
        # 对每个关节点应用旋转
        rotated_data = np.zeros_like(data_reshaped)
        for i in range(data_reshaped.shape[1]):
            rotated_data[:, i] = np.dot(data_reshaped[:, i], R)
        
        # 重塑回原始形状
        return rotated_data.reshape(n_frames, -1)
    
    def time_warping(self, data):
        """时间扭曲"""
        n_frames = data.shape[0]
        # 生成扭曲点
        warp_points = np.sort(np.random.randint(0, n_frames, 5))
        warp_points = np.concatenate([[0], warp_points, [n_frames-1]])
        # 生成扭曲后的时间点
        warped_points = np.sort(np.random.randint(0, n_frames, 5))
        warped_points = np.concatenate([[0], warped_points, [n_frames-1]])
        # 使用插值进行时间扭曲
        warped_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            f = interp1d(warp_points, data[warp_points, i])
            warped_data[:, i] = f(np.arange(n_frames))
        return warped_data
    
    def augment(self, data):
        """应用所有增强方法"""
        augmented_data = data.copy()
        if np.random.random() < 0.7:
            augmented_data = self.add_gaussian_noise(augmented_data)
        if np.random.random() < 0.7:
            augmented_data = self.random_scaling(augmented_data)
        if np.random.random() < 0.7:
            augmented_data = self.random_rotation(augmented_data)
        if np.random.random() < 0.7:
            augmented_data = self.time_warping(augmented_data)
        return augmented_data

def augment_dataset(X, y, subject_ids, augment_times=4):
    """对数据集进行增强"""
    aug = DataAugmentation()
    X_aug = []
    y_aug = []
    subject_ids_aug = []
    
    for i in range(len(X)):
        # 原始数据
        X_aug.append(X[i])
        y_aug.append(y[i])
        subject_ids_aug.append(subject_ids[i])
        # 增强数据
        for _ in range(augment_times):
            X_aug.append(aug.augment(X[i]))
            y_aug.append(y[i])
            subject_ids_aug.append(subject_ids[i])
    
    return np.array(X_aug), np.array(y_aug), np.array(subject_ids_aug)

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
    
    # 绘制混淆矩阵
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
    
def split_dataset(X, y, subject_ids, test_size=0.2, random_state=42):
    """
    改进的数据集划分方法，确保训练集和测试集中受试者分布均衡
    """
    np.random.seed(random_state)
    
    unique_actions = np.unique(y)
    train_indices = []
    test_indices = []
    
    for action in unique_actions:
        action_indices = np.where(y == action)[0]
        np.random.shuffle(action_indices)
        
        split_idx = int(len(action_indices) * (1 - test_size))
        train_indices.extend(action_indices[:split_idx])
        test_indices.extend(action_indices[split_idx:])
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    train_subjects = np.unique(subject_ids[train_indices])
    test_subjects = np.unique(subject_ids[test_indices])
    
    print(f"训练集中受试者分布: {train_subjects}")
    print(f"测试集中受试者分布: {test_subjects}")
    
    return train_indices, test_indices
class DynamicFeatureExtractor:
    def __init__(self, mask_length=50):
        """
        初始化动态特征提取器
        
        参数:
        mask_length : int
            掩码矩阵的长度M
        """
        self.mask_length = mask_length
        self.mask_matrix = None
        
    def _generate_mask_matrix(self, input_dim):
        """
        生成掩码矩阵
        
        参数:
        input_dim : int
            输入维度(3*K)
        """
        # 生成随机掩码矩阵，值为1或-1
        self.mask_matrix = np.random.choice([-1, 1], size=(input_dim, self.mask_length))
        
    def compute_dynamic_features(self, X):
        """
        计算动态特征
        
        参数:
        X : np.ndarray
            输入数据，形状为 [N, T, V]
            N = 样本数, T = 时间步数, V = 变量数
            
        返回:
        dynamic_features : np.ndarray
            动态特征矩阵，形状为 [N, T, 2*M]
            包含相对于第一帧和前一帧的变化特征
        """
        N, T, V = X.shape
        
        # 如果掩码矩阵未初始化，则生成
        if self.mask_matrix is None:
            self._generate_mask_matrix(V)
            
        # 初始化结果数组，保持时间步维度，特征维度翻倍
        dynamic_features = np.zeros((N, T, self.mask_length))
        
        for n in range(N):
            # 计算相对于第一帧的变化
            delta_first = X[n] - X[n, 0:1, :]  # 减去第一帧
            
            # # 计算相对于前一帧的变化
            # delta_prev = np.zeros_like(X[n])
            # delta_prev[0] = X[n, 0] - X[n, 0]  # 第一帧保持不变
            # for t in range(1, T):
            #     delta_prev[t] = X[n, t] - X[n, t-1]  # 减去前一帧
            
            # 对每一帧进行掩码映射
            for t in range(T):
                # 获取当前帧的动态变化
                delta_frame_first = delta_first[t]  # 形状: [V]
                # delta_frame_prev = delta_prev[t]    # 形状: [V]
                
                # 应用掩码矩阵
                virtual_nodes_first = np.dot(delta_frame_first, self.mask_matrix)  # 形状: [M]
                # virtual_nodes_prev = np.dot(delta_frame_prev, self.mask_matrix)    # 形状: [M]
                
                # 拼接两种特征
                dynamic_features[n, t] = virtual_nodes_first
                
        return dynamic_features
def load_homemade_data(data_path):
    """
    加载Home-made数据集
    
    参数:
    data_path : str
        数据集路径
    
    返回:
    X_train, y_train, X_test, y_test : tuple
        训练和测试数据及标签
    """
    # 定义动作类别
    action_classes = ['1-squart', '2-stretch', '3-leftfall', '4-rightfall', '5-downfall']
    
    # 加载数据
    X = []
    y = []
    subject_ids = []
    
    # 遍历所有受试者文件夹
    for subject_id in os.listdir(data_path):
        subject_path = os.path.join(data_path, subject_id)
        if not os.path.isdir(subject_path):
            continue
            
        # 提取受试者ID的数字部分
        subject_num = int(subject_id.split('-')[0])
            
        # 遍历所有动作类别文件夹
        for action_idx, action_name in enumerate(action_classes):
            action_path = os.path.join(subject_path, action_name)
            if not os.path.isdir(action_path):
                continue
                
            # 遍历所有重复实验文件
            for file_name in os.listdir(action_path):
                if not file_name.endswith('.mat'):
                    continue
                    
                file_path = os.path.join(action_path, file_name)
                
                # 使用h5py加载mat文件
                with h5py.File(file_path, 'r') as f:
                    # 获取骨架数据
                    skeleton_data = np.array(f['input']).T  # 转置为 T x 75 格式
                    # print(f"文件 {file_path} 的数据形状: {skeleton_data.shape}")
                
                X.append(skeleton_data)
                y.append(action_idx)
                subject_ids.append(subject_num)
    
    if len(X) == 0:
        raise ValueError("未加载到任何数据，请检查数据集路径和文件格式")
    
    # 统一时间步长和特征维度
    target_length = 100  # 设置目标时间步长
    target_features = 75  # 设置目标特征维度
    X_unified = []
    
    for i, skeleton_data in enumerate(X):
        current_length = skeleton_data.shape[0]
        current_features = skeleton_data.shape[1]
        # print(f"样本 {i} 的原始形状: {skeleton_data.shape}")
        
        # 统一时间步长
        if current_length > target_length:
            # 如果当前长度大于目标长度，进行降采样
            indices = np.linspace(0, current_length-1, target_length, dtype=int)
            skeleton_data = skeleton_data[indices]
        elif current_length < target_length:
            # 如果当前长度小于目标长度，进行插值
            x_old = np.linspace(0, 1, current_length)
            x_new = np.linspace(0, 1, target_length)
            skeleton_data_new = np.zeros((target_length, current_features))
            
            for j in range(current_features):
                f = interp1d(x_old, skeleton_data[:, j])
                skeleton_data_new[:, j] = f(x_new)
            
            skeleton_data = skeleton_data_new
        
        # 统一特征维度
        if current_features != target_features:
            # 如果特征维度不匹配，进行插值
            x_old = np.linspace(0, 1, current_features)
            x_new = np.linspace(0, 1, target_features)
            skeleton_data_new = np.zeros((target_length, target_features))
            
            for j in range(target_length):
                f = interp1d(x_old, skeleton_data[j, :])
                skeleton_data_new[j, :] = f(x_new)
            
            skeleton_data = skeleton_data_new
        
        # print(f"样本 {i} 的统一后形状: {skeleton_data.shape}")
        X_unified.append(skeleton_data)
    
    # 确保所有样本具有相同的形状
    X_unified = np.array(X_unified)
    # print(f"统一后的数据形状: {X_unified.shape}")
    
    y = np.array(y)
    subject_ids = np.array(subject_ids)
    
    print(f"成功加载数据: {len(X)} 个样本, {len(action_classes)} 个动作类别")
    # print(f"数据形状: X={X_unified.shape}, y={y.shape}")
    
    # 划分数据集
    train_indices, test_indices = split_dataset(X_unified, y, subject_ids)
    
    X_train = X_unified[train_indices]
    y_train = y[train_indices]
    X_test = X_unified[test_indices]
    y_test = y[test_indices]
    
    # # 对训练集进行数据增强
    # print("对训练集进行数据增强...")
    # X_train, y_train, _ = augment_dataset(X_train, y_train, subject_ids[train_indices], augment_times=4)
    # print(f"增强后训练集形状: X={X_train.shape}, y={y_train.shape}")
    
    # 应用卷积前端
    # conv_frontend = CustomConvFrontend()
    # X_train_conv = conv_frontend.transform(X_train)
    # X_test_conv = conv_frontend.transform(X_test)
    # print('x_train_conv size is', X_train_conv.shape)
    # print('x_test_conv size is', X_test_conv.shape)

    # 应用动态特征提取
    extractor = DynamicFeatureExtractor(mask_length=100)
    X_train_dynamic = extractor.compute_dynamic_features(X_train)
    X_test_dynamic = extractor.compute_dynamic_features(X_test)
    print('x_train_dynamic size is', X_train_dynamic.shape)
    print('x_test_dynamic size is', X_test_dynamic.shape)
    
    # 标签one-hot编码
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_train_oh = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_oh = onehot_encoder.transform(y_test.reshape(-1, 1))
    
    # 初始化RC模型
    classifier = RC_model(
        n_internal_units=100,
        nonlinearity='relu',
        spectral_radius=0.95,
        leak=0.9,  # 修改为单个值
        w_l2=0.1,
        mlp_layout=(30),
        num_epochs=30,
        readout_type='mlp',
        multi_timescale=False,
        mts_rep='concat'
    )
    
    # 训练模型
    print("训练模型...")
    start_time = time.time()
    classifier.fit(X_train_dynamic, y_train_oh)
    training_time = time.time() - start_time
    print(f"训练时间: {training_time:.3f}秒")
    
    # 评估训练集
    evaluate_model(classifier, X_train_dynamic, y_train_oh, action_classes, "train_HomeMade")
    
    # 评估测试集
    evaluate_model(classifier, X_test_dynamic, y_test_oh, action_classes, "test_HomeMade")
    
    return X_train, y_train, X_test, y_test

def main():
    # 设置数据路径
    data_path = r"D:\PycharmProjects\pythonProject\Time-series-classification-and-clustering-with-Reservoir-Computing\data\Home-made+falling+dataset"
    
    # 加载数据
    print("正在加载Home-made数据集...")
    X_train, y_train, X_test, y_test = load_homemade_data(data_path)

if __name__ == "__main__":
    main() 