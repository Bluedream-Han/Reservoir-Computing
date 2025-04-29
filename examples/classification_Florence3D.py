import numpy as np
import sys
import os
from regex import R
import torch
import torch.nn as nn
import time
import urllib.request
import zipfile
import cv2
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
    使用20个3x3的随机卷积核，取值范围为[-0.5, 0.5]
    """
    def __init__(self, n_kernels=30, kernel_size=5, in_channels=24):
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
            N = 样本数, T = 时间步数, V = 变量数（x,y,z坐标）
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
        
        # 重塑数据为8个关节点，每个点3个坐标
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
        warp_points = np.sort(np.random.randint(0, n_frames, 5))  # 增加扭曲点数量
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
        # 随机选择增强方法，增加应用概率
        if np.random.random() < 0.7:  # 增加噪声应用概率
            augmented_data = self.add_gaussian_noise(augmented_data)
        if np.random.random() < 0.7:  # 增加缩放应用概率
            augmented_data = self.random_scaling(augmented_data)
        if np.random.random() < 0.7:  # 增加旋转应用概率
            augmented_data = self.random_rotation(augmented_data)
        if np.random.random() < 0.7:  # 增加时间扭曲应用概率
            augmented_data = self.time_warping(augmented_data)
        return augmented_data

def augment_dataset(X, y, actor_ids, augment_times=4):  # 增加增强次数
    """对数据集进行增强"""
    aug = DataAugmentation()
    X_aug = []
    y_aug = []
    actor_ids_aug = []
    
    for i in range(len(X)):
        # 原始数据
        X_aug.append(X[i])
        y_aug.append(y[i])
        actor_ids_aug.append(actor_ids[i])
        # 增强数据
        for _ in range(augment_times):
            X_aug.append(aug.augment(X[i]))
            y_aug.append(y[i])
            actor_ids_aug.append(actor_ids[i])
    
    return np.array(X_aug), np.array(y_aug), np.array(actor_ids_aug)

def download_florence3d_dataset(data_path):
    """
    检测并解压Florence 3D数据集
    """
    zip_path = os.path.join(data_path, 'Florence3D.zip')
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    if not os.path.exists(os.path.join(data_path, 'Florence3D')):
        if os.path.exists(zip_path):
            print("正在解压文件...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_path)
                print("数据集解压完成。")
                os.remove(zip_path)
            except Exception as e:
                print(f"解压过程中出错: {str(e)}")
                print("请确保zip文件完整且未损坏")
                sys.exit(1)
        else:
            print(f"未找到数据集文件: {zip_path}")
            print("请将Florence3D.zip文件放在data目录下")
            sys.exit(1)

def extract_skeleton_from_video(video_path):
    """
    从视频中提取骨架数据
    
    参数:
    video_path : str
        视频文件路径
    
    返回:
    skeleton_data : np.ndarray
        骨架数据，形状为 [T, 3]，T为时间步数
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    skeleton_data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # TODO: 使用OpenCV或其他库提取骨架数据
        # 这里需要根据实际视频格式和骨架提取方法进行修改
        # 暂时使用随机数据作为示例
        skeleton_data.append(np.random.randn(3))
    
    cap.release()
    return np.array(skeleton_data)

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

def split_dataset(X, y, actor_ids, test_size=0.1, random_state=42):
    """
    改进的数据集划分方法
    
    参数:
    X : np.ndarray, 特征数据
    y : np.ndarray, 标签
    actor_ids : np.ndarray, 演员ID
    test_size : float, 测试集比例
    random_state : int, 随机种子
    """
    np.random.seed(random_state)
    
    # 获取所有唯一的动作类别
    unique_actions = np.unique(y)
    train_indices = []
    test_indices = []
    
    # 对每个动作类别进行划分
    for action in unique_actions:
        # 获取当前动作的所有数据索引
        action_indices = np.where(y == action)[0]
        np.random.shuffle(action_indices)
        
        # 按比例划分
        split_idx = int(len(action_indices) * (1 - test_size))
        train_indices.extend(action_indices[:split_idx])
        test_indices.extend(action_indices[split_idx:])
    
    # 转换为numpy数组
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    # 确保训练集和测试集中演员分布均衡
    train_actors = np.unique(actor_ids[train_indices])
    test_actors = np.unique(actor_ids[test_indices])
    
    print(f"训练集中演员分布: {train_actors}")
    print(f"测试集中演员分布: {test_actors}")
    
    return train_indices, test_indices

def load_florence3d_data(data_path):
    """
    加载Florence 3D数据集
    
    参数:
    data_path : str
        数据集路径
    
    返回:
    X_train, y_train, X_test, y_test : tuple
        训练和测试数据及标签
    """
    # 定义动作类别
    action_classes = ['wave', 'drink', 'phone', 'clap', 'tight_lace', 
                     'sit_down', 'stand_up', 'read_watch', 'bow']
    
    # 加载数据
    video_data = {}  # 用于存储每个视频的数据
    video_labels = {}  # 用于存储每个视频的标签
    video_actors = {}  # 用于存储每个视频的演员ID
    
    # 加载特征数据
    features_file = os.path.join(data_path, 'Florence3D', 'Florence_dataset_WorldCoordinates.txt')
    if not os.path.exists(features_file):
        raise ValueError(f"未找到特征文件: {features_file}")
    
    print("正在加载特征数据...")
    with open(features_file, 'r') as f:
        for line in f:
            if line.startswith('%'):
                continue
                
            # 解析每行数据
            # 格式: idvideo idactor idcategory f1....fn
            parts = line.strip().split()
            if len(parts) < 4:
                continue
                
            video_id = int(parts[0])
            actor_id = int(parts[1])
            category = int(parts[2])
            # 最后一个值是时间步，其他是特征
            features = np.array([float(x) for x in parts[3:]])
            # timestamp = float(parts[-1])
            
            # 提取骨架特征
            # elbows: f1-f6
            # wrists: f13-f18
            # knees: f7-f12
            # ankles: f19-f24
            # normalized frame value: f25
            skeleton_features = features  # 只使用骨架特征
            
            # 将数据添加到对应的视频中
            if video_id not in video_data:
                video_data[video_id] = []
                video_labels[video_id] = category - 1  # 转换为0-8的索引
                video_actors[video_id] = actor_id
            
            video_data[video_id].append(skeleton_features)
    
    if len(video_data) == 0:
        raise ValueError("未加载到任何数据，请检查数据集路径和文件格式")
    
    # 将视频数据转换为数组
    X = []
    y = []
    actor_ids = []
    
    # 统一使用25帧
    target_frames = 25
    print(f"统一视频帧数: {target_frames}")
    
    for video_id in video_data:
        # 获取当前视频的帧数
        current_frames = len(video_data[video_id])
        
        if current_frames < target_frames:
            # 如果帧数不足，使用最后一帧进行填充
            video_frames = np.array(video_data[video_id])
            padding = np.tile(video_frames[-1], (target_frames - current_frames, 1))
            video_frames = np.vstack((video_frames, padding))
        else:
            # 如果帧数过多，只取前target_frames帧
            video_frames = np.array(video_data[video_id][:target_frames])
        
        X.append(video_frames)
        y.append(video_labels[video_id])
        actor_ids.append(video_actors[video_id])
    
    X = np.array(X)
    y = np.array(y)
    actor_ids = np.array(actor_ids)
    
    print(f"成功加载数据: {len(X)} 个视频, {len(action_classes)} 个动作类别")
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 获取所有唯一的actor_id
    unique_actors = np.unique(actor_ids)
    train_indices, test_indices = split_dataset(X, y, actor_ids)
    
    # 划分数据
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # 只对训练集进行数据增强
    print("对训练集进行数据增强...")
    X_train, y_train, _ = augment_dataset(X_train, y_train, actor_ids[train_indices], augment_times=4)
    print(f"增强后训练集形状: X={X_train.shape}, y={y_train.shape}")
    
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    print(f"训练集中actor分布: {np.unique(actor_ids[train_indices])}")
    print(f"测试集中actor分布: {np.unique(actor_ids[test_indices])}")
    
    # 数据标准化
    # scaler = StandardScaler()
    # # 对每个视频的每个特征进行标准化
    # for i in range(X_train.shape[0]):
    #     X_train[i] = scaler.fit_transform(X_train[i])
    # for i in range(X_test.shape[0]):
    #     X_test[i] = scaler.transform(X_test[i])
    
    # 应用卷积前端
    # conv_frontend = CustomConvFrontend()
    # X_train_conv = conv_frontend.transform(X_train)
    # X_test_conv = conv_frontend.transform(X_test)
    # print('x_train_conv size is',X_train_conv.shape)
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
        leak=[0.9, 0.9, 0.9],
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
    evaluate_model(classifier, X_train_dynamic, y_train_oh, action_classes, "train_Florence3D")
    
    # 评估测试集
    evaluate_model(classifier, X_test_dynamic, y_test_oh, action_classes, "test_Florence3D")
    
    return X_train, y_train, X_test, y_test

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

def main():
    # 设置数据路径
    data_path = r"D:\PycharmProjects\pythonProject\Time-series-classification-and-clustering-with-Reservoir-Computing\data"
    
    # 加载数据
    print("正在加载Florence 3D数据集...")
    X_train, y_train, X_test, y_test = load_florence3d_data(data_path)
    

if __name__ == "__main__":
    main() 