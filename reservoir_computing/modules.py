import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from scipy.spatial.distance import pdist, cdist, squareform
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPClassifier

from .reservoir import Reservoir
from .tensorPCA import tensorPCA

class CustomMLPClassifier:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', alpha=0.0001, batch_size=32,
                 learning_rate='constant', learning_rate_init=0.001, max_iter=200, shuffle=True,
                 random_state=42, dropout_rate=0.2):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.dropout_rate = dropout_rate
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
    def _build_model(self, input_size, output_size):
        layers = []
        prev_size = input_size
        
        # 添加隐藏层
        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_size = hidden_size
        
        # 添加输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        return nn.Sequential(*layers)
    
    def fit(self, X, y):
        X = torch.FloatTensor(X).to(self.device)
        y = torch.LongTensor(np.argmax(y, axis=1)).to(self.device)
        
        # 构建模型
        input_size = X.shape[1]
        output_size = y.max().item() + 1
        self.model = self._build_model(input_size, output_size).to(self.device)
        
        # 设置优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        
        # 训练模式
        self.model.train()
        
        for epoch in range(self.max_iter):
            if self.shuffle:
                indices = torch.randperm(len(X))
                X = X[indices]
                y = y[indices]
            
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size]
                batch_y = y[i:i+self.batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
    
    def predict(self, X):
        X = torch.FloatTensor(X).to(self.device)
        self.model.eval()  # 评估模式
        with torch.no_grad():
            outputs = self.model(X)
            return outputs.cpu().numpy()
    
    def predict_proba(self, X):
        X = torch.FloatTensor(X).to(self.device)
        self.model.eval()  # 评估模式
        with torch.no_grad():
            outputs = self.model(X)
            return torch.softmax(outputs, dim=1).cpu().numpy()

            
class RC_model(object):
    r"""Build and evaluate a RC-based model for time series classification or clustering.

    The training and test Multivariate Time Series (MTS) are multidimensional arrays of shape ``[N,T,V]``, where ``N`` is the number of samples, ``T`` is the number of time steps in each sample, ``V`` is the number of variables in each sample.
    
    Training and test labels have shape ``[N,C]``, with ``C`` being the number of classes.
    
    The dataset consists of training data and respective labels ``(X, Y)`` and test data and respective labels ``(Xte, Yte)``.
    
    **Reservoir parameters:**
    
    :param reservoir: object of class ``Reservoir`` (default ``None``) 
        Precomputed reservoir. If ``None``, the following structural hyperparameters must be specified.
    :param n_internal_units: int (default ``100``) 
        Processing units in the reservoir.
    :param spectral_radius: float (default ``0.99``) 
        Largest eigenvalue of the reservoir matrix of connection weights.
        To ensure the Echo State Property, set ``spectral_radius <= leak <= 1``)
    :param leak: float (default ``None``) 
        Amount of leakage in the reservoir state update.
        If ``None`` or ``1.0``, no leakage is used.
    :param connectivity: float (default ``0.3``) 
        Percentage of nonzero connection weights.
        Unused in circle reservoir.
    :param input_scaling: float (default ``0.2``)
        Scaling of the input connection weights.
        Note that the input weights are randomly drawn from ``{-1,1}``.
    :param noise_level: float (default ``0.0``) 
        Standard deviation of the Gaussian noise injected in the state update.
    :param n_drop: int (default ``0``)
        Number of transient states to drop.
    :param bidir: bool (default ``False``)
        Use a bidirectional reservoir (``True``) or a standard one (``False``).
    :param circle: bool (default ``False``)
        Generate determinisitc reservoir with circle topology where each connection 
        has the same weight.
    
    **Dimensionality reduction parameters:**
    
    :param dimred_method: str (default ``None``)
        Procedure for reducing the number of features in the sequence of reservoir states. 
        Possible options are: ``None`` (no dimensionality reduction), ``'pca'`` (standard PCA), 
        or ``'tenpca'`` (tensorial PCA for multivariate time series data).
    :param n_dim: int (default ``None``) 
        Number of resulting dimensions after the dimensionality reduction procedure.
    
    **Representation parameters:**
    
    :param mts_rep: str (default ``'concat'``) 
        Type of MTS representation. 
        It can be ``'last'`` (last state), ``'mean'`` (mean of all the states),
        ``'output'`` (output model space), ``'reservoir'`` (reservoir model space), 
        ``'ols'`` (ordinary least squares model space), or
        ``'concat'`` (flattened concatenation of all time steps and features).
    :param w_ridge_embedding: float (default ``1.0``) 
        Regularization parameter of the ridge regression in the output model space 
        and reservoir model space representation; ignored if ``mts_rep == None``.
    
    **Readout parameters:**
    
    :param readout_type: str (default ``'lin'``) 
        Type of readout used for classification. It can be ``'lin'`` (ridge regression), 
        ``'mlp'`` (multiplayer perceptron), ``'svm'`` (support vector machine), 
        or ``None``. 
        If ``None``, the input representations will be saved in the ``.input_repr`` attribute: 
        this is useful for clustering and visualization.
        Also, if ````None````, the other Readout hyperparameters can be left unspecified.
    :param w_ridge: float (default ``1.0``) 
        Regularization parameter of the ridge regression readout (only for ``readout_type=='lin'``).
    :param mlp_layout: tuple (default ``None``) 
        Tuple with the sizes of MLP layers, e.g., ``(20, 10)`` defines a MLP with 2 layers of 20 and 10 units respectively.
        Used only when ``readout_type=='mlp'``.
    :param num_epochs: int (default ``None``) 
        Number of iterations during the optimization.
        Used only when ``readout_type=='mlp'``.
    :param w_l2: float (default ``None``) 
        Weight of the L2 regularization.
        Used only when ``readout_type=='mlp'``.
    :param nonlinearity: str (default ``None``) 
        Type of activation function ``{'relu', 'tanh', 'logistic', 'identity'}``.
        Used only when ``readout_type=='mlp'``.
    :param svm_gamma: float (default ``1.0``) 
        Bandwidth of the RBF kernel.
        Used only when ``readout_type=='svm'``.
    :param svm_C: float (default ``1.0``) 
        Regularization for SVM hyperplane.
        Used only when ``readout_type=='svm'``.
    :param dropout_rate: float (default ``0.2``)
        Dropout rate for the MLP readout.
    """
    
    def __init__(self,
              # reservoir
              reservoir=None,     
              n_internal_units=100,
              spectral_radius=0.99,
              leak=0.9,  # 可以是float或list类型，例如[0.3, 0.5, 0.4]
              connectivity=0.3,
              input_scaling=0.2,
              noise_level=0.0,
              n_drop=0,
              bidir=False,
              circle=False,
              # dim red
              dimred_method=None, 
              n_dim=None,
              # representation
              mts_rep='concat',
              w_ridge_embedding=1.0,
              # readout
              readout_type='mlp',               
              w_ridge=1.0,              
              mlp_layout=(100,),
              num_epochs=200,
              w_l2=0.1,
              nonlinearity='relu', 
              svm_gamma=1.0,
              svm_C=1.0,
              dropout_rate=0.2,
              multi_timescale = False):

        self.n_drop=n_drop
        self.bidir=bidir
        self.dimred_method=dimred_method
        self.mts_rep=mts_rep
        self.readout_type=readout_type
        self.svm_gamma=svm_gamma
        self.dropout_rate = dropout_rate
        self.multi_timescale = multi_timescale
                        
        # Initialize reservoir
        if reservoir is None:
            self._reservoir = Reservoir(n_internal_units=n_internal_units,
                                  spectral_radius=spectral_radius,
                                  leak=leak,
                                  connectivity=connectivity,
                                  input_scaling=input_scaling,
                                  noise_level=noise_level,
                                  circle=circle)
        else:
            self._reservoir = reservoir
                
        # Initialize dimensionality reduction method
        if dimred_method is not None:
            if dimred_method.lower() == 'pca':
                self._dim_red = PCA(n_components=n_dim)            
            elif dimred_method.lower() == 'tenpca':
                self._dim_red = tensorPCA(n_components=n_dim)
            else:
                raise RuntimeError('Invalid dimred method ID')
                
        # Initialize ridge regression model
        if mts_rep=='output' or mts_rep=='reservoir' or mts_rep=='ols':
            self._ridge_embedding = Ridge(alpha=w_ridge_embedding, fit_intercept=True)
            self._ols_embedding = LinearRegression(fit_intercept=True)
                        
        # Initialize readout type            
        if self.readout_type is not None:
            
            if self.readout_type == 'lin': # Ridge regression
                self.readout = Ridge(alpha=w_ridge)        
            elif self.readout_type == 'svm': # SVM readout
                self.readout = SVC(C=svm_C, kernel='precomputed')          
            elif readout_type == 'mlp': # MLP (deep readout)  
                self.readout = MLPClassifier(
                    hidden_layer_sizes=mlp_layout, 
                    activation=nonlinearity, 
                    alpha=w_l2,  # L2正则化系数
                    batch_size=64, 
                    learning_rate='adaptive',
                    learning_rate_init=0.001, 
                    max_iter=num_epochs, 
                    shuffle=True,
                    random_state=42,
                    early_stopping=True,  # 启用早停
                    validation_fraction=0.1,  # 使用10%的数据作为验证集
                    n_iter_no_change=10,  # 10轮无改善则停止
                    tol=1e-4  # 早停的容忍度
                )
            else:
                raise RuntimeError('Invalid readout type')  
        
        
    def fit(self, X, Y=None, verbose=True):
        r"""Train the RC model.

        Parameters:
        -----------
        X : np.ndarray 
            Array of of shape ``[N, T, V]`` representin the training data.

        Y : np.ndarray 
            Array of shape ``[N, C]`` representing the target values.

        verbose : bool
            If ``True``, print the training time.

        Returns:
        --------
        None
        """
                
        time_start = time.time()
        
        # 打印输入数据的形状
        print(f"输入数据形状: X = {X.shape}")
        
        # ============ Compute reservoir states ============ 
        print('time scale is',self.multi_timescale)
        res_states = self._reservoir.get_states(X, multi_timescale = self.multi_timescale, n_drop=self.n_drop, bidir=self.bidir)
        
        
        # 打印储备池状态的形状
        print(f"储备池状态形状: res_states = {res_states.shape}")
        
        # ============ Dimensionality reduction of the reservoir states ============  
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                # matricize
                N_samples = res_states.shape[0]
                res_states = res_states.reshape(-1, res_states.shape[2])                   
                # ..transform..
                red_states = self._dim_red.fit_transform(res_states)          
                # ..and put back in tensor form
                red_states = red_states.reshape(N_samples,-1,red_states.shape[1])          
            elif self.dimred_method.lower() == 'tenpca':
                red_states = self._dim_red.fit_transform(res_states)       
            
            # 打印降维后的形状
            print(f"降维后的形状: red_states = {red_states.shape}")
        else: # Skip dimensionality reduction
            red_states = res_states
            print(f"未降维的形状: red_states = {red_states.shape}")

        # ============ Generate representation of the MTS ============
        coeff_tr = []
        biases_tr = []   
        
        # Output model space representation
        if self.mts_rep=='output':
            if self.bidir:
                X = np.concatenate((X,X[:, ::-1, :]),axis=2)                
                
            for i in range(X.shape[0]):
                # self._ridge_embedding.fit(red_states[i, 0:-1, :], X[i, self.n_drop+1:, :])
                self._ols_embedding.fit(red_states[i, 0:-1, :], red_states[i, 1:, :])
                coeff_tr.append(self._ols_embedding.coef_.ravel())
                biases_tr.append(self._ols_embedding.intercept_.ravel())
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
            
        # Reservoir model space representation
        elif self.mts_rep=='reservoir':
            for i in range(X.shape[0]):
                self._ridge_embedding.fit(red_states[i, 0:-1, :], red_states[i, 1:, :])
                coeff_tr.append(self._ridge_embedding.coef_.ravel())
                biases_tr.append(self._ridge_embedding.intercept_.ravel())
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
            
        # Ordinary Least Squares model space representation
        elif self.mts_rep=='ols':
            for i in range(X.shape[0]):
                self._ols_embedding.fit(red_states[i, 0:-1, :], red_states[i, 1:, :])
                coeff_tr.append(self._ols_embedding.coef_.ravel())
                biases_tr.append(self._ols_embedding.intercept_.ravel())
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
        
        # Last state representation        
        elif self.mts_rep=='last':
            input_repr = red_states[:, -1, :]
            
        # Mean state representation        
        elif self.mts_rep=='mean':
            input_repr = np.mean(red_states, axis=1)
            
        # Concatenate representation - flatten the time and features dimensions
        elif self.mts_rep=='concat':
            # 将形状从[N, T, F]转换为[N, T*F]
            N, T, F = red_states.shape
            input_repr = red_states.reshape(N, T*F)
            print(f"拼接后的形状: input_repr = {input_repr.shape}, 从 {N}×{T}×{F} 重塑而来")
            
        else:
            raise RuntimeError('Invalid representation ID')            
            
        # ============ Train readout ============
        if self.readout_type == None: # Just store the input representations
            self.input_repr = input_repr
            
        elif self.readout_type == 'lin': # Ridge regression
            self.readout.fit(input_repr, Y)          
            
        elif self.readout_type == 'svm': # SVM readout
            Ktr = squareform(pdist(input_repr, metric='sqeuclidean')) 
            Ktr = np.exp(-self.svm_gamma*Ktr)
            self.readout.fit(Ktr, np.argmax(Y,axis=1))
            self.input_repr_tr = input_repr # store them to build test kernel
            
        elif self.readout_type == 'mlp': # MLP (deep readout)
            self.readout.fit(input_repr, Y)
            # 计算训练集预测结果
            train_pred = self.readout.predict(input_repr)
            train_pred = np.argmax(train_pred, axis=1)
            train_true = np.argmax(Y, axis=1)
            
            # 计算评估指标
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            accuracy = accuracy_score(train_true, train_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(train_true, train_pred, average='weighted')
            
            print(f"\n训练集评估指标:")
            print(f"准确率: {accuracy:.4f}")
            print(f"精确率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分数: {f1:.4f}")
            
        if verbose:
            tot_time = (time.time()-time_start)/60
            print(f"Training completed in {tot_time:.2f} min")

            
    def predict(self, Xte):
        r"""Computes predictions for out-of-sample (test) data.

        Parameters:
        -----------
        Xte : np.ndarray
            Array of shape ``[N, T, V]`` representing the test data.

        Returns:
        --------
        pred_class : np.ndarray
            Array of shape ``[N]`` representing the predicted classes.
        """

        # 打印测试数据的形状
        print(f"测试数据形状: Xte = {Xte.shape}")

        # ============ Compute reservoir states ============
        res_states_te = self._reservoir.get_states(Xte, multi_timescale=self.multi_timescale,n_drop=self.n_drop, bidir=self.bidir) 
        
        # 打印测试储备池状态的形状
        print(f"测试储备池状态形状: res_states_te = {res_states_te.shape}")
        
        # ============ Dimensionality reduction of the reservoir states ============ 
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                # matricize
                N_samples_te = res_states_te.shape[0]
                res_states_te = res_states_te.reshape(-1, res_states_te.shape[2])                    
                # ..transform..
                red_states_te = self._dim_red.transform(res_states_te)            
                # ..and put back in tensor form
                red_states_te = red_states_te.reshape(N_samples_te,-1,red_states_te.shape[1])            
            elif self.dimred_method.lower() == 'tenpca':
                red_states_te = self._dim_red.transform(res_states_te)        
        else: # Skip dimensionality reduction
            red_states_te = res_states_te             
        
        # ============ Generate representation of the MTS ============
        coeff_te = []
        biases_te = []   
        
        # Output model space representation
        if self.mts_rep=='output':
            if self.bidir:
                Xte = np.concatenate((Xte,Xte[:, ::-1, :]),axis=2)  
                    
            for i in range(Xte.shape[0]):
                self._ridge_embedding.fit(red_states_te[i, 0:-1, :], Xte[i, self.n_drop+1:, :])
                coeff_te.append(self._ridge_embedding.coef_.ravel())
                biases_te.append(self._ridge_embedding.intercept_.ravel())
            input_repr_te = np.concatenate((np.vstack(coeff_te), np.vstack(biases_te)), axis=1)
        
        # Reservoir model space representation
        elif self.mts_rep=='reservoir':    
            for i in range(Xte.shape[0]):
                self._ridge_embedding.fit(red_states_te[i, 0:-1, :], red_states_te[i, 1:, :])
                coeff_te.append(self._ridge_embedding.coef_.ravel())
                biases_te.append(self._ridge_embedding.intercept_.ravel())
            input_repr_te = np.concatenate((np.vstack(coeff_te), np.vstack(biases_te)), axis=1)
            
        # Ordinary Least Squares model space representation
        elif self.mts_rep=='ols':    
            for i in range(Xte.shape[0]):
                self._ols_embedding.fit(red_states_te[i, 0:-1, :], red_states_te[i, 1:, :])
                coeff_te.append(self._ols_embedding.coef_.ravel())
                biases_te.append(self._ols_embedding.intercept_.ravel())
            input_repr_te = np.concatenate((np.vstack(coeff_te), np.vstack(biases_te)), axis=1)
    
        # Last state representation        
        elif self.mts_rep=='last':
            input_repr_te = red_states_te[:, -1, :]
            
        # Mean state representation        
        elif self.mts_rep=='mean':
            input_repr_te = np.mean(red_states_te, axis=1)
            
        # Concatenate representation - flatten the time and features dimensions
        elif self.mts_rep=='concat':
            # 将形状从[N, T, F]转换为[N, T*F]
            N, T, F = red_states_te.shape
            input_repr_te = red_states_te.reshape(N, T*F)
            print(f"测试拼接后的形状: input_repr_te = {input_repr_te.shape}, 从 {N}×{T}×{F} 重塑而来")
            
        else:
            raise RuntimeError('Invalid representation ID')   
            
        # ============ Apply readout ============
        if self.readout_type == 'lin': # Ridge regression        
            logits = self.readout.predict(input_repr_te)
            pred_class = np.argmax(logits, axis=1)
            
        elif self.readout_type == 'svm': # SVM readout
            Kte = cdist(input_repr_te, self.input_repr_tr, metric='sqeuclidean')
            Kte = np.exp(-self.svm_gamma*Kte)
            pred_class = self.readout.predict(Kte)
            
        elif self.readout_type == 'mlp': # MLP (deep readout)
            pred_class = self.readout.predict(input_repr_te)
            pred_class = np.argmax(pred_class, axis=1)
            
        return pred_class

    def save_weights(self, path):
        """保存模型权重到指定路径
        
        Parameters:
        -----------
        path : str
            保存权重的文件路径
        """
        if self.readout_type == 'mlp':
            import joblib
            joblib.dump(self.readout, path)
        else:
            raise RuntimeError('只有MLP类型的readout支持保存权重')

    def load_weights(self, path):
        """从指定路径加载模型权重
        
        Parameters:
        -----------
        path : str
            权重文件的路径
        """
        if self.readout_type == 'mlp':
            import joblib
            self.readout = joblib.load(path)
        else:
            raise RuntimeError('只有MLP类型的readout支持加载权重')

class RC_forecaster(object):
    r"""Class to perform time series forecasting with RC.

    The training and test data are multidimensional arrays of shape ``[T,V]``, with

    - ``T`` = number of time steps in each sample,
    - ``V`` = number of variables in each sample.

    Given a time series ``X``, the training data are supposed to be as follows:
    
        ``Xtr, Ytr = X[0:-forecast_horizon,:], X[forecast_horizon:,:]``

    Once trained, the model can be used to compute prediction ``forecast_horizon`` steps ahead:
        
            ``Yhat[t,:] = Xte[t+forecast_horizon,:]``

    **Reservoir parameters:**

    :param reservoir: object of class ``Reservoir`` (default ``None``)
        Precomputed reservoir. If ``None``, the following structural hyperparameters must be specified.
    :param n_internal_units: int (default ``100``) 
        Processing units in the reservoir.
    :param spectral_radius: float (default ``0.99``) 
        Largest eigenvalue of the reservoir matrix of connection weights.
        To ensure the Echo State Property, set ``spectral_radius <= leak <= 1``)
    :param leak: float (default ``None``) 
        Amount of leakage in the reservoir state update.
    :param connectivity: float (default ``0.3``)
        Percentage of nonzero connection weights.
    :param input_scaling: float (default ``0.2``) 
        Scaling of the input connection weights.
        Note that the input weights are randomly drawn from ``{-1,1}``.
    :param noise_level: float (default ``0.0``)
        Standard deviation of the Gaussian noise injected in the state update.
    :param n_drop: int (default ``0``)
        Number of transient states to drop.
    :param circle: bool (default ``False``)
        Generate determinisitc reservoir with circle topology where each connection 
        has the same weight.

    **Dimensionality reduction parameters:**

    :param dimred_method: str (default ``None``)
        Procedure for reducing the number of features in the sequence of reservoir states. 
        Possible options are: ``None`` (no dimensionality reduction), ``'pca'`` (standard PCA), 
        or ``'tenpca'`` (tensorial PCA for multivariate time series data).
    :param n_dim: int (default ``None``) 
        Number of resulting dimensions after the dimensionality reduction procedure.

    **Readout parameters:**

    :param w_ridge: float (default ``1.0``) 
        Regularization parameter of the ridge regression readout.
    """
    
    def __init__(self,
                # reservoir
                reservoir=None,     
                n_internal_units=100,
                spectral_radius=0.99,
                leak=None,
                connectivity=0.3,
                input_scaling=0.2,
                noise_level=0.0,
                n_drop=0,
                circle=False,
                # dim red
                dimred_method=None, 
                n_dim=None,
                # readout              
                w_ridge=1.0):
        self.n_drop=n_drop
        self.dimred_method=dimred_method  
                        
        # Initialize reservoir
        if reservoir is None:
            self._reservoir = Reservoir(n_internal_units=n_internal_units,
                                        spectral_radius=spectral_radius,
                                        leak=leak,
                                        connectivity=connectivity,
                                        input_scaling=input_scaling,
                                        noise_level=noise_level,
                                        circle=circle)
        else:
            self._reservoir = reservoir
                
        # Initialize dimensionality reduction method
        if dimred_method is not None:
            if dimred_method.lower() == 'pca':
                self._dim_red = PCA(n_components=n_dim)            
            else:
                raise RuntimeError('Invalid dimred method ID')
            
        # Initialize readout
        self.readout = Ridge(alpha=w_ridge)


    def fit(self, X, Y, verbose=True):
        r"""Train the RC model for forecasting.

        Parameters:
        -----------
        X : np.ndarray 
            Array of shape ``[T, V]`` representing the training data.

        Y : np.ndarray
            Array of shape ``[T, V]`` representing the target values.

        verbose : bool
            If ``True``, print the training time.

        Returns:
        --------
        red_states : np.ndarray
            Array of shape ``[T, n_dim]`` representing the reservoir states of the time steps used for training.
        """
        
        time_start = time.time()
        
        # ============ Compute reservoir states ============ 
        res_states = self._reservoir.get_states(X[None,:,:], n_drop=self.n_drop, bidir=False)
        
        # ============ Dimensionality reduction of the reservoir states ============  
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                red_states = self._dim_red.fit_transform(res_states[0])          
        else: # Skip dimensionality reduction
            red_states = res_states[0]

        self._fitted_states = red_states

        # ============ Train readout ============
        self.readout.fit(red_states, Y[self.n_drop:,:])          
            
        if verbose:
            tot_time = (time.time()-time_start)/60
            print(f"Training completed in {tot_time:.2f} min")

        return red_states

    def predict(self, Xte, return_states=False):
        r"""Computes predictions for out-of-sample (test) data.

        Parameters:
        -----------
        Xte : np.ndarray
            Array of shape ``[T, V]`` representing the test data.
        
        return_states : bool
            If ``True``, return the predicted states.

        Returns:
        --------
        Yhat : np.ndarray
            Array of shape ``[T, V]`` representing the predicted values.
        
        red_states_te : np.ndarray
            Array of shape ``[T, n_dim]`` representing the reservoir states of the new time steps.
        """

        # ============ Compute reservoir states ============
        res_states_te = self._reservoir.get_states(Xte[None,:,:], n_drop=self.n_drop, bidir=False) 
        
        # ============ Dimensionality reduction of the reservoir states ============ 
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                red_states_te = self._dim_red.transform(res_states_te[0])                          
        else: # Skip dimensionality reduction
            red_states_te = res_states_te[0]        

        self._predicted_states = red_states_te

        # ============ Apply readout ============
        Yhat = self.readout.predict(red_states_te)

        if return_states:
            return Yhat, red_states_te
        return Yhat

    def get_fitted_states(self):
        r"""Return the fitted reservoir states.

        Returns:
        --------
        fitted_states : np.ndarray
            Array of shape ``[T, n_dim]`` representing the fitted reservoir states.
        """
        return self._fitted_states

    def get_predicted_states(self):
        r"""Return the predicted reservoir states.

        Returns:
        --------
        predicted_states : np.ndarray
            Array of shape ``[T, n_dim]`` representing the predicted reservoir states.
        """
        return self._predicted_states
