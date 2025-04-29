from calendar import c
import numpy as np
from scipy import sparse
from typing import Union, List, Optional, Any, cast

class Reservoir(object):    
    r"""
        Build a reservoir and compute the sequence of the internal states.
        
        Parameters:
        ------------
        n_internal_units : int (default ``100``)
            Processing units in the reservoir.
        spectral_radius : float (default ``0.99``)
            Largest eigenvalue of the reservoir matrix of connection weights.
            To ensure the Echo State Property, set ``spectral_radius <= leak <= 1``)
        leak : float or list (default ``None``)
            Amount of leakage in the reservoir state update. 
            If ``None`` or ``1.0``, no leakage is used.
            If a list is provided, multiple sub-reservoirs with different leakage rates will be created.
        connectivity : float (default ``0.3``)
            Percentage of nonzero connection weights.
            Unused in circle reservoir.
        input_scaling : float or list (default ``0.2``)
            Scaling of the input connection weights.
            If a list is provided, different scaling will be applied to each sub-reservoir.
            Note that the input weights are randomly drawn from ``{-1,1}``.
        noise_level : float (default ``0.0``)
            Standard deviation of the Gaussian noise injected in the state update.
        circle : bool (default ``False``)
            Generate determinisitc reservoir with circle topology where each connection 
            has the same weight.
        multi_timescale : bool (default ``False``)
            If True, creates multiple sub-reservoirs with different timescales.
        n_timescales : int (default ``3``)
            Number of different timescales (sub-reservoirs) to use when multi_timescale is True.
        """

    def __init__(self, 
                 n_internal_units=100, 
                 spectral_radius=0.99, 
                 leak=[0.3, 0.5, 0.4],
                 connectivity=0.3, 
                 input_scaling=0.2, 
                 noise_level=0.0, 
                 circle=False,
                 n_timescales=3,
                 multi_timescale=True):
       
        # Initialize hyperparameters
        self.multi_timescale = multi_timescale
        self._n_timescales = n_timescales
        self._noise_level = noise_level
        
        if self.multi_timescale:
            # 如果启用多时间尺度，将参数转换为列表
            if isinstance(n_internal_units, int):
                # 平均分配每个时间尺度的神经元数量
                self._n_internal_units = n_internal_units
                self._sub_units = n_internal_units // n_timescales
                # 确保每个子储备池至少有一个单元
                if self._sub_units == 0:
                    self._sub_units = 1
                    self._n_internal_units = n_timescales
            
            # 确保leak是列表类型
            if isinstance(leak, float) or leak is None:
                leak_value = 0.1 if leak is None else leak
                self._leak = [leak_value] * n_timescales
            elif isinstance(leak, list):
                if len(leak) != n_timescales:
                    # 如果提供的列表长度不匹配，调整它
                    if len(leak) > n_timescales:
                        self._leak = leak[:n_timescales]
                    else:
                        # 如果提供的列表太短，复制最后一个值
                        self._leak = leak + [leak[-1]] * (n_timescales - len(leak))
                else:
                    self._leak = leak
            else:
                # 如果是其他类型，创建默认列表
                self._leak = [0.3] * n_timescales
            
            # 为每个子储备池设置不同的输入缩放
            if isinstance(input_scaling, float):
                self._input_scaling = [input_scaling] * n_timescales
            elif isinstance(input_scaling, list):
                assert len(input_scaling) == n_timescales, "提供的输入缩放列表必须与时间尺度数量匹配"
                self._input_scaling = input_scaling
            
            # 初始化多个子储备池的内部权重列表
            self._internal_weights = []  # type: List[np.ndarray]
            # 初始化为空列表，稍后在计算时填充
            self._input_weights = []     # type: List[np.ndarray]
            
            for i in range(n_timescales):
                if circle:
                    internal_w = self._initialize_internal_weights_Circ(
                        self._sub_units, spectral_radius)
                else:
                    internal_w = self._initialize_internal_weights(
                        self._sub_units, connectivity, spectral_radius)
                self._internal_weights.append(internal_w)
        else:
            # 原始单一时间尺度实现
            self._n_internal_units = n_internal_units
            self._input_scaling = input_scaling
            self._leak = leak
            # 单一时间尺度时初始化为空列表而非None，以保持类型一致性
            self._input_weights = []     # type: List[np.ndarray]
            # 生成内部权重
            if circle:
                self._internal_weights = self._initialize_internal_weights_Circ(
                        n_internal_units, spectral_radius)
            else:
                self._internal_weights = self._initialize_internal_weights(
                    n_internal_units, connectivity, spectral_radius)


    def _initialize_internal_weights_Circ(self, n_internal_units, spectral_radius):
        """Generate internal weights with circular topology.
        """
        
        # Construct reservoir with circular topology
        internal_weights = np.zeros((n_internal_units, n_internal_units))
        internal_weights[0,-1] = 1.0
        for i in range(n_internal_units-1):
            internal_weights[i+1,i] = 1.0
            
        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max)/spectral_radius 
                
        return internal_weights
    
    
    def _initialize_internal_weights(self, n_internal_units,
                                     connectivity, spectral_radius):
        """Generate internal weights with a sparse, uniformly random topology.
        """

        # Generate sparse, uniformly distributed weights.
        internal_weights = sparse.rand(n_internal_units,
                                       n_internal_units,
                                       density=connectivity).todense()

        # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
        internal_weights[np.where(internal_weights > 0)] -= 0.5
        
        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max)/spectral_radius       

        return internal_weights


    def _compute_state_matrix(self, X, multi_timescale, n_drop=0, previous_state=None):
        """Compute the reservoir states on input data X.
        
        """
        self.multi_timescale = multi_timescale
        N, T, V = X.shape
        
        if self.multi_timescale:
            print('multi time scales processing')
            # print('leak rate is :', self._leak)
            # 确保类型正确处理
            input_scaling_list = cast(List[float], self._input_scaling)
            leak_list = cast(List[float], self._leak)  # 确保leak_list是列表类型
            
            # 初始化状态
            if previous_state is None:
                # 每个子储备池有自己的初始状态
                previous_states = [np.zeros((N, self._sub_units), dtype=float) for _ in range(self._n_timescales)]
            else:
                # 如果提供了初始状态，将其分割为多个子储备池状态
                previous_states = []
                start_idx = 0
                for i in range(self._n_timescales):
                    end_idx = start_idx + self._sub_units
                    if end_idx <= previous_state.shape[1]:
                        previous_states.append(previous_state[:, start_idx:end_idx])
                    else:
                        # 如果初始状态尺寸不匹配，使用零矩阵
                        previous_states.append(np.zeros((N, self._sub_units), dtype=float))
                    start_idx = end_idx
            
            # 初始化输入权重（如果尚未初始化）
            if isinstance(self._input_weights, list) and len(self._input_weights) == 0:
                for i in range(self._n_timescales):
                    scaling = input_scaling_list[i]
                    iw = (2.0 * np.random.binomial(1, 0.5, [self._sub_units, V]) - 1.0) * scaling
                    self._input_weights.append(iw)
            
            # 确保_input_weights是List[np.ndarray]类型
            input_weights_list = cast(List[np.ndarray], self._input_weights)
            internal_weights_list = cast(List[np.ndarray], self._internal_weights)
            
            # 存储状态
            if T - n_drop > 0:
                window_size = T - n_drop
            else:
                window_size = T
            
            # 为所有子储备池的状态分配空间
            total_units = self._sub_units * self._n_timescales
            state_matrix = np.empty((N, window_size, total_units), dtype=float)
            
            for t in range(T):
                current_input = X[:, t, :]
                all_states = []
                # print('x size is :', X.shape)
                
                # 更新每个子储备池
                for i in range(self._n_timescales):
                    time_scale = int(V/self._n_timescales)
                    # print('time_scale is :', time_scale)
                    state_before_tanh = current_input[:,i*time_scale:time_scale*(i+1)]
                    # 使用矩阵点乘操作
                    # iw = input_weights_list[i]
                    # iwt = iw.T  # 确保使用numpy数组的T属性而不是列表的
                    # state_before_tanh = np.dot(previous_states[i], internal_weights_list[i]) + np.dot(current_input, iwt)
                    #
                    # # 添加噪声
                    # if self._noise_level > 0:
                    #     state_before_tanh += np.random.normal(0, self._noise_level, state_before_tanh.shape)
                    
                    # 应用非线性与泄漏 - 确保泄漏率是浮点数
                    leak_rate = leak_list[i]  # 现在leak_list保证是列表类型
                    new_state = (1.0 - leak_rate) * previous_states[i] + leak_rate*np.maximum(0, state_before_tanh)
                    previous_states[i] = new_state
                    all_states.append(new_state)
                
                # 合并所有子储备池的状态
                combined_state = np.hstack(all_states)
                
                # 存储状态
                if T - n_drop > 0 and t >= n_drop:
                    state_matrix[:, t - n_drop, :] = combined_state
                elif T - n_drop <= 0:
                    state_matrix[:, t, :] = combined_state
            
            return state_matrix
        else:
            # 原始的单一时间尺度逻辑
            if previous_state is None:
                previous_state = np.zeros((N, self._n_internal_units), dtype=float)

            # 初始化输入权重（如果尚未初始化）
            # if len(self._input_weights) == 0:
            #     input_weights = (2.0 * np.random.binomial(1, 0.5, [self._n_internal_units, V]) - 1.0) * self._input_scaling
            #     self._input_weights = [input_weights]  # 保持类型一致性

            # Storage
            if T - n_drop > 0:
                window_size = T - n_drop
            else:
                window_size = T
            state_matrix = np.empty((N, window_size, self._n_internal_units), dtype=float)

            for t in range(T):
                current_input = X[:, t, :]
                
                

                # 计算状态 - 确保_input_weights是numpy数组
                # input_weights = self._input_weights[0]  # 获取第一个（也是唯一一个）输入权重矩阵
                # internal_weights = cast(np.ndarray, self._internal_weights)
                # state_before_tanh = np.dot(previous_state, internal_weights) + np.dot(current_input, input_weights.T)

                # 添加噪声
                # if self._noise_level > 0:
                #     state_before_tanh += np.random.normal(0, self._noise_level, state_before_tanh.shape)
                #
                # # 应用非线性和泄漏（可选）
                if self._leak is None:
                    previous_state = np.tanh(current_input)
                elif isinstance(self._leak, list):
                    # 如果是列表，使用第一个值
                    leak_rate = float(self._leak[0])
                    # print('leak rate is :', leak_rate)
                    previous_state = (1.0 - leak_rate) * previous_state + np.tanh(current_input)
                else:
                    # 否则，直接使用值
                    leak_rate = float(self._leak)
                    # print('leak rate is :', leak_rate)
                    previous_state = (1.0 - leak_rate) * previous_state + np.tanh(current_input)

                # 存储状态
                if T - n_drop > 0 and t >= n_drop:
                    state_matrix[:, t - n_drop, :] = previous_state
                elif T - n_drop <= 0:
                    state_matrix[:, t, :] = previous_state

            return state_matrix


    def get_states(self, X, multi_timescale=True, n_drop=0, bidir=True, initial_state=None):
        r"""
        Compute reservoir states and return them.

        Parameters:
        ------------
        X : np.ndarray
            Time series, 3D array of shape ``[N,T,V]``, where ``N`` is the number of time series,
            ``T`` is the length of each time series, and ``V`` is the number of variables in each
            time point.
        n_drop : int (default is ``0``)
            Washout period, i.e., number of initial samples to drop due to the transient phase.
        bidir : bool (default is ``True``)
            If ``True``, use bidirectional reservoir
        initial_state : np.ndarray (default is ``None``)
            Initialize the first state of the reservoir to the given value.
            If ``None``, the initial states is a zero-vector. 

        Returns:
        ------------
        states : np.ndarray
            Reservoir states, 3D array of shape ``[N,T,n_internal_units]``, where ``N`` is the number
            of time series, ``T`` is the length of each time series, and ``n_internal_units`` is the
            number of processing units in the reservoir.
        """

        # 计算储备池状态序列
        print('time scale is :',multi_timescale)
        states = self._compute_state_matrix(X,  multi_timescale, n_drop, previous_state=initial_state)
    
        # 反向时间输入的储备池状态（如果启用双向）
        if bidir is True:
            X_r = X[:, ::-1, :]
            states_r = self._compute_state_matrix(X_r, n_drop)
            states = np.concatenate((states, states_r), axis=2)

        return states