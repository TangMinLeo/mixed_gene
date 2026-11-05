import json
import numpy as np
from datetime import timezone, timedelta
import datetime
import math
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
import logging
from tqdm import tqdm
from zoneinfo import ZoneInfo
from pyproj import Transformer
from scipy.interpolate import interp1d


R = 6371000.0  # 地球半径，单位为米

# mea = 7.3930369290044915
# std = 1.8804091151810995

mea_delta = 3.1487522331724422
std_delta = 0.9709251038940904

# min_lon = 113.683333
# max_lon = 115.083333
# min_lat = 29.966667
# max_lat = 31.366667

min_lon = 115.416667
max_lon = 117.500000
min_lat = 39.433333
max_lat = 41.050000

# select_len = 520
num_freqs = 32

# def get_original_data(file_name_json):
#     with open(file_name_json, 'r') as f:
#         data = json.load(f)
#     # print(data[0])
    
#     time_seq = []
#     posi_seq = []
    
#     for one_day in data:
        
#         one_day_time_seq = []
#         one_day_posi_seq = []
#         for one_point in one_day:
#             one_day_posi_seq.append((one_point[0], one_point[1]))
#             # 已经除以1000转化到单位秒
#             one_day_time_seq.append(one_point[2]/1000)
#         time_seq.append(one_day_time_seq)
#         posi_seq.append(one_day_posi_seq)
#     # time_seq = np.array(time_seq, dtype=np.float32)
#     # posi_seq = np.array(posi_seq, dtype=np.float32)
    
#     return time_seq, posi_seq

import torch

def check_input_layer_gradients(model, flag_dim=128, physics_dim=128):
    """
    检查第一层线性层的梯度分布，以诊断 Flag 信号是否过于强大或弱小。
    假设拼接顺序是 [物理特征, Embedding_Flag] 
    """
    # 1. 定位层：请根据你代码中实际的名称修改，例如 model.input_proj 或 model.fusion[0]
    # 这里假设是 model.input_block.proj
    target_layer = None
    for name, module in model.m_p_proj.named_modules():
        if isinstance(module, nn.Linear):
            target_layer = module
            print(f"检测到第一层线性层: {name}, 形状: {target_layer.weight.shape}")
            break # 找到第一个线性层就跳出
            
    if target_layer is None or target_layer.weight.grad is None:
        print("警告：未找到线性层或梯度尚未计算（请确保在 backward 之后调用）")
        return

    # 2. 提取梯度
    # weight.grad 形状通常是 [out_features, in_features]
    grads = target_layer.weight.grad
    
    # 3. 拆分物理维度和 Flag 维度的梯度
    # ！！！请务必确认你拼接时的顺序 ！！！
    # 假设拼接代码是: torch.cat([physics_feat, flag_feat], dim=-1)
    physics_grads = grads[:, :physics_dim]
    flag_grads = grads[:, physics_dim:physics_dim + flag_dim]

    # 4. 计算平均梯度范数 (Frobenius Norm)
    physics_norm = physics_grads.norm(p=2).item()
    flag_norm = flag_grads.norm(p=2).item()
    
    # 5. 计算信噪比 (Mean / Std) 也可以帮助判断梯度是否剧烈震荡
    print("-" * 30)
    print(f"【梯度诊断】")
    print(f"物理特征通道梯度范数: {physics_norm:.6f}")
    print(f"Embedding Flag 通道梯度范数: {flag_norm:.6f}")
    
    if flag_norm > 10 * physics_norm:
        print("结论：Flag 信号过于霸道，可能掩盖了物理特征，建议减小 Flag 初始化增益或加 LayerNorm。")
    elif physics_norm > 10 * flag_norm:
        print("结论：物理特征占据主导，Flag 信号较弱，可能导致起点锚定不稳。")
    else:
        print("结论：两者梯度量级相对平衡。")
    print("-" * 30)

    return physics_norm, flag_norm



def resample_trip_to_fixed_N(df_trip, target_N=64):
    """
    将单段变长行程重采样为固定长度 N，并提取时空生成特征。
    
    参数:
    df_trip: DataFrame, 必须包含 ['time', 'lng', 'lat']，且按时间排序
    target_N: int, 目标轨迹点数 (如 64, 128)
    
    返回:
    np.ndarray, 形状为 (target_N, 5)，特征包含 [lng, lat, rel_time, delta_t, progress]
    """
    orig_len = len(df_trip)
    if orig_len < 2:
        raise ValueError("行程点数过少，无法插值")

    # 1. 提取原始空间坐标和相对时间
    lngs = df_trip['lng'].values
    lats = df_trip['lat'].values
    
    # 计算相对时间（单位：秒，起点为 0）
    start_time = df_trip['time'].iloc[0]
    rel_times = (df_trip['time'] - start_time).dt.total_seconds().values
    
    # 2. 构建插值基准 (基于序列索引归一化到 [0, 1])
    # 为什么用索引？因为原始 GPS 点的密集程度代表了车速慢，稀疏代表车速快
    # 按索引插值可以完美保留这种“时空密度”的非线性分布
    orig_idx = np.linspace(0, 1, orig_len)
    target_idx = np.linspace(0, 1, target_N)
    
    # 3. 构建时空插值函数 (推荐线性插值，避免三次样条在路口拐角处产生振荡偏移)
    f_lng = interp1d(orig_idx, lngs, kind='linear')
    f_lat = interp1d(orig_idx, lats, kind='linear')
    f_time = interp1d(orig_idx, rel_times, kind='linear')
    
    # 4. 执行插值，获取固定长度 N 的新序列
    new_lng = f_lng(target_idx)
    new_lat = f_lat(target_idx)
    new_rel_time = f_time(target_idx)
    
    # 5. 计算生成模型的关键辅助特征
    # 特征 A: 进度索引 Progress Index (0.0 到 1.0)
    progress = target_idx 
    
    # 特征 B: 时间差变量 Delta Time (\Delta t_i = t_i - t_{i-1})
    # prepend=0 确保第一个点的 \Delta t 为 0
    delta_times = np.diff(new_rel_time, prepend=0.0)
    
    # 浮点数截断保护（确保时间严格单调递增，没有负数）
    delta_times = np.clip(delta_times, a_min=0.0, a_max=None)
    
    # 6. 堆叠为特征矩阵
    # shape: (target_N, 5)
    resampled_tensor = np.stack([
        new_lng,       # 经度
        new_lat,       # 纬度
        new_rel_time,  # 累计相对时间 (可选，有时模型只需要 delta_t)
        delta_times,   # 时间差变量 (你要去噪的核心目标之一)
        progress       # 进度索引
    ], axis=1)
    
    return resampled_tensor


class TimePeriodicEncoder():
    def __init__(self, out_dim=32, min_period=5.0, max_period=3600.0):
        """
        out_dim: 输出的总维度 (必须是偶数)
        min_period: 最小周期 (单位: 秒)，对应最高频，捕捉 5s 级的变化
        max_period: 最大周期 (单位: 秒)，对应最低频，捕捉 1小时 级的进度
        """
        assert out_dim % 2 == 0
        self.out_dim = out_dim
        
        # 计算频率系数 omega
        # 我们让周期从 min_period 变到 max_period
        # T = 2 * pi / omega => omega = 2 * pi / T
        num_frequencies = out_dim // 2
        periods = np.exp(np.linspace(
            np.log(min_period), 
            np.log(max_period), 
            num_frequencies
        ))
        
        # omega: [num_frequencies]
        self.omega = 2 * np.pi / periods

    def encode(self, t):
        """
        t: [Batch, Seq_Len, 1] 累积时间 T_total
        return: [Batch, Seq_Len, out_dim]
        """
        # [B, L, 1] * [F] -> [B, L, F]
        out = t * self.omega
        
        # 拼接 sin 和 cos
        # 结果维度: [B, L, out_dim]
        return np.concatenate([np.sin(out), np.cos(out)], axis=-1)


class ProgressPeriodicEncoder():
    def __init__(self, device, out_dim=32, min_period=5.0, max_period=3600.0):
        """
        out_dim: 输出的总维度 (必须是偶数)
        min_period: 最小周期 (单位: 秒)，对应最高频，捕捉 5s 级的变化
        max_period: 最大周期 (单位: 秒)，对应最低频，捕捉 1小时 级的进度
        """
        assert out_dim % 2 == 0
        self.out_dim = out_dim
        
        # 计算频率系数 omega
        # 我们让周期从 min_period 变到 max_period
        # T = 2 * pi / omega => omega = 2 * pi / T
        num_frequencies = out_dim // 2
        periods = torch.exp(torch.linspace(
            torch.log(torch.tensor(min_period)), 
            torch.log(torch.tensor(max_period)), 
            num_frequencies
        ))
        
        # omega: [num_frequencies]
        self.omega = 2 * torch.pi / periods
        self.omega = self.omega.to(device)  # 确保 omega 在 GPU 上，以便后续计算时不需要频繁迁移

    def encode(self, t):
        """
        t: [Batch, Seq_Len, 1] 累积时间 T_total
        return: [Batch, Seq_Len, out_dim]
        """
        # [B, L, 1] * [F] -> [B, L, F]
        out = t * self.omega
        
        # 拼接 sin 和 cos
        # 结果维度: [B, L, out_dim]
        return torch.concat([torch.sin(out), torch.cos(out)], dim=-1)


def get_geolife_data(file_name_json):
    f = open(file_name_json, 'r')
    data = json.load(f)
    f.close()
    
    time_seq = []
    posi_seq = []
    
    for one_trajectory in data:
        one_day_time_seq = []
        one_day_posi_seq = []
        for one_point in one_trajectory:
            datetime_str = f"{one_point[2]} {one_point[3]}"
    
            # 解析字符串为datetime对象
            dt = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            dt_local = dt.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
            
            # 转换为时间戳（单位为秒）
            timestamp = int(dt_local.timestamp())
            
            one_day_posi_seq.append((one_point[1], one_point[0]))
            one_day_time_seq.append(timestamp)
            # time_seq.append(one_point[2]/1000)
            
        # 轨迹点过密，下采样每隔10个点取一个点
        time_seq.append(one_day_time_seq[::5])
        posi_seq.append(one_day_posi_seq[::5])
        # time_seq.append(one_day_time_seq)
        # posi_seq.append(one_day_posi_seq)
    print(f'第0,1条轨迹点数量: {len(time_seq[0])}, {len(time_seq[1])}')
    print(f'第0,1条轨迹点数量: {len(posi_seq[0])}, {len(posi_seq[1])}')
    return time_seq, posi_seq

        
def form_features(time_seq, posi_seq, actual_l):
    
    tz_utc8 = timezone(timedelta(hours=8))
    
    # lengths = np.array([len(seq) for seq in time_seq])
    # sorted_indices = np.argsort(lengths)
    
    sorted_time_seq = []
    sorted_posi_seq = []
    mask_ = []
    
    select_len = actual_l
    emb_time_seq = []
    emb_posi_seq = []
    
    for i in range(len(time_seq)):
        len_seq = len(time_seq[i])
        # print(f'Original length of trajectory {i}: {len_seq}')
        # 如果轨迹长度小于5（10倍下采样后的5），则直接丢弃该轨迹；如果轨迹长度小于select_len，则在末尾进行0填充；如果轨迹长度大于等于select_len，则直接截断到select_len。
        if len_seq < 5:
            continue
        elif len_seq < select_len:
            sorted_time_seq.append(time_seq[i] + [0] * (select_len - len_seq))
            sorted_posi_seq.append(posi_seq[i] + [(0.0, 0.0)] * (select_len - len_seq))
            mask_.append([0]*len_seq+[1]*(select_len - len_seq))
        else:
            sorted_time_seq.append(time_seq[i][:select_len])
            sorted_posi_seq.append(posi_seq[i][:select_len])
            mask_.append([0]*select_len)
    
    sorted_time_seq = np.array(sorted_time_seq)
    sorted_posi_seq = np.array(sorted_posi_seq)
    # print(mask_[100])
    # print(sorted_time_seq[100])
    # print(sorted_posi_seq[100])
    
    for i in range(len(sorted_time_seq)):
        # the i-th time sequence
        time_seri = sorted_time_seq[i]
        # print(f'time_seri: {time_seri}')
        # print(f'mask_[i]: {mask_[i]}')
        # print(f'len time_seri: {len(time_seri)}')
        actual_len = len(time_seri) - sum(mask_[i])
        
        
        one_seq_time = []
        
        first_time = time_seri[0]
        timestamp_seconds = int(first_time)
        # print(f'timestamp_seconds: {timestamp_seconds}')
        # dt = datetime.datetime.fromtimestamp(datetime.UTC)(timestamp_seconds)
        dt_local = datetime.datetime.fromtimestamp(timestamp_seconds, tz=tz_utc8)
        # dt_local = dt + datetime.timedelta(hours=8)  # 转换为本地时间
        # if i < 1:
        # print(f"dt_local.weekday(), dt_local.hour,  dt_local.minute, dt_local.second: {dt_local.weekday()} {dt_local.hour} {dt_local.minute} {dt_local.second}")
        
        sin_h = math.sin(2 * math.pi * dt_local.hour / 24)
        cos_h = math.cos(2 * math.pi * dt_local.hour / 24)
        sin_d = math.sin(2 * math.pi * dt_local.weekday() / 7)
        cos_d = math.cos(2 * math.pi * dt_local.weekday() / 7)
        sin_m = math.sin(2 * math.pi * dt_local.minute / 60)
        cos_m = math.cos(2 * math.pi * dt_local.minute / 60)
        # minite = dt_local.minute
        # second = dt_local.second + minite * 60
        second = dt_local.second
        offset = (np.log(second+1e-6) - mea_delta) / std_delta
        one_seq_time.append([1.0, sin_m, cos_m, sin_h, cos_h, sin_d, cos_d, offset])
        
        for j in range(1, actual_len):
            # time_point = time_seri[j]
            # 这里时间单位已经为秒了，所以不需要再除以1000了
            timestamp_d = int(time_seri[j])-int(time_seri[j-1])
            abs_timestamp = int(time_seri[j])
            dt_local = datetime.datetime.fromtimestamp(abs_timestamp, tz=tz_utc8)
            
            sin_h = math.sin(2 * math.pi * dt_local.hour / 24)
            cos_h = math.cos(2 * math.pi * dt_local.hour / 24)
            sin_d = math.sin(2 * math.pi * dt_local.weekday() / 7)
            cos_d = math.cos(2 * math.pi * dt_local.weekday() / 7)
            sin_m = math.sin(2 * math.pi * dt_local.minute / 60)
            cos_m = math.cos(2 * math.pi * dt_local.minute / 60)
            # print(f'timestamp_d: {timestamp_d}')
            one_seq_time.append([0.0, sin_m, cos_m, sin_h, cos_h, sin_d, cos_d, (np.log(timestamp_d+1e-6) - mea_delta) / std_delta])
            # print(f'log and normed time: ', {(np.log(timestamp_d+1e-6) - mea_delta) / std_delta})

        if actual_len < select_len:
            for j in range(actual_len, select_len):
                one_seq_time.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                
        emb_time_seq.append(one_seq_time)
        
    emb_time_seq = np.array(emb_time_seq)
    
    for i in range(len(sorted_posi_seq)):
        # the i-th position sequence
        trace = sorted_posi_seq[i]
        actual_len = len(trace) - sum(mask_[i])
        one_seq_pos = []
        
        for j in range(actual_len):
            lon, lat = trace[j]
            norm_lon = (lon - min_lon) / (max_lon-min_lon) * 2 - 1
            norm_lat = (lat - min_lat) / (max_lat-min_lat) * 2 - 1
            
            # norm_lon = np.clip(norm_lon, -1.0, 1.0)
            # norm_lat = np.clip(norm_lat, -1.0, 1.0)
            
            freqs = np.exp(np.linspace(0.0, 5.0, num_freqs))*math.pi
            fx_sin = np.sin(norm_lon * freqs)
            fx_cos = np.cos(norm_lon * freqs)
            
            fy_sin = np.sin(norm_lat * freqs)
            fy_cos = np.cos(norm_lat * freqs)
            
            lon_lat_emb = np.concatenate([fx_sin, fx_cos, fy_sin, fy_cos])
            # if j<2:
            #     print('lon, lat:', lon, lat)
            #     print('norm_lon, norm_lat: ', norm_lon, norm_lat)
            #     print('freqs: ', freqs)
            #     print('fx_sin: ', fx_sin)
            #     print('fx_cos: ', fx_cos)
            #     print('fy_sin: ', fy_sin)
            #     print('fy_cos: ', fy_cos)
                
            one_seq_pos.append(lon_lat_emb.tolist())
            
        if actual_len < select_len:
            for j in range(actual_len, select_len):
                one_seq_pos.append([0.0]*(num_freqs*4))
                
        emb_posi_seq.append(one_seq_pos)
        
    emb_posi_seq = np.array(emb_posi_seq)
       
    combined = torch.tensor(np.concatenate([emb_posi_seq, emb_time_seq], axis=-1), dtype=torch.float32)
        
    return combined, torch.tensor(mask_, dtype=torch.bool)


def construct_features(time_seq, posi_seq, sel_l):
    
    time_encoder = TimePeriodicEncoder(out_dim=32)
    emb_posi_seq = []
    
    select_len = sel_l
    
    padded_time_seq = []
    padded_posi_seq = []
    mask_ = []  
    
    for i in range(len(time_seq)):
        len_seq = len(time_seq[i])
        # print(f'Original length of trajectory {i}: {len_seq}')
        # geolife如果轨迹长度小于5（10倍下采样后的5），则直接丢弃该轨迹；如果轨迹长度小于select_len，则在末尾进行0填充；如果轨迹长度大于等于select_len，则直接截断到select_len。
        # if len_seq < 5:
        if len_seq < 20:     #t_drive数据已经过滤掉长度小于20的轨迹了，所以这里直接设置为20了
            continue
        elif len_seq < select_len:
            padded_time_seq.append(time_seq[i] + [0.0] * (select_len - len_seq))
            padded_posi_seq.append(posi_seq[i] + [(0.0, 0.0)] * (select_len - len_seq))
            mask_.append([0]*len_seq+[1]*(select_len - len_seq))
        else:
            padded_time_seq.append(time_seq[i][:select_len])
            padded_posi_seq.append(posi_seq[i][:select_len])
            mask_.append([0]*select_len)
        
    print(f'去掉长度小于5的轨迹数量: {len(padded_time_seq)}, {len(padded_posi_seq)}, {len(mask_)}')
    
    
    # 对位置的绝对坐标进行编码，得到每个位置的多尺度特征表示(128维)
    # 1. 初始化转换器: WGS84 (4326) -> UTM (32650)
    # always_xy=True 确保输出顺序为 (East, North) 即 (x, y)
    transformer = Transformer.from_crs("epsg:4326", "epsg:32650", always_xy=True)
    # city_center_lon = 116.388055555555556
    # city_center_lat = 39.9075
    # 2. 转换城市中心作为原点
    # center_x, center_y = transformer.transform(city_center_lon, city_center_lat)
    # test_rgn_invariance = []
    left_up_x, left_up_y = transformer.transform(min_lon, max_lat)
    right_down_x, right_down_y = transformer.transform(max_lon, min_lat)
    left_down_x, left_down_y = transformer.transform(min_lon, min_lat)
    right_up_x, right_up_y = transformer.transform(max_lon, max_lat)
    
    min_x = min(left_up_x, left_down_x)
    max_x = max(right_down_x, right_up_x)
    min_y = min(right_down_y, left_down_y)
    max_y = max(left_up_y, right_up_y)
    
    
    start_end_utm = []
    
    for i in range(len(padded_posi_seq)):
        # the i-th position sequence
        trace = padded_posi_seq[i]
        actual_len = int(len(trace) - sum(mask_[i]))
        one_seq_pos = []
        
        one_start_end_utm = []
        # for j in range(actual_len):
        # for j in range(1):
        for j in [0, actual_len-1]:
            lon, lat = trace[j]
            # 3. 转换轨迹点到 UTM 坐标系
            raw_x, raw_y = transformer.transform(lon, lat)
            one_start_end_utm.append([raw_x, raw_y])
            
            # # 4. 减去中心偏置，得到以“米”为单位的局部坐标
            # # 此时 (0, 0) 代表城市中心，(100, 200) 代表中心以东100米，以北200米
            # x_meters = raw_x - center_x
            # y_meters = raw_y - center_y
            
            # # 1. 尺度对齐：将米转换为“千米”
            # # 这样 P 的量级在 0~20 之间
            # P = np.array([[x_meters/1000.0, y_meters/1000.0]]) 
            
            # # 2. 确定性 B 矩阵
            # # 此时 sigma = 0.5 意味着捕捉公里级的变化
            # # 如果要捕捉“路口级”（20米），sigma 应该设为 5.0 左右
            # sigma = 10.0
            # rng = np.random.default_rng(42)
            # B = rng.standard_normal((2, 2*num_freqs)) * sigma
            
            # # 3. 计算映射
            # proj = np.matmul(P, B) * 2 * np.pi
            # spatial_emb = np.concatenate([np.sin(proj), np.cos(proj)], axis=-1)
            
            # one_seq_pos.append(np.squeeze(spatial_emb).tolist())
            
            norm_x = (raw_x - min_x) / (max_x-min_x) * 2 - 1
            norm_y = (raw_y - min_y) / (max_y-min_y) * 2 - 1
            
            freqs = (2**np.arange(10)) * math.pi
            fx_sin = np.sin(norm_x * freqs)
            fx_cos = np.cos(norm_x * freqs)
            
            fy_sin = np.sin(norm_y * freqs)
            fy_cos = np.cos(norm_y * freqs)
            
            lon_lat_emb = np.concatenate([fx_sin, fx_cos, fy_sin, fy_cos])
        
            one_seq_pos.append(lon_lat_emb.tolist())
        
        emb_posi_seq.append(one_seq_pos)
        start_end_utm.append(one_start_end_utm)
    
    emb_posi_seq = np.array(emb_posi_seq, dtype=np.float32)
    start_end_utm = np.array(start_end_utm, dtype=np.float32)
    
    # 对位置的相对坐标进行编码，得到每个位置相对于前一个位置的位移的多尺度特征表示(相邻点间的后前相对位置差，2维，单位为米)
    emb_posi_seq_d = []
    for i in range(len(padded_posi_seq)):
        # the i-th position sequence
        trace = padded_posi_seq[i]
        actual_len = len(trace) - sum(mask_[i])
        one_seq_pos = [[0.0, 0.0]]
        
        for j in range(1, actual_len):
            pre_lon, pre_lat = trace[j-1]
            lon, lat = trace[j]
            
            pre_x, pre_y = transformer.transform(pre_lon, pre_lat)
            cur_x, cur_y = transformer.transform(lon, lat)
            
            # d_lon = lon - pre_lon
            # d_lat = lat - pre_lat
            
            # # lat_ave = (lat + pre_lat) / 2
            # lat_ave = pre_lat
            
            # lat_ave_rad = math.radians(lat_ave)
            # d_lon_rad = math.radians(d_lon)
            # d_lat_rad = math.radians(d_lat)
            
            # dx = R * d_lon_rad * math.cos(lat_ave_rad)
            # dy = R * d_lat_rad
            dx = cur_x - pre_x
            dy = cur_y - pre_y
            one_seq_pos.append([dx, dy])
            
        if actual_len < select_len:
            for j in range(actual_len, select_len):
                one_seq_pos.append([0.0, 0.0])
        emb_posi_seq_d.append(one_seq_pos)
    
    emb_posi_seq_d = np.array(emb_posi_seq_d, dtype=np.float32)
    
    
    emb_time_seq = []
    # abs_emb_time_seq = []
    # abs_emb_time = []
    for i in range(len(padded_time_seq)):
        # the i-th time sequence
        time_seri = padded_time_seq[i]
        actual_len = len(time_seri) - sum(mask_[i])
        
        rel_diff_time = [[0.0]]
        # abs_diff_time = [[0.0]]
        # abs_start_time = [[0.0]]
        for j in range(1, actual_len):
            time_point = time_seri[j]
            pre_time_point = time_seri[j-1]
            rel_diff_time.append([time_point - pre_time_point])
            # abs_diff_time.append([time_point-time_seri[0]])
            # abs_start_time.append([time_point-time_seri[0]])
        
        if actual_len < select_len:
            for j in range(actual_len, select_len):
                rel_diff_time.append([0.0])
                # abs_diff_time.append(0.0)
                # abs_start_time.append([0.0])
        emb_time_seq.append(np.array(rel_diff_time, dtype=np.float32))
        
        # abs_emb_time.append(np.array(abs_start_time, dtype=np.float32))
        # emb_time_seq = np.array(emb_time_seq)
        
        # abs_diff_time = np.array(abs_diff_time, dtype=np.float32)
        # abs_time_emb = time_encoder.encode(abs_diff_time)
        # if actual_len < select_len:
        #     abs_time_emb = np.concatenate([abs_time_emb, np.zeros((select_len-actual_len, abs_time_emb.shape[1]))], axis=0)

        # abs_emb_time_seq.append(abs_time_emb)
    
    
    start_time_emb = []
    tz_utc8 = timezone(timedelta(hours=8))
    for i in range(len(padded_time_seq)):
        
        time_seri = padded_time_seq[i]
        
        start_timestamp = int(time_seri[0])
        dt_local = datetime.datetime.fromtimestamp(start_timestamp, tz=tz_utc8)
        
        sin_h = math.sin(2 * math.pi * dt_local.hour / 24)
        cos_h = math.cos(2 * math.pi * dt_local.hour / 24)
        sin_d = math.sin(2 * math.pi * dt_local.weekday() / 7)
        cos_d = math.cos(2 * math.pi * dt_local.weekday() / 7)
        sin_m = math.sin(2 * math.pi * dt_local.minute / 60)
        cos_m = math.cos(2 * math.pi * dt_local.minute / 60)
        sin_s = math.sin(2 * math.pi * (dt_local.second + dt_local.microsecond/1e6) / 60)
        cos_s = math.cos(2 * math.pi * (dt_local.second + dt_local.microsecond/1e6) / 60)
        
        start_time_emb.append([[sin_d, cos_d, sin_h, cos_h, sin_m, cos_m, sin_s, cos_s]])
    start_time_emb = np.array(start_time_emb, dtype=np.float32)
    
    # return emb_posi_seq, emb_posi_seq_d, emb_time_seq, abs_emb_time, abs_emb_time_seq, start_time_emb, np.array(mask_, dtype=np.bool_)
    return emb_posi_seq, emb_posi_seq_d, emb_time_seq, start_time_emb, start_end_utm, np.array(mask_, dtype=np.bool_)


def make_features(resampled_traj, start_time):
    
    emb_start_point_seq = []
    
    # 对位置的绝对坐标进行编码，得到每个位置的多尺度特征表示(128维)
    # 1. 初始化转换器: WGS84 (4326) -> UTM (32650)
    # always_xy=True 确保输出顺序为 (East, North) 即 (x, y)
    transformer = Transformer.from_crs("epsg:4326", "epsg:32650", always_xy=True)
    city_center_lon = 116.388055555555556
    city_center_lat = 39.9075
    # 2. 转换城市中心作为原点
    center_x, center_y = transformer.transform(city_center_lon, city_center_lat)
    test_rgn_invariance = []
    
    for i in range(len(start_time)):
        # 获取第 i 个轨迹的起始点坐标
        lon, lat = resampled_traj[i][0,0], resampled_traj[i][0,1]
        
        # 转换轨迹点到 UTM 坐标系
        raw_x, raw_y = transformer.transform(lon, lat)
        # 减去中心偏置，得到以“米”为单位的局部坐标
        # 此时 (0, 0) 代表城市中心，(100, 200) 代表中心以东100米，以北200米
        x_meters = raw_x - center_x
        y_meters = raw_y - center_y
        
        # 尺度对齐：将米转换为“千米”
        # 这样 P 的量级在 0~20 之间
        P = np.array([[x_meters/1000.0, y_meters/1000.0]]) 
        
        # 确定性 B 矩阵
        # 此时 sigma = 0.5 意味着捕捉公里级的变化
        # 如果要捕捉“路口级”（20米），sigma 应该设为 5.0 左右
        sigma = 10.0
        rng = np.random.default_rng(42)
        B = rng.standard_normal((2, 2*num_freqs)) * sigma
        
        # state = np.random.RandomState(42)
        # b_low = state.randn(2, 16) * 1.0
        # b_mid = state.randn(2, 32) * 10.0
        # b_high = state.randn(2, 16) * 50.0
        # B = np.concatenate([b_low, b_mid, b_high], axis=1)
        
        # 3. 计算映射
        proj = np.matmul(P, B) * 2 * np.pi
        spatial_emb = np.concatenate([np.sin(proj), np.cos(proj)], axis=-1)
        
        emb_start_point_seq.append(spatial_emb)
        
        if i < 200:
            test_rgn_invariance.append(B.tolist())
    
    f = open('./test_rgn_invariance_in_train.json', 'w')
    json.dump(test_rgn_invariance, f, ensure_ascii=False)
    f.close()
    
    emb_start_point_seq = np.array(emb_start_point_seq, dtype=np.float32)
    
    
    # 对位置的相对坐标进行编码，得到每个位置相对于前一个位置的位移的多尺度特征表示(相邻点间的后前相对位置差，2维，单位为米)
    emb_posi_seq_d = []
    emb_time_seq =[]
    progress_seq = []
    
    for i in range(len(start_time)):
        # the i-th position sequence
        trace = resampled_traj[i]
        emb_time_seq.append(trace[:,3:4])
        progress_seq.append(trace[:,4:5])
        
        one_seq_pos = [[0.0, 0.0]]
        
        for j in range(1, len(trace)):
            pre_lon, pre_lat = trace[j-1,0], trace[j-1,1]
            lon, lat = trace[j,0], trace[j,1]
            
            # pre_x, pre_y = transformer.transform(pre_lon, pre_lat)
            # cur_x, cur_y = transformer.transform(lon, lat)
            
            d_lon = lon - pre_lon
            d_lat = lat - pre_lat
            
            # lat_ave = (lat + pre_lat) / 2
            lat_ave = pre_lat
            
            lat_ave_rad = math.radians(lat_ave)
            d_lon_rad = math.radians(d_lon)
            d_lat_rad = math.radians(d_lat)
            
            dx = R * d_lon_rad * math.cos(lat_ave_rad)
            dy = R * d_lat_rad
            
            # dx = cur_x - pre_x
            # dy = cur_y - pre_y
            
            one_seq_pos.append([dx, dy])
            
        emb_posi_seq_d.append(one_seq_pos)
    
    emb_posi_seq_d = np.array(emb_posi_seq_d, dtype=np.float32)
    emb_time_seq = np.array(emb_time_seq, dtype=np.float32)
    progress_seq = np.array(progress_seq, dtype=np.float32)
    
    # 对起始时间进行编码，得到每个起始时间的多尺度特征表示(8维)
    start_time_emb = []
    tz_utc8 = timezone(timedelta(hours=8))
    for i in range(len(start_time)):
        
        start_timestamp = int(start_time[i])
        dt_local = datetime.datetime.fromtimestamp(start_timestamp, tz=tz_utc8)
        
        sin_h = math.sin(2 * math.pi * dt_local.hour / 24)
        cos_h = math.cos(2 * math.pi * dt_local.hour / 24)
        sin_d = math.sin(2 * math.pi * dt_local.weekday() / 7)
        cos_d = math.cos(2 * math.pi * dt_local.weekday() / 7)
        sin_m = math.sin(2 * math.pi * dt_local.minute / 60)
        cos_m = math.cos(2 * math.pi * dt_local.minute / 60)
        sin_s = math.sin(2 * math.pi * (dt_local.second + dt_local.microsecond/1e6) / 60)
        cos_s = math.cos(2 * math.pi * (dt_local.second + dt_local.microsecond/1e6) / 60)
        
        start_time_emb.append([[sin_d, cos_d, sin_h, cos_h, sin_m, cos_m, sin_s, cos_s]])
    start_time_emb = np.array(start_time_emb, dtype=np.float32)
        
    
    return emb_start_point_seq, emb_posi_seq_d, start_time_emb, emb_time_seq, progress_seq
    

class CustomDataset(Dataset):
    def __init__(self, data, mask, transform=None):
        """
        初始化数据集
        Args:
            data: 你的数据
            transform: 数据预处理/增强
        """
        self.data = data
        self.mask = mask
        self.transform = transform
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.data[idx]
        mask = self.mask[idx]
        # 如果有预处理/数据增强
        if self.transform:
            sample = self.transform(sample)
            
        # 返回 (特征, 标签) 对
        # 或者只返回特征，取决于你的任务
        return sample, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=512):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        # print(f"x is on cuda: {x.is_cuda}")
        # print(f"t is on cuda: {t.is_cuda}")
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb
    

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=512):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    # d_model表示离散值嵌入为向量时向量的长度
    # size表示一条样本对应着多少个嵌入向量
    def __init__(self, d_model, size, padding_mask=None):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.size = size
        self.padding_mask = padding_mask
        self.mha = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.ln = nn.LayerNorm([d_model])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([d_model]),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, padding_mask):
        # x = x.view(-1, self.channels, self.size[0] * self.size[1]).swapaxes(1, 2)
        assert x.size(1) == self.size, '输入的张量中词的个数与模型中的不一致'
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln, key_padding_mask=padding_mask)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        # return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size[0], self.size[1])
        return attention_value


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        # self.padding_mask = padding_mask
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        # 用于预测 AdaLN 的参数：gamma1, beta1, gamma2, beta2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim) 
        )

    def modulate(self, x, shift, scale):
        # 执行 AdaLN 调制：(1 + scale) * x + shift
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
    def forward(self, x, c, padding_mask):
        # x: 轨迹序列 [B, L, D]
        # c: 融合后的全局条件 [B, D]
        # 预测缩放和偏移
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
        
        # 注入全局条件到 Attention 层
        modulated_x = self.modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + self.attn(modulated_x, modulated_x, modulated_x, key_padding_mask=padding_mask)[0]
        # 注入全局条件到 MLP 层
        x = x + self.mlp(self.modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512, base=10000):
        super().__init__()
        # d_model 必须是偶数
        self.d_model = d_model
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def _get_cos_sin_embeddings(self, seq_len, device):
        # 生成序列位置索引 t = [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        # 计算外积得到相位矩阵 [seq_len, d_model/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # 复制一份以对应 (x1, x2) 结构 [seq_len, d_model]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def rotate_half(self, x):
        """将向量前半部分和后半部分进行旋转变换"""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        """
        x: [Batch, seq_len, d_model]
        """
        seq_len = x.size(-2)
        cos, sin = self._get_cos_sin_embeddings(seq_len, x.device)
        
        # print(f"cos shape: {cos.shape}, sin shape: {sin.shape}, x shape: {x.shape}")
        # 旋转公式：x_rot = x * cos + rotate_half(x) * sin
        # cos/sin 形状扩展为 [1, seq_len, d_model] 以支持 Batch
        cos = cos.view(*([1] * (x.ndim - 2)), seq_len, self.d_model)
        sin = sin.view(*([1] * (x.ndim - 2)), seq_len, self.d_model)
        # print(f"cos shape: {cos.shape}, sin shape: {sin.shape}, x shape: {x.shape}")
        
        return (x * cos) + (self.rotate_half(x) * sin)


class DiTBlock_RoPE(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 针对每个 Head 的维度进行旋转
        self.rope = RotaryPositionalEmbedding(self.head_dim) 
        
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim) # 一次性投影 QKV
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # ... 其他层 (norm, mlp, adaLN) 保持不变 ...
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        # 用于预测 AdaLN 的参数：gamma1, beta1, gamma2, beta2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim) 
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0.0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0.0)

    def modulate(self, x, shift, scale):
        # 执行 AdaLN 调制：(1 + scale) * x + shift
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
    def forward(self, x, c, padding_mask):
        B, L, D = x.shape
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
        
        # 1. 调制后的输入
        norm_x = self.modulate(self.norm1(x), shift_msa, scale_msa)
        
        # 2. 投影 QKV 并拆分多头
        # [B, L, 3*D] -> [B, L, 3, num_heads, head_dim] -> [3, B, num_heads, L, head_dim]
        qkv = self.qkv_proj(norm_x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # --- 核心改动：只对 Q 和 K 进行旋转 ---
        q = self.rope(q) 
        k = self.rope(k)
        
        scale = q.size(-1) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        scores = scores.masked_fill(mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        attn_out = torch.matmul(attention_weights, v)  # [B, num_heads, L, head_dim]
        
        # 3. 手动计算 Scaled Dot-Product Attention
        # 使用 F.scaled_dot_product_attention (PyTorch 2.0+) 自动处理 padding_mask
        # mask = (~padding_mask).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        # attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)  # [B, num_heads, L, head_dim]
        # # 官方实现（需要转换 mask）
        # # 注意：官方实现没有直接的 padding_mask 参数，需要转换为 attn_mask
        # mask = padding_mask.unsqueeze(1)  # (batch, 1, seq_len)
        # mask = mask.expand(-1, seq_len, -1)  # (batch, seq_len, seq_len)
        # # 简化：使用加法掩码
        # attn_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=q.dtype)
        # attn_mask = attn_mask.masked_fill(mask, float('-inf'))
        
        # official_output = F.scaled_dot_product_attention(
        #     q, k, v, attn_mask=attn_mask
        # )
        
        # 4. 合并多头并输出
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, L, D)
        x = x + self.out_proj(attn_out)
        
        # 5. MLP
        x = x + self.mlp(self.modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Net_model(nn.Module):
    def __init__(self, c_in=216, c_out=216, emb_dim=512, first_tim_dim=8, first_pos_dim=40, length_dim=128, device=torch.device("cuda")):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        select_len = c_in
        self.first_tim_dim = first_tim_dim
        self.first_pos_dim = first_pos_dim
        self.length_dim = length_dim
        
        ###################### 条件嵌入设计 ##################
        # 第一点的时间初始表示是8维，通过线性映射投影到128维的空间中
        self.time_proj = nn.Sequential(
            nn.Linear(self.first_tim_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 256),
            nn.LayerNorm(256) # 引入 LayerNorm 稳定时间特征
        )
        # 第一点的位置信息初始表示是128维，通过线性映射投影到128维的空间中
        self.pos_proj = nn.Sequential(
            nn.Linear(2*self.first_pos_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256) # 引入 LayerNorm 稳定位置特征
        )
        # 把轨迹长度的编码也加入到条件嵌入中，得到一个新的轨迹长度特征，维度为128
        self.length_proj = nn.Sequential(
            nn.Linear(self.length_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256) # 引入 LayerNorm 稳定长度特征
        )
        # 把第一点的时间和位置的嵌入拼接起来进行融合，得到一个新的时间位置融合特征，维度为256
        self.fusion_layer = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512), # 引入 LayerNorm 稳定融合特征
        )
        
        # 把时空融合特征与扩散步骤t的时间嵌入进行拼接后融合，得到一个全局条件特征，维度为512，用于后续的AdaLN
        self.global_cond = nn.Sequential(
            nn.Linear(512+512, 512),
            nn.LayerNorm(512), # 引入 LayerNorm 稳定全局条件特征
            nn.SiLU(),
            nn.Linear(512, 512)
        )
        
        ################## 扩散特征嵌入设计 ##################
        # 针对联合位置、位置差分量和时间特征的输入，设计分层嵌，将它们映射到与Transformer输入相同的维度空间中，维度为512
        # 1. 动力学嵌入 (dx, dy, dt)
        self.motion_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.SiLU(),
            nn.Linear(64, 128)
        )
        
        self.progress_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64)
        )
        
        # 2. 时间背景投影 (32 -> 128)，累积时间（1)和累积时间的向量特征(32)
        self.temporal_proj = nn.Sequential(
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        
        self.progress_fusion_proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        # # 3. 空间背景投影 (129 -> 128) (Spatial + Start Flag)
        # self.spatial_proj = nn.Sequential(
        #     nn.Linear(128+1, 128),
        #     nn.LayerNorm(128), # 引入 LayerNorm 稳定 Flag 信号
        #     nn.SiLU(),
        #     nn.Linear(128, 128)
        # )
        
        # 4. 最终融合层
        # 此时拼接维度为 128 (M) + 128 (T) + 128 (S) = 384
        # self.final_fuse = nn.Sequential(
        #     nn.Linear(256, self.emb_dim),
        #     nn.LayerNorm(self.emb_dim),
        #     nn.SiLU(),
        #     nn.Linear(self.emb_dim, self.emb_dim)
        # )
        self.flag_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256)
            )
        
        # self.in_liner = nn.Linear(4*num_freqs+36, emb_dim)
        # self.in_liner = CustomizedEmbLayer(c_in, time_dim, device)
        # self.in_liner = BatchPositionEncoder(c_in, time_dim, device=device)
        
        # B. 位置嵌入 (Learned Position Embedding)
        # 改成了ROPE编码，不再用正弦余弦位置编码了
        # self.pos_embed = nn.Embedding(select_len, emb_dim)
        self.type_embed = nn.Embedding(2, 128) # 0: Start, 1: subsequent data point
        # 将拼接后的特征映射到 Transformer 的输入维度
        # self.input_proj = nn.Sequential(
        #     nn.Linear(self.emb_dim+128, self.emb_dim),
        #     nn.LayerNorm(self.emb_dim),
        #     nn.SiLU(),
        #     nn.Linear(self.emb_dim, self.emb_dim)
        #     )
        
        self.input_proj = nn.Sequential(
            nn.Linear(512, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
            )  
        
        # C. 类别嵌入 (Type Embedding)
        # 0: Start, 1: Data, 2: Pad
        # self.type_embed = nn.Embedding(3, emb_dim)
        # self.layer_norm = nn.LayerNorm(emb_dim)
        
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        
        
        self.attention1 = DiTBlock_RoPE(emb_dim, 8)
        self.attention2 = DiTBlock_RoPE(emb_dim, 8)
        self.attention3 = DiTBlock_RoPE(emb_dim, 8)
        
        self.attention4 = DiTBlock_RoPE(emb_dim, 8)
        self.attention5 = DiTBlock_RoPE(emb_dim, 8)
        self.attention6 = DiTBlock_RoPE(emb_dim, 8)
        
        self.attention7 = DiTBlock_RoPE(emb_dim, 8)
        self.attention8 = DiTBlock_RoPE(emb_dim, 8)
        self.attention9 = DiTBlock_RoPE(emb_dim, 8)
        
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * emb_dim)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0.0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0.0)
        
        # self.linear = nn.Linear(emb_dim, 164, bias=True)
        self.norm_final = nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-6)
        self.linear_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 5)
        )
        


    def pos_encoding(self, t, channels):

        # inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        inv_freq = torch.exp(torch.arange(0, channels, 2, device=self.device) * (-torch.log(torch.tensor(10000)) / channels))
        # print(f'inv_freq  is on cuda: {inv_freq.is_cuda}')
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)

        # pos_enc = torch.zeros(pos_enc_a.size(0), 2*pos_enc_a.size(1)).to(self.device)
        pos_enc = torch.zeros(pos_enc_a.size(0), 2*pos_enc_a.size(1), device=self.device)

        pos_enc[:, 0::2] = pos_enc_a
        pos_enc[:, 1::2] = pos_enc_b
        # pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)

        return pos_enc
    
    
    def total_len_encoding(self, x, dim, max_period=10000):
        # x 形状: [Batch] 的真实轨迹点数，如 [12, 45, 120, 22]
        # 取对数平滑分布，防止长尾长轨迹变成孤立点
        x = torch.log(x + 1e-6)  # 对累积时间取对数，避免数值过大
        inv_freq = torch.exp(torch.arange(0, dim, 2, device=self.device).float() * (-torch.log(torch.tensor(max_period)) / dim))
        
        args = x.unsqueeze(1)*inv_freq.unsqueeze(0) # [B, 1] * [1, dim/2] -> [B, dim/2]
        embeddig = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # [B, dim]

        return embeddig
        
    
    def forward(self, x, t, padding_mask, start_time_emb, od_pos_emb, start_flag_idx):
        # print(f't in model forward is on cuda: {t.is_cuda}')
        # print(f"start_time_emb dtype: {start_time_emb.dtype}")
        # print(f"start_pos_emb dtype: {start_pos_emb.dtype}")
        # 检查 time_proj 的权重类型
        # print(f"time_proj weight dtype: {self.time_proj[0].weight.dtype}")
        
        t = t.unsqueeze(-1).type(torch.float32)
        # t = t.type(torch.float)
        # 对batch中的n个时刻的编码0:T-1共T个可能时刻,扩散步编码
        t = self.pos_encoding(t, self.emb_dim)
        t.to(self.device)
        t = self.time_mlp(t)

        # 轨迹长度编码
        b_len = (~padding_mask).sum(dim=1).float()
        len_emb = self.total_len_encoding(b_len, self.length_dim, max_period=10000)
        len_emb = self.length_proj(len_emb)
        len_emb = len_emb.unsqueeze(1)  # [B, 1, length_dim]
        
        # 起始位置和起始时间的嵌入经过线性变换后得到一个256维的特征表示
        t_fet = self.time_proj(start_time_emb)
        od_fusion_emb = torch.cat([od_pos_emb[:,:1,:], od_pos_emb[:,1:,:]], dim=-1)
        
        p_fet = self.pos_proj(od_fusion_emb)
        
        # fusion_emb is of shape bx1x512
        fusion_emb = self.fusion_layer(torch.cat([p_fet, t_fet, len_emb], dim=-1)).to(self.device).squeeze(1)
        
        condi_emb = torch.cat([fusion_emb, t], dim=-1)
        global_cond_emb = self.global_cond(condi_emb)
        
        # 线性变换之后再加位置编码
        # print(x.dtype)
        # print(self.in_liner.weight.dtype)
        x = x.to(torch.float32)
        # print('x.shape: ', x.shape)
        # x0 = self.in_liner(x)
        
        # start_pos_emb = start_pos_emb.to(torch.float32)
        # start_time_emb = start_time_emb.to(torch.float32)
        
        flag_feat = self.type_embed(start_flag_idx).to(torch.float32)
        # flag_feat = flag_feat.masked_fill(padding_mask.unsqueeze(-1), 0.0) # 将 padding 位置的 flag 特征置零
        # # abs_pi = x[:, :, :129]
        # rel_p = x[:, :, 0:2]
        # rel_t = x[:, :, 2:3]
        # # total_t = x[:, :, 3:4]
        # # total_t_vec = x[:, :, 4:]
        
        # 把相对位置和相对时间的特征拼接起来输入动力学特征提取模块进行处理，得到一个128维的动力学特征表示
        # m_feat = self.motion_mlp(torch.cat([rel_p, rel_t], dim=-1))
        m_feat = self.motion_mlp(x)
        
        
        # 累积时间的特征和累积时间的向量特征拼接后输入时间背景投影模块进行处理
        # t_feat = self.temporal_proj(torch.cat([total_t, total_t_vec], dim=-1))
        # padding_mask = padding_mask.to(self.device)
        
        prog_encoder = ProgressPeriodicEncoder(self.device, out_dim=32)
        
        lengths = (~padding_mask).sum(dim=1, keepdim=True).float()  # 计算每个样本的有效长度,形状为 (batch_size, 1)
        indices = torch.arange(padding_mask.size(1), device=self.device).float().unsqueeze(0).expand(padding_mask.size(0), -1)  # 生成位置索引,形状为 (1, seq_len) -> (batch_size, seq_len)
        progress = indices / (lengths-1)  # 计算进度，形状为 (batch_size, seq_len)，值在 [0, 1] 之间
        progress = progress.masked_fill(padding_mask, 0.0).unsqueeze(-1)  # 将 padding 部分的进度设置为 0，并扩展维度为 (batch_size, seq_len, 1)
        
        
        progress_periodic = prog_encoder.encode(progress)  # 计算周期性进度特征, 形状为 (batch_size, seq_len, 32)
        # progress_periodic = progress_periodic.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        # t_feat = self.temporal_proj(total_t_vec)
        prog_single = self.progress_proj(progress)
        prog_vec = self.temporal_proj(progress_periodic)
        
        # 进度的单一特征和周期性特征拼接后输入进度融合模块进行处理，得到一个128维的进度融合特征表示
        t_feat = self.progress_fusion_proj(torch.cat([prog_single, prog_vec], dim=-1))
        
        # x0 = self.final_fuse(torch.cat([s_feat, t_feat, m_feat], dim=-1))
        # 标志位的嵌入经过线性变换后得到一个256维的Flag特征表示
        f_feat = self.flag_proj(flag_feat)
        
        # x_input = self.final_fuse(torch.cat([m_feat, t_feat], dim=-1))
        
        x0 = torch.cat([m_feat, t_feat, f_feat], dim=-1)
        
        x0 = self.input_proj(x0)
        x0 = x0.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # 将 padding 位置的输入特征置零，防止其对后续的注意力计算产生干扰
        
        # 不用绝对位置编码了，改成ROPE了
        # seq_len = x0.size(1)
        # position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(x0.size(0), -1)
        # x0 = x0 + self.pos_embed(position_ids)
        
        # assert x0.size(0) == padding_mask.size(0), '批量中样本的个数与填充掩码中样本数量不一致'
        # 位置编码后再加类型编码
        # print(f'padding_mask is on cuda: {padding_mask.device}')
        
        # 下面的类型编码方式是：如果是padding位置则编码为2，如果是数据位置则编码为1，第一点的位置无论如何都编码为0
        # 由于不再采取将第一点的时间和位置信息直接作为待扩散的特征，而是通过time_proj和pos_proj先进行线性映射得到一个融合特征后再与扩散步骤t的时间嵌入进行拼接融合得到全局条件特征
        # 面对相邻差分计算，第一点的相对时间和相对位置特征都为0，所以第一点的类型编码也设置为0，表示既不是数据也不是padding；数据位置的类型编码设置为1；padding位置的类型编码设置为2。
        # rst_mask = torch.where(padding_mask, torch.tensor(2, device=self.device), torch.tensor(1, device=self.device))
        # rst_mask[:, 0] = torch.tensor(0, device=self.device)
        
        # x0 = x0 + self.type_embed(rst_mask)
        # selfattention模块中已经包含了LayerNorm，所以这里就不再单独加一个LayerNorm了
        # x0 = self.layer_norm(x0)
        
        x1 = self.attention1(x0, global_cond_emb, padding_mask)
        x1 = self.attention2(x1, global_cond_emb, padding_mask)
        x1 = self.attention3(x1, global_cond_emb, padding_mask)
        x1 = self.attention4(x1, global_cond_emb, padding_mask)
        x1 = self.attention5(x1, global_cond_emb, padding_mask)
        x1 = self.attention6(x1, global_cond_emb, padding_mask)
        x1 = self.attention7(x1, global_cond_emb, padding_mask)
        x1 = self.attention8(x1, global_cond_emb, padding_mask)
        x1 = self.attention9(x1, global_cond_emb, padding_mask)
        
        shift, scale = self.adaLN_modulation(global_cond_emb).chunk(2, dim=1)
        
        x2 = self.norm_final(x1)
        x2 = x2 * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # motion_out = self.motion_head(x2)
        # pos_out = self.pos_head(x2)
        # time_out = self.time_head(x2) 
        # output = torch.cat([pos_out, motion_out, time_out], dim=-1)
        output = self.linear_out(x2)
        return output


    # def forward(self, x, t, padding_mask, start_time_emb, start_pos_emb, start_flag_idx):
    #     # print(f't in model forward is on cuda: {t.is_cuda}')
    #     t = t.unsqueeze(-1).type(torch.float32)
    #     # t = t.type(torch.float)
    #     # 对batch中的n个时刻的编码0:T-1共T个可能时刻
    #     t = self.pos_encoding(t, self.emb_dim)
    #     t.to(self.device)
    #     t = self.time_mlp(t)

    #     t_fet = self.time_proj(start_time_emb)
    #     p_fet = self.pos_proj(start_pos_emb)
    #     # fusion_emb is of shape bx1x512
    #     fusion_emb = self.fusion_layer(torch.cat([t_fet, p_fet], dim=-1)).to(self.device).squeeze(1)
        
    #     condi_emb = torch.cat([fusion_emb, t], dim=-1)
    #     global_cond_emb = self.global_cond(condi_emb)
        
    #     # 线性变换之后再加位置编码
    #     # print(x.dtype)
    #     # print(self.in_liner.weight.dtype)
    #     x = x.to(torch.float32)
    #     # print('x.shape: ', x.shape)
    #     # x0 = self.in_liner(x)
        
    #     start_pos_emb = start_pos_emb.to(torch.float32)
    #     start_time_emb = start_time_emb.to(torch.float32)
        
    #     flag_feat = self.type_embed(start_flag_idx).to(torch.float32)
    #     flag_feat = flag_feat.masked_fill(padding_mask.unsqueeze(-1), 0.0) # 将 padding 位置的 flag 特征置零
    #     # # abs_pi = x[:, :, :129]
    #     # rel_p = x[:, :, 0:2]
    #     # rel_t = x[:, :, 2:3]
    #     # # total_t = x[:, :, 3:4]
    #     # # total_t_vec = x[:, :, 4:]
        
    #     # 把相对位置和相对时间的特征拼接起来输入动力学特征提取模块进行处理，得到一个128维的动力学特征表示
    #     # m_feat = self.motion_mlp(torch.cat([rel_p, rel_t], dim=-1))
    #     m_feat = self.motion_mlp(x)
        
        
    #     # 累积时间的特征和累积时间的向量特征拼接后输入时间背景投影模块进行处理
    #     # t_feat = self.temporal_proj(torch.cat([total_t, total_t_vec], dim=-1))
    #     padding_mask = padding_mask.to(self.device)
        
    #     prog_encoder = ProgressPeriodicEncoder(self.device, out_dim=32)
    #     lengths = (~padding_mask).sum(dim=1, keepdim=True).float()  # 计算每个样本的有效长度,形状为 (batch_size, 1)
    #     indices = torch.arange(padding_mask.size(1), device=self.device).float().unsqueeze(0).expand(padding_mask.size(0), -1)  # 生成位置索引,形状为 (1, seq_len) -> (batch_size, seq_len)
    #     progress = indices / (lengths-1)  # 计算进度，形状为 (batch_size, seq_len)，值在 [0, 1] 之间
    #     progress = progress.masked_fill(padding_mask, 0.0).unsqueeze(-1)  # 将 padding 部分的进度设置为 0，并扩展维度为 (batch_size, seq_len, 1)
    #     progress_periodic = prog_encoder.encode(progress)  # 计算周期性进度特征, 形状为 (batch_size, seq_len, 32)
    #     progress_periodic = progress_periodic.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
    #     # t_feat = self.temporal_proj(total_t_vec)
    #     prog_single = self.progress_proj(progress)
    #     prog_vec = self.temporal_proj(progress_periodic)
        
    #     # 进度的单一特征和周期性特征拼接后输入进度融合模块进行处理，得到一个128维的进度融合特征表示
    #     t_feat = self.progress_fusion_proj(torch.cat([prog_single, prog_vec], dim=-1))
        
    #     # x0 = self.final_fuse(torch.cat([s_feat, t_feat, m_feat], dim=-1))
    #     # 标志位的嵌入经过线性变换后得到一个256维的Flag特征表示
    #     f_feat = self.flag_proj(flag_feat)
        
    #     # x_input = self.final_fuse(torch.cat([m_feat, t_feat], dim=-1))
        
    #     x0 = torch.cat([m_feat, t_feat, f_feat], dim=-1)
        
    #     x0 = self.input_proj(x0)
    #     x0 = x0.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # 将 padding 位置的输入特征置零，防止其对后续的注意力计算产生干扰
        
    #     # 不用绝对位置编码了，改成ROPE了
    #     # seq_len = x0.size(1)
    #     # position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(x0.size(0), -1)
    #     # x0 = x0 + self.pos_embed(position_ids)
        
    #     assert x0.size(0) == padding_mask.size(0), '批量中样本的个数与填充掩码中样本数量不一致'
    #     # 位置编码后再加类型编码
    #     # print(f'padding_mask is on cuda: {padding_mask.device}')
        
    #     # 下面的类型编码方式是：如果是padding位置则编码为2，如果是数据位置则编码为1，第一点的位置无论如何都编码为0
    #     # 由于不再采取将第一点的时间和位置信息直接作为待扩散的特征，而是通过time_proj和pos_proj先进行线性映射得到一个融合特征后再与扩散步骤t的时间嵌入进行拼接融合得到全局条件特征
    #     # 面对相邻差分计算，第一点的相对时间和相对位置特征都为0，所以第一点的类型编码也设置为0，表示既不是数据也不是padding；数据位置的类型编码设置为1；padding位置的类型编码设置为2。
    #     # rst_mask = torch.where(padding_mask, torch.tensor(2, device=self.device), torch.tensor(1, device=self.device))
    #     # rst_mask[:, 0] = torch.tensor(0, device=self.device)
        
    #     # x0 = x0 + self.type_embed(rst_mask)
    #     # selfattention模块中已经包含了LayerNorm，所以这里就不再单独加一个LayerNorm了
    #     # x0 = self.layer_norm(x0)
        
    #     x1 = self.attention1(x0, global_cond_emb, padding_mask)
    #     x1 = self.attention2(x1, global_cond_emb, padding_mask)
    #     x1 = self.attention3(x1, global_cond_emb, padding_mask)
    #     x1 = self.attention4(x1, global_cond_emb, padding_mask)
    #     x1 = self.attention5(x1, global_cond_emb, padding_mask)
    #     x1 = self.attention6(x1, global_cond_emb, padding_mask)
    #     x1 = self.attention7(x1, global_cond_emb, padding_mask)
    #     x1 = self.attention8(x1, global_cond_emb, padding_mask)
    #     x1 = self.attention9(x1, global_cond_emb, padding_mask)
        
    #     shift, scale = self.adaLN_modulation(global_cond_emb).chunk(2, dim=1)
        
    #     x2 = self.norm_final(x1)
    #     x2 = x2 * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
    #     # motion_out = self.motion_head(x2)
    #     # pos_out = self.pos_head(x2)
    #     # time_out = self.time_head(x2) 
    #     # output = torch.cat([pos_out, motion_out, time_out], dim=-1)
    #     output = self.linear_out(x2)
    #     return output


class Pred_model(nn.Module):
    def __init__(self, c_in=216, c_out=216, emb_dim=512, first_tim_dim=8, first_pos_dim=40, length_dim=128, device=torch.device("cuda")):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        select_len = c_in
        self.first_tim_dim = first_tim_dim
        self.first_pos_dim = first_pos_dim
        self.length_dim = length_dim
        
        pe = torch.zeros(select_len+2, emb_dim)
        position = torch.arange(0, select_len+2, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        div_term = div_term.unsqueeze(0)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.to(self.device).unsqueeze(0)
        self.register_buffer('pe', pe)
        
        ###################### 条件嵌入设计 ##################
        # 第一点的时间初始表示是8维，通过线性映射投影到128维的空间中，换算成星期几，第几个小时，第几分钟，第多少秒，然后分别进行周期性的正弦余弦编码，得到8维的起始时间编码
        self.time_proj = nn.Sequential(
            nn.Linear(self.first_tim_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 256),
            nn.LayerNorm(256) # 引入 LayerNorm 稳定时间特征
        )
        # 起点位置初始表示是40维，通过线性映射投影到128维的空间中
        self.start_pos_proj = nn.Sequential(
            nn.Linear(self.first_pos_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512) # 引入 LayerNorm 稳定位置特征
        )
        # 终点位置初始表示是40维，通过线性映射投影到128维的空间中
        self.end_pos_proj = nn.Sequential(
            nn.Linear(self.first_pos_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512) # 引入 LayerNorm 稳定位置特征
        )
        # 起点和终点拼接，通过线性映射投影到256维的空间中
        self.pos_proj = nn.Sequential(
            nn.Linear(1024, 256),
            nn.SiLU(),
            nn.LayerNorm(256) # 引入 LayerNorm 稳定位置特征
        )
        # 把轨迹长度的编码也加入到全局条件中，得到一个轨迹长度特征，维度为256
        self.length_proj = nn.Sequential(
            nn.Linear(self.length_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256) # 引入 LayerNorm 稳定长度特征
        )
        # 把起点时间、起点位置终点位置、轨迹长度的嵌入拼接起来进行融合，得到一个新的时间位置融合特征，维度为512
        self.fusion_layer = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512), # 引入 LayerNorm 稳定融合特征
        )
        
        # 把时空融合特征与扩散步骤t的时间嵌入进行拼接后融合，得到一个全局条件特征，维度为512，用于后续的AdaLN
        self.global_cond = nn.Sequential(
            nn.Linear(512+512, 512),
            nn.LayerNorm(512), # 引入 LayerNorm 稳定全局条件特征
            nn.SiLU(),
            nn.Linear(512, 512)
        )
        
        ################## 扩散特征嵌入设计 ##################
        # 针对联合位置、位置差分量和时间特征的输入，设计分层嵌，将它们映射到与Transformer输入相同的维度空间中，维度为512
        # 动力学嵌入 (dx, dy, dt)
        self.motion_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, 128)
        )
        
        # 基于轨迹长度计算的进度特征嵌入 (1 -> 32)，输入是一个标量，表示当前点在轨迹中的相对位置（0-1之间）i/(L-1)，输出是一个64维的进度特征向量
        self.progress_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64)
        )
        
        # 对基于轨迹长度计算的进度先进行32维的高斯编码后，应用下面的线性层进行映射，得到一个64维的时间背景特征表示
        self.temporal_proj = nn.Sequential(
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        
        # 将进度的单一特征和周期性特征拼接后输入进度融合模块进行处理，得到一个128维的进度融合特征表示
        self.progress_fusion_proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

        self.flag_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256)
            )
        
        # 位置嵌入 (Learned Position Embedding)
        # self.pos_embed = nn.Embedding(select_len+2, self.emb_dim)
        
        # 改成了ROPE编码，不再用正弦余弦位置编码了
        # self.type_embed = nn.Embedding(2, 128) # 0: Start, 1: subsequent data point
        
        # 将拼接后的特征映射到 Transformer 的输入维度
        self.input_proj = nn.Sequential(
            nn.Linear(512, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
            )
        
        # 动力学特征与进度特征拼接，通过线性映射投影到256维的空间中
        self.m_p_proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
            )
        
        # C. 类别嵌入 (Type Embedding)
        # 0: Start, 1: Data, 2: Pad
        # self.type_embed = nn.Embedding(3, emb_dim)
        # self.layer_norm = nn.LayerNorm(emb_dim)
        
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        
        
        self.attention1 = DiTBlock_RoPE(emb_dim, 8)
        self.attention2 = DiTBlock_RoPE(emb_dim, 8)
        self.attention3 = DiTBlock_RoPE(emb_dim, 8)
        
        self.attention4 = DiTBlock_RoPE(emb_dim, 8)
        self.attention5 = DiTBlock_RoPE(emb_dim, 8)
        self.attention6 = DiTBlock_RoPE(emb_dim, 8)
        
        self.attention7 = DiTBlock_RoPE(emb_dim, 8)
        self.attention8 = DiTBlock_RoPE(emb_dim, 8)
        self.attention9 = DiTBlock_RoPE(emb_dim, 8)
        
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * emb_dim)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0.0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0.0)
        
        # self.linear = nn.Linear(emb_dim, 164, bias=True)
        self.norm_final = nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-6)
        self.linear_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 4)
        )
        

    # 扩散时间编码 (Temporal Encoding) - 使用正弦余弦位置编码
    def pos_encoding(self, t, channels):

        # inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        inv_freq = torch.exp(torch.arange(0, channels, 2, device=self.device) * (-torch.log(torch.tensor(10000)) / channels))
        # print(f'inv_freq  is on cuda: {inv_freq.is_cuda}')
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)

        # pos_enc = torch.zeros(pos_enc_a.size(0), 2*pos_enc_a.size(1)).to(self.device)
        pos_enc = torch.zeros(pos_enc_a.size(0), 2*pos_enc_a.size(1), device=self.device)

        pos_enc[:, 0::2] = pos_enc_a
        pos_enc[:, 1::2] = pos_enc_b
        # pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)

        return pos_enc
    
        
    
    def total_len_encoding(self, x, dim, max_period=10000):
        # x 形状: [Batch] 的真实轨迹点数，如 [12, 45, 120, 22]
        # 取对数平滑分布，防止长尾长轨迹变成孤立点
        x = torch.log(x + 1e-6)  # 对累积时间取对数，避免数值过大
        inv_freq = torch.exp(torch.arange(0, dim, 2, device=self.device).float() * (-torch.log(torch.tensor(max_period)) / dim))
        
        args = x.unsqueeze(1)*inv_freq.unsqueeze(0) # [B, 1] * [1, dim/2] -> [B, dim/2]
        embeddig = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # [B, dim]

        return embeddig
        
    
    def forward(self, x, t, padding_mask, start_time_emb, od_pos_emb):
        # print(f't in model forward is on cuda: {t.is_cuda}')
        # 检查 time_proj 的权重类型
        
        t = t.unsqueeze(-1).type(torch.float32)
        # t = t.type(torch.float)
        # 对batch中的n个时刻的编码0:T-1共T个可能时刻,扩散步编码
        t = self.pos_encoding(t, self.emb_dim)
        t.to(self.device)
        t = self.time_mlp(t)

        # 轨迹长度编码
        b_len = (~padding_mask).sum(dim=1).float()
        len_emb = self.total_len_encoding(b_len, self.length_dim, max_period=10000)
        len_emb = self.length_proj(len_emb)
        len_emb = len_emb.unsqueeze(1)  # [B, 1, 256]
        
        # 起始位置和起始时间的嵌入经过线性变换后得到一个256维的特征表示
        t_fet = self.time_proj(start_time_emb)
        
        # 起点位置和终点位置的嵌入经过线性变换后得到一个512维的特征表示
        start_pos_emb = self.start_pos_proj(od_pos_emb[:,:1,:])
        end_pos_emb = self.end_pos_proj(od_pos_emb[:,1:,:])
        
        # od_fusion_emb = torch.cat([od_pos_emb[:,:1,:], od_pos_emb[:,1:,:]], dim=-1)
        od_fusion_emb = torch.cat([start_pos_emb, end_pos_emb], dim=-1)
        
        p_fet = self.pos_proj(od_fusion_emb)
        
        # fusion_emb is of shape bx1x512
        fusion_emb = self.fusion_layer(torch.cat([p_fet, t_fet, len_emb], dim=-1)).to(self.device).squeeze(1)
        
        condi_emb = torch.cat([fusion_emb, t], dim=-1)
        global_cond_emb = self.global_cond(condi_emb)
        
        # 线性变换之后再加位置编码
        x = x.to(torch.float32)
        # x0 = self.in_liner(x)
        
        # start_pos_emb = start_pos_emb.to(torch.float32)
        # start_time_emb = start_time_emb.to(torch.float32)
        
        # 区分轨迹起点和其它点的类型编码，第一点的类型编码为0，表示起点；数据位置的类型编码设置为1；padding位置的类型编码设置为2。
        # flag_feat = self.type_embed(start_flag_idx).to(torch.float32)
 
        # 把相对位置和相对时间的特征拼接起来输入动力学特征提取模块进行处理，得到一个128维的动力学特征表示
        m_feat = self.motion_mlp(x)
        
        # 累积时间的特征和累积时间的向量特征拼接后输入时间背景投影模块进行处理
        # t_feat = self.temporal_proj(torch.cat([total_t, total_t_vec], dim=-1))
        
        prog_encoder = ProgressPeriodicEncoder(self.device, out_dim=32)
        
        lengths = (~padding_mask).sum(dim=1, keepdim=True).float()  # 计算每个样本的有效长度,形状为 (batch_size, 1)
        indices = torch.arange(padding_mask.size(1), device=self.device).float().unsqueeze(0).expand(padding_mask.size(0), -1)  # 生成位置索引,形状为 (1, seq_len) -> (batch_size, seq_len)
        progress = indices / (lengths-1)  # 计算进度，形状为 (batch_size, seq_len)，值在 [0, 1] 之间
        progress = progress.masked_fill(padding_mask, 1.0).unsqueeze(-1)  # 将 padding 部分的进度设置为 0，并扩展维度为 (batch_size, seq_len, 1)
        
        
        progress_periodic = prog_encoder.encode(progress)  # 计算周期性进度特征, 形状为 (batch_size, seq_len, 32)
        # progress_periodic = progress_periodic.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        # t_feat = self.temporal_proj(total_t_vec)
        prog_single = self.progress_proj(progress)
        prog_vec = self.temporal_proj(progress_periodic)
        
        # 进度的单一特征和周期性特征拼接后输入进度融合模块进行处理，得到一个128维的进度融合特征表示
        p_feat = self.progress_fusion_proj(torch.cat([prog_single, prog_vec], dim=-1))
        
        # x0 = self.final_fuse(torch.cat([s_feat, t_feat, m_feat], dim=-1))
        # 标志位的嵌入经过线性变换后得到一个256维的Flag特征表示
        # f_feat = self.flag_proj(flag_feat)
        
        # x_input = self.final_fuse(torch.cat([m_feat, t_feat], dim=-1))
        # x0 = torch.cat([m_feat, p_feat, f_feat], dim=-1)
        x0 = torch.cat([m_feat, p_feat], dim=-1)
        
        # 动力学特征与进度特征拼接，通过线性映射投影到512维的空间中
        x0 = self.m_p_proj(x0)
        
        # 夹心饼干拼接：在输入序列的开头和结尾分别添加一个特殊的标志位，表示轨迹的起点和终点，并为它们分别添加位置嵌入。这样做的目的是为了让模型更好地捕捉轨迹的边界信息，尤其是在处理变长序列时，可以明确区分有效数据和填充部分。
        full_seq = torch.zeros(x0.size(0), x0.size(1) + 2, self.emb_dim, device=self.device)  # 创建一个新的张量来存储夹心饼干拼接后的序列，形状为 [B, seq_len+2, emb_dim]
        full_seq[:, 0, :] = start_pos_emb.squeeze(1)  # 将起点位置嵌入放在序列的开头
        full_seq[:, 1:(x0.size(1)+1), :] = x0 # 将原始序列放在中间
        
        batch_indices = torch.arange(x0.size(0), device=self.device) # 创建一个批量索引，形状为 [B],[0,1,2,...,B-1]
        end_index = (~padding_mask).sum(dim=1)  # 计算每个样本的有效长度，得到终点位置的索引，形状为 [B]
        full_seq[batch_indices, end_index+1, :] = end_pos_emb.squeeze(1)  # 将终点位置嵌入放在序列的末尾
        
        full_seq = full_seq + self.pe[:, :full_seq.size(1), :]  # 加上位置编码，full_seq的形状是[B, seq_len+2, emb_dim]
        
        grid = torch.arange(x0.size(1) + 2, device=self.device).unsqueeze(0)  # 创建一个索引网格，形状为 [1, seq_len+2]
        valid_total_lens = (end_index + 2).unsqueeze(1)  # 每个样本的有效长度加2，考虑到起点和终点的特殊标志位,形状为 [B, 1]
        
        traj_pad_mask = grid >= valid_total_lens  # 生成一个掩码，用于标识 padding 位置的索引，形状为 [B, seq_len+2]，其中有效位置为 False，padding 位置为 True
        
        
        # full_seq = torch.cat([start_pos_emb, x0, end_pos_emb], dim=1).to(self.device) + self.pos_embed(torch.arange(x0.size(1)+2).to(self.device)).unsqueeze(0).to(self.device)  # 加上位置编码，full_seq的形状是[B, seq_len+2, emb_dim]
        
        
        # x0 = self.input_proj(x0)
        # x0 = x0.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # 将 padding 位置的输入特征置零，防止其对后续的注意力计算产生干扰
        
        # 不用绝对位置编码了，改成ROPE了
        # seq_len = x0.size(1)
        # position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(x0.size(0), -1)
        # x0 = x0 + self.pos_embed(position_ids)
        
        # start_mask_pad = torch.zeros((x0.size(0), 1), dtype=torch.bool, device=self.device)  # 起点位置的掩码，全为 False
        # end_mask_pad = torch.zeros((x0.size(0), 1), dtype=torch.bool, device=self.device)  # 终点位置的掩码，全为 False
        # total_padding_mask = torch.cat([start_mask_pad, padding_mask, end_mask_pad], dim=1)  # 将起点和终点的掩码添加到原始的 padding 掩码中
        
        # assert x0.size(0) == padding_mask.size(0), '批量中样本的个数与填充掩码中样本数量不一致'
        # 位置编码后再加类型编码
        # print(f'padding_mask is on cuda: {padding_mask.device}')
        
        # 下面的类型编码方式是：如果是padding位置则编码为2，如果是数据位置则编码为1，第一点的位置无论如何都编码为0
        # 由于不再采取将第一点的时间和位置信息直接作为待扩散的特征，而是通过time_proj和pos_proj先进行线性映射得到一个融合特征后再与扩散步骤t的时间嵌入进行拼接融合得到全局条件特征
        # 面对相邻差分计算，第一点的相对时间和相对位置特征都为0，所以第一点的类型编码也设置为0，表示既不是数据也不是padding；数据位置的类型编码设置为1；padding位置的类型编码设置为2。
        # rst_mask = torch.where(padding_mask, torch.tensor(2, device=self.device), torch.tensor(1, device=self.device))
        # rst_mask[:, 0] = torch.tensor(0, device=self.device)
        
        # x0 = x0 + self.type_embed(rst_mask)
        # selfattention模块中已经包含了LayerNorm，所以这里就不再单独加一个LayerNorm了
        # x0 = self.layer_norm(x0)
        
        # x1 = self.attention1(x0, global_cond_emb, padding_mask)
        
        x1 = self.attention1(full_seq, global_cond_emb, traj_pad_mask)
        x1 = self.attention2(x1, global_cond_emb, traj_pad_mask)
        x1 = self.attention3(x1, global_cond_emb, traj_pad_mask)
        x1 = self.attention4(x1, global_cond_emb, traj_pad_mask)
        x1 = self.attention5(x1, global_cond_emb, traj_pad_mask)
        x1 = self.attention6(x1, global_cond_emb, traj_pad_mask)
        x1 = self.attention7(x1, global_cond_emb, traj_pad_mask)
        x1 = self.attention8(x1, global_cond_emb, traj_pad_mask)
        x1 = self.attention9(x1, global_cond_emb, traj_pad_mask)
        
        traj_output = x1[:, 1:-1, :]  # 去掉起点和终点的特征，得到与输入序列长度相同的输出特征
        
        shift, scale = self.adaLN_modulation(global_cond_emb).chunk(2, dim=1)
        
        # x2 = self.norm_final(x1)
        x2 = self.norm_final(traj_output)
        x2 = x2 * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # motion_out = self.motion_head(x2)
        # pos_out = self.pos_head(x2)
        # time_out = self.time_head(x2) 
        # output = torch.cat([pos_out, motion_out, time_out], dim=-1)
        output = self.linear_out(x2)
        return output


    
class Diffusion:
    def __init__(self, noise_steps=2000, beta_start=1e-4, beta_end=0.02, channels=216, emb_dim=164, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.channels = channels
        self.device = device
        self.emb_dim = emb_dim
        # self.emb_model = Embedding(vocab, emb_size)

        # self.beta = self.prepare_noise_schedule().to(device)
        # self.beta = self.sqrt_beta_schedule().to(device)
        self.beta = self.cosine_beta_schedule().to(device)

        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        print(self.alpha_hat[0])
        print(self.alpha_hat[-1])
        print(self.alpha[-1])
        print(self.beta[-1])

    def cosine_beta_schedule(self, s=0.008):
        steps = self.noise_steps + 1
        x = torch.linspace(0, self.noise_steps, steps)
        alphas_cumprod = torch.cos(((x / self.noise_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas
        # return torch.clip(betas, 0, 0.999)

    def sqrt_beta_schedule(self, s=0.0001):
        x = torch.linspace(0, self.noise_steps-1, self.noise_steps)  # 0,1,...,999
        alphas_cumprod = 1-torch.sqrt(x/self.noise_steps + s)
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas

    # 重写betas使得最后一步全是噪声
    def prepare_noise_schedule(self):
        betas = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        alphas = 1 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        print(alphas_bar_sqrt_T)
        print(alphas_bar_sqrt_0)

        alphas_bar_sqrt = alphas_bar_sqrt-alphas_bar_sqrt_T

        alphas_bar_sqrt = alphas_bar_sqrt * alphas_bar_sqrt_0/(alphas_bar_sqrt_0-alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:]/alphas_bar[:-1]

        alphas = torch.cat([alphas_bar[0:1], alphas])

        betas = 1 - alphas
        return betas

        # return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    # 这里t是一个批量时间，针对一个batch样本，每个样本对应一个噪声时刻t
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        # Ɛ = torch.randn_like(x)
        # return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
        yup = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * yup, yup

    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.noise_steps, size=(n,))

    def sample(self, model, n, new_values, probabilities, start_time_emb, start_pos_emb):
        logging.info(f"Sampling {n} new populations....")
        model.eval()
        with torch.no_grad():
            select_len = self.channels
            actual_lens = np.random.choice(new_values, size=n, p=probabilities)
            # actual_lens = [13]
            
            x = torch.randn((n, self.channels, self.emb_dim)).to(self.device)
            padding_mask = torch.zeros((n, select_len), dtype=torch.bool, device=self.device)
           
            if actual_lens is not None:
                for ii in range(n):
                    assert n == len(actual_lens), '掩码表示的数量与生成样本数量不一致'
                    
                    actual_len = actual_lens[ii]
                    if actual_len < select_len:
                        padding_mask[ii, actual_len:] = True
                    else:
                        padding_mask[ii] = False
            padding_mask = padding_mask.to(self.device)
            # print('padding_mask: ', padding_mask)
            
            start_flag_idx = torch.zeros_like(padding_mask, dtype=torch.int32).to(self.device)  # 创建与 mask_ 第一列相同形状的张量
            start_flag_idx[:, 0] = 1
            
            for i in tqdm(reversed(range(0, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                predicted_v = model(x, t, padding_mask, start_time_emb, start_pos_emb, start_flag_idx)
                # predicted_noise = model(x, t)

                # print('predicted_noise: ', predicted_noise)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]

                pre_x_0 = torch.sqrt(alpha_hat)*x - torch.sqrt(1-alpha_hat)*predicted_v

                if i > 0:
                    alpha_hat_minus = self.alpha_hat[t-1][:, None, None]
                    # sigma = (1-alpha_hat_minus)/(1-alpha_hat)*beta
                    noise = torch.randn_like(x)

                    pre_u_0 = (torch.sqrt(alpha_hat_minus)*beta/(1-alpha_hat))*pre_x_0 + (torch.sqrt(alpha)*(1-alpha_hat_minus)/(1-alpha_hat))*x
                    x = pre_u_0 + torch.sqrt(beta) * noise

                    # x = pre_u_0 + torch.sqrt(beta) * noise
                else:
                    x = pre_x_0
                    # sigma = torch.zeros_like(x)
                    # noise = torch.zeros_like(x)

                # x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
            return x.detach().cpu().numpy(), padding_mask.detach().cpu().numpy()

        # # return (out/100.0+1.0)/2.0
        # return out/100.0, out_robust/100.0
        
        
    def sample_with_mask(self, model, n, padding_mask, start_time_emb, start_pos_emb):
        logging.info(f"Sampling {n} new populations....")
        model.eval()
        with torch.no_grad():
            
            x = torch.randn((n, self.channels, self.emb_dim)).float().to(self.device)
            
            start_flag_idx = torch.zeros_like(padding_mask, dtype=torch.int32).to(self.device)  # 创建与 mask_ 第一列相同形状的张量
            start_flag_idx[:, 0] = 1
            
            for i in tqdm(reversed(range(0, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                predicted_v = model(x, t, padding_mask, start_time_emb, start_pos_emb, start_flag_idx)
                # predicted_noise = model(x, t)

                # print('predicted_noise: ', predicted_noise)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]

                pre_x_0 = torch.sqrt(alpha_hat)*x - torch.sqrt(1-alpha_hat)*predicted_v

                if i > 0:
                    alpha_hat_minus = self.alpha_hat[t-1][:, None, None]
                    # sigma = (1-alpha_hat_minus)/(1-alpha_hat)*beta
                    noise = torch.randn_like(x)

                    pre_u_0 = (torch.sqrt(alpha_hat_minus)*beta/(1-alpha_hat))*pre_x_0 + (torch.sqrt(alpha)*(1-alpha_hat_minus)/(1-alpha_hat))*x
                    x = pre_u_0 + torch.sqrt(beta) * noise

                    # x = pre_u_0 + torch.sqrt(beta) * noise
                else:
                    x = pre_x_0
                    # sigma = torch.zeros_like(x)
                    # noise = torch.zeros_like(x)

                # x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
            return x.detach().cpu().numpy()    



def decode(emb, mask):
    
    decoded_seqs = []
    decoded_tims = []

    # 假设计算出的弧度是 rad
    for i in range(0, len(mask)):
        
        decoded_one_seq = []
        decoded_one_tim = []
        
        # 第i条轨迹的掩码和嵌入向量
        trajmask = mask[i]
        trajectory = emb[i]
        
        actual_l = trajmask.count(False)
        
        # 先解码第一个点
        pos_emb = trajectory[0][:4*num_freqs]
        tim_emb = trajectory[0][4*num_freqs:]
        
        rad_m = math.atan2(tim_emb[1], tim_emb[2])
        rad_m_norm = rad_m % (2 * math.pi)
        m_final = int(rad_m_norm / (2 * math.pi) * 60)
        
        rad_h = math.atan2(tim_emb[3], tim_emb[4])
        rad_h_norm = rad_h % (2 * math.pi)
        h_final = int(rad_h_norm / (2 * math.pi) * 24)
        
        rad_d = math.atan2(tim_emb[5], tim_emb[6])
        rad_d_norm = rad_d % (2 * math.pi)
        d_final = int(rad_d_norm / (2 * math.pi) * 7)
        
        second_offset = np.exp(tim_emb[7] * std_delta + mea_delta)
        # minutes = int(second_offset // 60)
        # seconds = int(second_offset % 60)
        seconds = int(second_offset)
        
        freqs = np.exp(np.linspace(0.0, 5.0, num_freqs))*math.pi
        
        # 1. 拆分嵌入向量
        fx_sin = np.array(pos_emb[0 : num_freqs])
        fx_cos = np.array(pos_emb[num_freqs : 2*num_freqs])
        
        fy_sin = np.array(pos_emb[2*num_freqs : 3*num_freqs])
        fy_cos = np.array(pos_emb[3*num_freqs : 4*num_freqs])
        
        # 2. 使用 atan2 获取每个频率对应的 (freq * norm_val)
        # atan2(y, x) 能够自动处理象限
        angles_lon = np.arctan2(fx_sin, fx_cos)
        angles_lat = np.arctan2(fy_sin, fy_cos)
        # norm_lon * freqs对应着 angles_lon
        # norm_lat * freqs对应着 angles_lat
        
        # 3. 最小二乘法求解: freqs * norm_val = angles
        # 方程形式为 Ax = B -> freqs.reshape(-1, 1) * norm_val = angles
        # 这里我们求出一个标量 norm_val
        norm_lon, _, _, _ = np.linalg.lstsq(freqs.reshape(-1, 1), angles_lon, rcond=None)
        norm_lat, _, _, _ = np.linalg.lstsq(freqs.reshape(-1, 1), angles_lat, rcond=None)
        
        # lstsq 返回的是列表，取第一个元素
        n_lon = norm_lon[0]
        n_lat = norm_lat[0]
        
        # 4. 反归一化映射回原始经纬度
        # 编码公式: norm = (val - min) / (max - min) * 2 - 1
        # 反推: val = (norm + 1) / 2 * (max - min) + min
        lon = (n_lon + 1) / 2 * (max_lon - min_lon) + min_lon
        lat = (n_lat + 1) / 2 * (max_lat - min_lat) + min_lat
        
        decoded_one_seq.append([lon, lat])
        decoded_one_tim.append([d_final, h_final, m_final, seconds])
        
        for j in range(1, actual_l):
            # 第i条轨迹的第j>=1个轨迹点的时间空间联合嵌入向量
            pos_emb = trajectory[j][:4*num_freqs]
            tim_emb = trajectory[j][4*num_freqs:]
            # 解码时间部分
            second_offset = np.exp(tim_emb[7] * std_delta + mea_delta)
            decoded_one_tim.append([0, 0, 0, int(second_offset)])
            # 解码位置部分
            fx_sin = np.array(pos_emb[0 : num_freqs])
            fx_cos = np.array(pos_emb[num_freqs : 2*num_freqs])
            fy_sin = np.array(pos_emb[2*num_freqs : 3*num_freqs])
            fy_cos = np.array(pos_emb[3*num_freqs : 4*num_freqs])
            angles_lon = np.arctan2(fx_sin, fx_cos)
            angles_lat = np.arctan2(fy_sin, fy_cos)
            norm_lon, _, _, _ = np.linalg.lstsq(freqs.reshape(-1, 1), angles_lon, rcond=None)
            norm_lat, _, _, _ = np.linalg.lstsq(freqs.reshape(-1, 1), angles_lat, rcond=None)
            n_lon = norm_lon[0]
            n_lat = norm_lat[0]
            lon = (n_lon + 1) / 2 * (max_lon - min_lon) + min_lon
            lat = (n_lat + 1) / 2 * (max_lat - min_lat) + min_lat
            decoded_one_seq.append([lon, lat])
            
        decoded_seqs.append(decoded_one_seq)
        decoded_tims.append(decoded_one_tim)
        
    return decoded_seqs, decoded_tims
