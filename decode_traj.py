import json
import math
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim


# from modules import decode
min_lon = 115.416667
max_lon = 117.500000
min_lat = 39.433333
max_lat = 41.050000

num_freqs = 32
mea = 7.3930369290044915
std = 1.8804091151810995


class ResBlock(nn.Module):
    """
    一个基础的残差块，保持维度不变。
    架构：Input -> LN -> Linear -> SiLU -> Linear -> + Input
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(), # SiLU 激活函数比 ReLU 更平滑，适合拟合正弦波
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        # 残差连接：x 是输入，self.net(x) 是非线性变换后的结果
        # 模型学习的是从 x 到目标之间的差异（Residual）
        return x + self.net(x)

class CoordinateDecoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, num_blocks=2):
        super().__init__()
        
        # 1. 输入投影层：将 128 维原始编码投影到 512 维隐空间
        self.initial_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 2. 堆叠残差块：深度提取高频特征的关联性
        # 在这里，信号一直在 512 维空间中流动和修正
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # 3. 输出头：将 512 维隐特征映射回 2 维坐标 [lon, lat]
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh() # 严格限制在 [-1, 1]
        )

    def forward(self, x):
        # x: [B, 128]
        h = self.initial_proj(x)
        
        for block in self.res_blocks:
            h = block(h)
            
        coords = self.output_head(h)
        return coords


def get_encoding_formula(coords, num_freqs=32, device='cuda'):
    """
    coords: [B, 2] 归一化后的坐标
    """
    # B = len(coords) coords为numpy数据[B, 2]尺寸
    lon = coords[:, 0:1] # [B, 1]
    lat = coords[:, 1:2] # [B, 1]
    
    # lon_ori = coords[:, 0:1] # [B, 1]
    # lat_ori = coords[:, 1:2] # [B, 1]
    
    # lon_norm = (lon_ori- min_lon)/(max_lon-min_lon)*2-1
    # lat_norm = (lat_ori- min_lat)/(max_lat-min_lat)*2-1
    
    # lon = torch.from_numpy(lon_norm)
    # lat = torch.from_numpy(lat_norm)
    
    # 频率设置：必须与你之前的逻辑严格一致 (exp(0~5) * pi)
    freqs = torch.exp(torch.linspace(0.0, 5.0, num_freqs, device=device)) * np.pi
    
    # lon = coords[:, 0:1] # [B, 1]
    # lat = coords[:, 1:2] # [B, 1]
    
    # 计算多尺度特征
    fx_sin = torch.sin(lon * freqs)
    fx_cos = torch.cos(lon * freqs)
    fy_sin = torch.sin(lat * freqs)
    fy_cos = torch.cos(lat * freqs)
    
    # 拼接成 [B, num_freqs * 4]
    return torch.cat([fx_sin, fx_cos, fy_sin, fy_cos], dim=-1)


def train_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_freqs = 32
    input_dim = num_freqs * 4
    
    decoder = CoordinateDecoder(input_dim=input_dim, hidden_dim=512, num_blocks=2).to(device)
    
    optimizer = optim.Adam(decoder.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    
    print("开始训练坐标解码器...")
    
    for step in range(20000): # 迭代 5000 次左右即可
        # 1. 随机生成训练数据 [-1, 1]
        raw_coords = torch.rand(1024, 2, device=device, requires_grad=False) * 2 - 1
        # ori_coords = np.random.rand(1024, 2)
        # ori_lon = ori_coords[:, 0:1]
        # ori_lat = ori_coords[:, 1:2]
        # lon_ = min_lon+ori_lon*(max_lon-min_lon)
        # lat_ = min_lat+ori_lat*(max_lat-min_lat)
        # sim_coords = np.concatenate((lon_, lat_), axis=-1)
        # raw_coords = torch.from_numpy(sim_coords)
        
        # 2. 转化为多尺度编码 (输入)
        with torch.no_grad():
            features = get_encoding_formula(raw_coords, num_freqs=32, device=device)
            
            # features = get_encoding_formula(raw_coords, num_freqs=num_freqs)
            # 模拟扩散模型输出的微小噪声 (注入 0.01 的高斯噪声增强鲁棒性)
            features += torch.randn_like(features) * 0.00001
            
        # 3. 前向传播与优化
        pred_coords = decoder(features)
        loss = criterion(pred_coords, raw_coords)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss.item():.8f}")

    # 4. 保存模型
    torch.save(decoder.state_dict(), "coord_decoder.pt")
    return decoder

# 运行训练
# my_decoder = train_decoder()
if "__main__" == __name__:
    # 直接加载预训练模型
    # 这里可以添加测试代码，验证解码器的性能
    trajs = [(116.318417, 39.984702), (116.316911, 39.984568), (116.314907, 39.984501), (116.313941, 39.984621), (116.311917, 39.984647), (116.310063, 39.984698), (116.308638, 39.98449), (116.307292, 39.984427), 
   (116.305466, 39.98441), (116.303672, 39.984429), (116.301888, 39.984359), (116.299943, 39.984288), (116.299566, 39.983726), (116.299321, 39.983337), (116.299031, 39.983675), (116.299292, 39.984397), (116.299508, 39.984463), 
   (116.299407, 39.984498), (116.299617, 39.984743), (116.300019, 39.984373), (116.301221, 39.984925), (116.303037, 39.985009), (116.304811, 39.985023), (116.306531, 39.985135), (116.308313, 39.985256), (116.30998, 39.985443), 
   (116.309893, 39.986697), (116.310059, 39.988008), (116.309905, 39.989287), (116.310277, 39.990406), (116.31009, 39.99069), (116.310657, 39.990874), (116.312473, 39.991058), (116.314385, 39.991095), (116.316172, 39.991189), 
   (116.318231, 39.991132), (116.320255, 39.991282), (116.322004, 39.991412), (116.3228, 39.992365), (116.322658, 39.99364), (116.322606, 39.995235), (116.322425, 39.996632), (116.322261, 39.998151), (116.322178, 39.999607), 
   (116.323605, 39.999558), (116.32451, 39.999641), (116.324498, 39.99956), (116.320158, 40.004775), (116.32126, 40.00525), (116.321447, 40.006813), (116.321425, 40.008337), (116.321625, 40.009021), (116.321544, 40.00898), 
   (116.321486, 40.009081), (116.321479, 40.009065), (116.321389, 40.008994), (116.321445, 40.009242), (116.321537, 40.008999), (116.321614, 40.009024), (116.321721, 40.009247), (116.322179, 40.009399), (116.321833, 40.009314), 
   (116.320964, 40.008694), (116.320573, 40.008348), (116.320241, 40.007959), (116.319986, 40.007661), (116.319436, 40.007459), (116.319313, 40.007555), (116.319321, 40.007654), (116.319371, 40.007479), (116.318852, 40.007548), 
   (116.318471, 40.008015), (116.318639, 40.007954), (116.318494, 40.007925), (116.319124, 40.007411), (116.318754, 40.007849), (116.318753, 40.008055), (116.319475, 40.007778), (116.31949, 40.008544), (116.318549, 40.006747), 
   (116.318625, 40.006724), (116.319398, 40.006791), (116.319816, 40.007582), (116.320343, 40.007853), (116.320851, 40.008306), (116.321395, 40.008724), (116.321933, 40.008864), (116.322297, 40.008821), (116.322259, 40.009068), 
   (116.321683, 40.008905), (116.321431, 40.009013)]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    feats = []
    for i in range(len(trajs)):
        lon, lat = trajs[i]
        lon_n = (lon-min_lon)/(max_lon-min_lon)*2-1
        lat_n = (lat-min_lat)/(max_lat-min_lat)*2-1
        feats.append([lon_n, lat_n])
    feat_np = np.array(feats)
    featus = torch.from_numpy(feat_np)
    featus = featus.to('cuda:0')
    features = get_encoding_formula(featus, num_freqs=32, device=device)        
    features = features.float()
    
    decoder = train_decoder()
    decoded = decoder(features)
    dec = decoded.detach().cpu().numpy()
    for i in range(len(dec)):
        poit = dec[i,:]
        print(f'{(poit[0]+1)*(max_lon-min_lon)/2+min_lon}, {(poit[1]+1)*(max_lat-min_lat)/2+min_lat},')
