import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from modules import get_original_data, form_features, CustomDataset
# ---------------------------------------------------------
# 第一部分：确定性的编码函数 (必须与你扩散模型训练时的一致)
# ---------------------------------------------------------

    
def get_deterministic_encoding(coords, delta_t_feat, num_freqs=32):
    """
    coords: [B, 2] (lon, lat 归一化到 -1~1)
    delta_t_feat: [B, 4] (sin_h, cos_h, sin_d, cos_d) 或 [B, 1] (offset/delta_t)
    """
    # 空间编码 (Multi-scale Sine/Cosine),这里频率到10，对应的最小周期约为10米左右，T=2*pi/exp(10)*pi, 武汉市经度和维度的跨度约为100km
    freqs = (torch.exp(torch.linspace(0, 10, num_freqs)) * math.pi).to(coords.device)
    x = coords.unsqueeze(-1) * freqs # [B, 2, F]
    space_enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1).view(coords.shape[0], -1)
    
    # 假设你最后把时间特征和空间特征拼接
    # 这里为了演示，简单拼接
    full_embedding = torch.cat([space_enc, delta_t_feat], dim=-1)
    return full_embedding

# ---------------------------------------------------------
# 第二部分：MLP 解码模型
# ---------------------------------------------------------
class TrajectoryDecoder(nn.Module):
    def __init__(self, input_dim, output_dim=3): # Output: [lon, lat, time_val]
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 第三部分：自监督训练逻辑
# ---------------------------------------------------------
def train_decoder():
    # 参数设置
    num_freqs = 32
    input_dim = num_freqs * 4 + 4 # 空间(F*4) + 时间(假设4维)
    model = TrajectoryDecoder(input_dim, output_dim=3) # lon, lat, t
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for step in range(10000): # 只需要迭代几千次即可收敛
        # 1. 随机产生物理真值 (模拟城市范围内的随机点)
        true_coords = (torch.rand(1024, 2) * 2 - 1) # [-1, 1]
        true_time = torch.rand(1024, 4) # 模拟时间特征
        
        # 2. 通过确定性公式获取 Embedding (这是扩散模型生成的类型)
        with torch.no_grad():
            input_emb = get_deterministic_encoding(true_coords, true_time, num_freqs)
            # 模拟扩散模型生成时的微小噪声 (提高鲁棒性)
            input_emb += torch.randn_like(input_emb) * 0.01 
        
        # 3. 训练解码器还原
        pred_phys = model(input_emb)
        
        # 这里假设只还原坐标作为演示
        target = torch.cat([true_coords, true_time[:, :1]], dim=-1) # 取部分作为目标
        loss = criterion(pred_phys, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")

    return model


# 损失函数的权重重分配（Loss Weighting）起始点（$i=0$）承载了整条轨迹的“时空锚点”信息，如果它预测错了，整条轨迹的时间链都会崩塌。因此，我们需要给第0个点更高的 Loss 权重。
def trajectory_loss_fn(pred_noise, target_noise, padding_mask, alpha=0.1):
    # 1. 基础 MSE Loss (噪声预测)
    mse_loss = F.mse_loss(pred_noise, target_noise, reduction='none')
    
    # 2. 增加起始点权重 (假设 batch_first=True)
    weights = torch.ones_like(mse_loss)
    weights[:, 0, :] *= 5.0  # 给第0个点（起始时间+位置）5倍权重
    
    # 3. 忽略 Padding 部分
    active_loss = (mse_loss * weights * (~padding_mask).unsqueeze(-1)).mean()
    
    return active_loss


class TrajectoryInputEncoder(nn.Module):
    def __init__(self, max_len, space_dim, time_dim, embed_dim, num_freqs=32):
        super().__init__()
        # A. 基础映射：将拼接后的 (空间编码 + 时间编码) 映射到统一的隐空间
        # 假设空间编码后是 space_dim, 时间编码后是 time_dim
        self.feature_proj = nn.Linear(space_dim + time_dim, embed_dim)
        
        # B. 位置嵌入 (Learned Position Embedding)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        
        # C. 类别嵌入 (Type Embedding)
        # 0: Start, 1: Data, 2: Pad
        self.type_embed = nn.Embedding(3, embed_dim)
        
        # D. 最终的标准化，防止梯度爆炸
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, space_features, time_features, actual_lengths):
        """
        space_features: [B, L, space_dim] (经纬度多尺度编码结果)
        time_features: [B, L, time_dim] (三层结构编码结果)
        actual_lengths: [B] (当前 Batch 中每个轨迹的真实长度)
        """
        B, L, _ = space_features.shape
        device = space_features.device

        # 1. 特征拼接与投影
        # 将确定的数学编码转换为可学习的向量
        combined_feat = torch.cat([space_features, time_features], dim=-1)
        x = self.feature_proj(combined_feat) # [B, L, embed_dim]

        # 2. 注入位置嵌入 (Sequence Order)
        pos_indices = torch.arange(L, device=device).unsqueeze(0) # [1, L]
        x = x + self.pos_embed(pos_indices)

        # 3. 注入类别嵌入 (Logic Type)
        # 初始化全为 Pad (2)
        type_ids = torch.full((B, L), 2, dtype=torch.long, device=device)
        for i, length in enumerate(actual_lengths):
            # 真实轨迹部分标记为 Data (1)
            type_ids[i, :length] = 1
            # 起始点特殊标记为 Start (0)
            type_ids[i, 0] = 0
        
        x = x + self.type_embed(type_ids)

        # 4. 归一化输出，进入 Transformer
        return self.norm(x)
    
    
    if __name__ == "__main__":
        joson_file = './day_traces_0.json'
        time_seq, posi_seq = get_original_data(joson_file)
        # print(time_seq[0])
        # print(posi_seq[0])
        print(len(time_seq), len(posi_seq))
        
        combined, mask = form_features(time_seq, posi_seq)
        print(combined.shape)
        print(mask.shape)
        print(combined[0][:6, -6:])
        print(combined[0][:6, :10])
        print(mask[0])
        
        dataset = CustomDataset(combined, mask)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
        
        for batch_data, batch_mask in dataloader:
            print(batch_data.shape)