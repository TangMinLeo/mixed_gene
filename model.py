import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalFusion(nn.Module):
    def __init__(self, pop_dim, traj_dim, hidden_dim, num_heads=8):
        super().__init__()
        self.pop_proj = nn.Linear(pop_dim, hidden_dim)
        self.traj_proj = nn.Linear(traj_dim, hidden_dim)
        
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 自注意力层用于各自模态的内部约束
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, pop_emb, traj_emb):
        # 投影到统一维度
        pop_proj = self.pop_proj(pop_emb)  # [batch, seq_len, hidden]
        traj_proj = self.traj_proj(traj_emb)  # [batch, seq_len, hidden]
        
        # 人口信息作为查询，轨迹信息作为键值（学习人口如何影响轨迹）
        pop_to_traj, _ = self.cross_attention(
            query=pop_proj, 
            key=traj_proj, 
            value=traj_proj
        )
        
        # 轨迹信息作为查询，人口信息作为键值（学习轨迹如何反映人口）
        traj_to_pop, _ = self.cross_attention(
            query=traj_proj,
            key=pop_proj,
            value=pop_proj
        )
        
        # 融合两种交叉信息
        fused_pop = self.layer_norm(pop_proj + pop_to_traj)
        fused_traj = self.layer_norm(traj_proj + traj_to_pop)
        
        # 最终拼接
        combined = torch.cat([fused_pop, fused_traj], dim=-1)
        return combined




class MultiModalDiffusion(nn.Module):
    def __init__(self, pop_dim, traj_dim, hidden_dim, num_timesteps=1000):
        super().__init__()
        
        # 融合模块
        self.fusion = CrossModalFusion(pop_dim, traj_dim, hidden_dim)
        
        # 扩散模型主干
        self.diffusion_backbone = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim * 2,  # 融合后维度
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=6
        )
        
        # 输出头 - 分别预测人口和轨迹
        self.pop_head = nn.Linear(hidden_dim * 2, pop_dim)
        self.traj_head = nn.Linear(hidden_dim * 2, traj_dim)
        
        # 扩散调度
        self.setup_diffusion(num_timesteps)
    
    def setup_diffusion(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def forward(self, noisy_pop, noisy_traj, timestep, mask):
        """
        前向传播
        noisy_pop: 带噪声的人口嵌入 [batch, seq_len, pop_dim]
        noisy_traj: 带噪声的轨迹嵌入 [batch, seq_len, traj_dim]  
        timestep: 时间步 [batch]
        mask: 序列掩码 [batch, seq_len]
        """
        # 1. 多模态融合
        fused = self.fusion(noisy_pop, noisy_traj)  # [batch, seq_len, hidden*2]
        
        # 2. 添加时间步信息
        t_embed = self.timestep_embedding(timestep, fused.size(-1))
        fused = fused + t_embed.unsqueeze(1)
        
        # 3. 通过扩散主干
        encoded = self.diffusion_backbone(fused, src_key_padding_mask=~mask.bool())
        
        # 4. 分别预测噪声
        pop_noise_pred = self.pop_head(encoded)
        traj_noise_pred = self.traj_head(encoded)
        
        return pop_noise_pred, traj_noise_pred
    
    def compute_loss(self, pop_pred, traj_pred, pop_target, traj_target, mask):
        """分别计算两部分损失"""
        mask_expanded = mask.unsqueeze(-1)
        
        # 人口信息损失
        pop_loss = F.mse_loss(pop_pred, pop_target, reduction='none')
        pop_loss = (pop_loss * mask_expanded).sum() / mask_expanded.sum()
        
        # 轨迹信息损失  
        traj_loss = F.mse_loss(traj_pred, traj_target, reduction='none')
        traj_loss = (traj_loss * mask_expanded).sum() / mask_expanded.sum()
        
        # 可选：添加相关性约束损失
        correlation_loss = self.correlation_constraint(pop_pred, traj_pred, mask)
        
        total_loss = pop_loss + traj_loss + 0.1 * correlation_loss
        return total_loss
    
    def correlation_constraint(self, pop_pred, traj_pred, mask):
        """鼓励学习人口-轨迹间的合理相关性"""
        # 计算批次内的相关性矩阵
        batch_size, seq_len, _ = pop_pred.shape
        
        # 只考虑有效位置
        valid_mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
        
        pop_valid = (pop_pred * valid_mask).view(batch_size, -1)
        traj_valid = (traj_pred * valid_mask).view(batch_size, -1)
        
        # 计算相关性（简化版本）
        pop_norm = F.normalize(pop_valid, dim=1)
        traj_norm = F.normalize(traj_valid, dim=1)
        correlation = torch.mm(pop_norm, traj_norm.t())
        
        # 鼓励适度的相关性（避免过强或过弱）
        target_correlation = 0.3  # 假设的合理相关性强度
        correlation_loss = F.mse_loss(correlation.mean(), 
                                    torch.tensor(target_correlation))
        
        return correlation_loss



class ConditionalDiffusionFusion(nn.Module):
    def __init__(self, pop_dim, traj_dim, hidden_dim):
        super().__init__()
        
        # 人口编码器
        self.pop_encoder = nn.Sequential(
            nn.Linear(pop_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 轨迹编码器  
        self.traj_encoder = nn.Sequential(
            nn.Linear(traj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 条件融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, noisy_traj, timestep, pop_emb):
        """
        以人口信息为条件生成轨迹
        noisy_traj: 带噪声的轨迹 [batch, seq_len, traj_dim]
        pop_emb: 人口嵌入 [batch, seq_len, pop_dim]
        """
        # 编码人口信息
        pop_encoded = self.pop_encoder(pop_emb)
        
        # 编码带噪声轨迹
        traj_encoded = self.traj_encoder(noisy_traj)
        
        # 融合条件信息
        combined = torch.cat([pop_encoded, traj_encoded], dim=-1)
        conditioned_output = self.fusion_layer(combined)
        
        return conditioned_output