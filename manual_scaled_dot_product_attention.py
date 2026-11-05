import torch
import torch.nn.functional as F

def manual_scaled_dot_product_attention(q, k, v, padding_mask=None, attn_mask=None, scale=None):
    """
    手工实现 scaled_dot_product_attention，支持 padding_mask
    
    Args:
        q: (batch, seq_len_q, dim)
        k: (batch, seq_len_k, dim)
        v: (batch, seq_len_v, dim)  # seq_len_v 应等于 seq_len_k
        padding_mask: (batch, seq_len_k), True 表示需要掩掉的位置（padding）
        attn_mask: (seq_len_q, seq_len_k) or (batch, seq_len_q, seq_len_k), 
                   True 表示需要掩掉的位置
        scale: 缩放因子，默认为 1/sqrt(dim)
    
    Returns:
        output: (batch, seq_len_q, dim)
        attention_weights: (batch, num_heads, seq_len_q, seq_len_k)  # 可选
    """
    
    # 1. 计算缩放因子
    if scale is None:
        scale = q.size(-1) ** -0.5
    
    # 2. 计算注意力分数: (batch, seq_len_q, seq_len_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # 3. 应用 padding_mask（掩掉 key 中的 padding 位置）
    if padding_mask is not None:
        # padding_mask: (batch, seq_len_k), True 表示需要掩掉
        # 扩展维度以便广播到 scores: (batch, 1, seq_len_k)
        padding_mask_expanded = padding_mask.unsqueeze(1)  # (batch, 1, seq_len_k)
        
        # 将需要掩掉的位置设为 -inf
        scores = scores.masked_fill(padding_mask_expanded, float('-inf'))
    
    # 4. 应用 attn_mask（掩掉特定位置）
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            # (seq_len_q, seq_len_k) -> (1, seq_len_q, seq_len_k)
            attn_mask_expanded = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            attn_mask_expanded = attn_mask
        else:
            raise ValueError(f"attn_mask dimension {attn_mask.dim()} not supported")
        
        scores = scores.masked_fill(attn_mask_expanded, float('-inf'))
    
    # 5. Softmax 获取注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    
    # 6. 应用注意力权重到 v
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights


def compare_with_official():
    """验证手工实现与官方 F.scaled_dot_product_attention 的一致性"""
    
    # 创建测试数据
    batch_size = 4
    seq_len = 16
    dim = 32
    
    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)
    v = torch.randn(batch_size, seq_len, dim)
    
    # 创建 padding_mask（随机）
    padding_mask = torch.rand(batch_size, seq_len) > 0.7
    
    # 手工实现
    manual_output, manual_weights = manual_scaled_dot_product_attention(
        q, k, v, padding_mask=padding_mask
    )
    
    # 官方实现（需要转换 mask）
    # 注意：官方实现没有直接的 padding_mask 参数，需要转换为 attn_mask
    mask = padding_mask.unsqueeze(1)  # (batch, 1, seq_len)
    mask = mask.expand(-1, seq_len, -1)  # (batch, seq_len, seq_len)
    # 简化：使用加法掩码
    attn_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=q.dtype)
    attn_mask = attn_mask.masked_fill(mask, float('-inf'))
    
    official_output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask
    )
    
    # 比较结果
    diff = (manual_output - official_output).abs().max()
    print(f"Maximum difference: {diff:.6f}")
    
    # 验证被掩掉的位置权重是否为 0
    print(f"\nPadding mask (first batch): {padding_mask[0].tolist()}")
    print(f"Attention weights to padding (first batch, first query):")
    for i in range(seq_len):
        if padding_mask[0, i]:
            weight_sum = manual_weights[0, 0, i].sum().item()
            print(f"  Position {i} (padding): total weight = {weight_sum:.6f}")
    
    assert diff < 1e-5, "Manual implementation differs from official!"

compare_with_official()