import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Q, K, V投影层
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states, attention_mask=None, use_cache=False, past_key_value=None):
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len] 或 [batch_size, 1, 1, seq_len]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. 投影到Q, K, V
        Q = self.q_proj(hidden_states)  # [batch, seq, hidden]
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        # 2. 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 现在形状: [batch, num_heads, seq_len, head_dim]
        
        # 3. 处理过去键值对（用于生成时的缓存）
        if past_key_value is not None:
            past_k, past_v = past_key_value
            K = torch.cat([past_k, K], dim=2)  # 沿序列维度拼接
            V = torch.cat([past_v, V], dim=2)
            
        # 4. 准备注意力掩码
        if attention_mask is not None:
            if attention_mask.dim() == 2:  # [batch, seq_len]
                # 转换为因果+padding组合掩码
                attention_mask = self._prepare_attention_mask(
                    attention_mask, hidden_states.dtype, hidden_states.device
                )
            # 现在attention_mask形状应为 [batch, 1, seq_len, seq_len] 或 [batch, 1, 1, seq_len]
        
        # 5. 计算注意力
        attn_output, attn_weights = self._attention(Q, K, V, attention_mask)
        
        # 6. 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # 7. 输出投影
        output = self.out_proj(attn_output)
        
        # 8. 返回结果和更新的键值对
        if use_cache:
            return output, (K, V)
        return output
    
    def _prepare_attention_mask(self, input_mask, dtype, device):
        """
        将输入的2D mask转换为4D注意力mask
        """
        batch_size, seq_len = input_mask.shape
        
        # 创建因果掩码
        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.uint8, device=device)
        ).view(1, 1, seq_len, seq_len)
        
        # 将因果掩码转换为注意力分数格式
        # 需要的位置为0，不需要的位置为负无穷
        causal_mask = (1.0 - causal_mask) * float('-inf')
        
        # 处理padding mask
        # input_mask: [batch, seq_len], 1表示有效，0表示padding
        padding_mask = (1.0 - input_mask) * float('-inf')
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        
        # 组合掩码：因果+padding
        # 注意：这里需要广播因果掩码到batch维度
        combined_mask = causal_mask + padding_mask
        
        return combined_mask.to(dtype=dtype)
    
    def _attention(self, Q, K, V, attention_mask):
        """核心注意力计算"""
        # 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 现在形状: [batch, num_heads, seq_len, seq_len]
        # 应用注意力掩码
        if attention_mask is not None:
            # 确保掩码形状正确
            if attention_mask.size(2) == 1:  # [batch, 1, 1, seq_len]
                # 这种情况通常只用于decoder的cross-attention或特定场景
                scores = scores + attention_mask
            else:  # [batch, 1, seq_len, seq_len]
                scores = scores + attention_mask
        
        # Softmax得到注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 注意力加权
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class CausalSelfAttentionWithCache(CausalSelfAttention):
    def forward_for_generation(self, hidden_states, attention_mask, past_key_value=None):
        batch_size, seq_len = hidden_states.shape[:2]
        
        # 对于生成，query是当前位置，key/value包含过去所有位置
        if past_key_value is not None:
            # 过去已计算了前n-1个token的K, V
            past_k, past_v = past_key_value
            key_length = past_k.shape[2] + seq_len  # 过去长度 + 当前长度
        else:
            key_length = seq_len
        
        # 准备因果掩码（只关注当前位置及以前）
        causal_mask = self._prepare_causal_mask_for_generation(
            batch_size, seq_len, key_length, hidden_states.device
        )
        
        # 如果有padding mask，需要调整形状
        if attention_mask is not None:
            # 扩展attention_mask以匹配key_length
            attention_mask = self._expand_mask_for_generation(
                attention_mask, seq_len, key_length
            )
            # 组合掩码
            combined_mask = causal_mask + attention_mask
        else:
            combined_mask = causal_mask
        
        # 正常计算注意力
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=combined_mask,
            use_cache=True,
            past_key_value=past_key_value
        )
    
    def _prepare_causal_mask_for_generation(self, batch_size, query_length, key_length, device):
        """为生成准备因果掩码"""
        # 创建因果掩码 [1, 1, query_length, key_length]
        causal_mask = torch.tril(
            torch.ones((key_length, key_length), device=device)
        )[-query_length:, :]  # 只取最后query_length行
        
        causal_mask = causal_mask.view(1, 1, query_length, key_length)
        
        # 转换为注意力掩码格式
        causal_mask = (1.0 - causal_mask) * float('-inf')
        
        return causal_mask.expand(batch_size, 1, query_length, key_length)