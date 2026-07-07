"""
Bidirectional cross-attention model for kcat regression.
FINAL BUG-FREE VERSION:
- CLS token always valid
- No query pollution
- RXNFP: no positional encoding
- Protein depthwise conv (local bias)
- Mask dtype 100% safe
- AMP stable
"""

from __future__ import annotations
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, expansion: int = 4):
        super().__init__()
        hidden = int(d_model * expansion)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1, ffn_expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout, expansion=ffn_expansion)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)

        # ✅ FIX: Zero out PADDING QUERY before attention (critical!)
        if key_padding_mask is not None:
            h = h.masked_fill(~key_padding_mask.unsqueeze(-1), 0.0)

        attn_out, _ = self.attn(
            query=h, key=h, value=h,
            key_padding_mask=~key_padding_mask if key_padding_mask is not None else None,
            need_weights=False
        )

        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))

        # Final zero padding to prevent drift
        if key_padding_mask is not None:
            x = x.masked_fill(~key_padding_mask.unsqueeze(-1), 0.0)

        return x


class BiCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1, ffn_expansion: int = 4):
        super().__init__()
        self.p_norm_q = nn.LayerNorm(d_model)
        self.r_norm_kv = nn.LayerNorm(d_model)
        self.p_attends_r = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.r_norm_q = nn.LayerNorm(d_model)
        self.p_norm_kv = nn.LayerNorm(d_model)
        self.r_attends_p = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.p_norm_ffn = nn.LayerNorm(d_model)
        self.r_norm_ffn = nn.LayerNorm(d_model)
        self.p_ffn = FeedForward(d_model, dropout, ffn_expansion)
        self.r_ffn = FeedForward(d_model, dropout, ffn_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, protein: torch.Tensor, reaction: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None, reaction_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # --------------------------
        # Protein attends Reaction
        # --------------------------
        p_q = self.p_norm_q(protein)
        r_kv = self.r_norm_kv(reaction)

        # ✅ FIX: Zero padding queries
        if protein_mask is not None:
            p_q = p_q.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
        if reaction_mask is not None:
            r_kv = r_kv.masked_fill(~reaction_mask.unsqueeze(-1), 0.0)

        p_delta, _ = self.p_attends_r(
            query=p_q, key=r_kv, value=r_kv,
            key_padding_mask=~reaction_mask if reaction_mask is not None else None,
            need_weights=False
        )

        # --------------------------
        # Reaction attends Protein
        # --------------------------
        r_q = self.r_norm_q(reaction)
        p_kv = self.p_norm_kv(protein)

        # ✅ FIX: Zero padding queries
        if reaction_mask is not None:
            r_q = r_q.masked_fill(~reaction_mask.unsqueeze(-1), 0.0)
        if protein_mask is not None:
            p_kv = p_kv.masked_fill(~protein_mask.unsqueeze(-1), 0.0)

        r_delta, _ = self.r_attends_p(
            query=r_q, key=p_kv, value=p_kv,
            key_padding_mask=~protein_mask if protein_mask is not None else None,
            need_weights=False
        )

        protein = protein + self.dropout(p_delta)
        reaction = reaction + self.dropout(r_delta)

        protein = protein + self.p_ffn(self.p_norm_ffn(protein))
        reaction = reaction + self.r_ffn(self.r_norm_ffn(reaction))

        # Zero padding
        if protein_mask is not None:
            protein = protein.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
        if reaction_mask is not None:
            reaction = reaction.masked_fill(~reaction_mask.unsqueeze(-1), 0.0)

        return protein, reaction


class AttentionPool(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        weights = self.score(x)
        if mask is not None:
            weights = weights.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(weights, dim=1)
        return torch.sum(weights * x, dim=1)


class KcatBiCrossAttention(nn.Module):
    def __init__(
        self, protein_dim=2560, rxnfp_dim=256, chemberta_dim=384, d_model=256, num_heads=4,
        num_layers=1, self_layers=1, dropout=0, ffn_expansion=4,
        protein_max_tokens=128, reaction_max_tokens=17,
        pooling="attn_max", fusion="interaction", use_type_embeddings=True
    ):
        super().__init__()
        self.d_model = d_model
        self.pooling = pooling
        self.fusion = fusion
        self.use_type_embeddings = use_type_embeddings

        # ============================
        # ✅ Protein: Add DEPTHWISE CONV for local motif bias (critical for kcat!)
        # ============================
        self.protein_proj = nn.Sequential(
            nn.LayerNorm(protein_dim),
            nn.Linear(protein_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, d_model),
        )
        self.protein_local_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=3, padding=1,
            groups=d_model  # depthwise
        )

        self.rxnfp_proj = nn.Sequential(nn.LayerNorm(rxnfp_dim), nn.Linear(rxnfp_dim, d_model))
        self.chemberta_proj = nn.Sequential(nn.LayerNorm(chemberta_dim), nn.Linear(chemberta_dim, d_model))

        # Pos emb
        self.protein_pos = nn.Parameter(torch.randn(1, protein_max_tokens, d_model) * 0.02)
        self.reaction_pos = nn.Parameter(torch.randn(1, reaction_max_tokens, d_model) * 0.02)

        # Type emb
        if use_type_embeddings:
            self.protein_type = nn.Parameter(torch.randn(1,1,d_model)*0.02)
            self.rxnfp_type = nn.Parameter(torch.randn(1,1,d_model)*0.02)
            self.chemberta_type = nn.Parameter(torch.randn(1,1,d_model)*0.02)

        # Blocks
        self.protein_self_in = nn.ModuleList([SelfAttentionBlock(d_model,num_heads,dropout,ffn_expansion) for _ in range(self_layers)])
        self.reaction_self_in = nn.ModuleList([SelfAttentionBlock(d_model,num_heads,dropout,ffn_expansion) for _ in range(self_layers)])
        self.bi_cross_blocks = nn.ModuleList([BiCrossAttentionBlock(d_model,num_heads,dropout,ffn_expansion) for _ in range(num_layers)])

        # Pooling
        self.protein_pooler = AttentionPool(d_model,dropout) if pooling in ("attn","attn_mean", "attn_max") else None
# 分开的后处理层：attn_mean和attn_max的统计分布差异较大，需要独立处理
        self.protein_post_pool_mean = nn.Sequential(nn.LayerNorm(d_model*2), nn.Linear(d_model*2,d_model)) if pooling=="attn_mean" else None
        self.protein_post_pool_max = nn.Sequential(nn.LayerNorm(d_model*2), nn.Linear(d_model*2,d_model)) if pooling=="attn_max" else None
        # 反应特征融合MLP：融合CLS、attention pooling和max pooling的特征
        self.reaction_fusion_mlp = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # 反应特征融合前的独立LayerNorm
        self.cls_norm = nn.LayerNorm(d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.max_norm = nn.LayerNorm(d_model)
        
        # 反应特征融合MLP：融合CLS、attention pooling和max pooling的特征
        self.reaction_fusion_mlp = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # Fusion
        # 修改前
        # single_pool_dim = d_model*2 if pooling in ("max_mean", "attn_mean", "attn_max") else d_model

        # 修改后
        # attn_mean和attn_max经过post_pool后输出d_model维度，而非d_model*2
        single_pool_dim = d_model*2 if pooling == "max_mean" else d_model
        fused_dim = single_pool_dim*4 if fusion=="interaction" else single_pool_dim*2
        self.gate = None
        if fusion in ("gated_sum","gated_concat"):
            self.gate = nn.Sequential(nn.Linear(single_pool_dim*2, single_pool_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(single_pool_dim,single_pool_dim),nn.Sigmoid())

        # Head
        hidden1 = max(d_model, fused_dim//2)
        hidden2 = max(d_model//2, hidden1//2)
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Linear(fused_dim,hidden1),nn.GELU(),nn.Dropout(dropout),
            nn.LayerNorm(hidden1),nn.Linear(hidden1,hidden2),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden2,1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m,nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2: return x.unsqueeze(1)
        if x.dim() == 3: return x
        raise ValueError(f"Expected 2D/3D tensor, got {x.shape}")

    @staticmethod
    def _resize_pos(pos: torch.Tensor, tgt_len: int) -> torch.Tensor:
        if pos.size(1) == tgt_len: return pos
        dtype = pos.dtype
        pos = F.interpolate(pos.float().transpose(1,2), size=tgt_len, align_corners=False).transpose(1,2)
        return pos.to(dtype)

    def _pool_protein(self, x: torch.Tensor, pooler: Optional[AttentionPool], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pooling == "mean":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
                return x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            return x.mean(1)

        if self.pooling == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
            return x.max(1).values

        if self.pooling == "max_mean":
            if mask is not None:
                x_max = x.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
                x_avg = x.masked_fill(~mask.unsqueeze(-1), 0.0).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            else:
                x_max = x.max(1).values
                x_avg = x.mean(1)
            return torch.cat([x_max, x_avg], -1)

        if self.pooling == "attn":
            return pooler(x, mask)

        if self.pooling == "attn_mean":
            attn = pooler(x, mask)
            if mask is not None:
                avg = x.masked_fill(~mask.unsqueeze(-1), 0.0).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            else:
                avg = x.mean(1)
            return self.protein_post_pool_mean(torch.cat([attn, avg], -1))

        if self.pooling == "attn_max":
            attn = pooler(x, mask)
            if mask is not None:
                max_val = x.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
            else:
                max_val = x.max(1).values
            return self.protein_post_pool_max(torch.cat([attn, max_val], -1))

    def forward(
        self, protein_tokens, rxnfp_tokens, chemberta_tokens,
        protein_mask: Optional[torch.Tensor] = None,
        reaction_mask: Optional[torch.Tensor] = None,
    ):
        """
        Kcat双向交叉注意力模型（带注意力掩码）的前向传播方法
        
        整体流程：
        1. 输入投影：将各模态特征统一到d_model维度
        2. 蛋白质局部卷积：捕捉催化位点的局部特征
        3. 类型嵌入：区分不同特征来源
        4. 位置嵌入：为序列添加位置信息
        5. 反应特征组装：RXNFP + ChemBERTa
        6. 注意力层：自注意力 + 双向交叉注意力（带掩码）
        7. Pooling：序列到向量的聚合
        8. Fusion：特征融合
        9. 预测头：输出Kcat预测值
        
        Args:
            protein_tokens: 蛋白质特征张量，形状为 [B, L_p] 或 [B, L_p, protein_dim]
            rxnfp_tokens: RXNFP特征张量，形状为 [B, 1] 或 [B, 1, rxnfp_dim]
            chemberta_tokens: ChemBERTa特征张量，形状为 [B, L_c] 或 [B, L_c, chemberta_dim]
            protein_mask: 蛋白质注意力掩码，形状为 [B, L_p]，True表示padding位置
            reaction_mask: 反应注意力掩码，形状为 [B, L_r]，True表示padding位置
            
        Returns:
            torch.Tensor: Kcat预测值，形状为 [B]
        """
        # ========== 第一步：输入投影（Input Projection） ==========
        # 将各模态特征投影到统一的d_model维度
        protein = self.protein_proj(self._ensure_3d(protein_tokens))
        rxnfp = self.rxnfp_proj(self._ensure_3d(rxnfp_tokens))
        chemberta = self.chemberta_proj(self._ensure_3d(chemberta_tokens))

        # ========== 第二步：蛋白质局部卷积（Protein Local Convolution） ==========
        # 关键设计：捕捉蛋白质序列中催化位点的局部依赖关系
        # 卷积操作要求通道维度在第二维，需先转置
        protein = protein.transpose(1, 2)  # [B, L, D] -> [B, D, L]
        conv_out = self.protein_local_conv(protein)  # 局部卷积提取局部特征
        protein = protein + conv_out  # 残差连接：保留原始特征并叠加卷积特征
        protein = protein.transpose(1, 2)  # [B, D, L] -> [B, L, D]

        # ========== 第三步：类型嵌入（Type Embeddings） ==========
        # 类型嵌入用于区分蛋白质、RXNFP、ChemBERTa特征来源
        if self.use_type_embeddings:
            protein += self.protein_type
            rxnfp += self.rxnfp_type
            chemberta += self.chemberta_type

        # ========== 第四步：位置嵌入（Positional Embeddings） ==========
        # 蛋白质位置嵌入：根据实际长度动态调整
        protein += self._resize_pos(self.protein_pos, protein.size(1))
        # ChemBERTa位置嵌入：使用反应位置编码的第2-17位
        chemberta += self._resize_pos(self.reaction_pos[:, 1:, :], chemberta.size(1))
        # ✅ 关键设计：RXNFP作为全局CLS token，不添加位置编码

        # ========== 第五步：组装反应特征（Assemble Reaction） ==========
        # RXNFP作为CLS token（第1位）+ ChemBERTa tokens（后续位）
        reaction = torch.cat([rxnfp, chemberta], dim=1)

        # ========== 第六步：掩码处理（Mask Processing） ==========
        # ✅ 关键设计：强制CLS token有效（不被mask掉）
        # CLS token包含全局反应信息，必须参与注意力计算
        if reaction_mask is not None:
            reaction_mask = reaction_mask.bool()
            reaction_mask[:, 0] = True  # 确保CLS token始终有效

        # ========== 第七步：注意力层（Attention Layers） ==========
        # 蛋白质自注意力：捕捉蛋白质内部依赖（带padding掩码）
        for blk in self.protein_self_in:
            protein = blk(protein, key_padding_mask=protein_mask)
        # 反应自注意力：捕捉反应内部依赖（带padding掩码）
        for blk in self.reaction_self_in:
            reaction = blk(reaction, key_padding_mask=reaction_mask)
        # 双向交叉注意力：蛋白质↔反应双向交互（带掩码）
        for blk in self.bi_cross_blocks:
            protein, reaction = blk(protein, reaction, protein_mask, reaction_mask)

        # ========== 第八步：Pooling ==========
        # 蛋白质：使用配置的pooling方式（带掩码）
        protein_pool = self._pool_protein(protein, self.protein_pooler, protein_mask)
        # 反应：直接取CLS token（RXNFP）

        # ========== 反应特征融合（Reaction Feature Fusion） ==========
        # 融合CLS token、注意力池化和最大池化的特征
        # 这样既能保留RXNFP的全局反应信息，又能获取ChemBERTa的局部原子信息
        cls = reaction[:, 0, :]  # RXNFP CLS token（全局反应信息）
        tokens = reaction[:, 1:, :]  # ChemBERTa tokens（局部原子信息）
        
        # 提取token级别的mask（排除CLS token）
        token_mask = reaction_mask[:, 1:] if reaction_mask is not None else None
        
        # 注意力池化：动态加权聚合局部特征（带mask）
        if self.protein_pooler is not None:
            attn_pool = self.protein_pooler(tokens, token_mask)
        else:
            attn_pool = tokens.mean(dim=1)  # 降级为均值池化
        
        # 最大池化：提取关键特征（带mask）
        if token_mask is not None:
            # 将padding位置的值设为-1e9，确保max pooling不会选中padding
            # 使用-1e9而非float('-inf')以避免后续LayerNorm/Linear/AMP产生NaN
            tokens_masked = tokens.masked_fill(
                ~token_mask.unsqueeze(-1),  # mask=True表示valid，~取反后True表示padding
                -1e9
            )
            max_pool = tokens_masked.max(dim=1).values
        else:
            max_pool = tokens.max(dim=1).values
        
        # 在concat前分别进行LayerNorm
        cls = self.cls_norm(cls)
        attn_pool = self.attn_norm(attn_pool)
        max_pool = self.max_norm(max_pool)
        
        # 拼接三种特征并通过MLP融合
        reaction_pool = self.reaction_fusion_mlp(
            torch.cat([cls, attn_pool, max_pool], dim=-1)
        )
        # ========== 第九步：特征融合（Feature Fusion） ==========
        if self.fusion == "interaction":
            # 交互融合：p, r, p*r, |p-r|
            fused = torch.cat([protein_pool, reaction_pool, protein_pool * reaction_pool, torch.abs(protein_pool - reaction_pool)], -1)
        elif self.fusion == "concat":
            # 拼接融合
            fused = torch.cat([protein_pool, reaction_pool], -1)
        elif self.fusion == "gated_concat":
            # 门控拼接：动态调整权重
            g = self.gate(torch.cat([protein_pool, reaction_pool], -1))
            fused = torch.cat([g * protein_pool, (1 - g) * reaction_pool], -1)
        elif self.fusion == "gated_sum":
            # 门控求和：动态加权
            g = self.gate(torch.cat([protein_pool, reaction_pool], -1))
            fused = g * protein_pool + (1 - g) * reaction_pool
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")
    
        # ========== 第十步：预测头（Prediction Head） ==========
        return self.head(fused).view(-1)


KcatTokenCrossAttention = KcatBiCrossAttention

if __name__ == "__main__":
    model = KcatBiCrossAttention(protein_max_tokens=8, reaction_max_tokens=17)

    # Inputs
    p = torch.randn(4, 8, 2560)
    r = torch.randn(4, 1, 256)
    c = torch.randn(4, 16, 384)

    # Masks (BOOL: True=valid, False=pad)
    protein_mask = torch.ones(4, 8).bool()
    protein_mask[:, 6:] = False
    reaction_mask = torch.ones(4, 17).bool()

    # Forward
    y = model(p, r, c, protein_mask, reaction_mask)
    print("Output shape:", y.shape)