from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableLRPairLayer(nn.Module):
    def __init__(self, n_ligand: int, n_receptor: int, lr_mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", torch.as_tensor(lr_mask, dtype=torch.float32))
        self.weights = nn.Parameter(torch.randn(n_ligand, n_receptor) * 0.02)

    def forward(self, ligand_vec: torch.Tensor, mode: str = "masked") -> torch.Tensor:
        if mode == "masked":
            weights = self.weights * self.mask
        elif mode == "dense":
            weights = self.weights
        else:
            raise ValueError(f"Unsupported LR mode: {mode}")
        return torch.matmul(ligand_vec, weights)


class EfficientTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout),
        )
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = x.shape
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(batch_size, num_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        x = x + attn_out
        return x + self.ffn(self.norm2(x))


class ScFoundationCellModel(nn.Module):
    """Checkpoint-compatible LITD model with optional parameter-free hidden-state export."""

    def __init__(self, n_genes, rec_indices, tf_indices, lr_mask, input_embed_dim=512,
                 hidden_dim=128, n_heads=4, tf_depth=2, gene_depth=2,
                 mlp_ratio=2.0, dropout=0.2, input_mode="scfoundation", lr_mode="masked"):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.input_mode = input_mode
        self.lr_mode = lr_mode
        self.register_buffer("rec_indices", torch.as_tensor(rec_indices, dtype=torch.long))
        self.register_buffer("tf_indices", torch.as_tensor(tf_indices, dtype=torch.long))
        self.input_proj = nn.Sequential(nn.Linear(input_embed_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                        nn.GELU(), nn.Dropout(dropout))
        self.expr_input_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim),
                                             nn.GELU(), nn.Dropout(dropout))
        self.lr_layer = LearnableLRPairLayer(lr_mask.shape[0], len(rec_indices), lr_mask)
        self.gamma_proj = nn.Linear(1, hidden_dim)
        self.beta_proj = nn.Linear(1, hidden_dim)
        self.rec_to_tf_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=dropout)
        self.tf_norm_cross = nn.LayerNorm(hidden_dim)
        self.tf_internal_layers = nn.ModuleList([
            EfficientTransformerBlock(hidden_dim, n_heads, mlp_ratio, dropout) for _ in range(tf_depth)
        ])
        self.tf_to_gene_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=dropout)
        self.gene_norm_cross = nn.LayerNorm(hidden_dim)
        self.gene_internal_layers = nn.ModuleList([
            EfficientTransformerBlock(hidden_dim, n_heads, mlp_ratio, dropout) for _ in range(gene_depth)
        ])
        self.shared_head_body = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2),
                                              nn.GELU(), nn.Dropout(dropout))
        final_dim = hidden_dim // 2
        self.gene_head_weight = nn.Parameter(torch.empty(n_genes, final_dim))
        self.gene_head_bias = nn.Parameter(torch.empty(n_genes))
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.gamma_proj.weight, mean=0, std=0.02)
        nn.init.normal_(self.beta_proj.weight, mean=0, std=0.02)
        nn.init.kaiming_uniform_(self.gene_head_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.gene_head_weight.size(1))
        nn.init.uniform_(self.gene_head_bias, -bound, bound)

    def forward(self, all_gene_embeddings, env_ligand_vec, return_grn=False,
                return_artifacts=False, return_hidden=False):
        raw_input = all_gene_embeddings
        if self.input_mode == "raw_expr":
            if all_gene_embeddings.dim() == 2:
                projected = self.expr_input_proj(all_gene_embeddings.unsqueeze(-1))
            elif all_gene_embeddings.dim() == 3 and all_gene_embeddings.size(-1) == 1:
                projected = self.expr_input_proj(all_gene_embeddings)
            else:
                raise ValueError(f"Expected raw expression input with shape [B, G], got {tuple(all_gene_embeddings.shape)}")
        else:
            if all_gene_embeddings.dim() == 2:
                all_gene_embeddings = all_gene_embeddings.unsqueeze(0).expand(env_ligand_vec.size(0), -1, -1)
            raw_input = all_gene_embeddings
            projected = self.input_proj(all_gene_embeddings)
        rec_embeddings = torch.index_select(projected, 1, self.rec_indices)
        tf_embeddings = torch.index_select(projected, 1, self.tf_indices)

        if self.lr_mode == "off":
            rec_act_score = torch.zeros(rec_embeddings.size(0), rec_embeddings.size(1),
                                        device=rec_embeddings.device, dtype=rec_embeddings.dtype)
            rec_features = rec_embeddings
        else:
            rec_act_score = self.lr_layer(env_ligand_vec, mode=self.lr_mode)
            film_input = rec_act_score.unsqueeze(-1)
            rec_features = self.gamma_proj(film_input) * rec_embeddings + self.beta_proj(film_input)

        need_weights = return_grn or return_artifacts
        tf_signal, rec_to_tf_weights = self.rec_to_tf_attn(
            query=tf_embeddings, key=rec_features, value=rec_features,
            need_weights=need_weights, average_attn_weights=True)
        x_tf = self.tf_norm_cross(tf_embeddings + tf_signal)
        x_tf_cross = x_tf
        tf_layers = []
        for layer in self.tf_internal_layers:
            x_tf = layer(x_tf)
            if return_hidden:
                tf_layers.append(x_tf)

        gene_signal, grn_matrix = self.tf_to_gene_attn(
            query=projected, key=x_tf, value=x_tf,
            need_weights=need_weights, average_attn_weights=True)
        x_gene = self.gene_norm_cross(projected + gene_signal)
        x_gene_cross = x_gene
        gene_layers = []
        for layer in self.gene_internal_layers:
            x_gene = layer(x_gene)
            if return_hidden:
                gene_layers.append(x_gene)

        x_hidden = self.shared_head_body(x_gene)
        out = (x_hidden * self.gene_head_weight.unsqueeze(0)).sum(dim=-1) + self.gene_head_bias

        if return_hidden:
            return out, {
                "raw_gene": raw_input,
                "projected_gene": projected,
                "rec_pre_film": rec_embeddings,
                "rec_post_film": rec_features,
                "rec_act_score": rec_act_score,
                "tf_cross": x_tf_cross,
                "tf_layers": tf_layers,
                "tf_final": x_tf,
                "gene_cross": x_gene_cross,
                "gene_layers": gene_layers,
                "gene_final": x_gene,
            }
        if return_artifacts:
            return out, {"rec_act_score": rec_act_score, "rec_to_tf_attn": rec_to_tf_weights,
                         "tf_to_gene_attn": grn_matrix}
        if return_grn:
            return out, grn_matrix
        return out
