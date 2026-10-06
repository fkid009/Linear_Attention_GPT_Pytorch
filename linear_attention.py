import torch.nn as nn
from torch.nn import functional as F


class CausalLinearAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, eps=1e-6):
        super().__init__()
        assert d_model % n_head == 0, "d_model은 n_head로 나누어 떨어져야 한다."

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.eps = eps

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_dropout = nn.Dropout(dropout)

    def feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        B, T, _ = x.shape

        # multi-head
        q = self.W_Q(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # feature map
        phi_q = self.feature_map(q)
        phi_k = self.feature_map(k)

        # phi(k_j) @ v_j^T
        # ponytail: (B, H, T, D, D)를 통째로 만들어 T < D^2 이면 softmax보다 메모리를 더 씀. 필요하면 chunkwise로 교체
        kv = phi_k.unsqueeze(-1) @ v.unsqueeze(-2)  # (B, H, T, D, 1) @ (B, H, T, 1, D) -> (B, H, T, D, D)

        # cumulative sum = causal
        S = kv.cumsum(dim=2)
        Z = phi_k.cumsum(dim=2)

        num = (phi_q.unsqueeze(-2) @ S).squeeze(-2)      # (B, H, T, 1, D) @ (B, H, T, D, D) -> (B, H, T, D)
        den = (phi_q * Z).sum(dim=-1, keepdim=True)      # (B, H, T, 1)
        out = num / (den + self.eps)                     # (B, H, T, D)

        # head 병합
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, D_MODEL)
        out = self.out_proj(out)
        return self.out_dropout(out)
