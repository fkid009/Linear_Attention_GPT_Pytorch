import torch
import torch.nn as nn

from linear_attention import CausalLinearAttention


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(d_model, d_ffn)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.output = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(self.gelu(self.input(x)))
        return self.dropout2(self.output(x))


class GPTBlock(nn.Module):
    """
    x + Attention(LayerNorm(x)) -> x + FFN(LayerNorm(x))
    """
    def __init__(self, d_model, n_head, d_ffn, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalLinearAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ffn, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ffn, n_layer, max_seq_len, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.gpt_blocks = nn.ModuleList([
            GPTBlock(d_model, n_head, d_ffn, dropout) for _ in range(n_layer)
        ])

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        # x: (B, T) 정수 텐서
        B, T = x.shape
        assert T <= self.max_seq_len, f"입력 길이 {T}가 max_seq_len({self.max_seq_len})을 초과했습니다."

        pos = torch.arange(T, device=x.device)
        x = self.emb_dropout(self.tok_emb(x) + self.pos_emb(pos))

        for block in self.gpt_blocks:
            x = block(x)

        return self.head(self.ln(x))  # (B, T, VOCAB_SIZE)
