import os
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # string to int
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # int to string

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class LLMDataset(Dataset):
    # __len__: 샘플이 총 몇 개인지
    # __getitem__(idx): idx번째 샘플이 무엇인지 return
    # DataLoader는 무작위 idx를 batch_size만큼 뽑아 __getitem__을 호출하고, 결과를 쌓아 (batch_size, seq_len) 배치를 만듦
    def __init__(self, data, seq_len, stride=1):
        self.data = data
        self.seq_len = seq_len
        self.stride = stride  # 다음 샘플의 시작 위치를 몇 칸 옮길지. Ex. abcd -> x = abc, y = bcd

    def __len__(self):
        return (len(self.data) - self.seq_len - 1) // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.data[start: start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def load_text(path="shakespeare.txt"):
    if not os.path.exists(path):
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_dataloaders(text, tokenizer, seq_len, batch_size, train_ratio=0.9, num_workers=2):
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(len(data) * train_ratio)

    train_ds = LLMDataset(data[:n], seq_len, stride=1)
    val_ds = LLMDataset(data[n:], seq_len, stride=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
