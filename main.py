import time

import torch
import lightning as L

from data import CharTokenizer, load_text, build_dataloaders
from gpt import GPT
from trainer import LitGPT

SEQ_LEN = 256
BATCH_SIZE = 64

D_MODEL = 256
N_HEAD = 8          # d_head = 256 / 8 = 32
D_FFN = D_MODEL * 4
N_LAYER = 4
DROPOUT = 0.1

LR = 1e-3
MAX_STEPS = 3000


def main():
    L.seed_everything(42)

    text = load_text()
    tokenizer = CharTokenizer(text)
    train_loader, val_loader = build_dataloaders(text, tokenizer, SEQ_LEN, BATCH_SIZE)

    model = GPT(tokenizer.vocab_size, D_MODEL, N_HEAD, D_FFN, N_LAYER, SEQ_LEN, DROPOUT)
    lit_model = LitGPT(model, LR)

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_steps=MAX_STEPS,
        val_check_interval=500,   # 500 step마다 val 실행
        gradient_clip_val=1.0,    # gradient 폭주 방지
        log_every_n_steps=50,
    )

    start = time.time()
    trainer.fit(lit_model, train_loader, val_loader)
    print(f"학습 시간: {(time.time() - start) / 60:.1f}분")
    if torch.cuda.is_available():
        print(f"최대 GPU 메모리: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":  # num_workers > 0 일 때 macOS/Windows(spawn)에서 필수
    main()
